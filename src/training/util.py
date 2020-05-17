"""
Helper functions for Trainers
"""
import torch.distributed as dist
from typing import Any, Union, Dict, Iterable, List, Optional
from collections import OrderedDict
import datetime
import logging
import os
import re
import shutil
from functools import partial
from pathlib import Path

import torch

from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common.params import Params
from allennlp.common.tqdm import Tqdm
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance, Vocabulary
from allennlp.data.iterators import DataIterator
from allennlp.models.model import Model
from allennlp.models.archival import CONFIG_NAME
from allennlp.nn import util as nn_util

logger = logging.getLogger(__name__)

INF = 1e-10

# We want to warn people that tqdm ignores metrics that start with underscores
# exactly once. This variable keeps track of whether we have.
class HasBeenWarned:
    tqdm_ignores_underscores = False

flatten  = lambda l: [item for sublist in l for item in sublist]

def clone(tensor):
    """Detach and clone a tensor including the ``requires_grad`` attribute.
    Arguments:
        tensor (torch.Tensor): tensor to clone.
    """
    cloned = tensor.detach().clone()
    cloned.requires_grad = tensor.requires_grad
    if tensor.grad is not None:
        cloned.grad = clone(tensor.grad)
    return cloned


def clone_state_dict(state_dict):
    """Clone a state_dict. If state_dict is from a ``torch.nn.Module``, use ``keep_vars=True``.
    Arguments:
        state_dict (OrderedDict): the state_dict to clone. Assumes state_dict is not detached from model state.
    """
    return OrderedDict([(name, clone(param)) for name, param in state_dict.items()])


def read_all_datasets(
    train_data_path: str,
    dataset_reader: DatasetReader,
    validation_dataset_reader: DatasetReader = None,
    validation_data_path: str = None,
    test_data_path: str = None,
) -> Dict[str, Iterable[Instance]]:
    """
    Reads all datasets (perhaps lazily, if the corresponding dataset readers are lazy) and returns a
    dictionary mapping dataset name ("train", "validation" or "test") to the iterable resulting from
    `reader.read(filename)`.
    """

    logger.info("Reading training data from %s", train_data_path)
    train_data = dataset_reader.read(train_data_path)

    datasets: Dict[str, Iterable[Instance]] = {"train": train_data}

    validation_dataset_reader = validation_dataset_reader or dataset_reader

    if validation_data_path is not None:
        logger.info("Reading validation data from %s", validation_data_path)
        validation_data = validation_dataset_reader.read(validation_data_path)
        datasets["validation"] = validation_data

    if test_data_path is not None:
        logger.info("Reading test data from %s", test_data_path)
        test_data = validation_dataset_reader.read(test_data_path)
        datasets["test"] = test_data

    return datasets


def datasets_from_params(params: Params) -> Dict[str, Dict[str, Iterable[Instance]]]:
    """
    Load all the datasets specified by the config.

    # Parameters

    params : `Params`
    cache_directory : `str`, optional
        If given, we will instruct the `DatasetReaders` that we construct to cache their
        instances in this location (or read their instances from caches in this location, if a
        suitable cache already exists).  This is essentially a `base` directory for the cache, as
        we will additionally add the `cache_prefix` to this directory, giving an actual cache
        location of `cache_directory + cache_prefix`.
    cache_prefix : `str`, optional
        This works in conjunction with the `cache_directory`.  The idea is that the
        `cache_directory` contains caches for all different parameter settings, while the
        `cache_prefix` captures a specific set of parameters that led to a particular cache file.
        That is, if you change the tokenization settings inside your `DatasetReader`, you don't
        want to read cached data that used the old settings.  In order to avoid this, we compute a
        hash of the parameters used to construct each `DatasetReader` and use that as a "prefix"
        to the cache files inside the base `cache_directory`.  So, a given `input_file` would
        be cached essentially as `cache_directory + cache_prefix + input_file`, where you specify
        a `cache_directory`, the `cache_prefix` is based on the dataset reader parameters, and
        the `input_file` is whatever path you provided to `DatasetReader.read()`.  In order to
        allow you to give recognizable names to these prefixes if you want them, you can manually
        specify the `cache_prefix`.  Note that in some rare cases this can be dangerous, as we'll
        use the `same` prefix for both train and validation dataset readers.
    """
    dataset_readers_params = params.pop("dataset_readers")
    validation_dataset_readers_params = params.pop("validation_dataset_readers", None)

    dataset_readers = {key: DatasetReader.from_params(dataset_reader_params)
        for key, dataset_reader_params in dataset_readers_params.items()}

    validation_and_test_dataset_readers: Dict[str, DatasetReader] = dataset_readers
    if validation_dataset_readers_params is not None:
        logger.info("Using a separate dataset reader to load validation and test data.")
        validation_and_test_dataset_readers = {key: DatasetReader.from_params(
            validation_dataset_reader_params
        ) for key, validation_dataset_reader_params in validation_dataset_readers_params.items()}

    train_data_paths = params.pop("train_data_paths")
    logger.info("Reading training data from %s", train_data_paths)
    train_datas = {key: dataset_reader.read(train_data_paths[key])
        for key, dataset_reader in dataset_readers.items()}

    datasets: Dict[str, Iterable[Iterable[Instance]]] = {"train": train_datas}

    validation_data_paths = params.pop("validation_data_paths", None)
    if validation_data_paths is not None:
        logger.info("Reading validation data from %s", validation_data_paths)
        validation_datas = {key: dataset_reader.read(validation_data_paths[key])
            for key, dataset_reader in validation_and_test_dataset_readers.items()}
        datasets["validation"] = validation_datas

    test_data_paths = params.pop("test_data_paths", None)
    if test_data_paths is not None:
        logger.info("Reading test data from %s", test_data_paths)
        test_datas = {key: dataset_reader.read(test_data_paths[key])
            for key, dataset_reader in validation_and_test_dataset_readers.items()}
        datasets["test"] = test_datas

    return datasets


def as_flat_dict(params):
    """
    Returns the parameters of a flat dictionary from keys to values.
    Nested structure is collapsed with periods.
    """
    flat_params = {}

    def recurse(parameters, path):
        for key, value in parameters.items():
            newpath = path + [key]
            if isinstance(value, dict):
                recurse(value, newpath)
            else:
                flat_params[".".join(newpath)] = value

    recurse(params, [])
    return flat_params


def get_tensors(obj):
    """
    Given a structure (possibly) containing Tensors on the CPU,
    move all the Tensors to the specified GPU (or do nothing, if they should be on the CPU).
    """

    tensors = []
    if not nn_util.has_tensor(obj):
        pass
    elif isinstance(obj, torch.Tensor):
        tensors += [obj]
    elif isinstance(obj, dict):
        tensors += flatten([get_tensors(value) for key, value in obj.items()])
    elif isinstance(obj, list):
        tensors += flatten([get_tensors(item) for item in obj])
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # This is the best way to detect a NamedTuple, it turns out.
        tensors += flatten([get_tensors(item) for item in obj])
    elif isinstance(obj, tuple):
        tensors += flatten([get_tensors(item) for item in obj])
    return tensors


def nan_hook(self, inp, output):
    from torch.nn.utils.rnn import PackedSequence
    outputs = get_tensors(output)

    for i, out in enumerate(outputs):
        if isinstance(out, PackedSequence):
            out = out.data
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            print("In", self.__class__.__name__)
            raise RuntimeError(f"Found NAN in output {i} at indices: ",
                               nan_mask.nonzero(),
                               "where:",
                               out[nan_mask.nonzero()[:, 0].unique(sorted=True)])


def filter_state_dict(state_dict, filter_func):
    return OrderedDict({k: v for k, v in state_dict.items() if filter_func(k, v)})


def move_to_device(obj, device: torch.device):
    """
    Given a structure (possibly) containing Tensors on the CPU,
    move all the Tensors to the specified GPU (or do nothing, if they should be on the CPU).
    """

    if not nn_util.has_tensor(obj):
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # This is the best way to detect a NamedTuple, it turns out.
        return obj.__class__(*(move_to_device(item, device) for item in obj))
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    else:
        return obj

def pad_batched_tensors(batched_tensors: List[torch.Tensor],
                        length_dim: int = 1):
    max_length = max(map(lambda x: x.size(length_dim), batched_tensors))
    def pad_to_len(tensor, max_len):
        new_shape = list(tensor.shape)
        new_shape[length_dim] = max_len
        new_tensor = tensor.new_zeros(new_shape)
        slicing_shape = list(tensor.shape)
        slices = tuple([slice(0, x) for x in slicing_shape])
        new_tensor[slices] = tensor
        return new_tensor

    return torch.cat(list(map(partial(pad_to_len, max_len=max_length), batched_tensors)), dim=0)

def get_lang_means(lang_mean_regex, vocab=None):
    ckpt_dirs = list(Path(".").glob(lang_mean_regex))
    lang_mean_list = [None] * vocab.get_vocab_size("lang_labels")
    for ckpt_dir in ckpt_dirs:
        lang_name = re.match("(.*)_mean", ckpt_dir.name).group(1)
        try:
            idx = vocab.get_token_index(lang_name, namespace="lang_labels")
        except KeyError:
            continue
        state_dict = torch.load(ckpt_dir.joinpath("model_state_epoch_1.th"))
        lang_mean_list[idx] = state_dict["mean"]

    exemplar = None
    for lang_mean in lang_mean_list:
        if lang_mean is not None:
            exemplar = lang_mean
            break

    lang_mean_list = list(map(
        lambda x: torch.zeros_like(exemplar) if x is None else x,
        lang_mean_list))

    lang_means = torch.stack(lang_mean_list)

    return lang_means

def get_lang_mean(lang_mean_dir):
    lang_mean_dir = Path(lang_mean_dir)
    lang_name = re.match("(.*)_mean", lang_mean_dir.name).group(1)
    state_dict = torch.load(lang_mean_dir.joinpath("model_state_epoch_1.th"))
    return state_dict["mean"]


def get_dir_mask(sent_lens):
    batch_size = len(sent_lens)
    max_sent_len = max(sent_lens)
    left_ids = torch.arange(max_sent_len).to(
        sent_lens.device).view(1, -1).expand(max_sent_len, -1)
    right_ids = torch.arange(max_sent_len).to(
        sent_lens.device).view(-1, 1).expand(-1, max_sent_len)
    left_mask = left_ids < right_ids
    right_mask = left_ids > right_ids
    length_mask = nn_util.get_mask_from_sequence_lengths(sent_lens, max_sent_len)
    mask = length_mask.unsqueeze(1) & length_mask.unsqueeze(2)
    left_mask = left_mask.unsqueeze(0).expand(batch_size, -1, -1) & mask
    right_mask = right_mask.unsqueeze(0).expand(batch_size, -1, -1) & mask
    return left_mask, right_mask


def avg_loss_func(loss, weights_batch_sum, average='batch'):
    if average == "batch":
        # shape : (batch_size,)
        per_batch_loss = loss / (weights_batch_sum + 1e-13)
        num_non_empty_sequences = (weights_batch_sum > 0).float().sum() + 1e-13
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return loss.sum() / (weights_batch_sum.sum() + 1e-13)
    else:
        # shape : (batch_size,)
        per_batch_loss = loss / (weights_batch_sum + 1e-13)
        return per_batch_loss


def memoize(func):
    mem = {}

    def helper(*args, **kwargs):
        key = (args, frozenset(list(kwargs.items())))
        if key not in mem:
            mem[key] = func(*args, **kwargs)
        return mem[key]

    return helper


@memoize
def constituent_index(sentence_length, multiroot):
    counter_id = 0
    basic_span = []
    id_2_span = {}
    for left_idx in range(sentence_length):
        for right_idx in range(left_idx, sentence_length):
            for dir in range(2):
                id_2_span[counter_id] = (left_idx, right_idx, dir)
                counter_id += 1

    span_2_id = {s: id for id, s in list(id_2_span.items())}

    for i in range(sentence_length):
        if i != 0:
            id = span_2_id.get((i, i, 0))
            basic_span.append(id)
        id = span_2_id.get((i, i, 1))
        basic_span.append(id)

    ijss = []
    ikcs = [[] for _ in range(counter_id)]
    ikis = [[] for _ in range(counter_id)]
    kjcs = [[] for _ in range(counter_id)]
    kjis = [[] for _ in range(counter_id)]

    for l in range(1, sentence_length):
        for i in range(sentence_length - l):
            j = i + l
            for dir in range(2):
                ids = span_2_id[(i, j, dir)]
                for k in range(i, j + 1):
                    if dir == 0:
                        if k < j:
                            # two complete spans to form an incomplete span
                            idli = span_2_id[(i, k, dir + 1)]
                            ikis[ids].append(idli)
                            idri = span_2_id[(k + 1, j, dir)]
                            kjis[ids].append(idri)
                            # one complete span,one incomplete span to form a complete span
                            idlc = span_2_id[(i, k, dir)]
                            ikcs[ids].append(idlc)
                            idrc = span_2_id[(k, j, dir)]
                            kjcs[ids].append(idrc)

                    else:
                        if k < j and ((not (i == 0 and k != 0) and not multiroot) or multiroot):
                            # two complete spans to form an incomplete span
                            idli = span_2_id[(i, k, dir)]
                            ikis[ids].append(idli)
                            idri = span_2_id[(k + 1, j, dir - 1)]
                            kjis[ids].append(idri)
                        if k > i:
                            # one incomplete span,one complete span to form a complete span
                            idlc = span_2_id[(i, k, dir)]
                            ikcs[ids].append(idlc)
                            idrc = span_2_id[(k, j, dir)]
                            kjcs[ids].append(idrc)

                ijss.append(ids)

    return span_2_id, id_2_span, ijss, ikcs, ikis, kjcs, kjis, basic_span


def _divmod(input, other):
    return input // other, input % other