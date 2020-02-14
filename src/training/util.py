"""
Helper functions for Trainers
"""
import torch.distributed as dist
from typing import Any, Union, Dict, Iterable, List, Optional
from collections import OrderedDict
import datetime
import logging
import os
import shutil

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


# We want to warn people that tqdm ignores metrics that start with underscores
# exactly once. This variable keeps track of whether we have.
class HasBeenWarned:
    tqdm_ignores_underscores = False


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


def datasets_from_params(params: Params) -> Dict[str, Iterable[Instance]]:
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
    dataset_reader_params = params.pop("dataset_reader")
    validation_dataset_reader_params = params.pop("validation_dataset_reader", None)

    dataset_reader = DatasetReader.from_params(dataset_reader_params)

    validation_and_test_dataset_reader: DatasetReader = dataset_reader
    if validation_dataset_reader_params is not None:
        logger.info("Using a separate dataset reader to load validation and test data.")
        validation_and_test_dataset_reader = DatasetReader.from_params(
            validation_dataset_reader_params
        )

    train_data_path = params.pop("train_data_path")
    logger.info("Reading training data from %s", train_data_path)
    train_data = dataset_reader.read(train_data_path)

    datasets: Dict[str, Iterable[Instance]] = {"train": train_data}

    validation_data_path = params.pop("validation_data_path", None)
    if validation_data_path is not None:
        logger.info("Reading validation data from %s", validation_data_path)
        validation_data = validation_and_test_dataset_reader.read(validation_data_path)
        datasets["validation"] = validation_data

    test_data_path = params.pop("test_data_path", None)
    if test_data_path is not None:
        logger.info("Reading test data from %s", test_data_path)
        test_data = validation_and_test_dataset_reader.read(test_data_path)
        datasets["test"] = test_data

    return datasets
