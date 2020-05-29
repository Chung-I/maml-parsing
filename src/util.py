"""
A collection of handy utilities
"""

from typing import List, Iterator, Tuple, Dict, Any, Optional

import os
import glob
import json
import sys
import logging
import tarfile
import traceback
import numpy as np
from tqdm import tqdm
from itertools import chain

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common import Params
from allennlp.common.params import with_fallback
#from allennlp.commands.make_vocab import make_vocab_from_params
from allennlp.commands.predict import _PredictManager
from allennlp.common.checks import check_for_gpu
from allennlp.common.file_utils import cached_path
from allennlp.common.util import lazy_groups_of
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor, JsonDict
from allennlp.data import Instance

from src.data.dataset_readers.conll18_ud_eval import evaluate, load_conllu_file, UDError

VOCAB_CONFIG_PATH = "config/create_vocab.json"

logger = logging.getLogger(__name__)


flatten  = lambda l: [item for sublist in l for item in sublist]

def merge_configs(configs: List[Params]) -> Params:
    """
    Merges a list of configurations together, with items with duplicate keys closer to the front of the list
    overriding any keys of items closer to the rear.
    :param configs: a list of AllenNLP Params
    :return: a single merged Params object
    """
    while len(configs) > 1:
        overrides, config = configs[-2:]
        configs = configs[:-2]

        if "udify_replace" in overrides:
            replacements = [replace.split(".") for replace in overrides.pop("udify_replace")]
            for replace in replacements:
                obj = config
                try:
                    for key in replace[:-1]:
                        obj = obj[key]
                except KeyError:
                    raise ConfigurationError(f"Config does not have key {key}")
                obj.pop(replace[-1])

        configs.append(Params(with_fallback(preferred=overrides.params, fallback=config.params)))

    return configs[0]


#def cache_vocab(params: Params, vocab_config_path: str = None):
#    """
#    Caches the vocabulary given in the Params to the filesystem. Useful for large datasets that are run repeatedly.
#    :param params: the AllenNLP Params
#    :param vocab_config_path: an optional config path for constructing the vocab
#    """
#    if "vocabulary" not in params or "directory_path" not in params["vocabulary"]:
#        return
#
#    vocab_path = params["vocabulary"]["directory_path"]
#
#    if os.path.exists(vocab_path):
#        if os.listdir(vocab_path):
#            return
#
#        # Remove empty vocabulary directory to make AllenNLP happy
#        try:
#            os.rmdir(vocab_path)
#        except OSError:
#            pass
#
#    vocab_config_path = vocab_config_path if vocab_config_path else VOCAB_CONFIG_PATH
#
#    params = merge_configs([params, Params.from_file(vocab_config_path)])
#    params["vocabulary"].pop("directory_path", None)
#    make_vocab_from_params(params, os.path.split(vocab_path)[0])


def get_ud_treebank_files(dataset_dir: str, treebanks: List[str] = None) -> Dict[str, Tuple[str, str, str]]:
    """
    Retrieves all treebank data paths in the given directory.
    :param dataset_dir: the directory where all treebank directories are stored
    :param treebanks: if not None or empty, retrieve just the subset of treebanks listed here
    :return: a dictionary mapping a treebank name to a list of train, dev, and test conllu files
    """
    datasets = {}
    treebanks = os.listdir(dataset_dir) if not treebanks else treebanks
    for treebank in treebanks:
        treebank_path = os.path.join(dataset_dir, treebank)
        conllu_files = [file for file in sorted(os.listdir(treebank_path)) if file.endswith(".conllu")]

        train_file = [file for file in conllu_files if file.endswith("train.conllu")]
        dev_file = [file for file in conllu_files if file.endswith("dev.conllu")]
        test_file = [file for file in conllu_files if file.endswith("test.conllu")]

        train_file = os.path.join(treebank_path, train_file[0]) if train_file else None
        dev_file = os.path.join(treebank_path, dev_file[0]) if dev_file else None
        test_file = os.path.join(treebank_path, test_file[0]) if test_file else None

        datasets[treebank] = (train_file, dev_file, test_file)
    return datasets


def get_ud_treebank_names(dataset_dir: str) -> List[Tuple[str, str]]:
    """
    Retrieves all treebank names from the given directory.
    :param dataset_dir: the directory where all treebank directories are stored
    :return: a list of long and short treebank names
    """
    treebanks = os.listdir(dataset_dir)
    short_names = []

    for treebank in treebanks:
        treebank_path = os.path.join(dataset_dir, treebank)
        conllu_files = [file for file in sorted(os.listdir(treebank_path)) if file.endswith(".conllu")]

        test_file = [file for file in conllu_files if file.endswith("test.conllu")]
        test_file = test_file[0].split("-")[0] if test_file else None

        short_names.append(test_file)

    treebanks = ["_".join(treebank.split("_")[1:]) for treebank in treebanks]

    return list(zip(treebanks, short_names))


def predict_model_with_archive(predictor: str, params: Params, archive: str,
                               input_file: str, output_file: str, batch_size: int = 1):
    cuda_device = params["trainer"]["cuda_device"]

    check_for_gpu(cuda_device)
    archive = load_archive(archive,
                           cuda_device=cuda_device,
                           overrides=json.dumps(params.as_dict()))

    predictor = Predictor.from_archive(archive, predictor)

    manager = _PredictManager(predictor,
                              input_file,
                              output_file,
                              batch_size,
                              print_to_console=False,
                              has_dataset_reader=True)
    manager.run()


def predict_and_evaluate_model_with_archive(predictor: str, params: Params, archive: str, gold_file: str,
                               pred_file: str, output_file: str, segment_file: str = None, batch_size: int = 1):
    if not gold_file or not os.path.isfile(gold_file):
        logger.warning(f"No file exists for {gold_file}")
        return

    segment_file = segment_file if segment_file else gold_file
    predict_model_with_archive(predictor, params, archive, segment_file, pred_file, batch_size)

    try:
        evaluation = evaluate(load_conllu_file(gold_file), load_conllu_file(pred_file))
        save_metrics(evaluation, output_file)
    except UDError:
        logger.warning(f"Failed to evaluate {pred_file}")
        traceback.print_exc()


def predict_model(predictor: str, params: Params, archive_dir: str,
                  input_file: str, output_file: str, batch_size: int = 1):
    """
    Predict output annotations from the given model and input file and produce an output file.
    :param predictor: the type of predictor to use, e.g., "udify_predictor"
    :param params: the Params of the model
    :param archive_dir: the saved model archive
    :param input_file: the input file to predict
    :param output_file: the output file to save
    :param batch_size: the batch size, set this higher to speed up GPU inference
    """
    archive = os.path.join(archive_dir, "model.tar.gz")
    predict_model_with_archive(predictor, params, archive, input_file, output_file, batch_size)


def predict_and_evaluate_model(predictor: str, params: Params, archive_dir: str, gold_file: str,
                               pred_file: str, output_file: str, segment_file: str = None, batch_size: int = 1):
    """
    Predict output annotations from the given model and input file and evaluate the model.
    :param predictor: the type of predictor to use, e.g., "udify_predictor"
    :param params: the Params of the model
    :param archive_dir: the saved model archive
    :param gold_file: the file with gold annotations
    :param pred_file: the input file to predict
    :param output_file: the output file to save
    :param segment_file: an optional file separate gold file that can be evaluated,
    useful if it has alternate segmentation
    :param batch_size: the batch size, set this higher to speed up GPU inference
    """
    archive = os.path.join(archive_dir, "model.tar.gz")
    predict_and_evaluate_model_with_archive(predictor, params, archive, gold_file,
                                            pred_file, output_file, segment_file, batch_size)


def save_metrics(evaluation: Dict[str, Any], output_file: str):
    """
    Saves CoNLL 2018 evaluation as a JSON file.
    :param evaluation: the evaluation dict calculated by the CoNLL 2018 evaluation script
    :param output_file: the output file to save
    """
    evaluation_dict = {k: v.__dict__ for k, v in evaluation.items()}

    with open(output_file, "w") as f:
        json.dump(evaluation_dict, f, indent=4)

    logger.info("Metric     | Correct   |      Gold | Predicted | Aligned")
    logger.info("-----------+-----------+-----------+-----------+-----------")
    for metric in ["Tokens", "Sentences", "Words", "UPOS", "XPOS", "UFeats",
                   "AllTags", "Lemmas", "UAS", "LAS", "CLAS", "MLAS", "BLEX"]:
        logger.info("{:11}|{:10.2f} |{:10.2f} |{:10.2f} |{}".format(
                    metric,
                    100 * evaluation[metric].precision,
                    100 * evaluation[metric].recall,
                    100 * evaluation[metric].f1,
                    "{:10.2f}".format(100 * evaluation[metric].aligned_accuracy)
                    if evaluation[metric].aligned_accuracy is not None else ""))


def cleanup_training(serialization_dir: str, keep_archive: bool = False, keep_weights: bool = False):
    """
    Removes files generated from training.
    :param serialization_dir: the directory to clean
    :param keep_archive: whether to keep a copy of the model archive
    :param keep_weights: whether to keep copies of the intermediate model checkpoints
    """
    if not keep_weights:
        for file in glob.glob(os.path.join(serialization_dir, "*.th")):
            os.remove(file)
    if not keep_archive:
        os.remove(os.path.join(serialization_dir, "model.tar.gz"))


def archive_bert_model(serialization_dir: str, config_file: str, output_file: str = None):
    """
    Extracts BERT parameters from the given model and saves them to an archive.
    :param serialization_dir: the directory containing the saved model archive
    :param config_file: the configuration file of the model archive
    :param output_file: the output BERT archive name to save
    """
    archive = load_archive(os.path.join(serialization_dir, "model.tar.gz"))

    model = archive.model
    model.eval()

    try:
        bert_model = model.text_field_embedder.token_embedder_bert.model
    except AttributeError:
        logger.warning(f"Could not find the BERT model inside the archive {serialization_dir}")
        traceback.print_exc()
        return

    weights_file = os.path.join(serialization_dir, "pytorch_model.bin")
    torch.save(bert_model.state_dict(), weights_file)

    if not output_file:
        output_file = os.path.join(serialization_dir, "bert-finetune.tar.gz")

    with tarfile.open(output_file, 'w:gz') as archive:
        archive.add(config_file, arcname="bert_config.json")
        archive.add(weights_file, arcname="pytorch_model.bin")

    os.remove(weights_file)


class _VisualizeManager:

    def __init__(self,
                 predictor: Predictor,
                 input_file: str,
                 output_file: Optional[str],
                 batch_size: int,
                 print_to_console: bool,
                 has_dataset_reader: bool) -> None:

        self._predictor = predictor
        self._input_file = input_file
        if output_file is not None:
            self._output_file = open(output_file, "w")
        else:
            self._output_file = None
        self._batch_size = batch_size
        self._print_to_console = print_to_console
        if has_dataset_reader:
            self._dataset_reader = predictor._dataset_reader # pylint: disable=protected-access
        else:
            self._dataset_reader = None

    def _predict_json(self, batch_data: List[JsonDict]) -> Iterator[str]:
        if len(batch_data) == 1:
            results = [self._predictor.predict_json(batch_data[0])]
        else:
            results = self._predictor.predict_batch_json(batch_data)
        for output in results:
            yield self._predictor.dump_line(output)

    def _predict_instances(self, batch_data: List[Instance]) -> Iterator[str]:
        if len(batch_data) == 1:
            results = [self._predictor.predict_instance(batch_data[0])]
        else:
            results = self._predictor.predict_batch_instance(batch_data)
        for output in results:
            yield self._predictor.dump_line(output)

    def _maybe_print_to_console_and_file(self,
                                         index: int,
                                         prediction: str,
                                         model_input: str = None) -> None:
        if self._print_to_console:
            if model_input is not None:
                print(f"input {index}: ", model_input)
            print("prediction: ", prediction)
        if self._output_file is not None:
            self._output_file.write(prediction)

    def _get_json_data(self) -> Iterator[JsonDict]:
        if self._input_file == "-":
            for line in sys.stdin:
                if not line.isspace():
                    yield self._predictor.load_line(line)
        else:
            input_file = cached_path(self._input_file)
            with open(input_file, "r") as file_input:
                for line in file_input:
                    if not line.isspace():
                        yield self._predictor.load_line(line)

    def _get_instance_data(self) -> Iterator[Instance]:
        if self._input_file == "-":
            raise ConfigurationError("stdin is not an option when using a DatasetReader.")
        elif self._dataset_reader is None:
            raise ConfigurationError("To generate instances directly, pass a DatasetReader.")
        elif isinstance(self._input_file, list):
            for f in self._input_file:
                yield from self._dataset_reader.read(f)
        else:
            yield from self._dataset_reader.read(self._input_file)

    def _get_embeddings(self):
        has_reader = self._dataset_reader is not None
        index = 0
        embeddings = []
        def get_sentence_embedding(result):
            length = len(result['words'])
            hidden_state = np.array(result['hidden_state'][:length])
            mean_pooled_state = np.mean(hidden_state, axis=0)
            return mean_pooled_state
        for batch_data in tqdm(lazy_groups_of(self._get_instance_data(), self._batch_size)):
            if len(batch_data) == 1:
                results = [self._predictor.predict_instance(batch_data[0])]
            else:
                results = self._predictor.predict_batch_instance(batch_data)
            for input_instance, output in zip(batch_data, results):
                result =  self._predictor.dump_line(output)
                self._maybe_print_to_console_and_file(index, result, str(input_instance))
                index = index + 1
            embeddings += list(map(get_sentence_embedding, results))
        embeddings = np.stack(embeddings)
        return embeddings

    def _get_lang_mean(self):
        has_reader = self._dataset_reader is not None
        index = 0
        states = []
        def get_sentence_embedding(result):
            length = len(result['words'])
            hidden_state = np.array(result['hidden_state'][:length])
            states.append(torch.from_numpy(hidden_state).float())
        for batch_data in tqdm(lazy_groups_of(self._get_instance_data(), self._batch_size)):
            if len(batch_data) == 1:
                results = [self._predictor.predict_instance(batch_data[0])]
            else:
                results = self._predictor.predict_batch_instance(batch_data)
            for input_instance, output in zip(batch_data, results):
                result =  self._predictor.dump_line(output)
                self._maybe_print_to_console_and_file(index, result, str(input_instance))
                index = index + 1
            list(map(get_sentence_embedding, results))
        all_states = torch.cat(states, dim=0)
        return torch.mean(all_states, dim=0)

    def _get_word_embeddings(self):
        has_reader = self._dataset_reader is not None
        index = 0
        all_embeddings = []
        all_pos_tags = []
        all_positions = []
        def get_sentence_embedding(result):
            length = len(result['words'])
            *hidden_states, = np.array(result['hidden_state'][:length])
            tags = result["upos"][:length]
            positions = list(range(len(tags)))
            return hidden_states, tags, positions
        for batch_data in tqdm(lazy_groups_of(self._get_instance_data(), self._batch_size)):
            if len(batch_data) == 1:
                results = [self._predictor.predict_instance(batch_data[0])]
            else:
                results = self._predictor.predict_batch_instance(batch_data)
            for input_instance, output in zip(batch_data, results):
                result =  self._predictor.dump_line(output)
                self._maybe_print_to_console_and_file(index, result, str(input_instance))
                index = index + 1
            embeddings, pos_tags, positions = zip(*map(get_sentence_embedding, results))
            all_embeddings += flatten(embeddings)
            all_pos_tags += flatten(pos_tags)
            all_positions += flatten(positions)
        return all_embeddings, all_pos_tags, all_positions

    def _get_arc_and_tag_representations(self):
        has_reader = self._dataset_reader is not None
        index = 0
        all_results = []
        def get_sentence_embedding(result):
            length = len(result['words'])
            *arcs, = np.array(result['arc_hidden_state'][:length])
            *tags, = np.array(result['tag_hidden_state'][:length])
            heads = result["gold_heads"]
            deprels = result["gold_tags"]
            poss = result["upos"][:length]
            predicted_deprels = result["predicted_dependencies"]
            predicted_heads = result["predicted_heads"]
            langs = [result["langs"] for _ in range(length)]
            positions = list(range(len(poss)))
            return {"arcs": arcs, "tags": tags, "poss": poss,
                    "deprels": deprels, "heads": heads,
                    "pred_deprels": predicted_deprels, "pred_heads": predicted_heads,
                    "langs": langs, "positions": positions}
        for batch_data in tqdm(lazy_groups_of(self._get_instance_data(), self._batch_size)):
            if len(batch_data) == 1:
                results = [self._predictor.predict_instance(batch_data[0])]
            else:
                results = self._predictor.predict_batch_instance(batch_data)
            for input_instance, output in zip(batch_data, results):
                result =  self._predictor.dump_line(output)
                self._maybe_print_to_console_and_file(index, result, str(input_instance))
                index = index + 1
            proc_results = list(map(get_sentence_embedding, results))
            batched_proc_results = {key: flatten([result[key] for result in proc_results])
                                    for key in proc_results[0].keys()}
            all_results.append(batched_proc_results)
        all_batched_results = {key: flatten([result[key] for result in all_results])
                               for key in all_results[0].keys()}
        return all_batched_results

    def run(self) -> None:
        has_reader = self._dataset_reader is not None
        index = 0
        if has_reader:
            for batch in lazy_groups_of(self._get_instance_data(), self._batch_size):
                for model_input_instance, result in zip(batch, self._predict_instances(batch)):
                    self._maybe_print_to_console_and_file(index, result, str(model_input_instance))
                    index = index + 1
        else:
            for batch_json in lazy_groups_of(self._get_json_data(), self._batch_size):
                for model_input_json, result in zip(batch_json, self._predict_json(batch_json)):
                    self._maybe_print_to_console_and_file(index, result, json.dumps(model_input_json))
                    index = index + 1

        if self._output_file is not None:
            self._output_file.close()
