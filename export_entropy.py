"""
Predict conllu files given a trained model
"""

import os
import shutil
import logging
import argparse
import tarfile
from pathlib import Path
import json
import numpy as np
import torch
#from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from tsnecuda import TSNE
sns.set()

from itertools import combinations, product
from collections import defaultdict

from allennlp.common import Params
from allennlp.common.util import import_submodules
from allennlp.common.checks import check_for_gpu
from allennlp.models.archival import load_archive

from src import util
from src.util import flatten
from src.predictors.predictor import Predictor


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def cos_sim(e1, e2):
    return np.sum(e1 * e2, axis=1) / (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1))

a = np.array([[0.3, 0.7], [0.4, 0.6]])
b = np.array([[1.2, 0.8], [0.1, 0.2]])
np.testing.assert_allclose(cos_sim(a, b), np.array([0.83761059683, 0.99227787671]))
np.testing.assert_allclose(np.linalg.norm(a - b, ord=2, axis=1),
                           np.array([0.90553851381, 0.5]))

def get_embed_diff_stats(lang_embeddings, langs):
    stats = defaultdict(dict)
    for (e1, lang1), (e2, lang2) in combinations(zip(lang_embeddings, langs), 2):
        l2 = np.linalg.norm(e1 - e2, ord=2, axis=1)
        l2_mean = np.mean(l2)
        l2_std = np.std(l2)
        key = '_'.join(sorted([lang1, lang2]))
        stats['l2_mean'][key] = l2_mean
        stats['l2_std'][key] = l2_std
        cos = cos_sim(e1, e2)
        cos_mean = np.mean(cos)
        cos_std = np.std(cos)
        stats['cos_mean'][key] = cos_mean
        stats['cos_std'][key] = cos_std
    return stats

parser = argparse.ArgumentParser()
parser.add_argument("archive", type=str, help="The archive file")
parser.add_argument("--include-package", type=str, help="The included package.")
parser.add_argument("--layer", type=str, choices=["embedding", "vib", "encoder", "projection", "adjacency"],
                    default="vib", help="Layer to inspect")
parser.add_argument("--langs", nargs='+', default=["en", "fr", "zh", "ar"], help="languages")
parser.add_argument("--cuda-device", default=0, type=int, help="CUDA device number; set to -1 for CPU")
parser.add_argument("--batch-size", default=1, type=int, help="The size of each prediction batch")
parser.add_argument("--first-n", default=-1, type=int, help="first n embeddings.")
parser.add_argument("--lazy", action="store_true", help="Lazy load dataset")
parser.add_argument("--overwrite", action="store_true", help="overwrite existing .npy file.")
parser.add_argument("--tsne", action="store_true", help="perform tsne visualization.")
parser.add_argument("--raw-text", action="store_true", help="Input raw sentences, one per line in the input file.")
parser.add_argument("--split", help='which dataset split to export')

args = parser.parse_args()

import_submodules(args.include_package)

archive_dir = Path(args.archive).resolve().parent

print(archive_dir)

selected_langs = args.langs
selected_tags = ["VERB", "NOUN", "ADJ", "ADV", "ADP"]
selected_locs = [0, 1, 2, 3, 4, 5]
ud_root = Path(os.environ['UD_GT'])
paths = list(ud_root.rglob(f"*-ud-{args.split}.conllu"))
paths = list(filter(lambda x: x.name.split('_')[0] in selected_langs, paths))

config_file = archive_dir / "config.json"
file_params = Params.from_file(config_file)
overrides = {"dataset_readers": {},
             "validation_dataset_readers": {},
             "dataset_reader": {
                 "type": "ud_multilang",
                 "languages": [path.name.split("_")[0] for path in paths],
                 "alternate": False,
                 "instances_per_file": 32,
                 "is_first_pass_for_vocab": False,
                 "lazy": True,
                 "read_language": False,
                 "read_dependencies": True,
                 "use_language_specific_deprel": False,
                 "deprel_file": "data/vocabulary/head_tags.txt",
                 "token_indexers": {
                     "bert": {
                         "type": "transformer_pretrained_mismatched",
                         "model_name": "bert-base-multilingual-cased",
                         "max_length": 512,
                     }
                 },
             },
             "model": {"inspect_layer": args.layer},
             "trainer": {"cuda_device": args.cuda_device}}

#try:
#    if os.environ["SHIFT"] == "1":
#        overrides['model']["ft_lang_mean_dir"] = f"ckpts/{os.environ['FT_LANG']}_mean"
#except KeyError:
#    pass

configs = [Params(overrides), file_params]
params = util.merge_configs(configs)
predictor_name = "ud_predictor"
assert not args.raw_text, "currently support only conllu input"

def maybe_first_n(predictions):
    if args.first_n > 0:
        return predictions[args.first_n]
    else:
        return predictions

def get_predictor(predictor_name: str, params: Params, archive: str):
    cuda_device = params["trainer"]["cuda_device"]

    check_for_gpu(cuda_device)
    archive = load_archive(archive,
                           cuda_device=cuda_device,
                           overrides=json.dumps(params.as_dict()))

    predictor = Predictor.from_archive(archive, predictor_name)
    return predictor

archive_path = Path(args.archive)
archive_dir = archive_path.parent.joinpath(archive_path.stem.split(".")[0])
archive_dir.mkdir(exist_ok=True)

npy_files = [archive_dir.joinpath(path.with_suffix(f".{args.layer}.npy").name) for path in paths]
if not all(map(lambda path: path.exists(), npy_files)) or args.overwrite:
    predictor = get_predictor(predictor_name, params, args.archive)
else:
    predictor = None

if args.layer == 'adjacency':
    all_outputs = []
    for idx, path in enumerate(paths):
        th_file = archive_dir.joinpath(path.with_suffix(f".{args.layer}.th").name)
        if (not os.path.exists(th_file)) or args.overwrite:
            manager = util._VisualizeManager(predictor,
                                             str(path),
                                             os.path.join(archive_dir, path.name),
                                             args.batch_size,
                                             print_to_console=False,
                                             has_dataset_reader=True)
            output_dict = manager._get_arc_and_tag_logits()
            torch.save(output_dict, str(th_file))
        else:
            output_dict = torch.load(th_file)

        all_outputs.append(output_dict)
