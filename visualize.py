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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict

from allennlp.common import Params
from allennlp.common.util import import_submodules
from allennlp.common.checks import check_for_gpu
from allennlp.models.archival import load_archive

from src import util
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
parser.add_argument("--cuda-device", default=0, type=int, help="CUDA device number; set to -1 for CPU")
parser.add_argument("--batch-size", default=1, type=int, help="The size of each prediction batch")
parser.add_argument("--first-n", default=-1, type=int, help="first n embeddings.")
parser.add_argument("--lazy", action="store_true", help="Lazy load dataset")
parser.add_argument("--overwrite", action="store_true", help="overwrite existing .npy file.")
parser.add_argument("--tsne", action="store_true", help="perform tsne visualization.")
parser.add_argument("--raw-text", action="store_true", help="Input raw sentences, one per line in the input file.")

args = parser.parse_args()

import_submodules(args.include_package)

archive_dir = Path(args.archive).resolve().parent

print(archive_dir)

ud_root = Path(os.environ['UD_ROOT'])
paths = list(ud_root.rglob("*pud-ud-test.conllu"))
paths = list(filter(lambda x: Path(f"ckpts/{x.name.split('_')[0]}_mean").exists(), paths))

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
                 "token_indexers": {
                     "roberta": {
                         "type": "transformer_pretrained_mismatched",
                         "model_name": "xlm-roberta-base",
                         "max_length": 512,
                     }
                 },
                 "use_language_specific_pos": False,
                 "read_language": True,
                 "read_dependencies": False,
             },
             "trainer": {"cuda_device": -1}}

#try:
#    if os.environ["SHIFT"] == "1":
#        overrides['model']["ft_lang_mean_dir"] = f"ckpts/{os.environ['FT_LANG']}_mean"
#except KeyError:
#    pass

configs = [Params(overrides), file_params]
params = util.merge_configs(configs)
predictor_name = "ud_predictor"
assert not args.raw_text, "currently support only conllu input"

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

npy_files = [archive_dir.joinpath(path.with_suffix('.npy').name) for path in paths]
if not all(map(lambda path: path.exists(), npy_files)) or args.overwrite:
    predictor = get_predictor(predictor_name, params, args.archive)
else:
    predictor = None

lang_embeddings = []
langs = []
lang_ids = []
for idx, path in enumerate(paths):
    npy_file = archive_dir.joinpath(path.with_suffix('.npy').name)
    if (not os.path.exists(npy_file)) or args.overwrite:
        manager = util._VisualizeManager(predictor,
                                         str(path),
                                         os.path.join(archive_dir, path.name),
                                         args.batch_size,
                                         print_to_console=False,
                                         has_dataset_reader=True)
        embeddings = manager._get_embeddings()
        np.save(npy_file, embeddings)
    else:
        embeddings = np.load(npy_file)

    embeddings = embeddings[:args.first_n] if args.first_n > 0 else embeddings
    lang_embeddings.append(embeddings)
    langs.append(path.name.split("_")[0])
    lang_ids.append(idx)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

stats = get_embed_diff_stats(lang_embeddings, langs)
with open(archive_dir.joinpath('stats.json'), 'w') as fp:
    json.dump(stats, fp)

if args.tsne:
    X = np.concatenate(lang_embeddings, axis=0)
    X_embedded = TSNE(n_components=2).fit_transform(X)
    start = 0
    for lang, lang_embedding, color in zip(langs, lang_embeddings, colors):
        end = start + lang_embedding.shape[0]
        plt.scatter(X_embedded[start:end,0],
                    X_embedded[start:end,1],
                    c=color,
                    label=lang)
        start = end
    
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc="center left", title="languages", bbox_to_anchor=(1, 0.5))
    ax.grid(True)
    plt.savefig(archive_dir.joinpath('tsne.png'))
