import glob
import argparse
import os
from pathlib import Path
from collections import defaultdict
from itertools import chain
import json
import shutil
import numpy as np

import stanfordnlp
from stanfordnlp.models.common.conll import CoNLLFile

np.random.seed(0)


def determine_treebank_factory(model_dir, tb_maps):
    def determine_treebank(tb):
        lang = None
        kwargs = {}
        lang = tb.split("_")[0]
        if tb == "th_pud":
            lang = None
        elif tb in tb_maps:
            tb = tb_maps[tb]
        elif tb == "fro_srcmf":
            kwargs['lemma_use_identity'] = True
        elif not model_dir.joinpath(f"{tb}_models").exists():
            tb = None
        return lang, tb, kwargs
    return determine_treebank

parser = argparse.ArgumentParser()
parser.add_argument('--langs')
parser.add_argument('--ud-root')
parser.add_argument('--model-dir')
parser.add_argument('--out-dir')
parser.add_argument('--split', choices=["traindev", "test"])
parser.add_argument('--num-splits', default=3)
parser.add_argument('--case', choices=["preprocess", "benchmark"])
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()

ud_root = Path(args.ud_root)
out_dir = Path(args.out_dir)
model_dir = Path(args.model_dir)
out_dir.mkdir(exist_ok=True)

tb_maps = {"br_keb": "ga_idt",
           "pcm_nsc": "en_ewt",
           "fo_oft": "no_nynorsk"}

determine_treebank = determine_treebank_factory(model_dir, tb_maps)

with open(args.langs) as fp:
    train_tbs = [line.split()[0] for line in fp.read().splitlines()]

def prep_conllu(tb, file_path, overwrite):
    out_file = out_dir.joinpath(file_path.name)
    if out_file.exists() and not overwrite:
        print(f"{out_file.name} exists; skipping")
        doc = stanfordnlp.Document('')
        doc.conll_file = CoNLLFile(out_file)
        return doc
    lang, tb, tb_kwargs = determine_treebank(tb)
    if not lang:
        shutil.copy(file_path, out_file)
        return None
    doc = stanfordnlp.Document('')
    doc.conll_file = CoNLLFile(file_path)
    nlp = stanfordnlp.Pipeline(lang=lang, treebank=tb,
                               processors='tokenize,pos',
                               tokenize_pretokenized=True)
    nlp.processors['pos'].process(doc)
    doc.load_annotations()
    return doc

if args.split == "traindev":
    for tb in train_tbs:
        # prepare dev.conllu if available
        has_dev = True
        conll_glob = list(ud_root.glob(f"*/{tb}-ud-dev.conllu"))
        if not conll_glob:
            has_dev = False
        else:
            assert len(conll_glob) == 1
            file_path = conll_glob[0]
            doc = prep_conllu(tb, file_path, args.overwrite)
            if doc is not None:
                out_file = out_dir.joinpath(file_path.name)
                doc.write_conll_to_file(str(out_file))

        # prepare train.conllu and dev.conllu (if has_dev == False)
        conll_glob = list(ud_root.glob(f"*/{tb}-ud-train.conllu"))
        assert len(conll_glob) == 1
        file_path = conll_glob[0]
        doc = prep_conllu(tb, file_path, args.overwrite)
        if doc is not None:
            if has_dev: # if dev has been written -> write train
                out_file = out_dir.joinpath(file_path.name)
                doc.write_conll_to_file(str(out_file))
            else:
                sents = doc.conll_file.sents
                permutations = np.random.permutation(np.arange(len(sents))).tolist()
                divides = [int(n * len(sents) * (1 / args.num_splits)) for n in range(args.num_splits + 1)]
                for file_num, (start, end) in enumerate(zip(divides[:-1], divides[1:])):
                    train_sents = [sents[idx] for idx in chain(permutations[:start],permutations[end:])]
                    dev_sents = [sents[idx] for idx in permutations[start:end]]

                    train_name = file_path.name.split("-")[0] + f"-{file_num}-ud-train.conllu"
                    out_train_file = out_dir.joinpath(train_name)
                    out_train_file.touch()
                    dev_name = file_path.name.split("-")[0] + f"-{file_num}-ud-dev.conllu"
                    out_dev_file = out_dir.joinpath(dev_name)
                    out_dev_file.touch()

                    train_conll = CoNLLFile(str(out_train_file))
                    dev_conll = CoNLLFile(str(out_dev_file))

                    train_conll._sents = train_sents
                    dev_conll._sents = dev_sents

                    train_conll.write_conll(out_train_file)
                    dev_conll.write_conll(out_dev_file)

elif args.split == "test":
    kwargs = {}
    if args.case == "preprocess":
        kwargs['processors'] = 'tokenize,mwt,pos,lemma'
    txt_files = list(ud_root.glob("conll18-ud-test/*_*.txt"))
    assert len(txt_files) == 82, f"number of txt_files = {len(txt_files)}; should be 82"
    for txt_file in sorted(txt_files):
        out_file = out_dir.joinpath(txt_file.stem + "-ud-test.conllu")
        if out_file.exists() and not args.overwrite:
            print(f"{out_file.name} exists; skipping")
            continue
        lang = None
        tb = txt_file.name.split(".")[0]
        lang, tb, tb_kwargs = determine_treebank(tb)
        if not lang:
            shutil.copy(txt_file.parent.joinpath(f"{tb}-udpipe.conllu"), out_file)
            continue
        kwargs.update(tb_kwargs)
        with open(txt_file) as fp:
            doc = stanfordnlp.Document(fp.read())
            nlp = stanfordnlp.Pipeline(lang=lang, treebank=tb, **kwargs)
            annotated = nlp(doc)
            annotated.write_conll_to_file(str(out_file))
else:
    raise NotImplementedError

