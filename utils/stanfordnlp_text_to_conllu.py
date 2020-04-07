import glob
import argparse
import os
from pathlib import Path
from collections import defaultdict
import itertools
import json
import shutil
import numpy as np

import stanza
from stanza.models.common.doc import Document
from stanza.utils.conll import CoNLL

np.random.seed(0)

def flatten(l):
    return [item for sublist in l for item in sublist]

def write_doc_to_file(doc, out_file):
    conll_string = CoNLL.conll_as_string(CoNLL.convert_dict(doc.to_dict()))
    with open(str(out_file), "w") as fp:
        fp.write(conll_string)

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
parser.add_argument('--lang-file')
parser.add_argument('--ud-root')
parser.add_argument('--model-dir')
parser.add_argument('--out-dir')
parser.add_argument('--split', choices=["traindev", "test"])
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

with open(args.lang_file) as fp:
    train_tbs = [line.split()[0] for line in fp.read().splitlines()]

def prep_conllu(tb, file_path, overwrite):
    out_file = out_dir.joinpath(file_path.name)
    if out_file.exists() and not overwrite:
        print(f"{out_file.name} exists; skipping")
        return None
    lang, tb, tb_kwargs = determine_treebank(tb)
    if not lang:
        shutil.copy(file_path, out_file)
        return None
    doc = Document(CoNLL.conll2dict(input_file=file_path))
    nlp = stanza.Pipeline(lang=lang,
                          processors='tokenize,mwt,pos',
                          tokenize_pretokenized=True)
    doc = nlp.processors['pos'].process(doc)
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
                write_doc_to_file(doc, out_file)

        # prepare train.conllu and dev.conllu (if has_dev == False)
        conll_glob = list(ud_root.glob(f"*/{tb}-ud-train.conllu"))
        assert len(conll_glob) == 1
        file_path = conll_glob[0]
        doc = prep_conllu(tb, file_path, args.overwrite)
        if doc is not None:
            if has_dev:
                out_file = out_dir.joinpath(file_path.name)
                write_doc_to_file(doc, out_file)
            else:
                out_train_file = out_dir.joinpath(file_path.name)
                dev_name = file_path.name.split("-")[0] + "-ud-dev.conllu"
                out_dev_file = out_dir.joinpath(dev_name)
                out_dev_file.touch()

                sents = doc.to_dict()

                permutations = np.random.permutation(np.arange(len(sents))).tolist()
                divide = len(sents) * 7 // 8
                train_sents = [sents[idx] for idx in permutations[:divide]]
                dev_sents = [sents[idx] for idx in permutations[divide:]]

                train_doc = Document(train_sents)
                write_doc_to_file(train_doc, out_train_file)

                dev_doc = Document(dev_sents)
                write_doc_to_file(dev_doc, out_dev_file)

elif args.split == "test":
    kwargs = {}
    if args.case == "preprocess":
        kwargs['processors'] = 'tokenize,mwt,pos,lemma'
    if not args.lang_file:
        txt_files = list(ud_root.glob("*/*_*.txt"))
        assert len(txt_files) == 82, f"number of txt_files = {len(txt_files)}; should be 82"
    else:
        with open(args.lang_file) as fp:
            tbs = fp.read().splitlines()
        txt_files = flatten([list(ud_root.rglob(f"{tb}-ud-test.txt")) for tb in tbs])
    for txt_file in sorted(txt_files):
        out_file = out_dir.joinpath(txt_file.stem.split("-")[0] + "-ud-test.conllu")
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
            nlp = stanza.Pipeline(lang=lang, **kwargs)
            doc = nlp(fp.read())
            write_doc_to_file(doc, out_file)
else:
    raise NotImplementedError

