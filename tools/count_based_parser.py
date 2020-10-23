from typing import List, Union
import argparse
import os
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import pickle
import  numpy as np
from stanza.models.common.doc import Document
from stanza.utils.conll import CoNLL

from stanza.models.common.chuliu_edmonds import chuliu_edmonds_one_root

from ud_utils import load_conllu_file
from ud_utils import ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC

EPS = 1e-10
ROOT = 'ROOT'
BLANK_FILES = ['ja_bccwj-ud-train', 
               'ar_nyuad-ud-train',
               'en_esl-ud-train',
               'fr_ftb-ud-train',
               'qhe_heincs-ud-train']


def get_gold_paths(lang, split='train'):
    ud_gt = Path(os.environ["UD_GT"].replace("**/", ""))
    gold_files = filter(lambda x: x.stem not in BLANK_FILES,
                        ud_gt.rglob(f"{lang}_*-ud-{split}.conllu"))
    
    return list(gold_files)

def get_pair(dep):
    head, rel, child = dep
    return (head.upos, child.upos)

def get_pair_and_dist(dep):
    head, rel, child = dep
    return (head.upos, child.upos, int(child.id) - int(head.id))

def get_counts(doc, key_func=get_pair_and_dist):
    counts = Counter()
    for sent in doc.sentences:
        for dep in sent.dependencies:
            counts.update([key_func(dep)])
    return counts

def normalize_counts(counts, normalization_const: Union[float, None]=None):
    if normalization_const is None:
        return counts
    else:
        total = sum(counts.values())
        normalized_counts = [(triple, (normalization_const * count) / total) for triple, count in counts.items()]
        return Counter({triple: count for triple, count in normalized_counts})


def train(args):
    paths_of_langs = [get_gold_paths(lang) for lang in args.training_languages]
    normalized_counts_langs = []
    key_funcs = {"pair": get_pair, "dist": get_pair_and_dist}
    for paths in paths_of_langs:
        counts_lang = Counter()
        for path in paths:
            doc = Document(CoNLL.conll2dict(input_file=str(path)))
            counts = get_counts(doc, key_funcs[args.key_func])
            counts_lang += counts
        normalized_counts_lang = normalize_counts(counts_lang, args.normalization_constant)
        normalized_counts_langs.append(normalized_counts_lang)
    averaged_normalized_counts = Counter()
    for normalized_counts in normalized_counts_langs:
        averaged_normalized_counts += normalized_counts
    overall_normalized_counts = normalize_counts(averaged_normalized_counts,
                                                 args.normalization_constant)

    with open(args.model_file, 'wb') as fp:
        pickle.dump(overall_normalized_counts, fp)

def in_span(span, possibly_bigger_span):
    return span.start >= possibly_bigger_span.start and span.end <= possibly_bigger_span.end

def get_ud_in_sents(ud):
    sents = []
    sent_idx = 0
    sent = []
    for word in ud.words:
        if in_span(word.span, ud.sentences[sent_idx]):
            sent.append(word)
        else:
            sents.append(sent)
            sent = [word]
            sent_idx += 1
    sents.append(sent)
    return sents

class Parser:
    def __init__(self, model, head_neutral=False, use_dist=False):
        self.model = model
        self.use_dist = use_dist
        self.head_neutral = head_neutral
    def filter_key(self, head_pos, child_pos, child_idx, head_idx):
        if self.use_dist:
            return head_pos, child_pos, child_idx - head_idx
        else:
            return head_pos, child_pos
    def get_prob(self, head_pos, child_pos, head_idx, child_idx):
        if self.head_neutral:
            cnt_bothside = [self.model.get(self.filter_key(head_pos, child_pos, child_idx, head_idx), 0),
                            self.model.get(self.filter_key(head_pos, child_pos, head_idx, child_idx), 0)]
            cnt = sum(cnt_bothside) / len(cnt_bothside)
        else:
            cnt = self.model.get(self.filter_key(head_pos, child_pos, child_idx, head_idx), 0)
        return cnt

    def parse(self, doc):
        return [self.parse_sent(sent) for sent in doc.sentences]
    def parse_sent(self, sent):
         # the prob of any word being parent of root should be zero
        probs = [[0 for _ in range(len(sent.words) + 1)]]
        #probs = []
        part_of_speeches = [word.upos for word in sent.words]
        for child_idx, child_pos in enumerate(part_of_speeches):
            unnormed_prob = []
            for head_idx, head_pos in enumerate([ROOT] + part_of_speeches):
                cnt = self.get_prob(head_pos, child_pos, head_idx, child_idx)
                unnormed_prob.append(cnt)
            total_weight = sum(unnormed_prob)
            if total_weight == 0:
                normed_prob = [1 / len(unnormed_prob) for p in unnormed_prob]
            else:
                normed_prob = [p / sum(unnormed_prob) for p in unnormed_prob]
            probs.append(normed_prob)
        probs = np.array(probs)
        probs = np.log(probs + EPS)
        pred_heads = chuliu_edmonds_one_root(probs)[1:]
        return pred_heads

    @classmethod
    def load_model(cls, model_file, head_neutral, use_dist):
        with open(model_file, 'rb') as fp:
            count_based_model = pickle.load(fp)
        return cls(count_based_model, head_neutral, use_dist)

def write_heads_to_doc(doc, list_pred_heads):
    assert len(doc.sentences) == len(list_pred_heads)
    for sent, pred_heads in zip(doc.sentences, list_pred_heads):
        assert len(sent.words) == len(pred_heads)
        for word, pred_head in zip(sent.words, pred_heads):
            word.head = pred_head
    return doc

def write_doc_to_file(doc, out_file):
    conll_string = CoNLL.conll_as_string(CoNLL.convert_dict(doc.to_dict()))
    with open(str(out_file), "w") as fp:
        fp.write(conll_string)

def predict(args):
    assert (args.output_file == None) != (args.output_dir == None), \
        'exactly one of output_file or output_dir must be specified'
    parser = Parser.load_model(args.model_file, args.head_neutral, args.use_dist)
    doc = Document(CoNLL.conll2dict(input_file=args.input_file))
    pred_heads = parser.parse(doc)
    write_heads_to_doc(doc, pred_heads)
    if args.output_file is not None:
        write_doc_to_file(doc, args.output_file)
    elif args.output_dir is not None:
        if args.result_file is not None:
            if args.save_as_tb:
                lang = Path(args.input_file).name.split("-")[0]
            else:
                lang = Path(args.input_file).name.split("_")[0]
            parent_dir = Path(f"{args.output_dir}_10_{lang}_zs")
            parent_dir.mkdir(parents=True, exist_ok=True)
            output_file = parent_dir.joinpath(args.result_file)
            #os.symlink(output_file.absolute(), output_file.absolute)
        else:
            name = Path(args.input_file).name
            output_file = Path(args.output_dir).joinpath(name)
        write_doc_to_file(doc, output_file)

if __name__ == '__main__':
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--key-func", choices=['pair', 'dist'], default='pair')
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    train_parser = subparsers.add_parser('train', parents=[parent_parser])
    train_parser.add_argument('--training-languages', nargs='+', required=True)
    train_parser.add_argument('--normalization-constant', type=float, default=10000.0,
                              help='normalize and multiply by this value to prevent numerical issues')
    train_parser.add_argument('--model-file', type=str, required=True,
                              help='location to store count-based model file')
    train_parser.set_defaults(func=train)
    predict_parser = subparsers.add_parser('predict', parents=[parent_parser])
    predict_parser.add_argument('--input-file', required=True)
    predict_parser.add_argument('--model-file', required=True)
    predict_parser.add_argument('--save-as-tb', action='store_true')
    predict_parser.add_argument('--output-file')
    predict_parser.add_argument('--output-dir', help='file name will be the name of input file if provided.')
    predict_parser.add_argument('--result-file')
    predict_parser.add_argument('--use-dist', action='store_true')
    predict_parser.add_argument('--head-neutral', action='store_true',
                                help='average the prob of (HEAD, CHILD, D) and (HEAD, CHILD, -D).')
    predict_parser.set_defaults(func=predict)
    args = parser.parse_args()
    args.func(args)
