from typing import Tuple

import argparse
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
from collections import OrderedDict
from functools import partial

import os
import re
import seaborn as sns
import torch
import numpy as np
from stanza.models.common.doc import Document
from stanza.utils.conll import CoNLL
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from adjustText import adjust_text

from plot_utils import linechart_plot

sns.set()

#matplotlib.rc('font',**{'family':'serif','serif':['Noto Serif CJK TC']})
plt.rcParams.update({"text.usetex": True})
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['text.latex.preamble'] = [
           '\\usepackage{CJK}',
           r'\AtBeginDocument{\begin{CJK}{UTF8}{bsmi}}',
           r'\AtEndDocument{\end{CJK}}',
]
matplotlib.use('Agg')
parser = argparse.ArgumentParser()
parser.add_argument("--archives", nargs='+')
parser.add_argument("--name-archives", nargs='+')
parser.add_argument("--langs", nargs='+')
parser.add_argument("--split", choices=['train', 'dev', 'test'])
parser.add_argument("--layer", default='adjacency')
parser.add_argument("--ckpt", default='ckpts')
parser.add_argument("--suffix")
parser.add_argument("--name-suffixes", nargs='+')
parser.add_argument("--suffixes", nargs='+')
parser.add_argument("--title")
parser.add_argument("--filename")
parser.add_argument("--xlabel")
parser.add_argument("--ylabel")
parser.add_argument("--colors", nargs='+')
parser.add_argument("--todo", choices=['entropy', 'dir', 'dir-all-langs'])
parser.add_argument("--separate", action='store_true')
parser.add_argument("--tb", action='store_true')
parser.add_argument("--regress", action='store_true')
parser.add_argument("--observation", choices=['head-dir', 'head-rule'], default='head-dir')
parser.add_argument('--exclude-rels', nargs='+', default=['root', 'aux', 'cop', 'mark', 'det', 'clf', 'case', 'cc'])

args = parser.parse_args()

ud_root = Path(os.environ['UD_GT'])
paths = list(ud_root.rglob(f"*-ud-{args.split}.conllu"))
langs = [re.sub('\d+', '', lang) for lang in args.langs]
paths = list(filter(lambda x: x.name.split('_')[0] in langs or x.name.split('-')[0] in langs, paths))

def linreg(x, y):
    model = LinearRegression(fit_intercept=True)
    
    model.fit(x[:, np.newaxis], y)
    
    xfit = np.linspace(0, 10, 1000)
    yfit = model.predict(xfit[:, np.newaxis])
    y_predicted = model.predict(y[:, np.newaxis])
    intercept = model.predict(np.array([1]).reshape(1, -1))[0]
    score = r2_score(y, y_predicted)

    return xfit, yfit, score, intercept

def flatten(l):
    return [item for sublist in l for item in sublist]

def postprocess_title(title):
    title = re.sub(r'\\n', r'\n', title)
    return title

def get_archive_dir(archive):
    archive_path = Path(archive)
    archive_dir = archive_path.parent.joinpath(archive_path.stem.split(".")[0])
    return archive_dir

archive_dirs = list(map(get_archive_dir, args.archives))

def get_conll(conll_file):
    doc = Document(CoNLL.conll2dict(input_file=conll_file))
    return doc

def get_state_dict(archive_dir, path, suffix):
    th_file = archive_dir.joinpath(path.with_suffix(f".{args.layer}.th").name)
    conll_file = archive_dir.joinpath(path.name)

    try:
        epoch = re.match(r'model_epoch_(\d+).*', archive_dir.stem).group(1)
        method = str(archive_dir).split("/")[-2]
        if args.tb:
            lang = path.name.split("-")[0]
        else:
            lang = path.name.split("_")[0]
        one_path = list(Path(args.ckpt).glob(f"{method}_{epoch}_96{lang}_{suffix}/result-gt.conllu"))
        one_path += list(Path(args.ckpt).glob(f"{method}_{epoch}_{lang}_{suffix}/result-gt.conllu"))
        assert len(one_path) == 1, f"{method}_{epoch}_{lang}_{suffix}"
        return get_conll(one_path[0])
    except (AssertionError, IndexError, AttributeError):
        if conll_file.exists():
            return get_conll(conll_file)
        elif th_file.exists():
            return torch.load(th_file)
        else:
            raise NotImplementedError

suffixes = args.suffixes if args.suffixes else [args.suffix]
all_stats = {str(path): {str(archive_dir): {suffix: get_state_dict(archive_dir, path, suffix)
                                            for suffix in suffixes}
                         for archive_dir in archive_dirs}
             for path in paths}

def get_entropies(stat):
    def get_entropy(tensor):
        return np.exp(-float(torch.sum(tensor.exp() * tensor))) / tensor.shape[0]
    return list(map(get_entropy, stat['arcs']))

def is_head_rule(dep, head_rule):
    head, rel, child = dep
    head_pos = head.upos
    child_pos = child.upos
    rev_head_rule = tuple(reversed(head_rule))
    dep_pair = (head_pos, child_pos)
    if dep_pair == head_rule:
        enum = 1
    elif dep_pair == rev_head_rule:
        enum = 0
    else:
        enum = -1
    return enum

def get_head_rule_matches(stat, head_key, head_rule: Tuple[str, str] = ('VERB', 'NOUN')):
    if isinstance(stat, Document):
        doc = stat
        deps = flatten([sent.dependencies for sent in doc.sentences])
        head_rule_matches = list(map(partial(is_head_rule, head_rule=head_rule), deps))
        #num_cis_head_rule = len(list(filter(lambda x: x > 0, head_rule_matcheses)))
        #num_all_head_rule = len(list(filter(lambda x: x >= 0, head_rule_matcheses)))
        return head_rule_matches 
            
    else:
        raise NotImplementedError

def get_dirs(stat, head_key):
    def get_dir(head, position):
        position += 1
        assert head != position
        return head - position
    if isinstance(stat, Document):
        doc = stat
        heads = flatten([[int(word.head) for word in sent.words] for sent in doc.sentences])
        positions = flatten([[int(word.id)-1 for word in sent.words] for sent in doc.sentences])
        return np.array(list(map(get_dir, heads, positions)))
    else:
        return np.array(list(map(get_dir, stat[head_key], stat['positions'])))

def path2name(path, has_tb=True, has_split=True, to_lang=False):
    tb, _, split = path.stem.split('-')
    if to_lang:
        tb = tb.split("_")[0]
    if has_tb and has_split:
        return f"{tb}_{split}"
    elif has_tb:
        return tb
    elif has_split:
        return split
    else:
        raise ValueError

def hist_plot(path):
    fig, ax = plt.subplots(1, 1)
    stats = [all_stats[str(path)][str(archive_dir)] for archive_dir in archive_dirs]
    entropies = list(map(get_entropies, stats))
    colors = args.colors
    if args.separate:
        ax.hist(entropies, bins=np.linspace(0, 1, 51),
                range=(0, 1), color=colors,
                label=args.name_archives)
    else:
        for entropy, color, label in zip(entropies, colors, args.name_archives):
            ax.hist(entropy, bins=np.linspace(0, 1, 51),
                    range=(0, 1), color=color,
                    label=label, alpha=0.5)
    lgd = ax.legend(bbox_to_anchor=(0.5, 1.20), fancybox=True,
                    loc = 9, ncol = len(colors), shadow = True, handletextpad=-2)
    ax.set_title(f"{Path(args.title).stem}: {path2name(path, has_split=False)}")
    ax.set_ylabel("次數")
    fig.savefig(f"{args.title}_{path2name(path)}.pdf",
                box_extra_artists=(lgd, ax.title), bbox_inches='tight')
    plt.close(fig=fig)

def dir_plot(path):
    fig, ax = plt.subplots(1, 1)
    stats = [all_stats[str(path)][str(archive_dir)] for archive_dir in archive_dirs]
    pred_dir_dists = list(map(get_dirs, stats, ['pred_heads' for stat in stats]))
    gold_dir_dist = get_dirs(stats[0], 'heads')
    dir_dists = pred_dir_dists + [gold_dir_dist]

    colors = args.colors + ['#ffd700']

    names = args.name_archives + ['正確答案']

    if args.separate:
        ax.hist(dir_dists, bins=np.arange(-20, 21),
                range=(-20, 20), color=colors,
                label=names)
    else:
        for dir_dist, color, label in zip(dir_dists, colors, names):
            ax.hist(dir_dist, bins=np.arange(-20, 21),
                    range=(-20, 20), color=color,
                    label=label, alpha=0.5)
    lgd = ax.legend(bbox_to_anchor=(0.5, 1.20), fancybox=True,
                    loc = 9, ncol = len(colors), shadow = True)
    ax.set_title(f"{Path(args.title).stem}: {path2name(path, has_split=False)}")
    ax.set_ylabel("次數")
    ax.set_xlabel("方向性")
    fig.savefig(f"{args.title}_{path2name(path)}.pdf",
                box_extra_artists=(lgd, ax.title), bbox_inches='tight')
    plt.close(fig=fig)

def get_leftness(dir_dist, mask):
    masked_dirs = dir_dist * np.array(mask)
    return sum(masked_dirs > 0) / sum(masked_dirs != 0)

def get_leftness_diffs(dir_dist, gold_dir_dist, mask):
    masked_dirs = ((np.array(dir_dist) > 0) == (np.array(gold_dir_dist) > 0)) * np.array(mask)
    return sum(masked_dirs > 0) / sum(mask)

def get_deprels(doc):
    return [[word.deprel for word in sent.words] for sent in doc.sentences]

def get_values_to_show(paths, obs_func, suffix):
    _paths = list(map(str, paths))
    _archive_dirs = list(map(str, archive_dirs))
    stats_of_methods = [[all_stats[path][archive_dir][suffix] for path in _paths] for archive_dir in _archive_dirs]
    gold_conlls = [get_conll(path) for path in _paths]
    masks = [list(map(lambda l: l not in args.exclude_rels, flatten(get_deprels(conll))))
             for conll in gold_conlls]
    gold_dir_dists = [obs_func(conll, None) for conll in gold_conlls]
    pred_dir_dists_of_methods = [[obs_func(stat, 'pred_heads') for stat in stats] for stats in stats_of_methods]
    return gold_dir_dists, pred_dir_dists_of_methods, masks

def avg_diffs(preds, gold):
    pred_vals = np.array(preds)
    gold_val = np.array(gold)
    return np.mean(np.abs(pred_vals - gold_val.reshape(1, -1)), axis=1)

def get_head_rule_ratio(head_rule_matches, mask=None):
    num_cis_head_rule = len(list(filter(lambda x: x > 0, head_rule_matches)))
    num_all_head_rule = len(list(filter(lambda x: x >= 0, head_rule_matches)))
    return num_cis_head_rule / num_all_head_rule

def get_head_rule_diffs(head_rule_matches, gold_head_rule_matches, mask=None):
    return np.abs(get_head_rule_ratio(head_rule_matches) - get_head_rule_ratio(gold_head_rule_matches))

def dir_all_langs_linechart(paths, obs_func, diff_func, ylabel, suffixes, name_suffixes,
                            name_methods):
    diff_methods = []
    import pdb
    pdb.set_trace()
    for gold, preds, masks in [get_values_to_show(paths, obs_func, suffix) for suffix in suffixes]:
        diff = np.array([diff_func(flatten(pred), flatten(gold), flatten(masks)) for pred in preds])
        diff_methods.append(diff)
    #diff_methods = \
    #    [np.array([reduce_func(flatten(pred), flatten(masks)) for pred in preds]) - reduce_func(flatten(gold), flatten(masks))
    #     for preds, gold, masks in [get_values_to_show(paths, obs_func, suffix) for suffix in suffixes]]
    diff_methods = np.array(diff_methods).transpose(1, 0)
    xss = [[i for i in range(len(diff_method))] for diff_method in diff_methods]
    title = postprocess_title(args.title)

    fig = linechart_plot(xss, diff_methods, name_suffixes, name_methods, title,
                         args.ylabel, colors=args.colors, ylim=None)
    fig.savefig(args.filename)
    plt.close(fig=fig)

def dir_all_langs_plot(paths, obs_func, reduce_func, xlabel, ylabel, suffix):
    fig, ax = plt.subplots(1, 1)
    gold_dir_dists, pred_dir_dists_of_methods, masks = get_values_to_show(paths, obs_func, suffix)

    pred_leftnesses_of_methods = \
        [[reduce_func(pred_dir_dist, mask) for pred_dir_dist, mask in zip(pred_dir_dists, masks)]
         for pred_dir_dists in pred_dir_dists_of_methods]
    gold_leftnesses = [reduce_func(gold_dir_dist, mask) for gold_dir_dist, mask in zip(gold_dir_dists, masks)]

    leftnesses = pred_leftnesses_of_methods
    colors = args.colors

    names = args.name_archives
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    xlabel = ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal', adjustable='box')

    text_positions = []
    for pred_leftnesses, color, name in zip(pred_leftnesses_of_methods, colors, names):
        print(name, gold_leftnesses, pred_leftnesses)
        ax.scatter(gold_leftnesses, pred_leftnesses, s=10, color=color, label=name)
        if args.regress:
            xfit, yfit, score, intercept = linreg(np.array(gold_leftnesses), np.array(pred_leftnesses))
            ax.plot(xfit, yfit, color=color, alpha=0.5, linestyle='--')
            text_positions.append((score, intercept, color))
        #ax.annotate(, (1, intercept), fontsize=8)

    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_yticks(np.linspace(0, 1, 11))
    #ax.set_xticklabels([path2name(path, has_split=False, to_lang=True) for path in paths],
    #                    rotation=45)
    left_orders = np.argsort(gold_leftnesses)
    #for leftness, path in zip(gold_leftnesses, paths):
    #for idx in left_orders:
    #idx = 0
    #lower_xlabel = False
    #while idx < len(left_orders):
    #    next_left = gold_leftnesses[left_orders[idx+1]] if idx+1 < len(left_orders) else np.inf
    #    this_left = gold_leftnesses[left_orders[idx]]
    #    if next_left - this_left < 0.03:
    #        lower_xlabel = True
    #        leftness = (next_left + this_left) / 2
    #        label = ", ".join([path2name(paths[left_orders[_idx]], has_split=False, to_lang=True)
    #                           for _idx in [idx, idx+1]])
    #        idx += 1
    #    else:
    #        leftness = gold_leftnesses[left_orders[idx]]
    #        label = 
    lang_texts = [ax.text(x=leftness, y=-0.01, s=path2name(path, has_split=False, to_lang=True),
                          rotation=90, va='top', fontsize=9)
             for leftness, path in zip(gold_leftnesses, paths)]
    #    idx += 1

    y_for_xlabel = -0.1
    ax.xaxis.set_label_coords(0.5, y_for_xlabel)

    plt.tick_params(
        axis='x',
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False
    )
    title = postprocess_title(args.title)
    ax.set_title(title, y=1.18)
    ax.plot([0, 1], [0.5, 0.5], transform=ax.transAxes, color='k', linestyle='--', alpha=0.5)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='k')
    lgd = ax.legend(bbox_to_anchor=(0.5, 1.20), fancybox=True,
                    loc = 9, ncol = 3, shadow = True,
                    handletextpad=-0.5, columnspacing=0)
    if args.regress:
        annotations = [ax.text(1, intercept, f"$R^{2} = {score:.3f}$", fontsize=8, color=color, ha='right')
                       for intercept, score, color in text_positions]
        adjust_text(annotations, add_objects=[ax.title, lgd], only_move={"text": "y"})
    adjust_text(lang_texts, add_objects=[ax.title, lgd], only_move={"text": "x"})
    fig.savefig(args.filename,
                box_extra_artists=(lgd, ax.title, ax.get_xlabel()), bbox_inches='tight')
    plt.close(fig=fig)

funcs = {"head-dir": get_dirs, "head-rule": get_head_rule_matches}
reduce_funcs = {"head-dir": get_leftness, "head-rule": get_head_rule_ratio}
diff_funcs = {"head-dir": get_leftness_diffs, "head-rule": get_head_rule_diffs}
         
if args.todo == 'entropy':
    list(map(hist_plot, paths))
elif args.todo == 'dir':
    list(map(dir_plot, paths))
elif args.todo == 'dir-all-langs':
    if args.suffixes is not None:
        dir_all_langs_linechart(paths, funcs[args.observation], diff_funcs[args.observation],
                                args.ylabel, args.suffixes, args.name_suffixes,
                                args.name_archives)
    else:
        dir_all_langs_plot(paths, funcs[args.observation], reduce_funcs[args.observation], 
                           args.xlabel, args.ylabel, args.suffix)
