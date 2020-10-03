import argparse
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product
from collections import OrderedDict
import os
import re
import seaborn as sns
import torch
import numpy as np

sns.set()

matplotlib.rc('font',**{'family':'serif','serif':['Noto Serif CJK TC']})
matplotlib.use('Agg')
parser = argparse.ArgumentParser()
parser.add_argument("--archives", nargs='+')
parser.add_argument("--name-archives", nargs='+')
parser.add_argument("--langs", nargs='+')
parser.add_argument("--split", choices=['train', 'dev', 'test'])
parser.add_argument("--layer", default='adjacency')
parser.add_argument("--title")
parser.add_argument("--colors", nargs='+')
parser.add_argument("--todo", choices=['entropy', 'dir', 'dir-all-langs'])
parser.add_argument("--separate", action='store_true')
parser.add_argument('--exclude-rels', nargs='+', default=['root', 'aux', 'cop', 'mark', 'det', 'clf', 'case', 'cc'])

args = parser.parse_args()

ud_root = Path(os.environ['UD_GT'])
paths = list(ud_root.rglob(f"*-ud-{args.split}.conllu"))
langs = [re.sub('\d+', '', lang) for lang in args.langs]
paths = list(filter(lambda x: x.name.split('_')[0] in langs or x.name.split('-')[0] in langs, paths))

def flatten(l):
    return [item for sublist in l for item in sublist]

def get_archive_dir(archive):
    archive_path = Path(archive)
    archive_dir = archive_path.parent.joinpath(archive_path.stem.split(".")[0])
    return archive_dir

archive_dirs = list(map(get_archive_dir, args.archives))

def get_state_dict(archive_dir, path):
    th_file = archive_dir.joinpath(path.with_suffix(f".{args.layer}.th").name)
    return torch.load(th_file)

all_stats = {str(path): {str(archive_dir): get_state_dict(archive_dir, path)
                         for archive_dir in archive_dirs}
             for path in paths}

def get_entropies(stat):
    def get_entropy(tensor):
        return np.exp(-float(torch.sum(tensor.exp() * tensor))) / tensor.shape[0]
    return list(map(get_entropy, stat['arcs']))

def get_dirs(stat, head_key):
    def get_dir(head, position):
        position += 1
        assert head != position
        return head - position
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
    masked_dirs = dir_dist * mask
    return sum(masked_dirs > 0) / sum(masked_dirs != 0)

def dir_all_langs_plot(paths):
    fig, ax = plt.subplots(1, 1)
    _paths = list(map(str, paths))
    _archive_dirs = list(map(str, archive_dirs))
    stats_of_methods = [[all_stats[path][archive_dir] for path in _paths] for archive_dir in _archive_dirs]
    gold_dir_dists = [get_dirs(stats_of_methods[0][idx], 'heads') for idx in range(len(paths))]
    masks = [np.array(list(map(lambda l: l not in args.exclude_rels, stats_of_methods[0][idx]['deprels'])))
             for idx in range(len(paths))]
    pred_dir_dists_of_methods = [[get_dirs(stat, 'pred_heads') for stat in stats] for stats in stats_of_methods]
    pred_leftnesses_of_methods = \
        [[get_leftness(pred_dir_dist, mask) for pred_dir_dist, mask in zip(pred_dir_dists, masks)]
         for pred_dir_dists in pred_dir_dists_of_methods]
    gold_leftnesses = [get_leftness(gold_dir_dist, mask) for gold_dir_dist, mask in zip(gold_dir_dists, masks)]

    leftnesses = pred_leftnesses_of_methods
    colors = args.colors

    names = args.name_archives
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    xlabel = ax.set_xlabel("正確的語言方向性")
    ax.set_ylabel("模型預測的語言方向性")
    ax.set_aspect('equal', adjustable='box')

    for pred_leftnesses, color, name in zip(pred_leftnesses_of_methods, colors, names):
        ax.scatter(gold_leftnesses, pred_leftnesses, s=10, color=color, label=name)

    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_yticks(np.linspace(0, 1, 11))
    #ax.set_xticklabels([path2name(path, has_split=False, to_lang=True) for path in paths],
    #                    rotation=45)
    left_orders = np.argsort(gold_leftnesses)
    #for leftness, path in zip(gold_leftnesses, paths):
    #for idx in left_orders:
    idx = 0
    lower_xlabel = False
    while idx < len(left_orders):
        next_left = gold_leftnesses[left_orders[idx+1]] if idx+1 < len(left_orders) else np.inf
        this_left = gold_leftnesses[left_orders[idx]]
        if next_left - this_left < 0.03:
            lower_xlabel = True
            leftness = (next_left + this_left) / 2
            label = ", ".join([path2name(paths[left_orders[_idx]], has_split=False, to_lang=True)
                               for _idx in [idx, idx+1]])
            idx += 1
        else:
            leftness = gold_leftnesses[left_orders[idx]]
            label = path2name(paths[left_orders[idx]], has_split=False, to_lang=True)
        ax.text(x=leftness, y=-0.01, s=label,
                rotation=90, va='top', fontsize=10)
        idx += 1

    y_for_xlabel = -0.15 if lower_xlabel else -0.1
    ax.xaxis.set_label_coords(0.5, y_for_xlabel)

    plt.tick_params(
        axis='x',
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False
    )
    ax.set_title(Path(args.title).stem, y=1.18)
    ax.plot([0, 1], [0.5, 0.5], transform=ax.transAxes, color='k', linestyle='--', alpha=0.5)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='k')
    lgd = ax.legend(bbox_to_anchor=(0.5, 1.20), fancybox=True,
                    loc = 9, ncol = 3, shadow = True,
                    handletextpad=-0.5, columnspacing=0)
    fig.savefig(f"{args.title}.pdf",
                box_extra_artists=(lgd, ax.title, ax.get_xlabel()), bbox_inches='tight')
    plt.close(fig=fig)

if args.todo == 'entropy':
    list(map(hist_plot, paths))
elif args.todo == 'dir':
    list(map(dir_plot, paths))
elif args.todo == 'dir-all-langs':
    dir_all_langs_plot(paths)
