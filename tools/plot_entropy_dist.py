import argparse
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import os
import re
import seaborn as sns
import torch
import numpy as np

sns.set()
matplotlib.rc('font',**{'family':'serif','serif':['Noto Serif CJK TC']})
parser = argparse.ArgumentParser()
parser.add_argument("--archives", nargs='+')
parser.add_argument("--name-archives", nargs='+')
parser.add_argument("--langs", nargs='+')
parser.add_argument("--split", choices=['train', 'dev', 'test'])
parser.add_argument("--layer", default='adjacency')
parser.add_argument("--title")
parser.add_argument("--colors", nargs='+')
parser.add_argument("--todo", choices=['entropy', 'dir'])

args = parser.parse_args()

ud_root = Path(os.environ['UD_GT'])
paths = list(ud_root.rglob(f"*-ud-{args.split}.conllu"))
langs = [re.sub('\d+', '', lang) for lang in args.langs]
paths = list(filter(lambda x: x.name.split('_')[0] in langs, paths))

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
    return list(map(get_dir, stat[head_key], stat['positions']))

def path2name(path):
    tb, _, split = path.stem.split('-')
    return f"{tb}_{split}"

def hist_plot(path):
    fig, ax = plt.subplots(1, 1)
    stats = [all_stats[str(path)][str(archive_dir)] for archive_dir in archive_dirs]
    entropies = list(map(get_entropies, stats))
    for entropy, color, label in zip(entropies, args.colors, args.name_archives):
        ax.hist(entropy, bins=np.linspace(0, 1, 51),
                range=(0, 1), color=color,
                label=label, alpha=0.5)
    ax.legend()
    fig.savefig(f"{args.title}_{path2name(path)}.pdf")
    plt.close(fig=fig)

def dir_plot(path):
    fig, ax = plt.subplots(1, 1)
    stats = [all_stats[str(path)][str(archive_dir)] for archive_dir in archive_dirs]
    pred_dir_dists = list(map(get_dirs, stats, ['pred_heads' for stat in stats]))
    gold_dir_dist = get_dirs(stats[0], 'heads')
    dir_dists = pred_dir_dists + [gold_dir_dist]

    colors = args.colors + ['#ffd700']

    names = args.name_archives + ['正確答案']
    for dir_dist, color, label in zip(dir_dists, colors, names):
        ax.hist(dir_dist, bins=np.arange(-20, 21),
                range=(-20, 20), color=color,
                label=label, alpha=0.5)
    ax.legend()
    fig.savefig(f"{args.title}_{path2name(path)}.pdf")
    plt.close(fig=fig)


if args.todo == 'entropy':
    list(map(hist_plot, paths))
elif args.todo == 'dir':
    list(map(dir_plot, paths))
