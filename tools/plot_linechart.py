import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import matplotlib
import logging
import json
import argparse
import glob
from pathlib import Path
from itertools import product
import numpy as np
import seaborn as sns
import re
import conllu
import os
from collections import OrderedDict, defaultdict
from subprocess import Popen, PIPE
lex_LAS = {"wo": 77.05, "gd": 70.81, "te": 79.89, "cop": 59.71,
           "be": 63.88, "mr": 52.64, "mt": 78.15, "ta": 55.76}

sns.set()
#matplotlib.rc('font',**{'family':'serif','serif':['Noto Serif CJK TC']})
plt.rcParams.update({"text.usetex": True})
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['text.latex.preamble'] = [
           '\\usepackage{CJK}',
           r'\AtBeginDocument{\begin{CJK}{UTF8}{bsmi}}',
           r'\AtEndDocument{\end{CJK}}',
]

parser = argparse.ArgumentParser()
parser.add_argument('--suffixes', nargs='+')
parser.add_argument('--methods', nargs='+')
parser.add_argument('--epochs', nargs='+')
parser.add_argument('--colors', nargs='+')
parser.add_argument('--name-methods', nargs='+')
parser.add_argument('--name-suffixes', nargs='+')
parser.add_argument('--name-epochs', nargs='+')
parser.add_argument('--markers', nargs='+')
parser.add_argument('--langs', nargs='+')
parser.add_argument('--title')
parser.add_argument('--ckpt', default='ckpts')
parser.add_argument('--no-ens', action='store_true')
parser.add_argument('--gt', action='store_true')
parser.add_argument('--logscale', action='store_true')
parser.add_argument('--epoch-steps', nargs='+', type=int)
parser.add_argument('--ylim', nargs=2, type=int, default=[0, 100])
parser.add_argument('--merge-lang', action='store_true',
                    help='merge different synthetic datasets of the same language')
parser.add_argument('--metric', default='LAS-F1 Score')
parser.add_argument('--out-dir')
parser.add_argument('--filename')
args = parser.parse_args()
out_dir = Path(args.out_dir)
print(args.title)

with open("data/ensemble_langs.txt") as fp:
    ens_langs = fp.read().splitlines()

def has_path_and_not_empty(path: Path):
    return path.exists() and path.stat().st_size

def cumsum(values):
    new_values = [0]
    for idx, value in enumerate(values):
        new_values.append(values[idx] + new_values[-1])
    return new_values[1:]

def get_metric(lines, metric, view):
    try:
        entries = [re.split("\ +\|\ ?", line) for line in lines]
    except FileNotFoundError:
        logging.warning(f"file {str(path)} not found")
        return None
    view2id = {name.strip(): idx for idx, name in enumerate(entries[0])}
    idx = view2id[view]
    metrics = {entry[0]: float(entry[idx].strip()) for entry in entries[2:]}
    return metrics[metric]

def maybe_ens(lang):
    if lang in ens_langs and not args.no_ens:
        return "_ens"
    else:
        return ""

def maybe_no_num(lang, suffix):
    if suffix == 'zs':
        lang = re.sub('\d+', '', lang)
    return lang

def postprocess_lang_or_tb(name, is_tb=False):
    if not is_tb:
        name = name.split("_")[0]
    return re.sub('\d', '', name)

def suffix2epoch(epoch, suffix):

    value = 0
    if int(epoch) == 10 and suffix.endswith('zs'):
        value = 60
    elif suffix.endswith('one_step'):
        value = 1
    elif int(epoch) == 10:
        match = re.match(r".*epoch_(\d+", suffix)
        value += 6 * int(match.group(1))
    return value


langs = args.langs
methods = args.methods
suffixes = args.suffixes
epochs = args.epochs

def get_label(method, epoch, suffix):
    method_name = method2name[method]
    suffix_name = suffix2name[suffix]
    epoch_name = epoch2name[epoch]
    return f"{method_name}, {suffix_name}, {epoch_name}"

result_file = "result-gt.conllu" if args.gt else "result.conllu"



xs = list(range(len(suffixes)))

def plot(xss, yss, xticks, line_names, title, ylabel, logscale=False, colors=None,
         ylim=(0, 100)):
    fig, ax = plt.subplots(1, 1)

    if logscale:
        ax.set_xscale('log')
        xss = [list(map(lambda x: x+1, xs)) for xs in xss]

    ax.set_xticks(xss[0])
    xticks = [re.sub(r'\\n', r'\n', xtick) for xtick in xticks]
    ax.set_xticklabels(xticks, fontsize=12)

    for idx, (xs, ys, line_name) in enumerate(zip(xss, yss, line_names)):
        if colors is not None:
            ax.plot(xs, ys, label=line_name, color=colors[idx], linestyle='-',
                    marker='o', linewidth=1, markersize=3)
        else:
            ax.plot(xs, ys, label=line_name, linestyle='-', marker='o',
                    linewidth=1, markersize=5)

    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel)
    ax.legend()
    title = re.sub(r'\\n', r'\n', title)
    ax.set_title(title)
    return fig, ax.get_xticklabels()

colors = args.colors

metric, view = args.metric.split("-")
def evaluate(gold_files, system_files):
    script = ['python', 'utils/multi_conll18_ud_eval.py']
    process = Popen(script +['--gold_files'] + gold_files + ['--system_files'] + system_files + ['-v'],
                    stdout=PIPE)
    output = process.communicate()[0].decode()
    return get_metric(output.split("\n")[:-1], metric, view)

ckpt = Path(args.ckpt)
ud_gt = Path(os.environ["UD_GT"].replace("**/", ""))
gold_files = [list(ud_gt.rglob(f"{maybe_no_num(lang, 'zs')}*-ud-test.conllu"))[0] for lang in langs]

xss = []
yss = []

for method in methods:
    system_files_of_settings = [[ckpt.joinpath(f"{method}_{epoch}_{maybe_no_num(lang, suffix)}_{suffix}/{result_file}")
                                for lang in langs] for suffix, epoch in zip(suffixes, epochs)]
    ys = [evaluate(gold_files, system_files) for system_files in system_files_of_settings]
    if args.epoch_steps:
        xs = args.epoch_steps
    else:
        xs = cumsum([suffix2epoch(epoch, suffix) for suffix, epoch in zip(suffixes, epochs)])
    print(xs, ys)
    xss.append(xs)
    yss.append(ys)

#name_methods = args.name_methods + ["Stanford"]
#xss.append(args.epoch_steps)
#yss.append([np.mean([lex_LAS[postprocess_lang_or_tb(lang)] for lang in langs])
#            for _ in system_files_of_settings])
#colors += ["#808080"]

title = f"{args.title}"
fig, xticklabels = plot(xss, yss, args.name_suffixes, args.name_methods, title, metric,
                        args.logscale, colors, args.ylim)
if args.filename is None:
    fig.savefig(out_dir.joinpath(f"{args.title}.pdf"), box_extra_artists=(xticklabels,),
                bbox_inches='tight')
else:
    fig.savefig(out_dir.joinpath(args.filename), box_extra_artists=(xticklabels,), 
                bbox_inches='tight')
plt.close(fig=fig)
