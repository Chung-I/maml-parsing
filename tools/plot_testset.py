import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import matplotlib
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

sns.set()

#font_path = '/home/nlpmaster/miniconda3/envs/maml-grammar/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/simkai.ttf'
#fontprop = matplotlib.font_manager.FontProperties(fname=font_path, size=10)
#matplotlib.font_manager._rebuild()
#matplotlib.rc('font', family='serif')
#plt.rcParams['font.family']=['serif']
#plt.rcParams["errorbar.capsize"] = 3.0
#plt.rcParams['font.serif']=['SimKai']
#plt.rcParams['font.sans-serif']=['SimHei']
matplotlib.rc('font',**{'family':'serif','serif':['Noto Serif CJK TC']})
#matplotlib.rc('font',**{'family':'serif','serif':['SimKai']})
#matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Noto Sans CJK TC']})
#matplotlib.rc('text', usetex=True)
#plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC']
#plt.rcParams['axes.unicode_minus'] = False

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='ckpts')
parser.add_argument('--methods', nargs='+')
parser.add_argument('--name-methods', nargs='+')
parser.add_argument('--seed-suffixes', nargs='+', default=[])
parser.add_argument('--control-method', default="multi-fixed-big", help="control group to be compared with")
parser.add_argument('--name-control-method')
parser.add_argument('--dev-tbs-file', default='data/interp_train_tbs.txt')
parser.add_argument('--affix')
parser.add_argument('--metric', default='LAS-F1 Score')
parser.add_argument('--dep-len-range', type=int, default=30)
parser.add_argument('--sent-len-bins', type=list, default=[(0, 10), (11, 20), (21, 30), (31, 40), (41, float('inf'))])
parser.add_argument('--langs', nargs='+')
parser.add_argument('--lang-file')
parser.add_argument('--by-step', action='store_true')
parser.add_argument('--model-selection', choices=["val", "test"], default="val", help="select model by performance of which split")
parser.add_argument('--test', action='store_true')
parser.add_argument('--no-ens', action='store_true')
parser.add_argument('--gt', action='store_true')
parser.add_argument('--ana', action='store_true')
parser.add_argument('--by-typology', action='store_true')
parser.add_argument('--by-tb-size', action='store_true')
parser.add_argument('--log-scale', action='store_true')
parser.add_argument('--scatter-typo-tbsize', action='store_true')
parser.add_argument('--decouple-model-selection', action='store_true', help="decouple model selection of each language")
parser.add_argument('--exclude-lang-file')
parser.add_argument('--epochs', nargs='+')#, type=int)
parser.add_argument('--out-dir')

args = parser.parse_args()
out_dir = Path(args.out_dir)


assert len(args.methods) == len(args.name_methods)

flatten = lambda l: [item for sublist in l for item in sublist]

seed_suffixes = [""] + [suffix.lstrip() for suffix in args.seed_suffixes]
methods = flatten([[method + suffix for suffix in seed_suffixes] for method in args.methods])

def postprocess_interp_method(name):
    start, end, alpha = name.split("__")
    if float(alpha) == 0:
        return f"\\textsc{{{start.split('-')[0].capitalize()}}}", 0
    if float(alpha) == 1.0:
        return f"\\textsc{{{end.split('-')[0].capitalize()}}}", 2
    if float(alpha) == 0.5:
        return f"\\textsc{{M\&M}}", 1

def postprocess_method(name):
    parts = name.split("-")
    return parts[0]

def get_tb_size(name, tb_sizes):
    matches = re.match('(\d+)[a-z]+', name)
    if matches:
        return int(matches.group(1))
    else:
        return tb_sizes[name]

def postprocess_lang_or_tb(name, is_tb=False):
    if not is_tb:
        name = name.split("_")[0]
    return re.sub('\d', '', name)

with open(args.dev_tbs_file) as fp:
    dev_tbs = fp.read().splitlines()

result_file = "result-gt.txt" if args.gt else "result.txt"

def is_empty_path(path: Path):
    return path.exists() and path.stat().st_size

def get_data_from_json(json_file: str, metric_name: str, lang: str, split: str):
    with open(json_file) as fp:
        data = json.load(fp)
    new = data.get(f"{split}_{metric_name}_{re.sub('[0-9]', '', lang)}")
    old = data.get(f"{split}_{metric_name}_{lang}")
    value = new if new else old
    return value

with open("data/ensemble_langs.txt") as fp:
    ens_langs = fp.read().splitlines()

def get_json_files(method, epoch, lang):
    if lang in ens_langs:
        return [folder.glob("metrics_epoch_*.json") for folder in Path(args.ckpt).glob(f"{method}_{epoch}_{lang}_cv*_{args.affix}")]
    else:
        return [Path(args.ckpt).glob(f"{method}_{epoch}_{lang}_{args.affix}/metrics_epoch_*.json")]

def get_metric(path, metric, view):
    with open(str(path)) as fp:
        entries = [re.split("\ +\|\ ?", line) for line in fp.read().splitlines()]
    view2id = {name.strip(): idx for idx, name in enumerate(entries[0])}
    idx = view2id[view]
    metrics = {entry[0]: float(entry[idx].strip()) for entry in entries[2:]}
    return metrics[metric]

def maybe_ens(lang):
    if lang in ens_langs and not args.no_ens:
        return "_ens"
    else:
        return ""

def adv_proc(method, epoch):
    if args.by_step and 'adv' in method:
        return epoch * 2
    return epoch

def rel_plot(methods, _langs, xs, get_perf, xlabel, ylabel, title, filename):
    fig, ax = plt.subplots(1,1)
    for method in methods:
        ys = [get_perf(method, lang) for lang in _langs]
        ax.plot(xs, ys, label=method)
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(filename)
    plt.close(fig=fig)

def scatter_plot(xs, ys, colors, xlabel, ylabel, clabel, labels, title, filename,
                 yscale='linear', vmin=0, vmax=100, cscale='linear', hline=False):
    fig, ax = plt.subplots(1,1)
    if cscale == 'log':
        cax = ax.scatter(xs, ys, c=colors, cmap='coolwarm', norm=plt_colors.LogNorm(vmin=vmin, vmax=vmax))# vmin=vmin, vmax=vmax)
    else:
        cax = ax.scatter(xs, ys, c=colors, cmap='coolwarm', vmin=vmin, vmax=vmax)
    if hline:
        ax.axhline(y=0, linestyle='--', color='grey')
    cbar = fig.colorbar(cax)
    cbar.ax.set_ylabel(clabel, rotation=270, labelpad=15)
    for i, label in enumerate(labels):
        x, y = xs[i], ys[i]
        #if label in ['20wo', '100wo']:
        #    x -= 0.02
        #    if yscale == 'log':
        #        y += y * 0.15
        #    else:
        #        y += 10
        ax.annotate(label, (x, y), rotation=15, fontsize=6, weight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    ax.set_title(title)
    fig.savefig(filename)
    plt.close(fig=fig)


if args.langs or args.lang_file:
    if args.lang_file:
        with open(args.lang_file) as fp:
            langs = fp.read().splitlines()
    else:
        langs = args.langs
    paths = {(method, epoch, lang): Path(f"{args.ckpt}/{method}_{adv_proc(method, epoch)}_{lang}{maybe_ens(lang)}_{args.affix}").joinpath(result_file)
               for method, epoch, lang in product(methods, args.epochs, langs)}
        
    assert all([is_empty_path(path) for key, path in paths.items()]),\
        f"path that doesn't exists: {[path for _, path in paths.items() if not is_empty_path(path)]}"
else:
    sets_of_paths = [list(Path(args.ckpt).glob(f"{method}_{epoch}_*_{args.affix}/{args.result_file}"))
             for method, epoch in product(methods, args.epochs)]
    langs = set.intersection(*[set([re.match(f".*?_\d+_(.*?)_{args.affix}", str(path)).group(1) for path in paths])
             for paths in sets_of_paths])
    if args.exclude_lang_file:
        with open(args.exclude_lang_file) as fp:
            other_langs = fp.read().splitlines()
            langs = langs - set(other_langs)
    paths = {(method, epoch, lang): Path(f"{args.ckpt}/{method}_{epoch}_{lang}_{args.affix}").joinpath(result_file)
               for method, epoch, lang in product(methods, args.epochs, langs)}

datas = {}
metric, view = args.metric.split("-")
for key, path in paths.items():
    value = get_metric(path, metric, view)
    datas[key] = value


colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

if args.model_selection == 'val':
    val_datas = defaultdict(list)
    for lang, epoch in product(langs, args.epochs):
        if args.affix != "zs":
            fig, ax = plt.subplots(1,1)
        for method, color in zip(methods, colors):
            if args.affix == "zs":
                result_file = "dev-result.txt" if lang in dev_tbs else "train-result.txt"
                _path = Path(f"{args.ckpt}/{method}_{epoch}_{lang}_{args.affix}").joinpath(result_file)
                value = get_metric(_path, metric, view)
                val_datas[(method, epoch, lang)].append(value)
            else:
                for split, linestyle in zip(["training", "validation"], ["solid", "dashed"]):
                    for json_files in get_json_files(method, epoch, lang):
                        run_datas = {int(json_file.stem.split("_")[-1]): \
                                     get_data_from_json(str(json_file), metric, lang, split)
                                     for json_file in json_files}
                        xs, ys = zip(*sorted(filter((lambda x: x[1] is not None), run_datas.items()), key=lambda x: x[0]))
                        if split == "validation":
                            val_datas[(method, epoch, lang)].append(max(ys))
                        ax.plot(xs, ys, label=method, color=color, linestyle=linestyle)
        if args.affix != "zs":
            ax.legend()
            ax.set_xlabel("run_epoch")
            ax.set_ylabel(metric)
            ax.set_title(f"val {metric} v.s. ft-epoch for lang={lang}, pretrain-epoch={epoch}")
            fig.savefig(out_dir.joinpath(f"{metric}_{lang}_{epoch}.png"))
            plt.close(fig=fig)
    
    for key in val_datas.keys():
        perfs = val_datas[key]
        val_datas[key] = np.mean(perfs)

if args.model_selection == "val":
    ref_datas = val_datas
elif args.model_selection == "test":
    ref_datas = datas
else:
    raise NotImplementedError

for _datas in [ref_datas, datas]:
    for method in args.methods:
        for epoch in args.epochs:
            for lang in langs:
                value = np.mean([ref_datas[(method + suffix, epoch, lang)] for suffix in seed_suffixes])
                _datas[(method, epoch, lang)] = value

for lang in langs:
    fig, ax = plt.subplots(1,1)
    for method, color in zip(args.methods, colors):
        ax.plot(args.epochs, [ref_datas[(method, epoch, lang)] for epoch in args.epochs],
                label=method, color=color)
    ax.legend()
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"{args.model_selection} {metric} per epoch: {lang}")
    fig.savefig(out_dir.joinpath(f"{metric}_{lang}.png"))
    plt.close(fig=fig)

print(f"languages: {langs}")
means = {}
stds = {}

for method, epoch in product(args.methods, args.epochs):
    stats = [ref_datas[(method, epoch, lang)] for lang in langs]
    means[(method, epoch)] = np.mean(stats)
    stds[(method, epoch)] = np.std(stats)
    print(f"{method} at epoch {epoch}: {np.mean(stats)}; std: {np.std(stats)}")


fig, ax = plt.subplots(1,1)
for method, color in zip(args.methods, colors):
    ax.errorbar(args.epochs, [means[(method, epoch)] for epoch in args.epochs],
                [stds[(method, epoch)] for epoch in args.epochs],
                label=method, color=color, marker='o')
    ax.legend()
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(f"{args.model_selection} {metric} per epoch")
fig.savefig(out_dir.joinpath(f"{metric}_overall.png"))
plt.close(fig=fig)


ud_roots = list(map(Path, glob.glob(os.environ['UD_ROOT'])))

def get_corpus_size(lang):
    files = [list(ud_root.rglob(f"**/{lang}*-ud-train.conllu")) for ud_root in ud_roots]
    files = [i for j in files for i in j]
    if len(files) == 0:
        return (lang, 0)
    conllu_file = files[0]
    with open(conllu_file) as fp:
        annotations = conllu.parse(fp.read())
    return (lang, len(annotations))


pos_accs = {}
with open("tools/editors.csv") as fp:
    column_names = fp.readline()[:-1].split(",")
    for line in fp:
        columns = line[:-1].split(",")
        pos_accs[columns[0]] = {a: float(b) for a, b in zip(column_names[1:], columns[1:])}

def get_corpus_pos_acc(lang):
    files = list(Path("/home/nlpmaster/ssd-1t/corpus/ud/ud-treebanks-v2.2").rglob(f"{lang}*-ud-train.conllu"))
    assert len(files) == 1
    conllu_file = files[0]
    corpus_name = str(conllu_file).split("/")[-2]
    return (lang, pos_accs[corpus_name]["UPOS"])


#corpus_sizes = dict(map(get_corpus_size, langs))
##corpus_pos_accs = dict(map(get_corpus_pos_acc, langs))
#for lang in langs:
#    mono_perfs = [ref_datas[(method, 0, lang)] for method in args.methods]
#    print(lang, mono_perfs)
#monolingual_perfs = {lang: ref_datas[(args.methods[0], 0, lang)] for lang in langs}
#
#stats_wrt_size = {}
#rel_stats_wrt_size = {}

#for corpus_props, x_axis, is_log in zip([corpus_sizes, monolingual_perfs],
#                                        ["corpus size", f"monolingual {metric}"],
#                                        [True, False]):
#    sorted_langs = sorted(langs, key=lambda lang: corpus_props[lang])
#    xs = [corpus_props[lang] for lang in sorted_langs]
#    for method in args.methods:
#        stat = [max([ref_datas[(method, epoch, lang)] for epoch in args.epochs]) for lang in sorted_langs]
#        stats_wrt_size[method] = np.array(stat)
#    for method in stats_wrt_size:
#            rel_stats_wrt_size[method] = stats_wrt_size[method] - stats_wrt_size[args.control_method]
#    for stats, title in zip([stats_wrt_size, rel_stats_wrt_size], ["", f"relative to {args.control_method}"]):
#        fig, ax = plt.subplots(1,1)
#        for method, color in zip(args.methods, colors):
#            ax.plot(xs,
#                    stats[method],
#                    label=method, color=color)
#        
#        if is_og:
#            ax.set_xscale("log")
#        ax.set_xticks(xs)
#        ax.set_xticklabels(sorted_langs)
#        ax.legend()
#        ax.set_xlabel(x_axis)
#        ax.set_ylabel(metric)
#        ax.set_title(f"{args.model_selection} {metric} v.s. {x_axis} {title}")
#        fig.savefig(out_dir.joinpath(f"{metric}_vs_{x_axis.replace(' ', '_')}_{title.replace(' ', '_')}.png"))
#        plt.close(fig=fig)

if args.ana:
    gt_paths = {(method, epoch, lang): Path(f"{args.ckpt}/{method}_{epoch}_{lang}{maybe_ens(lang)}_{args.affix}").joinpath("result-gt.txt")
               for method, epoch, lang in product(args.methods, args.epochs, langs)}
    deplen_datas = {}
    for key, path in gt_paths.items():
        for i in range(1, args.dep_len_range):
            value = get_metric(path, f"LASdep{i}", "F1 Score")
            deplen_datas[(*key, i)] = value
    
    sentlen_datas = {}
    for key, path in gt_paths.items():
        for (start, end) in args.sent_len_bins:
            value = get_metric(path, f"LASlen{start}{str(end)[:2]}", "F1 Score")
            sentlen_datas[(*key, (start, end))] = value

if args.test:
    test_perfs = {}
    if args.decouple_model_selection:
        for method in args.methods:
            max_epoch, _ = max([(epoch, ref_datas[(method, epoch, lang)])\
                                for epoch in args.epochs], key=lambda x: x[1])
            for lang in langs:
                test_perfs[(method, lang)] = datas[(method, max_epoch, lang)]
    else:
        for method in args.methods:
            max_epoch, _ = max([(epoch, np.mean([ref_datas[(method, epoch, lang)] for lang in langs]))\
                                for epoch in args.epochs], key=lambda x: x[1])
            for lang in langs:
                test_perfs[(method, lang)] = datas[(method, max_epoch, lang)]

    for method in args.methods:
        mean_perfs = np.mean([test_perfs[(method, lang)] for lang in langs])
        print(f"{method}: {mean_perfs}")
        for lang in langs:
            print(f"{method} {lang}: {test_perfs[(method, lang)]}")
    avg_perfs = np.mean([test_perfs[(method, lang)] for method, lang in product(args.methods, langs)])
    print(f"avg perfs: {avg_perfs}")

    for lang in args.langs:
        mean_perfs = np.mean([test_perfs[(method, lang)] for method in args.methods])
        print(f"{lang}: {mean_perfs}")

    if args.ana:
        for lang in langs:
            fig, ax = plt.subplots(1,1)
            for method, color in zip(args.methods, colors):
                ax.plot(list(range(1, args.dep_len_range)), [deplen_datas[(method, epoch, lang, dep_len)] for dep_len in range(1, args.dep_len_range)],
                        label=method, color=color)
            ax.legend()
            ax.set_xlabel("dependency length")
            ax.set_ylabel(metric)
            ax.set_title(f"{args.model_selection} {metric} by dependency length: {lang}")
            fig.savefig(out_dir.joinpath(f"{metric}_{lang}_by_dep_len.png"))
            plt.close(fig=fig)

        for lang in langs:
            fig, ax = plt.subplots(1,1)
            for method, color in zip(args.methods, colors):
                ax.plot([len_bin[-1] for len_bin in args.sent_len_bins],
                        [sentlen_datas[(method, epoch, lang, len_bin)] for len_bin in args.sent_len_bins],
                        label=method, color=color)
            ax.legend()
            ax.set_xlabel("sentence length")
            ax.set_ylabel(metric)
            ax.set_title(f"{args.model_selection} {metric} by sentence length: {lang}")
            fig.savefig(out_dir.joinpath(f"{metric}_{lang}_by_sent_len.png"))
            plt.close(fig=fig)
        def gen_colors(num_tags, num_langs):
            return [sns.hls_palette(num_tags, l=l) for l in np.linspace(0.25, 0.75, num_langs)]
        
        plt.rc('text', usetex=True)
        colors = gen_colors(8, 3)
        fig, ax = plt.subplots(1,1)
        for method in args.methods:
            label, color_idx = postprocess_interp_method(method)
            ax.plot([len_bin[-1] for len_bin in args.sent_len_bins],
                    [np.mean([sentlen_datas[(method, epoch, lang, len_bin)] for lang in langs]) for len_bin in args.sent_len_bins],
                    label=label, color=colors[1][color_idx*3-2])
        ax.legend()
        ax.set_xlabel("sentence length")
        ax.set_ylabel(metric)
        ax.set_title(f"{args.model_selection} {metric} by sentence length over \\textit{{testing}} languages")
        fig.savefig(out_dir.joinpath(f"{metric}_by_sent_len.pdf"))
        plt.close(fig=fig)

    if args.by_typology:
        lang_leftness = {}
        with open('data/directionality.csv') as fp:
            for line in fp.read().splitlines():
                lang, leftness = line.split(",")
                lang_leftness[lang] = float(leftness)
    
        leftnesses = [lang_leftness[postprocess_lang_or_tb(lang)] for lang in langs]
        indices_by_leftness = np.argsort(leftnesses)
        langs_by_leftness = [langs[idx] for idx in indices_by_leftness]

        rel_plot(args.methods,
                 [langs[idx] for idx in indices_by_leftness],
                 [leftnesses[idx] for idx in indices_by_leftness],
                 lambda m, l: test_perfs[(m, l)],
                 'leftness',
                 metric,
                 f"{args.model_selection} {metric} w.r.t. leftness",
                 out_dir.joinpath(f"{metric}_by_leftness.pdf"))

        rel_plot(args.methods,
                 [langs[idx] for idx in indices_by_leftness],
                 [leftnesses[idx] for idx in indices_by_leftness],
                 lambda m, l: test_perfs[(m, l)] - test_perfs[(args.control_method, l)],
                 'leftness',
                 metric,
                 f"{args.model_selection} {metric} w.r.t. leftness relative to {args.control_method}",
                 out_dir.joinpath(f"{metric}_by_leftness_rel.pdf"))

    if args.by_tb_size:
        with open('data/all_tb_sizes.json') as fp:
            tb_sizes = json.load(fp)
    
        selected_tb_sizes = [get_tb_size(lang, tb_sizes) for lang in langs]
        indices_by_sizes = np.argsort(selected_tb_sizes)
        langs_by_sizes = [langs[idx] for idx in indices_by_sizes]

        rel_plot(args.methods,
                 [langs[idx] for idx in indices_by_sizes],
                 [selected_tb_sizes[idx] for idx in indices_by_sizes],
                 lambda m, l: test_perfs[(m, l)],
                 'treebank size',
                 metric,
                 f"{args.model_selection} {metric} by treebank size",
                 out_dir.joinpath(f"{metric}_by_tb_size.pdf"))

        rel_plot(args.methods,
                 [langs[idx] for idx in indices_by_sizes],
                 [selected_tb_sizes[idx] for idx in indices_by_sizes],
                 lambda m, l: test_perfs[(m, l)] - test_perfs[(args.control_method, l)],
                 'treebank size',
                 metric,
                 f"{args.model_selection} {metric} by treebank size relative to {args.control_method}",
                 out_dir.joinpath(f"{metric}_by_tb_size_rel.pdf"))

    setting = "zs" if args.affix == "zs" else "ft"
    name_control_method = args.name_control_method
    if args.scatter_typo_tbsize:
        assert args.by_typology and args.by_tb_size
        perfs = [test_perfs[(args.control_method, lang)] for lang in langs]
        tb_zh = "語料庫"
        scatter_plot(leftnesses, selected_tb_sizes, perfs, "方向性", f"{tb_zh}大小", metric, langs,
                     f"{tb_zh}大小與語言方向性對{metric}之影響：\n{name_control_method}",
                     out_dir.joinpath(f"dir_size_{metric.lower()}_{setting}_{args.control_method}.pdf"),
                     yscale='log' if args.log_scale else 'linear')

        scatter_plot(leftnesses, perfs, selected_tb_sizes, "方向性", metric, f"{tb_zh}大小", langs,
                     f"{tb_zh}大小與語言方向性對{metric}之影響：\n{name_control_method}",
                     out_dir.joinpath(f"dir_size_{metric.lower()}_{setting}_colortbsize_{args.control_method}.pdf"),
                     vmin=1, vmax=50000, cscale='log' if args.log_scale else 'linear')

        for method, name_method in zip(args.methods, args.name_methods):
            perfs = [test_perfs[(method, lang)] - test_perfs[(args.control_method, lang)] for lang in langs]
            scatter_plot(leftnesses, selected_tb_sizes, perfs, "方向性", f"{tb_zh}大小", f"{metric}進步量", langs,
                         f"{tb_zh}大小與語言方向性對{metric}之影響：\n{name_method}相對於{name_control_method}",
                         out_dir.joinpath(f"dir_size_{metric.lower()}_{setting}_{method}-to-{args.control_method}.pdf"),
                         yscale='log' if args.log_scale else 'linear',
                         vmin=-6, vmax=6)

            filename = out_dir.joinpath(f"dir_size_{metric.lower()}_{setting}_colortbsize_{method}-to-{args.control_method}.pdf")
            scatter_plot(leftnesses, perfs, selected_tb_sizes, "方向性", f"{metric}進步量", f"{tb_zh}大小", langs,
                         f"{tb_zh}大小與語言方向性對{metric}之影響：\n{name_method}相對於{name_control_method}",
                         filename, vmin=1, vmax=50000, cscale='log' if args.log_scale else 'linear', hline=True)
