import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors
import matplotlib
from matplotlib.patches import Rectangle
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
from plot_utils import get_path, get_metric, maybe_no_num, read_t_test_stats
from plot_utils import suffix2epoch, postprocess_lang_or_tb, has_path_and_not_empty
import jsonlines

sns.set()
#matplotlib.rc('font',**{'family':'serif','serif':['Noto Serif CJK TC']})
plt.rcParams.update({"text.usetex": True})
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['text.latex.preamble'] = [
           '\\usepackage{CJK}',
           r'\AtBeginDocument{\begin{CJK}{UTF8}{bsmi}}',
           r'\AtEndDocument{\end{CJK}}',
]

stanford_perfs = {"wo":  [83.25, 77.05],
                  "gd":  [77.90, 70.81],
                  "te":  [89.32, 79.89],
                  "cop": [61.94, 59.71],
                  "be":  [69.28, 63.88],
                  "mr":  [66.42, 52.64],
                  "mt":  [83.31, 78.15],
                  "ta":  [61.23, 55.76]}

def main(args):

    out_dir = Path(args.out_dir)
    with open("data/ensemble_langs.txt") as fp:
        ens_langs = fp.read().splitlines()

    langs = args.langs
    methods = args.methods
    suffixes = args.suffixes
    epochs = args.epochs

    result_file = "result-gt.txt" if args.gt else "result.txt"

    maybe_merge_lang = lambda lang: postprocess_lang_or_tb(lang) if args.merge_lang else lang

    paths = {(method, epoch, lang, suffix): get_path(args.ckpt, method, epoch, lang, suffix, result_file)
                for method, epoch, lang, suffix in product(methods, epochs, langs, suffixes)}
        
    if not args.merge_lang:
        assert all([has_path_and_not_empty(path) for key, path in paths.items()]),\
            f"path that doesn't exists: {[path for _, path in paths.items() if not has_path_and_not_empty(path)]}"
    else:
        paths = {(method, epoch, maybe_merge_lang(lang), suffix): path
                for (method, epoch, lang, suffix), path in paths.items() if has_path_and_not_empty(path)}


    flatten = lambda l: [item for sublist in l for item in sublist]
    # setup x axes data for langs

    metricviews = [metric.split("-") for metric in args.metrics]
    metrics = [metric for metric, _ in metricviews]
    datas = {key: [get_metric(path, metric, view) for metric, view in metricviews] for key, path in paths.items()}

    t_test_stats = read_t_test_stats(args.t_test_stats, methods, epochs, langs, suffixes, metrics)

    xs = list(range(len(suffixes)))

    def plot(xss, yss, xticks, xtick_colors, line_names, title, xlabel, ylabels, highlights,
            logscale=False, ylim=(0, 100), colors=None, hatches=(None, '.')):
        fig, ax = plt.subplots(1, 1)

        if logscale:
            ax.set_xscale('log')
            xss = [list(map(lambda x: x+1, xs)) for xs in xss]

        ax.set_xticks(xss[0])
        xticks = [re.sub(r'\\n', r'\n', xtick) for xtick in xticks]
        ax.set_xticklabels(xticks, color=xtick_colors)
        [t.set_color(color) for color, t in zip(xtick_colors, ax.get_xticklabels())]

        ax.set_ylim(ylim)

        width = 0.7
        n_bars = len(xss)
        shifts = np.linspace(-0.5, 0.5, n_bars + 1) + 0.5 / n_bars
        shifts = shifts * width
        extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        list_handles = [[extra for i in range(len(ylabels)+1)]]

        for idx, (xs, ys, line_name, shift, highlight) in enumerate(zip(xss, yss, line_names, shifts, highlights)):
            handles = [extra]
            for jdx, (hatch, ylabel) in enumerate(zip(hatches, ylabels)):
                color = colors[idx] if colors is not None else None
                ec = str(color)
                #color = 'none' if hatch is not None else color
                #alpha = 0.7 if hatch is not None else 1
                handle = ax.bar(list(map(lambda x: x + shift, xs)),
                                list(map(lambda x: x[jdx], ys)),
                                width / n_bars,
                                #[(width / n_bars) * (1 + 0.2 * (hl[ylabel] - 1)) for hl in highlight],
                                #label=f"{line_name}-{ylabel}",
                                color=color,
                                #alpha=alpha,
                                linewidth=[0.5 + 0.5 * hl[ylabel] for hl in highlight],
                                hatch=hatch, ec='k')
                handles.append(handle)
            list_handles.append(handles)
                #ax.bar(list(map(lambda x: x + shift, xs)),
                #       list(map(lambda x: x[jdx], ys)),
                #       width / n_bars,
                #       color='none',
                #       ec='k')
        label_empty = [""]
        list_labels = [label_empty + ylabels]
        list_labels += [[line_name] + label_empty * len(ylabels) for line_name in line_names]

        ax.set_xlabel(xlabel)
        ax.set_ylabel("/".join(ylabels))
        #ax.legend(frameon=False)
        legend = ax.legend(flatten(list_handles), flatten(list_labels),
                        bbox_to_anchor=(0.5, 1.25), fancybox=True,
                        loc = 9, ncol = len(xss) + 1, shadow = True, handletextpad=-2)
        ax.set_title(title, y=1.28)
        return fig, ax.title, legend

    colors = args.colors
    lang_colors = ['k' if lang not in args.langs_not_in_mbert else 'b' for lang in langs]
    if args.stanford:
        colors.append("#FFFFFF")
        name_methods = args.name_methods
        name_methods.append("Stanford")
        _methods = methods + ["stanford"]
    for (name_suffix, suffix), epoch in product(zip(args.name_suffixes, args.suffixes), epochs):
        #xss = [[i for i, _ in enumerate(suffixes)] for method in methods]
        xss = [list(range(len(langs))) for method in methods]
        yss = [[datas.get((method, epoch, maybe_merge_lang(lang), suffix)) for lang in langs] for method in methods]
        if args.stanford:
            xss.append(list(range(len(langs))))
            stanford_ys = [stanford_perfs[lang] for lang in langs]
            yss.append(stanford_ys)
        xss, yss = zip(*[zip(*list(filter(lambda z: z[0] != None and z[1] != None, zip(xs, ys)))) for xs, ys in zip(xss, yss)])
        highlights = [[t_test_stats.get((method, epoch, lang, suffix),
                                        {m: 0 for m in metrics})
                       for lang in langs]
                      for method in _methods]
        #title = f"{args.title}: {name_suffix}"
        title = args.title
        fig, plttitle, lgd = plot(xss, yss, list(map(postprocess_lang_or_tb, langs)),
                                  lang_colors,
                                  name_methods, title, args.xlabel,
                                  metrics, highlights, args.logscale,
                                  colors=colors, hatches=args.hatches)
        if args.filename:
            out_filename = f"{args.filename}_{suffix}.pdf"
        else:
            out_filename = f"{args.title}_{suffix}_epoch{epoch}.pdf"
        fig.savefig(out_dir.joinpath(out_filename),
                    box_extra_artists=(lgd,plttitle), bbox_inches='tight')
        plt.close(fig=fig)

#for epoch in epochs:
#    xss = [[suffix2epoch(suffix) for suffix in suffixes] for method in methods]
#    yss = [[np.mean([datas.get((method, epoch, maybe_merge_lang(lang), suffix)) for lang in langs]) for suffix in suffixes] for method in methods]
#    title = f"{args.title}"
#    fig = plot(xss, yss, args.name_suffixes, args.name_methods, title, metric, args.logscale, colors)
#    fig.savefig(out_dir.joinpath(f"{args.title}_epoch{epoch}.pdf"))
#    plt.close(fig=fig)

if __name__ == '__main__':
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
    parser.add_argument('--merge-lang', action='store_true',
                        help='merge different synthetic datasets of the same language')
    parser.add_argument('--metrics', default=['UAS-F1 Score', 'LAS-F1 Score'])
    parser.add_argument('--hatches', default=[r'////', None])
    parser.add_argument('--langs-not-in-mbert', nargs='+', default=[])
    parser.add_argument('--out-dir')
    parser.add_argument('--t-test-stats')
    parser.add_argument('--filename')
    parser.add_argument('--stanford', action='store_true')
    parser.add_argument('--xlabel', default="語言")
    args = parser.parse_args()
    main(args)
