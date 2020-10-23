import argparse
from subprocess import Popen, PIPE
from pathlib import Path
from itertools import permutations
from plot_utils import get_path, has_path_and_not_empty, get_gold_path
import jsonlines

def run_t_test(script_path, gold_path, patha, pathb, n_trials):
    script = ['python3', script_path]
    options = [f"--n-trials", str(n_trials), '--skip-if-less']
    full_command = script + [gold_path, patha, pathb] + options
    print(" ".join(full_command))
    proc = Popen(full_command,
                 stdout=PIPE)
    output = proc.communicate()[0].decode()
    return eval(output)

def main(args):
    result_file = 'result-gt.conllu' if args.gt else 'result.conllu'
    with jsonlines.open(args.out_file, mode='w') as writer:
        for epoch, suffix in zip(args.epochs, args.suffixes):
            for lang in args.langs:
                gold_path = get_gold_path(lang)
                for ma, mb in permutations(args.methods, 2):
                    patha = get_path(args.ckpt, ma, epoch, lang, suffix, result_file)
                    pathb = get_path(args.ckpt, mb, epoch, lang, suffix, result_file)
                    result = run_t_test(args.script_path, str(gold_path), str(patha),
                                        str(pathb), args.n_trials)
                    key = ",".join([lang, epoch, suffix, ma, mb])
                    writer.write({key: result})
                    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffixes', nargs='+')
    parser.add_argument('--epochs', nargs='+')
    parser.add_argument('--langs', nargs='+')
    parser.add_argument('--methods', nargs='+')
    parser.add_argument('--ckpt', default='ckpts')
    parser.add_argument('--gt', action='store_true')
    parser.add_argument('--script-path', default='utils/t_test.py')
    parser.add_argument('--n-trials', type=int, default=10000)
    parser.add_argument('--out-file', type=str)

    args = parser.parse_args()

    assert len(args.suffixes) == len(args.epochs)

    main(args)
    
