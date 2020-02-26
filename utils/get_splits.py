import glob
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--langs')
parser.add_argument('--ud-root')
args = parser.parse_args()

with open(args.langs) as fp:
    train_tbs = [line.split()[0] for line in fp.read().splitlines()]

print(f"total {len(train_tbs)} train treebanks: {train_tbs}")

dev_tbs = set()
for train_tb in train_tbs:
    if list(glob.glob(os.path.join(args.ud_root, f"*/{train_tb}*dev.conllu"))):
        dev_tbs.add(train_tb)

print(f"total {len(dev_tbs)} dev treebanks: {list(dev_tbs)}")

test_tbs = [conllu.name.split("-")[0] for conllu in Path(args.ud_root).glob("conll18-ud-test/*.conllu")]

print(f"total {len(test_tbs)} test treebanks: {list(test_tbs)}")
