import sys
from allennlp.common import Params
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('src')
parser.add_argument('tgt')
parser.add_argument('--collect-emb', action='store_true')
args = parser.parse_args()

params = Params.from_file(args.src)
if args.collect_emb:
    params['model']['type'] = 'collect-emb'
params.to_file(args.tgt)
