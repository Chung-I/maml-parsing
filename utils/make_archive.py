import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-s')
parser.add_argument('-n')
args = parser.parse_args()

from allennlp.models.archival import archive_model
weights = f"model_state_epoch_{args.n}.th"
archive_path = os.path.join(args.s, f"model_epoch_{args.n}.tar.gz")
archive_model(args.s, weights, archive_path)
