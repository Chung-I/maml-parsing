import torch
import argparse
from pathlib import Path
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--start', type=str, help='start model')
parser.add_argument('--end', type=str, help='end model')
parser.add_argument('--theta', type=float, help='proportion')
parser.add_argument('--out-dir', type=str, help='path to save the model')

args = parser.parse_args()

start_model = torch.load(args.start)
end_model = torch.load(args.end)
assert start_model.keys() == end_model.keys()

interp_model = {}
for key in start_model.keys():
    interp_model[key] = start_model[key] * (1 - args.theta) + end_model[key] * args.theta

out_dir = Path(args.out_dir)
out_dir.mkdir(exist_ok=True)
if not out_dir.joinpath('config.json').exists():
    shutil.copy(Path(args.start).parent.joinpath('config.json'), out_dir.joinpath('config.json'))
if not out_dir.joinpath('vocabulary').exists():
    shutil.copytree(Path(args.start).parent.joinpath('vocabulary'), out_dir.joinpath('vocabulary'))
torch.save(interp_model, out_dir.joinpath(Path(args.end).name))
