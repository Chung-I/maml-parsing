import argparse
import os
import torch
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-s')
parser.add_argument('-n')
args = parser.parse_args()

def maybe_add_pretrained_embeddings(serialization_dir, weights_file, epoch):
    if torch.cuda.is_available():
        model_state = torch.load(os.path.join(serialization_dir, weights_file))
        pretrained_model_state = torch.load(os.path.join("ckpts/meta-fixed-base",
                                                         "model_state_epoch_0.th"))
    else:
        model_state = torch.load(os.path.join(serialization_dir, weights_file),
                                 map_location=torch.device('cpu'))
        pretrained_model_state = torch.load(os.path.join("ckpts/meta-fixed-base",
                                                         "model_state_epoch_0.th"),
                                            map_location=torch.device('cpu'))

    if len(set(pretrained_model_state.keys()) - set(model_state.keys())) > 0:
        full_model_state = {}
        for key, value in pretrained_model_state.items():
            if not key in model_state:
                full_model_state[key] = value
            else:
                full_model_state[key] = model_state[key]
        full_model_weights_file = f"full_epoch_{epoch}.th"
        torch.save(full_model_state, os.path.join(serialization_dir, full_model_weights_file))
    else:
        new_model_state = model_state
        full_model_weights_file = weights
    return full_model_weights_file


from allennlp.models.archival import archive_model
if args.n:
    weights = f"model_state_epoch_{args.n}.th"
    archive_path = os.path.join(args.s, f"model_epoch_{args.n}.tar.gz")
else:
    weights = "best.th"
    archive_path = os.path.join(args.s, f"model.tar.gz")

weights = maybe_add_pretrained_embeddings(args.s, weights, args.n)

archive_model(args.s, weights, archive_path)
os.remove(os.path.join(args.s, weights))
