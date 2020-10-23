from typing import List

import argparse
import os
import shutil

import torch
import json
from pathlib import Path

from allennlp.common.checks import ConfigurationError
from allennlp.common import Params
from allennlp.common.params import with_fallback

parser = argparse.ArgumentParser()
parser.add_argument('-s')
args = parser.parse_args()

def maybe_add_pretrained_embeddings(serialization_dir, weights_file):
    is_tmp_weight = False
    if torch.cuda.is_available():
        model_state = torch.load(os.path.join(serialization_dir, weights_file))
        pretrained_model_state = torch.load(os.path.join(f"{os.environ['BASE_MODEL']}",
                                                         weights_file))
    else:
        model_state = torch.load(os.path.join(serialization_dir, weights_file),
                                 map_location=torch.device('cpu'))
        pretrained_model_state = torch.load(os.path.join(f"{os.environ['BASE_MODEL']}",
                                                         weights_file),
                                            map_location=torch.device('cpu'))

    if len(set(pretrained_model_state.keys()) - set(model_state.keys())) > 0:
        is_tmp_weight = True
        full_model_state = {}
        for key, value in pretrained_model_state.items():
            if not key in model_state:
                full_model_state[key] = value
            else:
                full_model_state[key] = model_state[key]
        for key, value in model_state.items():
            if not key in pretrained_model_state:
                full_model_state[key] = value

        torch.save(full_model_state, os.path.join(serialization_dir, weights_file))

weights = "best.th"

maybe_add_pretrained_embeddings(args.s, weights)
