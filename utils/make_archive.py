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


def merge_configs(configs: List[Params]) -> Params:
    """
    Merges a list of configurations together, with items with duplicate keys closer to the front of the list
    overriding any keys of items closer to the rear.
    :param configs: a list of AllenNLP Params
    :return: a single merged Params object
    """
    while len(configs) > 1:
        overrides, config = configs[-2:]
        configs = configs[:-2]

        if "udify_replace" in overrides:
            replacements = [replace.split(".") for replace in overrides.pop("udify_replace")]
            for replace in replacements:
                obj = config
                try:
                    for key in replace[:-1]:
                        obj = obj[key]
                except KeyError:
                    raise ConfigurationError(f"Config does not have key {key}")
                obj.pop(replace[-1])

        configs.append(Params(with_fallback(preferred=overrides.params, fallback=config.params)))

    return configs[0]


CONFIG_NAME = "config.json"
OLD_CONFIG_NAME = "old_config.json"

parser = argparse.ArgumentParser()
parser.add_argument('-s')
parser.add_argument('-n')
parser.add_argument('-m')
args = parser.parse_args()

def maybe_add_pretrained_embeddings(serialization_dir, weights_file, epoch):
    is_tmp_weight = False
    if torch.cuda.is_available():
        model_state = torch.load(os.path.join(serialization_dir, weights_file))
        pretrained_model_state = torch.load(os.path.join(f"ckpts/{os.environ['BASE_MODEL']}",
                                                         "model_state_epoch_0.th"))
    else:
        model_state = torch.load(os.path.join(serialization_dir, weights_file),
                                 map_location=torch.device('cpu'))
        pretrained_model_state = torch.load(os.path.join(f"ckpts/{os.environ['BASE_MODEL']}",
                                                         "model_state_epoch_0.th"),
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

        full_model_weights_file = f"full_epoch_{epoch}.th"
        torch.save(full_model_state, os.path.join(serialization_dir, full_model_weights_file))
    else:
        new_model_state = model_state
        full_model_weights_file = weights
    return full_model_weights_file, is_tmp_weight


from allennlp.models.archival import archive_model
if args.n:
    weights = f"model_state_epoch_{args.n}.th"
    archive_path = os.path.join(args.s, f"model_epoch_{args.n}.tar.gz")
else:
    weights = "best.th"
    archive_path = os.path.join(args.s, f"model.tar.gz")

if os.environ.get("BASE_MODEL") is None:
    is_tmp_weight = False
else:
    weights, is_tmp_weight = maybe_add_pretrained_embeddings(args.s, weights, args.n)

config_fname =os.path.join(args.s, CONFIG_NAME)
old_config_fname =os.path.join(args.s, OLD_CONFIG_NAME)
if args.m:
    shutil.copy(config_fname, old_config_fname)
    full_config = merge_configs([Params.from_file(args.m), Params.from_file(config_fname)])
    full_config.to_file(config_fname)

archive_model(args.s, weights, archive_path)
if is_tmp_weight:
    os.remove(os.path.join(args.s, weights))

if args.m:
    os.rename(old_config_fname, config_fname)
