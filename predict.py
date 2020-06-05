"""
Predict conllu files given a trained model
"""

import os
import shutil
import logging
import argparse
import tarfile
from pathlib import Path
import json

from allennlp.common import Params
from allennlp.common.util import import_submodules
from allennlp.models.archival import archive_model

from src import util

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("archive", type=str, help="The archive file")
parser.add_argument("input_file", type=str, help="The input file to predict")
parser.add_argument("pred_file", type=str, help="The output prediction file")
parser.add_argument("--include-package", type=str, help="The included package.")
# parser.add_argument("--overrides", type=str, help="config files to be overrided")
parser.add_argument("--eval-file", default=None, type=str,
                    help="If set, evaluate the prediction and store it in the given file")
parser.add_argument("--cuda-device", default=0, type=int, help="CUDA device number; set to -1 for CPU")
parser.add_argument("--batch-size", default=1, type=int, help="The size of each prediction batch")
parser.add_argument("--lazy", action="store_true", help="Lazy load dataset")
parser.add_argument("--raw-text", action="store_true", help="Input raw sentences, one per line in the input file.")

args = parser.parse_args()

import_submodules(args.include_package)

archive_dir = Path(args.archive)

config_file = archive_dir / "config.json"
overrides = {"dataset_reader": {"read_dependencies": False,
                                "max_len": 10000},
             "validation_dataset_reader": {"read_dependencies": False,
                                           "max_len": 10000,}}
configs = [Params(overrides), Params.from_file(config_file)]

params = util.merge_configs(configs)
if params["model"]["type"] == "from_archive":
    model_config_file = str(Path(params["model"]["archive_file"]).parent.joinpath("config.json"))
    model_config = Params.from_file(model_config_file)["model"]
    params['model'] = model_config.as_dict(quiet=True)
    try:
        lm_name = f"_{os.environ['LM']}" if os.environ.get("LM") else ""
        if os.environ["SHIFT"] == "1":
            params['model']["ft_lang_mean_dir"] = f"ckpts/{os.environ['FT_LANG']}_mean{lm_name}"
    except (AttributeError, KeyError) as e:
        pass
    try:
        lm_name = f"_{os.environ['LM']}" if os.environ.get("LM") else ""
        if os.environ["ZS_SHIFT"] == "1":
            params['model']["zs_lang_mean_dir"] = [f"ckpts/{os.environ['ANCHOR']}_mean{lm_name}",
                                                   f"ckpts/{os.environ['FT_LANG']}_mean{lm_name}"]
    except (AttributeError, KeyError) as e:
        pass
    try:
        shutil.copy(str(config_file), str(archive_dir / "old_config.json"))
        params.to_file(str(archive_dir / "config.json"))
    except:
        shutil.copy(str(archive_dir / "old_config.json"), str(config_file))
    
#predictor = "udify_predictor" if not args.raw_text else "udify_text_predictor"
predictor = "ud_predictor"
assert not args.raw_text, "currently support only conllu input"

if not args.eval_file:
    util.predict_model_with_archive(predictor, params, archive_dir, args.input_file, args.pred_file,
                                    batch_size=args.batch_size)
else:
    util.predict_and_evaluate_model_with_archive(predictor, params, archive_dir, args.input_file,
                                                 args.pred_file, args.eval_file, batch_size=args.batch_size)
