import os
import shutil
import logging
import argparse
import tarfile
from pathlib import Path
import json

from allennlp.common import Params

CONFIG_FILE = "config.json"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("archive", type=str, help="The archive file")
parser.add_argument("outdir", type=str, help="The output directory")
args = parser.parse_args()

outdir = Path(args.outdir)
outdir.mkdir(exist_ok=True)
archive_dirs = list(Path(".").glob(args.archive))
assert len(archive_dirs) > 1
config_file = archive_dirs[0] / CONFIG_FILE
config = Params.from_file(config_file)
config_submodels = []
for archive_dir in archive_dirs:
    #submodel_config_file = archive_dir / CONFIG_FILE
    #submodel_config = Params.from_file(submodel_config_file)["model"]
    #config_submodels.append(submodel_config.as_dict(quiet=True)
    submodel_config = {
        "type": "from_archive",
        "archive_file": str(archive_dir),
    }
    config_submodels.append(submodel_config)
config["model"] = {
    "type": "parser-ensemble",
    "submodels": config_submodels,
}

config.to_file(outdir / CONFIG_FILE)

