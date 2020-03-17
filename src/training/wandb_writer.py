from typing import Any, Set, Optional, Callable, Dict
import logging
import os

import wandb
import torch

from allennlp.common.from_params import FromParams
from allennlp.models.model import Model

logger = logging.getLogger(__name__)


class WandBWriter(FromParams):
    """
    Class that handles WandB(and other) logging.

    # Parameters

    get_batch_num_total : Callable[[], int]
        A thunk that returns the number of batches so far. Most likely this will
        be a closure around an instance variable in your `Trainer` class.
    serialization_dir : str, optional (default = None)
        If provided, this is where the Tensorboard logs will be written.
    summary_interval : int, optional (default = 100)
        Most statistics will be written out only every this many batches.
    histogram_interval : int, optional (default = None)
        If provided, activation histograms will be written out every this many batches.
        If None, activation histograms will not be written out.
    should_log_parameter_statistics : bool, optional (default = True)
        Whether to log parameter statistics.
    should_log_learning_rate : bool, optional (default = False)
        Whether to log learning rate.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: Model,
        wandb_config: Dict[str, Any],
    ) -> None:

        log = wandb_config.pop("log", None)
        wandb.init(config=config, **wandb_config)
        wandb.watch(model, log=log)

    def log(self, metrics, step, epoch=None, prefix=None):
        log_dict = {'epoch': epoch} if epoch is not None else {}
        for k, v in metrics.items():
            key = f"{prefix}_{k}" if prefix else k
            log_dict[key] = v
        wandb.log(log_dict, step=step)
