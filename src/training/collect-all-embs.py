import datetime
import logging
import math
import os
import re
import time
import traceback
from collections import defaultdict
from typing import Dict, Optional, Tuple, Union, Iterable, Any

import torch
import torch.distributed as dist
import torch.optim.lr_scheduler
from torch.nn.parallel import DistributedDataParallel

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError, parse_cuda_device, check_for_gpu
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import dump_metrics, gpu_memory_mb, peak_memory_mb, lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.models.model import Model
from allennlp.nn import util as nn_util
from allennlp.training import util as training_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler, SlantedTriangular
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer_base import TrainerBase

from src.training.wandb_writer import WandBWriter
from src.training.tensorboard_writer import TensorboardWriter
from src.training.util import as_flat_dict, filter_state_dict

logger = logging.getLogger(__name__)


@TrainerBase.register("collect-all-embs")
class Trainer(TrainerBase):
    def __init__(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        iterator: DataIterator,
        train_dataset: Iterable[Instance],
        validation_dataset: Optional[Iterable[Instance]] = None,
        patience: Optional[int] = None,
        validation_metric: str = "-loss",
        validation_iterator: DataIterator = None,
        shuffle: bool = True,
        num_epochs: int = 20,
        serialization_dir: Optional[str] = None,
        num_serialized_models_to_keep: int = 20,
        keep_serialized_model_every_num_seconds: int = None,
        checkpointer: Checkpointer = None,
        model_save_interval: float = None,
        cuda_device: int = -1,
        grad_norm: Optional[float] = None,
        grad_clipping: Optional[float] = None,
        learning_rate_scheduler: Optional[LearningRateScheduler] = None,
        momentum_scheduler: Optional[MomentumScheduler] = None,
        summary_interval: int = 100,
        histogram_interval: int = None,
        should_log_parameter_statistics: bool = True,
        should_log_learning_rate: bool = False,
        log_batch_size_period: Optional[int] = None,
        moving_average: Optional[MovingAverage] = None,
        distributed: bool = False,
        local_rank: int = 0,
        world_size: int = 1,
        num_gradient_accumulation_steps: int = 1,
        writer: WandBWriter = None,
    ) -> None:
        """
        A trainer for doing supervised learning. It just takes a labeled dataset
        and a `DataIterator`, and uses the supplied `Optimizer` to learn the weights
        for your model over some fixed number of epochs. You can also pass in a validation
        dataset and enable early stopping. There are many other bells and whistles as well.

        # Parameters

        model : `Model`, required.
            An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
            their `forward` method returns a dictionary with a "loss" key, containing a
            scalar tensor representing the loss function to be optimized.

            If you are training your model using GPUs, your model should already be
            on the correct device. (If you use `Trainer.from_params` this will be
            handled for you.)
        optimizer : `torch.nn.Optimizer`, required.
            An instance of a Pytorch Optimizer, instantiated with the parameters of the
            model to be optimized.
        iterator : `DataIterator`, required.
            A method for iterating over a `Dataset`, yielding padded indexed batches.
        train_dataset : `Dataset`, required.
            A `Dataset` to train on. The dataset should have already been indexed.
        validation_dataset : `Dataset`, optional, (default = None).
            A `Dataset` to evaluate on. The dataset should have already been indexed.
        patience : Optional[int] > 0, optional (default=None)
            Number of epochs to be patient before early stopping: the training is stopped
            after `patience` epochs with no improvement. If given, it must be `> 0`.
            If None, early stopping is disabled.
        validation_metric : str, optional (default="loss")
            Validation metric to measure for whether to stop training using patience
            and whether to serialize an `is_best` model each epoch. The metric name
            must be prepended with either "+" or "-", which specifies whether the metric
            is an increasing or decreasing function.
        validation_iterator : `DataIterator`, optional (default=None)
            An iterator to use for the validation set.  If `None`, then
            use the training `iterator`.
        shuffle : `bool`, optional (default=True)
            Whether to shuffle the instances in the iterator or not.
        num_epochs : int, optional (default = 20)
            Number of training epochs.
        serialization_dir : str, optional (default=None)
            Path to directory for saving and loading model files. Models will not be saved if
            this parameter is not passed.
        num_serialized_models_to_keep : `int`, optional (default=20)
            Number of previous model checkpoints to retain.  Default is to keep 20 checkpoints.
            A value of None or -1 means all checkpoints will be kept.
        keep_serialized_model_every_num_seconds : `int`, optional (default=None)
            If num_serialized_models_to_keep is not None, then occasionally it's useful to
            save models at a given interval in addition to the last num_serialized_models_to_keep.
            To do so, specify keep_serialized_model_every_num_seconds as the number of seconds
            between permanently saved checkpoints.  Note that this option is only used if
            num_serialized_models_to_keep is not None, otherwise all checkpoints are kept.
        checkpointer : `Checkpointer`, optional (default=None)
            An instance of class Checkpointer to use instead of the default. If a checkpointer is specified,
            the arguments num_serialized_models_to_keep and keep_serialized_model_every_num_seconds should
            not be specified. The caller is responsible for initializing the checkpointer so that it is
            consistent with serialization_dir.
        model_save_interval : `float`, optional (default=None)
            If provided, then serialize models every `model_save_interval`
            seconds within single epochs.  In all cases, models are also saved
            at the end of every epoch if `serialization_dir` is provided.
        cuda_device : `int`, optional (default = -1)
            An integer specifying the CUDA device(s) to use for this process. If -1, the CPU is used.
            Data parallelism is controlled at the allennlp train level, so each trainer will have a single
            GPU.
        grad_norm : `float`, optional, (default = None).
            If provided, gradient norms will be rescaled to have a maximum of this value.
        grad_clipping : `float`, optional (default = `None`).
            If provided, gradients will be clipped `during the backward pass` to have an (absolute)
            maximum of this value.  If you are getting `NaNs` in your gradients during training
            that are not solved by using `grad_norm`, you may need this.
        learning_rate_scheduler : `LearningRateScheduler`, optional (default = None)
            If specified, the learning rate will be decayed with respect to
            this schedule at the end of each epoch (or batch, if the scheduler implements
            the `step_batch` method). If you use :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`,
            this will use the `validation_metric` provided to determine if learning has plateaued.
            To support updating the learning rate on every batch, this can optionally implement
            `step_batch(batch_num_total)` which updates the learning rate given the batch number.
        momentum_scheduler : `MomentumScheduler`, optional (default = None)
            If specified, the momentum will be updated at the end of each batch or epoch
            according to the schedule.
        summary_interval : `int`, optional, (default = 100)
            Number of batches between logging scalars to tensorboard
        histogram_interval : `int`, optional, (default = `None`)
            If not None, then log histograms to tensorboard every `histogram_interval` batches.
            When this parameter is specified, the following additional logging is enabled:
                * Histograms of model parameters
                * The ratio of parameter update norm to parameter norm
                * Histogram of layer activations
            We log histograms of the parameters returned by
            `model.get_parameters_for_histogram_tensorboard_logging`.
            The layer activations are logged for any modules in the `Model` that have
            the attribute `should_log_activations` set to `True`.  Logging
            histograms requires a number of GPU-CPU copies during training and is typically
            slow, so we recommend logging histograms relatively infrequently.
            Note: only Modules that return tensors, tuples of tensors or dicts
            with tensors as values currently support activation logging.
        should_log_parameter_statistics : `bool`, optional, (default = True)
            Whether to send parameter statistics (mean and standard deviation
            of parameters and gradients) to tensorboard.
        should_log_learning_rate : `bool`, optional, (default = False)
            Whether to send parameter specific learning rate to tensorboard.
        log_batch_size_period : `int`, optional, (default = `None`)
            If defined, how often to log the average batch size.
        moving_average : `MovingAverage`, optional, (default = None)
            If provided, we will maintain moving averages for all parameters. During training, we
            employ a shadow variable for each parameter, which maintains the moving average. During
            evaluation, we backup the original parameters and assign the moving averages to corresponding
            parameters. Be careful that when saving the checkpoint, we will save the moving averages of
            parameters. This is necessary because we want the saved model to perform as well as the validated
            model if we load it later. But this may cause problems if you restart the training from checkpoint.
        distributed : `bool`, optional, (default = False)
            If set, PyTorch's `DistributedDataParallel` is used to train the model in multiple GPUs. This also
            requires `world_size` to be greater than 1.
        local_rank : `int`, optional, (default = 0)
            This is the unique identifier of the `Trainer` in a distributed process group. The GPU device id is
            used as the rank.
        world_size : `int`, (default = 1)
            The number of `Trainer` workers participating in the distributed training.
        num_gradient_accumulation_steps : `int`, optional, (default = 1)
            Gradients are accumulated for the given number of steps before doing an optimizer step. This can
            be useful to accommodate batches that are larger than the RAM size. Refer Thomas Wolf's
            [post](https://tinyurl.com/y5mv44fw) for details on Gradient Accumulation.
        """
        super().__init__(serialization_dir, cuda_device, distributed, local_rank, world_size)

        # I am not calling move_to_gpu here, because if the model is
        # not already on the GPU then the optimizer is going to be wrong.
        self.model = model

        self.iterator = iterator
        self._validation_iterator = validation_iterator
        self.shuffle = shuffle
        self.optimizer = optimizer
        self.train_data = train_dataset
        self._validation_data = validation_dataset
        self._means = defaultdict(list)
        self._num_tokens = defaultdict(list)
        self.overall_mean = {}

        if patience is None:  # no early stopping
            if validation_dataset:
                logger.warning(
                    "You provided a validation dataset but patience was set to None, "
                    "meaning that early stopping is disabled"
                )
        elif (not isinstance(patience, int)) or patience <= 0:
            raise ConfigurationError(
                '{} is an invalid value for "patience": it must be a positive integer '
                "or None (if you want to disable early stopping)".format(patience)
            )

        # For tracking is_best_so_far and should_stop_early
        self._metric_tracker = MetricTracker(patience, validation_metric)
        # Get rid of + or -
        self._validation_metric = validation_metric[1:]

        self._num_epochs = num_epochs

        if checkpointer is not None:
            # We can't easily check if these parameters were passed in, so check against their default values.
            # We don't check against serialization_dir since it is also used by the parent class.
            if (
                num_serialized_models_to_keep != 20
                or keep_serialized_model_every_num_seconds is not None
            ):
                raise ConfigurationError(
                    "When passing a custom Checkpointer, you may not also pass in separate checkpointer "
                    "args 'num_serialized_models_to_keep' or 'keep_serialized_model_every_num_seconds'."
                )
            self._checkpointer = checkpointer
        else:
            self._checkpointer = Checkpointer(
                serialization_dir,
                keep_serialized_model_every_num_seconds,
                num_serialized_models_to_keep,
            )

        self._model_save_interval = model_save_interval

        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping

        self._learning_rate_scheduler = learning_rate_scheduler
        self._momentum_scheduler = momentum_scheduler
        self._moving_average = moving_average

        # We keep the total batch number as an instance variable because it
        # is used inside a closure for the hook which logs activations in
        # `_enable_activation_logging`.
        self._batch_num_total = 0

        if writer is not None:
            self._writer = writer
        else:
            self._writer = TensorboardWriter(
                    get_batch_num_total=lambda: self._batch_num_total,
                    serialization_dir=serialization_dir,
                    summary_interval=summary_interval,
                    histogram_interval=histogram_interval,
                    should_log_parameter_statistics=should_log_parameter_statistics,
                    should_log_learning_rate=should_log_learning_rate)

        self._log_batch_size_period = log_batch_size_period

        self._last_log = 0.0  # time of last logging

        self._num_gradient_accumulation_steps = num_gradient_accumulation_steps

        # Using `DistributedDataParallel`(ddp) brings in a quirk wrt AllenNLP's `Model` interface and its
        # usage. A `Model` object is wrapped by `ddp`, but assigning the wrapped model to `self.model`
        # will break the usages such as `Model.get_regularization_penalty`, `Model.get_metrics`, etc.
        #
        # Hence a reference to Pytorch's object is maintained in the case of distributed training and in the
        # normal case, reference to `Model` is retained. This reference is only used in
        # these places: `model.__call__`, `model.train` and `model.eval`.
        if self._distributed:
            self._pytorch_model = DistributedDataParallel(
                self.model, device_ids=[self.cuda_device], find_unused_parameters=True
            )
        else:
            self._pytorch_model = self.model

    def rescale_gradients(self) -> Optional[float]:
        return training_util.rescale_gradients(self.model, self._grad_norm)

    def batch_loss(self, batch: TensorDict, for_training: bool) -> torch.Tensor:
        """
        Does a forward pass on the given batches and returns the `loss` value in the result.
        If `for_training` is `True` also applies regularization penalty.
        """
        batch = nn_util.move_to_device(batch, self.cuda_device)
        output_dict = self._pytorch_model(**batch)
        mean = output_dict.pop("mean")
        num_tokens = output_dict.pop("num_tokens")
        self._means["overall"].append(mean)
        self._num_tokens["overall"].append(num_tokens)
        matches = [re.match("(.*)_mean", key) for key, value in output_dict.items()]
        tags = [match.group(1) for match in matches if match is not None]
        for tag in tags:
            self._means[tag].append(output_dict[f"{tag}_mean"])
            self._num_tokens[tag].append(output_dict[f"{tag}_num_tokens"])

        return output_dict

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs)
        peak_cpu_usage = peak_memory_mb()
        logger.info(f"Peak CPU memory usage MB: {peak_cpu_usage}")
        gpu_usage = []
        for gpu, memory in gpu_memory_mb().items():
            gpu_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage MB: {memory}")

        train_loss = 0.0
        # Set the model to "train" mode.
        self._pytorch_model.train()

        # Get tqdm for the training batches
        batch_generator = self.iterator(self.train_data, num_epochs=1, shuffle=self.shuffle)
        batch_group_generator = lazy_groups_of(
            batch_generator, self._num_gradient_accumulation_steps
        )
        num_training_batches = math.ceil(
            self.iterator.get_num_batches(self.train_data) / self._num_gradient_accumulation_steps
        )

        # Having multiple tqdm bars in case of distributed training will be a mess. Hence only the master's
        # progress is shown
        if self._master:
            batch_group_generator_tqdm = Tqdm.tqdm(
                batch_group_generator, total=num_training_batches
            )
        else:
            batch_group_generator_tqdm = batch_group_generator

        self._last_log = time.time()
        last_save_time = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        logger.info("Training")

        cumulative_batch_group_size = 0
        for batch_group in batch_group_generator_tqdm:
            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            self.optimizer.zero_grad()

            for batch in batch_group:
                output_dict = self.batch_loss(batch, for_training=True)

        # Let all workers finish their epoch before computing
        # the final statistics for the epoch.
        if self._distributed:
            dist.barrier()

        for tag in self._means:
            means = self._means[tag]
            num_tokens = self._num_tokens[tag]
            # total_val = torch.stack([mean * num_token \
            #     for mean, num_token in zip(means, num_tokens)], dim=0).sum(dim=0)
            total_val = sum([mean * num_token for mean, num_token in zip(means, num_tokens)])
            total_num_tokens = sum(num_tokens)
            self.overall_mean[f"{tag}_mean"] = total_val / total_num_tokens

        means = self._means["overall"]
        num_tokens = self._num_tokens["overall"]
        total_val = torch.stack([mean * num_token \
            for mean, num_token in zip(means, num_tokens)], dim=0).sum(dim=0)
        total_num_tokens = sum(num_tokens)
        self.overall_mean["mean"] = total_val / total_num_tokens

    def _validation_loss(self) -> Tuple[float, int]:
        """
        Computes the validation loss. Returns it and the number of batches.
        """
        logger.info("Validating")

        self._pytorch_model.eval()

        # Replace parameter values with the shadow values from the moving averages.
        if self._moving_average is not None:
            self._moving_average.assign_average_value()

        if self._validation_iterator is not None:
            val_iterator = self._validation_iterator
        else:
            val_iterator = self.iterator

        val_generator = val_iterator(self._validation_data, num_epochs=1, shuffle=False)
        num_validation_batches = val_iterator.get_num_batches(self._validation_data)
        val_generator_tqdm = Tqdm.tqdm(val_generator, total=num_validation_batches)
        batches_this_epoch = 0
        val_loss = 0
        for batch in val_generator_tqdm:

            loss = self.batch_loss(batch, for_training=False)
            if loss is not None:
                # You shouldn't necessarily have to compute a loss for validation, so we allow for
                # `loss` to be None.  We need to be careful, though - `batches_this_epoch` is
                # currently only used as the divisor for the loss function, so we can safely only
                # count those batches for which we actually have a loss.  If this variable ever
                # gets used for something else, we might need to change things around a bit.
                batches_this_epoch += 1
                val_loss += loss.detach().cpu().numpy()

            # Update the description with the latest metrics
            val_metrics = training_util.get_metrics(
                self.model,
                val_loss,
                batches_this_epoch,
                world_size=self._world_size,
                cuda_device=[self.cuda_device],
            )
            description = training_util.description_from_metrics(val_metrics)
            val_generator_tqdm.set_description(description, refresh=False)

        # Now restore the original parameter values.
        if self._moving_average is not None:
            self._moving_average.restore()

        return val_loss, batches_this_epoch

    def train(self) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters.
        """
        try:
            epoch_counter = self._restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise ConfigurationError(
                "Could not recover training from the checkpoint.  Did you mean to output to "
                "a different serialization directory or delete the existing serialization "
                "directory?"
            )

        training_util.enable_gradient_clipping(self.model, self._grad_clipping)

        logger.info("Beginning training.")

        train_metrics: Dict[str, float] = {}
        val_metrics: Dict[str, float] = {}
        this_epoch_val_metric: float = None
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()

        metrics["best_epoch"] = self._metric_tracker.best_epoch
        for key, value in self._metric_tracker.best_epoch_metrics.items():
            metrics["best_validation_" + key] = value

        num_training_batches = math.ceil(
            self.iterator.get_num_batches(self.train_data) / self._num_gradient_accumulation_steps
        )

        if isinstance(self._learning_rate_scheduler, SlantedTriangular):
            self._learning_rate_scheduler.num_steps_per_epoch = num_training_batches

        for epoch in range(epoch_counter, self._num_epochs + 1):
            epoch_start_time = time.time()
            if epoch > 0:
                self._train_epoch(epoch)

            if self._master:
                self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch: Union[int, str]) -> None:
        """
        Saves a checkpoint of the model to self._serialization_dir.
        Is a no-op if self._serialization_dir is None.

        # Parameters

        epoch : Union[int, str], required.
            The epoch of training.  If the checkpoint is saved in the middle
            of an epoch, the parameter is a string with the epoch and timestamp.
        """
        # If moving averages are used for parameters, we save
        # the moving average values into checkpoint, instead of the current values.
        if self._moving_average is not None:
            self._moving_average.assign_average_value()

        # These are the training states we need to persist.
        training_states = {
            "metric_tracker": self._metric_tracker.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "batch_num_total": self._batch_num_total,
        }

        self._checkpointer.save_checkpoint(
            model_state=self.overall_mean,
            epoch=epoch,
            training_states={},
            is_best_so_far=self._metric_tracker.is_best_so_far(),
        )

        # Restore the original values for parameters so that training will not be affected.
        if self._moving_average is not None:
            self._moving_average.restore()

    def _restore_checkpoint(self) -> int:
        """
        Restores the model and training state from the last saved checkpoint.
        This includes an epoch count and optimizer state, which is serialized separately
        from model parameters. This function should only be used to continue training -
        if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        ` model.load_state_dict(torch.load("/path/to/model/weights.th"))`

        If `self._serialization_dir` does not exist or does not contain any checkpointed weights,
        this function will do nothing and return 0.

        # Returns

        epoch: int
            The epoch at which to resume training, which should be one after the epoch
            in the saved training state.
        """
        model_state, training_state = self._checkpointer.restore_checkpoint()

        if not training_state:
            # No checkpoint to restore, start at 0
            return 0

        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(training_state["optimizer"])
        if (
            self._learning_rate_scheduler is not None
            and "learning_rate_scheduler" in training_state
        ):
            self._learning_rate_scheduler.load_state_dict(training_state["learning_rate_scheduler"])
        if self._momentum_scheduler is not None and "momentum_scheduler" in training_state:
            self._momentum_scheduler.load_state_dict(training_state["momentum_scheduler"])
        training_util.move_optimizer_to_cuda(self.optimizer)

        # Currently the `training_state` contains a serialized `MetricTracker`.
        if "metric_tracker" in training_state:
            self._metric_tracker.load_state_dict(training_state["metric_tracker"])
        # It used to be the case that we tracked `val_metric_per_epoch`.
        elif "val_metric_per_epoch" in training_state:
            self._metric_tracker.clear()
            self._metric_tracker.add_metrics(training_state["val_metric_per_epoch"])
        # And before that we didn't track anything.
        else:
            self._metric_tracker.clear()

        if isinstance(training_state["epoch"], int):
            epoch_to_return = training_state["epoch"] + 1
        else:
            epoch_to_return = int(training_state["epoch"].split(".")[0]) + 1

        # For older checkpoints with batch_num_total missing, default to old behavior where
        # it is unchanged.
        batch_num_total = training_state.get("batch_num_total")
        if batch_num_total is not None:
            self._batch_num_total = batch_num_total

        return epoch_to_return

    # Requires custom from_params.
    @classmethod
    def from_params(  # type: ignore
        cls,
        params: Params,
        serialization_dir: str,
        recover: bool = False,
        local_rank: int = 0,
    ) -> "Trainer":

        from allennlp.training.trainer import Trainer
        from allennlp.training.trainer_pieces import TrainerPieces

        config = dict(as_flat_dict(params.as_dict()))
        pieces = TrainerPieces.from_params(params, serialization_dir, recover)
        model = pieces.model
        serialization_dir = serialization_dir
        iterator = pieces.iterator
        train_data = pieces.train_dataset
        validation_data = pieces.validation_dataset
        params = pieces.params
        validation_iterator = pieces.validation_iterator

        patience = params.pop_int("patience", None)
        validation_metric = params.pop("validation_metric", "-loss")
        shuffle = params.pop_bool("shuffle", True)
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = parse_cuda_device(params.pop("cuda_device", -1))
        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)
        lr_scheduler_params = params.pop("learning_rate_scheduler", None)
        momentum_scheduler_params = params.pop("momentum_scheduler", None)

        check_for_gpu(cuda_device)
        if cuda_device >= 0:
            # Moving model to GPU here so that the optimizer state gets constructed on
            # the right device.
            model = model.cuda(cuda_device)

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))
        if "moving_average" in params:
            moving_average = MovingAverage.from_params(
                params.pop("moving_average"), parameters=parameters
            )
        else:
            moving_average = None

        if lr_scheduler_params:
            lr_scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
        else:
            lr_scheduler = None
        if momentum_scheduler_params:
            momentum_scheduler = MomentumScheduler.from_params(optimizer, momentum_scheduler_params)
        else:
            momentum_scheduler = None

        if "checkpointer" in params:
            if (
                "keep_serialized_model_every_num_seconds" in params
                or "num_serialized_models_to_keep" in params
            ):
                raise ConfigurationError(
                    "Checkpointer may be initialized either from the 'checkpointer' key or from the "
                    "keys 'num_serialized_models_to_keep' and 'keep_serialized_model_every_num_seconds'"
                    " but the passed config uses both methods."
                )
            checkpointer = Checkpointer.from_params(params.pop("checkpointer"))
        else:
            num_serialized_models_to_keep = params.pop_int("num_serialized_models_to_keep", 20)
            keep_serialized_model_every_num_seconds = params.pop_int(
                "keep_serialized_model_every_num_seconds", None
            )
            checkpointer = Checkpointer(
                serialization_dir=serialization_dir,
                num_serialized_models_to_keep=num_serialized_models_to_keep,
                keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds,
            )
        model_save_interval = params.pop_float("model_save_interval", None)
        summary_interval = params.pop_int("summary_interval", 100)
        histogram_interval = params.pop_int("histogram_interval", None)
        should_log_parameter_statistics = params.pop_bool("should_log_parameter_statistics", True)
        should_log_learning_rate = params.pop_bool("should_log_learning_rate", False)
        log_batch_size_period = params.pop_int("log_batch_size_period", None)

        distributed = params.pop_bool("distributed", False)
        world_size = params.pop_int("world_size", 1)

        num_gradient_accumulation_steps = params.pop("num_gradient_accumulation_steps", 1)

        writer = None
        wandb_config = params.pop("wandb", None)
        if wandb_config is not None:
            writer = WandBWriter(config, model, wandb_config)

        params.assert_empty(cls.__name__)
        return cls(
            model,
            optimizer,
            iterator,
            train_data,
            validation_data,
            patience=patience,
            validation_metric=validation_metric,
            validation_iterator=validation_iterator,
            shuffle=shuffle,
            num_epochs=num_epochs,
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            learning_rate_scheduler=lr_scheduler,
            momentum_scheduler=momentum_scheduler,
            checkpointer=checkpointer,
            model_save_interval=model_save_interval,
            summary_interval=summary_interval,
            histogram_interval=histogram_interval,
            should_log_parameter_statistics=should_log_parameter_statistics,
            should_log_learning_rate=should_log_learning_rate,
            log_batch_size_period=log_batch_size_period,
            moving_average=moving_average,
            distributed=distributed,
            local_rank=local_rank,
            world_size=world_size,
            num_gradient_accumulation_steps=num_gradient_accumulation_steps,
            writer=writer,
        )
