import logging
import math
import datetime
import os
import time
import traceback
from collections import defaultdict
from typing import Dict, Optional, Tuple, Union, Iterable, Any
import numpy as np

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
from src.training.wrapper import Wrapper
from src.modules.adv import TaskDiscriminator
from src.training.util import as_flat_dict, filter_state_dict

logger = logging.getLogger(__name__)


@TrainerBase.register("meta")
class MetaTrainer(TrainerBase):
    def __init__(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        iterator: DataIterator,
        train_datasets: Dict[str, Iterable[Instance]],
        validation_datasets: Optional[Dict[str, Iterable[Instance]]] = None,
        patience: Optional[int] = None,
        validation_metric: str = "-loss",
        validation_iterator: DataIterator = None,
        shuffle: bool = True,
        num_epochs: int = 20,
        serialization_dir: Optional[str] = None,
        num_serialized_models_to_keep: int = 20,
        save_embedder: bool = True,
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
        log_grad_norm: str = "total",
        wrapper: Optional[Wrapper] = None,
        task_discriminator: Optional[TaskDiscriminator] = None,
        discriminator_optimizer: Optional[torch.optim.Optimizer] = None,
        tasks_per_step: int = 0,
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
        self.train_datas = train_datasets
        self._validation_datas = validation_datasets
        self._save_embedder = save_embedder

        if patience is None:  # no early stopping
            if validation_datasets:
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

        self._tasks_per_step = tasks_per_step if tasks_per_step > 0 else len(self.train_datas.items())
        self.wrapper = wrapper
        self.task_D = task_discriminator
        self.optim_D = discriminator_optimizer

        def update_hook(norms):
            assert log_grad_norm in ["none", 'total', 'var']
            if log_grad_norm in ['total', 'var']:
                total_task_grad_norm = 0.0
                total_summed_grad_norm = 0.0
                for name, norm_list in norms.items():
                    if len(norm_list) == 1:
                        logger.info(f"{name} has no gradient; skipping")
                        continue
                    avg_task_grad_norm = (sum(norm_list[:-1]) / len(norm_list[:-1]))
                    total_task_grad_norm += avg_task_grad_norm
                    summed_grad_norm = norm_list[-1]
                    total_summed_grad_norm += summed_grad_norm
                    if log_grad_norm == 'var':
                        ratio = summed_grad_norm / (avg_task_grad_norm + 1e-10)
                        self._writer.log({f"avg_task_grad_norm_{name}": avg_task_grad_norm,
                                          f"summed_grad_norm_{name}": summed_grad_norm,
                                          f"task-total_norm_ratio_{name}": ratio},
                                         step=self._batch_num_total)
                avg_ratio = total_summed_grad_norm / total_task_grad_norm
                self._writer.log({"avg_task-total_norm_ratio": avg_ratio,
                                  "total_grad_norm": total_summed_grad_norm,
                                  "total_task_grad_norm": total_task_grad_norm},
                                 step=self._batch_num_total)

        self.wrapper.update_hook = update_hook

    def rescale_gradients(self) -> Optional[float]:
        return training_util.rescale_gradients(self.model, self._grad_norm)

    def batch_loss(self, batch: TensorDict, for_training: bool) -> torch.Tensor:
        """
        Does a forward pass on the given batches and returns the `loss` value in the result.
        If `for_training` is `True` also applies regularization penalty.
        """
        batch = nn_util.move_to_device(batch, self.cuda_device)
        output_dict = self._pytorch_model(**batch)

        try:
            loss = output_dict["loss"]
            if for_training:
                loss += self.model.get_regularization_penalty()
        except KeyError:
            if for_training:
                raise RuntimeError(
                    "The model you are trying to optimize does not contain a"
                    " 'loss' key in the output of model.forward(inputs)."
                )
            loss = None

        return loss

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

        batch_generators = {task: self.iterator(train_data, num_epochs=1, shuffle=self.shuffle)
            for task, train_data in self.train_datas.items()}
        batch_group_generators = {task: lazy_groups_of(
            batch_generator, self._num_gradient_accumulation_steps
        ) for task, batch_generator in batch_generators.items()}
        num_training_batches = [math.ceil(
            self.iterator.get_num_batches(train_data) / self._num_gradient_accumulation_steps
        ) for task, train_data in self.train_datas.items()]
        assert len(set(num_training_batches)) == 1, "num_training_batches doesn't agree"
        tasks = list(batch_group_generators.keys())
        num_tasks = len(tasks)

        if isinstance(self._learning_rate_scheduler, SlantedTriangular):
            old_num_steps_per_epoch = self._learning_rate_scheduler.num_steps_per_epoch
            self._learning_rate_scheduler.num_steps_per_epoch = num_training_batches[0]
            logger.info(f"modify num_steps_per_epoch of lr scheduler from"
                        f"{old_num_steps_per_epoch} to {num_training_batches}")

        self._last_log = time.time()
        last_save_time = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        logger.info("Training")

        cumulative_batch_group_size = 0
        tqdm_bar = Tqdm.tqdm(range(num_training_batches[0]))
        for _ in tqdm_bar:
            randperms = torch.randperm(len(tasks)).tolist()
            sampled_tasks = [tasks[idx] for idx in randperms[:self._tasks_per_step]]
            sampled_task_generators = [next(batch_group_generators[task]) for task in sampled_tasks]

            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            self.optimizer.zero_grad()

            task_metrics = self.wrapper(tasks=sampled_task_generators, train=True, meta_train=True)
            losses = [list(map(lambda x: x["loss"], metrics)) for metrics in task_metrics]
            LASes = [list(map(lambda x: x["metric"]["LAS"], metrics)) for metrics in task_metrics]

            for name, values in zip(["loss", "LAS"], [losses, LASes]):
                self._writer.log({f"step_{name}_{task}_{i}": value
                                  for task, task_values in zip(sampled_tasks, values)
                                  for i, value in enumerate(task_values)},
                                 step=self._batch_num_total)
                values_inner_steps = list(map(np.mean, zip(*values)))
                self._writer.log({f"step_{name}_{i}": value for i, value in
                                  enumerate(values_inner_steps)},
                                 step=self._batch_num_total)
                if name == "loss":
                    train_loss += values_inner_steps[0]

            batch_grad_norm = self.rescale_gradients()

            # This does nothing if batch_num_total is None or you are using a
            # scheduler which doesn't update per batch.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(batch_num_total)
            if self._momentum_scheduler:
                self._momentum_scheduler.step_batch(batch_num_total)

            if self.task_D and self.optim_D:
                # D training
                steps_per_update = self.task_D.steps_per_update
                if (batch_num_total - 1) % steps_per_update == 0:
                    self.optim_D.zero_grad()
                    hidden_states, labels, masks = self.task_D.get_hidden_states(
                        self.model,
                        sampled_task_generators
                    )
                    D_loss, _, acc = self.task_D(hidden_states, labels, masks, detach=True)
                    D_loss.backward()
                    self.optim_D.step()
                    self._writer.log({"D_loss": D_loss.detach().item(),
                                      "D_acc": acc},
                                     step=self._batch_num_total)

                # G training
                hidden_states, labels, masks = self.task_D.get_hidden_states(
                    self.model,
                    sampled_task_generators
                )
                _, g_loss, acc = self.task_D(hidden_states, labels, masks)
                if self.task_D.weight:
                    alpha = self.task_D.weight
                else:
                    alpha = self.task_D.get_alpha(self._batch_num_total,
                                                  num_training_batches[0] * self._num_epochs)
                G_loss = -alpha * g_loss
                G_loss.backward()
                self._writer.log({"G_loss": g_loss.detach().item(), "alpha": alpha, "G_acc": acc},
                                 step=self._batch_num_total)

            self.optimizer.step()

            # Update moving averages
            if self._moving_average is not None:
                self._moving_average.apply(batch_num_total)

            # Update the description with the latest metrics
            metrics = training_util.get_metrics(
                self.wrapper.container,
                train_loss,
                batches_this_epoch,
                world_size=self._world_size,
                cuda_device=[self.cuda_device],
            )

            # Updating tqdm only for the master as the trainers wouldn't have one
            if self._master:
                description = training_util.description_from_metrics(metrics)
                tqdm_bar.set_description(description, refresh=False)

            # log learning rate.
            self._writer.log({"lr": self.optimizer.param_groups[0]['lr']},
                             step=self._batch_num_total)

            # Save model if needed.
            if (
                self._model_save_interval is not None
                and (time.time() - last_save_time > self._model_save_interval)
                and self._master
            ):
                last_save_time = time.time()
                self._save_checkpoint(
                    "{0}.{1}".format(epoch, training_util.time_to_str(int(last_save_time)))
                )

        # Let all workers finish their epoch before computing
        # the final statistics for the epoch.
        if self._distributed:
            dist.barrier()

        metrics = training_util.get_metrics(
            self.wrapper.container,
            train_loss,
            batches_this_epoch,
            reset=True,
            world_size=self._world_size,
            cuda_device=[self.cuda_device],
        )
        metrics["cpu_memory_MB"] = peak_cpu_usage
        for (gpu_num, memory) in gpu_usage:
            metrics["gpu_" + str(gpu_num) + "_memory_MB"] = memory
        return metrics

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

        batches_this_epoch = 0
        val_loss = 0
        val_generators = {key: val_iterator(val_data, num_epochs=1, shuffle=False)
            for key, val_data in self._validation_datas.items()}
        num_validation_batches = {key: val_iterator.get_num_batches(val_data)
            for key, val_data in self._validation_datas.items()}
        val_generators_tqdm = [Tqdm.tqdm(val_generator, total=num_validation_batches[key])
            for key, val_generator in val_generators.items()]
        for val_generator_tqdm in val_generators_tqdm:
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
                for val_generator_tqdm in val_generators_tqdm:
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

        if self._master:
            self._save_checkpoint(epoch_counter - 1)

        for epoch in range(epoch_counter, self._num_epochs + 1):
            epoch_start_time = time.time()
            train_metrics = self._train_epoch(epoch)

            # get peak of memory usage
            if "cpu_memory_MB" in train_metrics:
                metrics["peak_cpu_memory_MB"] = max(
                    metrics.get("peak_cpu_memory_MB", 0), train_metrics["cpu_memory_MB"]
                )
            for key, value in train_metrics.items():
                if key.startswith("gpu_"):
                    metrics["peak_" + key] = max(metrics.get("peak_" + key, 0), value)

            if self._validation_datas is not None:
                with torch.no_grad():
                    # We have a validation set, so compute all the metrics on it.
                    val_loss, num_batches = self._validation_loss()

                    # It is safe again to wait till the validation is done. This is
                    # important to get the metrics right.
                    if self._distributed:
                        dist.barrier()

                    val_metrics = training_util.get_metrics(
                        self.model,
                        val_loss,
                        num_batches,
                        reset=True,
                        world_size=self._world_size,
                        cuda_device=[self.cuda_device],
                    )

                    # Check validation metric for early stopping
                    this_epoch_val_metric = val_metrics[self._validation_metric]
                    self._metric_tracker.add_metric(this_epoch_val_metric)

                    if self._metric_tracker.should_stop_early():
                        logger.info("Ran out of patience.  Stopping training.")
                        break

            if self._master:
                self._writer.log(train_metrics, step=self._batch_num_total,
                                 epoch=epoch, prefix="train")
                self._writer.log(val_metrics, step=self._batch_num_total,
                                 epoch=epoch, prefix="val")

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
            metrics["training_start_epoch"] = epoch_counter
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value

            if self._metric_tracker.is_best_so_far():
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                metrics["best_epoch"] = epoch
                for key, value in val_metrics.items():
                    metrics["best_validation_" + key] = value

                self._metric_tracker.best_epoch_metrics = val_metrics

            if self._serialization_dir and self._master:
                dump_metrics(
                    os.path.join(self._serialization_dir, f"metrics_epoch_{epoch}.json"), metrics
                )

            # The Scheduler API is agnostic to whether your schedule requires a validation metric -
            # if it doesn't, the validation metric passed here is ignored.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step(this_epoch_val_metric, epoch)
            if self._momentum_scheduler:
                self._momentum_scheduler.step(this_epoch_val_metric, epoch)

            if self._master:
                self._save_checkpoint(epoch)

            # Wait for the master to finish saving the checkpoint
            if self._distributed:
                dist.barrier()

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

            if epoch < self._num_epochs:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * (
                    (self._num_epochs + 1 - epoch_counter) / float(epoch - epoch_counter + 1) - 1
                )
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            epochs_trained += 1

        # Load the best model state before returning
        best_model_state = self._checkpointer.best_model_state()
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return metrics

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

        # If we have a learning rate or momentum scheduler, we should persist them too.
        if self._learning_rate_scheduler is not None:
            training_states["learning_rate_scheduler"] = self._learning_rate_scheduler.state_dict()
        if self._momentum_scheduler is not None:
            training_states["momentum_scheduler"] = self._momentum_scheduler.state_dict()

        if self.task_D is not None:
            training_states["task_discriminator"] = self.task_D.state_dict()
        if self.optim_D is not None:
            training_states["discriminator_optimizer"] = self.optim_D.state_dict()

        if self._save_embedder:
            model_state = self.model.state_dict()
        else:
            model_state = filter_state_dict(self.model.state_dict(),
                lambda k, v: 'text_field_embedder' not in k)

        self._checkpointer.save_checkpoint(
            model_state=model_state,
            epoch=epoch,
            training_states=training_states,
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
            return 1

        missing_keys, _ = self.model.load_state_dict(model_state, strict=False)
        self.optimizer.load_state_dict(training_state["optimizer"])
        if (
            self._learning_rate_scheduler is not None
            and "learning_rate_scheduler" in training_state
        ):
            self._learning_rate_scheduler.load_state_dict(training_state["learning_rate_scheduler"])
        if self._momentum_scheduler is not None and "momentum_scheduler" in training_state:
            self._momentum_scheduler.load_state_dict(training_state["momentum_scheduler"])
        training_util.move_optimizer_to_cuda(self.optimizer)

        if self.task_D is not None and "task_discriminator" in training_state:
            self.task_D.load_state_dict(training_state["task_discriminator"])
        if self.optim_D is not None and "discriminator_optimizer" in training_state:
            self.optim_D.load_state_dict(training_state["discriminator_optimizer"])

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
    ) -> "MetaTrainer":

        from allennlp.training.trainer import Trainer
        from src.training.trainer_pieces import MetaTrainerPieces

        config = dict(as_flat_dict(params.as_dict()))
        pieces = MetaTrainerPieces.from_params(params, serialization_dir, recover)
        model = pieces.model
        serialization_dir = serialization_dir
        iterator = pieces.iterator
        train_datas = pieces.train_datasets
        validation_datas = pieces.validation_datasets
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

        log_grad_norm = params.pop("log_grad_norm", "total")
        save_embedder = params.pop_bool("save_embedder", True)
        model_save_interval = params.pop_float("model_save_interval", None)
        summary_interval = params.pop_int("summary_interval", 100)
        histogram_interval = params.pop_int("histogram_interval", None)
        should_log_parameter_statistics = params.pop_bool("should_log_parameter_statistics", True)
        should_log_learning_rate = params.pop_bool("should_log_learning_rate", False)
        log_batch_size_period = params.pop_int("log_batch_size_period", None)

        distributed = params.pop_bool("distributed", False)
        world_size = params.pop_int("world_size", 1)

        num_gradient_accumulation_steps = params.pop("num_gradient_accumulation_steps", 1)
        tasks_per_step = params.pop_int("tasks_per_step", 0)
        wrapper = Wrapper.from_params(
            params.pop("wrapper"),
            model=model,
            meta_optimizer=optimizer,
        )

        task_discriminator_params = params.pop("task_discriminator", None)
        if task_discriminator_params:
            num_tasks = model.vocab.get_vocab_size("lang_labels")
            task_discriminator = TaskDiscriminator.from_params(task_discriminator_params,
                                                               num_tasks=num_tasks)
            if cuda_device >= 0:
                task_discriminator = task_discriminator.cuda(cuda_device)

            discriminator_parameters = \
                [[n, p] for n, p in task_discriminator.named_parameters() if p.requires_grad]
            discriminator_optimizer = Optimizer.from_params(discriminator_parameters,
                                                            params.pop("discriminator_optimizer"))
        else:
            task_discriminator = None
            discriminator_optimizer = None

        writer = None
        wandb_config = params.pop("wandb", None)
        if wandb_config is not None:
            writer = WandBWriter(config, wrapper.container, wandb_config)

        params.assert_empty(cls.__name__)
        return cls(
            model,
            optimizer,
            iterator,
            train_datas,
            validation_datas,
            patience=patience,
            validation_metric=validation_metric,
            validation_iterator=validation_iterator,
            shuffle=shuffle,
            num_epochs=num_epochs,
            serialization_dir=serialization_dir,
            save_embedder=save_embedder,
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
            log_grad_norm=log_grad_norm,
            wrapper=wrapper,
            task_discriminator=task_discriminator,
            discriminator_optimizer=discriminator_optimizer,
            tasks_per_step=tasks_per_step,
            writer=writer,
        )
