"""Meta-learners."""
import random
from copy import deepcopy
from abc import abstractmethod
from collections import OrderedDict, defaultdict
from typing import List, Dict, Optional, Tuple, Union, Iterable, Any, Callable

import torch
import higher

from allennlp.training.trainer_base import TrainerBase
from allennlp.common import Params, Registrable, FromParams
from allennlp.common.checks import ConfigurationError, parse_cuda_device, check_for_gpu
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import dump_metrics, gpu_memory_mb, peak_memory_mb, lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.models.model import Model
from allennlp.training import util as training_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer
from allennlp.training.tensorboard_writer import TensorboardWriter

from src.training.util import clone_state_dict, move_to_device

class Wrapper(Registrable, FromParams):
    def __init__(self):
        pass


@Wrapper.register("multi")
class MultiWrapper(Wrapper):
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer):
        super(MultiWrapper, self).__init__()
        self._counter = 0
        self.model = model
        self.optimizer = optimizer

    @property
    def container(self):
        return self.model

    def __call__(self, tasks, train=True, meta_train=False):
        # argument key meta_train is dummy
        total_loss = 0.0
        self._counter = len(tasks)
        for task in tasks:
            task_loss = self.run_task(task, train=train)
            total_loss += task_loss

        return total_loss

    def run_task(self, task, train):
        if train:
            self.model.train()
        else:
            self.model.eval()
        return self.run_batches(task, train=train)

    def run_batches(self, batches, train=False):
        train_loss = 0.0
        device = next(self.model.parameters()).device

        for n, inputs in enumerate(batches):
            inputs = move_to_device(inputs, device)

            output_dict = self.model(**inputs)
            loss = output_dict["loss"]
            loss = loss / len(batches)
            loss = loss / self._counter
            train_loss += loss.item()
            if not train:
                continue
            loss.backward()

        return train_loss

    @classmethod
    def from_params(
        cls,
        model: Model,
        meta_optimizer: torch.optim.Optimizer,
        params: Params,
    ) -> "Wrapper":
        params.assert_empty(cls.__name__)
        return  cls(model, meta_optimizer)


@Wrapper.register("default")
class BaseWrapper(Wrapper):

    """Generic training wrapper.

    Arguments:
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.

    """

    def __init__(
        self,
        model: Model,
        meta_optimizer: torch.optim.Optimizer,
        optimizer_cls: str,
        optimizer_kwargs: Dict[str, Any],
        grad_norm: Optional[float] = None,
        grad_clipping: Optional[float] = None,
        update_hook: Callable = None,
    ):
        super(BaseWrapper, self).__init__()
        self.model = model
        self.meta_optimizer = meta_optimizer
        self._grad_clipping = grad_clipping
        self._grad_norm = grad_norm
        self._container = deepcopy(self.model)
        training_util.enable_gradient_clipping(self.model, self._grad_clipping)
        self.optimizer_cls = getattr(torch.optim, optimizer_cls)
        self.optimizer_kwargs = optimizer_kwargs
        self._update_hook = update_hook

    @property
    def update_hook(self):
        return self._update_hook

    @update_hook.setter
    def update_hook(self, new_update_hook):
        self._update_hook = new_update_hook

    def rescale_gradients(self) -> Optional[float]:
        return training_util.rescale_gradients(self._container, self._grad_norm)

    def __call__(self, tasks, train=True, meta_train=True):
        return self.run_tasks(tasks, train=train, meta_train=meta_train)

    @property
    def container(self):
        return self._container

    @abstractmethod
    def _partial_meta_update(self, num_batches):
        """Meta-model specific meta update rule.

        Arguments:
            loss (nn.Tensor): loss value for given mini-batch.
            final (bool): whether iteration is the final training step.
        """
        NotImplementedError('Implement in meta-learner class wrapper.')

    @abstractmethod
    def _final_meta_update(self):
        """Meta-model specific meta update rule."""
        NotImplementedError('Implement in meta-learner class wrapper.')

    def run_tasks(self, tasks, train, meta_train):
        """Train on a mini-batch tasks and evaluate test performance.

        Arguments:
            tasks (list, torch.utils.data.DataLoader): list of task-specific dataloaders.
            meta_train (bool): whether current run in during meta-training.
        """
        metrics = []
        for task in tasks:
            metric = self.run_task(task, train=train, meta_train=meta_train)
            metrics.append(metric)

        # Meta gradient step
        if meta_train:
            self._final_meta_update()

        return metrics

    def run_task(self, task, train, meta_train):
        """Run model on a given task.

        Arguments:
            task (torch.utils.data.DataLoader): task-specific dataloaders.
            train (bool): whether to train on task.
            meta_train (bool): whether to meta-train on task.
        """
        optimizer = None
        if train:
            self._container.train()
            optimizer = self.optimizer_cls(
                self._container.parameters(), **self.optimizer_kwargs)
            optimizer.zero_grad()
        else:
            self._container.eval()

        return self.run_batches(task, optimizer, train=train, meta_train=meta_train)

    def run_batches(self, batches, optimizer, train=False, meta_train=False):
        """Iterate over task-specific batches.

        Arguments:
            batches (torch.utils.data.DataLoader): task-specific dataloaders.
            optimizer (torch.nn.optim): optimizer instance if training is True.
            train (bool): whether to train on task.
            meta_train (bool): whether to meta-train on task.
        """

        metrics = []
        N = len(batches)
        device = next(self._container.parameters()).device

        for n, inputs in enumerate(batches):
            optimizer.zero_grad()
            # task specific
            inputs = move_to_device(inputs, device)

            # Evaluate model
            output_dict = self._container(**inputs, return_metric=True)
            loss = output_dict["loss"]
            metric = output_dict["metric"]
            if torch.isnan(loss):
                raise ValueError("nan loss encountered")
            metrics.append({"loss": loss.item(), "metric": metric})
            # TRAINING #
            if not train:
                continue

            loss.backward()

            grad_norm = self.rescale_gradients()

            optimizer.step()

        if meta_train:
            self._partial_meta_update(len(batches))

        return metrics


class NoWrapper(BaseWrapper):

    """Wrapper for baseline without any meta-learning.

    Arguments:
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
    """

    def __init__(self, *args, **kwargs):
        super(NoWrapper, self).__init__(*args, **kwargs)

    def __call__(self, tasks, meta_train=False):
        return super(NoWrapper, self).__call__(tasks, meta_train=False)

    def run_task(self, *args, **kwargs):
        out = super(NoWrapper, self).run_task(*args, **kwargs)
        self._container.load_state_dict(self.model.state_dict(keep_vars=True))
        return out

    def _partial_meta_update(self, num_batches):
        pass

    def _final_meta_update(self):
        pass


class _FOWrapper(BaseWrapper):

    """Base wrapper for First-order MAML and Reptile.

    Arguments:
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        meta_optimizer_cls: meta optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
        meta_optimizer_kwargs (dict): kwargs to pass to meta optimizer upon construction.
    """

    _all_grads = None

    def __init__(
        self,
        model: Model,
        meta_optimizer: torch.optim.Optimizer,
        optimizer_cls: str,
        optimizer_kwargs: Dict[str, Any],
        grad_norm: Optional[float] = None,
        grad_clipping: Optional[float] = None,
        update_hook: Callable = None,
    ):
        super(_FOWrapper, self).__init__(
            model=model,
            meta_optimizer=meta_optimizer,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            update_hook=update_hook,
        )
        self._counter = 0
        self._updates = None
        self._norms = defaultdict(list)

    def run_task(self, task, train, meta_train):
        if meta_train:
            self._counter += 1
        if train:
            self._container.load_state_dict(self.model.state_dict(keep_vars=True))
        return super(_FOWrapper, self).run_task(task, train, meta_train)

    def _partial_meta_update(self, num_batches):
        if self._updates is None:
            self._updates = {}
            for n, p in self.model.state_dict(keep_vars=True).items():
                if not getattr(p, 'requires_grad', False):
                    continue

                if p.size():
                    self._updates[n] = p.new(*p.size()).zero_()
                else:
                    self._updates[n] = p.clone().zero_()

        for n, p in self._container.state_dict(keep_vars=True).items():
            if n not in self._updates or p.grad is None:
                continue

            print(self._all_grads)
            if self._all_grads is True:
                trajectory = (self.model.state_dict()[n].data - p.data) / num_batches
                self._updates[n].add_(trajectory)
            else:
                self._updates[n].add_(p.grad.data)

            self._norms[n].append(p.grad.data.norm(2))

    def _final_meta_update(self):
        for n, p in self._updates.items():
            p.data.div_(self._counter)

        for n, p in self.model.state_dict(keep_vars=True).items():
            if n not in self._updates:
                continue
            p.grad = self._updates[n]
            self._norms[n].append(p.grad.data.norm(2))

        if self._update_hook is not None:
            self._update_hook(self._norms)

        self._counter = 0
        self._updates = None
        self._norms = defaultdict(list)


@Wrapper.register("reptile")
class ReptileWrapper(_FOWrapper):

    """Wrapper for Reptile.

    Arguments:
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        meta_optimizer_cls: meta optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
        meta_optimizer_kwargs (dict): kwargs to pass to meta optimizer upon construction.
    """

    _all_grads = True


@Wrapper.register("fomaml")
class FOMAMLWrapper(_FOWrapper):
    """Wrapper for FOMAML.

    Arguments:
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        meta_optimizer_cls: meta optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
        meta_optimizer_kwargs (dict): kwargs to pass to meta optimizer upon construction.
    """

    _all_grads = False


@Wrapper.register("maml")
class MAMLWrapper(Wrapper):
    def __init__(
        self,
        model: Model,
        meta_optimizer: torch.optim.Optimizer,
        optimizer_cls: str,
        optimizer_kwargs: Dict[str, Any],
        shuffle_label_namespaces: List[str] = [],
        grad_norm: Optional[float] = None,
        grad_clipping: Optional[float] = None,
        update_hook: Callable = None,
    ):
        super(MAMLWrapper, self).__init__()
        self._counter = 0
        self.model = model
        self.meta_optimizer = meta_optimizer
        self.inner_optimizer = getattr(torch.optim, optimizer_cls)(
            self.model.parameters(),
            **optimizer_kwargs
        )
        self._shuffler_factory: Dict[str, Callable] = {}
        for namespace in shuffle_label_namespaces:
            num_labels = self.model.vocab.get_vocab_size(namespace)
            self._shuffler_factory[namespace] = \
                lambda: torch.randperm(num_labels)

    @property
    def container(self):
        return self.model

    def shuffle_labels(self, inputs, label_shufflers):
        for key, value in inputs.items():
            if key in label_shufflers:
                shuffler = label_shufflers[key].to(value.device)
                inputs[key] = shuffler[value]

        return inputs

    def __call__(self, tasks, train=True, meta_train=True):
        losses = []
        for task in tasks:
            loss = self.run_task(task, train=train, meta_train=meta_train)
            losses.append(loss)

        return losses

    def run_task(self, task, train, meta_train):
        if train:
            self.model.train()
        else:
            self.model.eval()
        return self.run_batches(task, train=train, meta_train=meta_train)

    def run_batches(self, batches, train=True, meta_train=True):
        metrics = []
        device = next(self.model.parameters()).device
        shufflers = {key: shuffler() for key, shuffler in self._shuffler_factory.items()}
        with higher.innerloop_ctx(
            self.model, self.inner_optimizer, copy_initial_weights=False
        ) as (fmodel, diffopt):
            for n, inputs in enumerate(batches[:-1]):
                inputs = self.shuffle_labels(inputs, shufflers)
                inputs = move_to_device(inputs, device)
                output_dict = self.model(**inputs, return_metric=True)
                loss = output_dict["loss"]
                metric = output_dict["metric"]
                diffopt.step(loss)
                metrics.append({"loss": loss.item(), "metric": metric})

            inputs = self.shuffle_labels(batches[-1], shufflers)
            inputs = move_to_device(inputs, device)
            output_dict = self.model(**inputs, return_metric=True)
            loss = output_dict["loss"]
            metric = output_dict["metric"]
            loss.backward()
            metrics.append({"loss": loss.item(), "metric": metric})

        return metrics
