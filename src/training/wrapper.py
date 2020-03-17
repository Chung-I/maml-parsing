"""Meta-learners."""
import random
from copy import deepcopy
from abc import abstractmethod
from collections import OrderedDict, defaultdict
from typing import Dict, Optional, Tuple, Union, Iterable, Any, Callable

import torch

from allennlp.training.trainer_base import TrainerBase
from allennlp.common import Params, Registrable
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

class Wrapper(Registrable):
    def __init__(self):
        pass

    @classmethod
    def from_params(  # type: ignore
        cls,
        model: Model,
        optimizer: Optimizer,
        params: Params,
    ) -> "Wrapper":
        typ3 = params.pop("type", "default")
        klass: Type[Wrapper] = Wrapper.by_name(typ3)  # type: ignore
        # Explicit check to prevent recursion.
        is_overriden = (
            klass.from_params.__func__ != Wrapper.from_params.__func__  # type: ignore
        )
        assert is_overriden, f"Class {klass.__name__} must override `from_params`."
        return klass.from_params(model, optimizer, params)


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
        #self.model.get_metrics = self._container.get_metrics
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
        total_loss = 0.0
        for task in tasks:
            task_loss = self.run_task(task, train=train, meta_train=meta_train)
            total_loss += task_loss

        avg_loss = total_loss / len(tasks)

        # Meta gradient step
        if meta_train:
            self._final_meta_update()

        return avg_loss

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

        train_loss = 0.0
        N = len(batches)
        device = next(self._container.parameters()).device

        for n, inputs in enumerate(batches):
            optimizer.zero_grad()
            # task specific
            inputs = move_to_device(inputs, device)

            # Evaluate model
            output_dict = self._container(**inputs)
            loss = output_dict["loss"]
            if torch.isnan(loss):
                raise ValueError("nan loss encountered")
            train_loss += loss.item() / len(batches)
            # TRAINING #
            if not train:
                continue

            loss.backward()

            grad_norm = self.rescale_gradients()

            optimizer.step()

        if meta_train:
            self._partial_meta_update(len(batches))

        return train_loss

    @classmethod
    def from_params(
        cls,
        model: Model,
        meta_optimizer: torch.optim.Optimizer,
        params: Params,
        update_hook: Callable = None,
    ) -> "Wrapper":

        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)
        optimizer_cls = params.pop("optimizer_cls", "SGD")
        optimizer_kwargs = params.pop("optimizer_kwargs", Params({})).as_dict()

        params.assert_empty(cls.__name__)

        return cls(
            model,
            meta_optimizer,
            optimizer_cls,
            optimizer_kwargs,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            update_hook=update_hook,
        )


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

    def __init__(self,
                 *args,
                 **kwargs):
        super(_FOWrapper, self).__init__(*args, **kwargs)
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
            if n not in self._updates:
                continue

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

    def __init__(self, *args, **kwargs):
        super(ReptileWrapper, self).__init__(*args, **kwargs)


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

    def __init__(self, *args, **kwargs):
        super(FOMAMLWrapper, self).__init__(*args, **kwargs)


class MAMLWrapper(object):

    """Wrapper around the MAML meta-learner.
    Arguments:
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        meta_optimizer_cls: meta optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
        meta_optimizer_kwargs (dict): kwargs to pass to meta optimizer upon construction.
        criterion (func): loss criterion to use.
    """

    def __init__(self, model, optimizer_cls, meta_optimizer_cls, optimizer_kwargs,
                 meta_optimizer_kwargs, criterion):
        self.criterion = criterion
        self.model = model

        self.optimizer_cls = maml.SGD if optimizer_cls.lower() == 'sgd' else maml.Adam

        self.meta = maml.MAML(optimizer_cls=self.optimizer_cls,
                              model=model, tensor=False, **optimizer_kwargs)

        self.meta_optimizer_cls = optim.SGD if meta_optimizer_cls.lower() == 'sgd' else optim.Adam

        self.optimizer_kwargs = optimizer_kwargs
        self.meta_optimizer = self.meta_optimizer_cls(self.meta.parameters(), **meta_optimizer_kwargs)

    def __call__(self, meta_batch, meta_train):
        tasks = []
        for t in meta_batch:
            t.dataset.train()
            inner = [b for b in t]
            t.dataset.eval()
            outer = [b for b in t]
            tasks.append((inner, outer))
        return self.run_meta_batch(tasks, meta_train=meta_train)

    def run_meta_batch(self, meta_batch, meta_train):
        """Run on meta-batch.
        Arguments:
            meta_batch (list): list of task-specific dataloaders
            meta_train (bool): meta-train on batch.
        """
        loss, results = self.meta(meta_batch, return_predictions=False, return_results=True, create_graph=meta_train)
        if meta_train:
            loss.backward()
            self.meta_optimizer.step()
            self.meta_optimizer.zero_grad()

        return results
