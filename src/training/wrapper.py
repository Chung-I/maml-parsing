"""Meta-learners."""
import random
from copy import deepcopy
from abc import abstractmethod
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union, Iterable, Any

import torch

from allennlp.training.trainer_base import TrainerBase
from allennlp.common import Params, Registrable
from allennlp.common.checks import ConfigurationError, parse_cuda_device, check_for_gpu
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import dump_metrics, gpu_memory_mb, peak_memory_mb, lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.models.model import Model
from allennlp.nn import util as nn_util
from allennlp.training import util as training_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer
from allennlp.training.tensorboard_writer import TensorboardWriter

from src.training.util import clone_state_dict


class MultiWrapper(object):
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 cuda_device: int):
        self.model = model
        self.optimizer = optimizer
        self.cuda_device = cuda_device

    def __call__(self, tasks, train=True, meta_train=False):
        # argument key meta_train is dummy
        train_loss = 0.0
        for task in tasks:
            loss = self.run_task(task, train=train)
            train_loss += loss

        return train_loss

    def run_task(self, task, train):
        if train:
            self.model.train()
        else:
            self.model.eval()
        return self.run_batches(task, train=train)

    def run_batches(self, batches, train=False):
        train_loss = 0.0

        for n, inputs in enumerate(batches):
            inputs = nn_util.move_to_device(inputs, self.cuda_device)

            output_dict = self.model(**inputs)
            loss = output_dict["loss"]
            loss.backward()
            train_loss += loss.item()

        return train_loss


    def state_dict(self):
        return {"model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state_dict, load_opt=False):
        self.model.load_state_dict(state_dict["model"])
        if load_opt:
            self.optimizer.load_state_dict(state_dict["optimizer"])


class BaseWrapper(Registrable):

    """Generic training wrapper.

    Arguments:
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.

    """
    default_implementation: str = "default"

    def __init__(
        self,
        model: Model,
        meta_optimizer: torch.optim.Optimizer,
        optimizer_cls: str,
        optimizer_kwargs: Dict[str, Any],
        cuda_device: int = -1,
        grad_norm: Optional[float] = None,
        grad_clipping: Optional[float] = None,
    ):
        self.model = model
        self.meta_optimizer = meta_optimizer
        self._grad_clipping = grad_clipping
        self._grad_norm = grad_norm
        self._container = deepcopy(self.model)
        #self.model.get_metrics = self._container.get_metrics
        training_util.enable_gradient_clipping(self.model, self._grad_clipping)
        self.optimizer_cls = getattr(torch.optim, optimizer_cls)
        self.optimizer_kwargs = optimizer_kwargs
        self.cuda_device = cuda_device

    def rescale_gradients(self) -> Optional[float]:
        return training_util.rescale_gradients(self._container, self._grad_norm)

    def __call__(self, tasks, train=True, meta_train=True):
        return self.run_tasks(tasks, train=train, meta_train=meta_train)

    @abstractmethod
    def _partial_meta_update(self, loss, final):
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

        for n, inputs in enumerate(batches):
            # task specific
            #inputs["pos_tags"] = inputs["pos_tags"].to(device)
            #inputs["head_tags"] = inputs["head_tags"].to(device)
            inputs = nn_util.move_to_device(inputs, self.cuda_device)

            # Evaluate model
            #loss = self.model(inputs)
            output_dict = self._container(**inputs)
            loss = output_dict["loss"]
            if torch.isnan(loss):
                raise ValueError("nan loss encountered")
            loss = loss / len(batches)
            train_loss += loss.item()
            # TRAINING #
            if not train:
                continue

            final = (n+1) == N
            loss.backward()

            grad_norm = self.rescale_gradients()

            if meta_train:
                self._partial_meta_update(loss, final)

            optimizer.step()
            optimizer.zero_grad()

            if final:
                break

        return train_loss

    @classmethod
    def from_params(
        cls,
        model: Model,
        meta_optimizer: torch.optim.Optimizer,
        params: Params,
        cuda_device: int = -1,
    ) -> "Wrapper":

        typ3 = params.pop("type", "default")

        klass: Type[BaseWrapper] = BaseWrapper.by_name(typ3)  # type: ignore

        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)
        optimizer_cls = params.pop("optimizer_cls", "SGD")
        optimizer_kwargs = params.pop("optimizer_kwargs", Params({})).as_dict()

        params.assert_empty(cls.__name__)

        return klass(
            model,
            meta_optimizer,
            optimizer_cls,
            optimizer_kwargs,
            cuda_device=cuda_device,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping
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

    def _partial_meta_update(self, loss, final):
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

    def run_task(self, task, train, meta_train):
        if meta_train:
            self._counter += 1
        if train:
            self._container.load_state_dict(self.model.state_dict(keep_vars=True))
        return super(_FOWrapper, self).run_task(task, train, meta_train)

    def _partial_meta_update(self, loss, final):
        if not final:
            return

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
                self._updates[n].add_(p.data)
            else:
                self._updates[n].add_(p.grad.data)

    def _final_meta_update(self):
        for n, p in self._updates.items():
            p.data.div_(self._counter)

        for n, p in self.model.state_dict(keep_vars=True).items():
            if n not in self._updates:
                continue

            if self._all_grads:
                p.grad = p.data - self._updates[n].data
            else:
                p.grad = self._updates[n]

        #self.meta_optimizer.step()
        #self.meta_optimizer.zero_grad()
        self._counter = 0
        self._updates = None


@BaseWrapper.register("reptile")
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

@BaseWrapper.register("fomaml")
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
