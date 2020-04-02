# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
# ==============================================================================

"""MAML

APIs for the MAML meta-learner.

"""
import torch
import torch.nn as nn
from .utils import build_dict, load_state_dict, build_iterator, Res, AggRes
from src.training.util import move_to_device


def maml_inner_step(inputs, model, optimizer, create_graph):
    """Create a computation graph through the gradient operation

    Arguments:
        input (torch.Tensor): input tensor.
        output (torch.Tensor): target tensor.
        model (torch.nn.Module): task learner.
        optimizer (maml.optim): optimizer for inner loop.
        criterion (func): loss criterion.
        create_graph (bool): create graph through gradient step.
    """
    new_parameters = None

    with torch.backends.cudnn.flags(enabled=False):
        output_dict = model(**inputs, return_metric=True)
    loss = output_dict["loss"]
    metric = output_dict["metric"]
    metrics = {"loss": loss.item(), "metric": metric}
    loss.backward(create_graph=create_graph, retain_graph=create_graph)

    if create_graph:
        _, new_parameters = optimizer.step(retain_graph=create_graph)
    else:
        optimizer.step(retain_graph=create_graph)

    return loss, new_parameters, metrics


def maml_task(data_inner, data_outer, model, optimizer, create_graph):
    """Adapt model parameters to task and use adapted params to predict new samples

    Arguments:
        data_inner (iterable): list of input-output for task adaptation.
        data_outer (iterable): list of input-output for task validation.
        model (torch.nn.Module): task learner.
        optimizer (maml.optim): optimizer for inner loop.
        criterion (func): loss criterion.
        create_graph (bool): create graph through gradient step.
    """
    metrics = []
    original_parameters = model.state_dict(keep_vars=True)
    device = next(model.parameters()).device

    # Adaptation of parameters to task
    for i, inputs in enumerate(data_inner):
        inputs = move_to_device(inputs, device)

        loss, new_params, metric = maml_inner_step(inputs, model, optimizer, create_graph)
        metrics.append(metric)

        if create_graph:
            load_state_dict(model, build_dict([n for n, _ in model.named_parameters()], new_params))

        for p in original_parameters.values():
            p.grad = None

    # Run with adapted parameters on task
    for i, inputs in enumerate(data_outer):
        inputs = move_to_device(inputs, device)
        with torch.backends.cudnn.flags(enabled=False):
            output_dict = model(**inputs, return_metric=True)
            loss += output_dict["loss"]
            metrics.append(output_dict["metric"])

    load_state_dict(model, original_parameters)

    return loss, metrics


def maml_outer_step(task_iterator, model, optimizer_cls,
                    create_graph=True, **optimizer_kwargs):
    """MAML objective.

    Run MAML on a batch of tasks.


    Arguments:
        task_iterator (iterator): data sampler for K tasks. Of the format
            [task1, task2, task3] where each task is of the format
            task1 = (data_iterator_inner, data_iterator_outer) and each
            data_iterator_ = [(input_batch1, target_batch1), ...]

            ::note::
                the inner data_iterator defines the number of gradient

        model (Module): task learner.
        optimizer_cls (maml.optim.SGD, maml.optim.Adam): inner optimizer class.
            Must allow backpropagation through gradient step.
        criterion (func): loss criterion.
        return_predictions (bool): whether to return.
        return_results (bool): return accumulated meta-data.
        create_graph (bool): create computational graph through gradient step.
        optimizer_kwargs (kwargs): kwargs to optimizer.
    """
    loss = 0
    metrics = []
    for i, task in enumerate(task_iterator):
        inner_iterator, outer_iterator = task[:-1], task[-1:]
        task_optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

        task_loss, metric = maml_task(
            inner_iterator, outer_iterator, model, task_optimizer, create_graph)
        metrics.append(metric)
        loss += task_loss

    loss = loss / (i + 1)

    return loss, metrics

###############################################################################


class MAML(nn.Module):

    """MAML

    Class Instance for the MAML objective

    Arguments:
        model (torch.nn.Module): task learner.
        optimizer_cls (maml.optim): task optimizer. Note: must allow backpropagation through gradient steps.
        criterion (func): loss criterion.
        tensor (bool): whether meta mini-batches come as a tensor or as a list of dataloaders.
        inner_bsz (int): if tensor=True, batch size in inner loop.
        outer_bsz (int): if tensor=True, batch size in outer loop.
        inner_steps (int): if tensor=True, number of steps in inner loop.
        outer_steps (int): if tensor=True, number of steps in outer loop.

    Example:
        >>> loss = maml.forward(task_iterator)
        >>> loss.backward()
        >>> meta_optimizer.step()
        >>> meta_optimizer.zero_grad()
    """

    def __init__(self, model, optimizer_cls, **optimizer_kwargs):
        super(MAML, self).__init__()

        self.model = model
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs

    def forward(self, tasks, create_graph=True):
        return maml_outer_step(
            task_iterator=tasks,
            model=self.model,
            optimizer_cls=self.optimizer_cls,
            create_graph=create_graph,
            **self.optimizer_kwargs)
