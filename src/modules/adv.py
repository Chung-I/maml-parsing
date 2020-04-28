from itertools import chain
import numpy as np

import torch
import torch.nn as nn

from allennlp.common import Params, FromParams
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules import FeedForward
from allennlp.training.optimizers import Optimizer
from allennlp.nn import Activation

from src.training.util import move_to_device, pad_batched_tensors

class TaskDiscriminator(nn.Module, FromParams):
    def __init__(self,
                 encoder: Seq2VecEncoder,
                 first_n_states: int = 1,
                 weight: float = None,
                 disc_grad_norm: float = None,
                 gen_grad_norm: float = None,
                 steps_per_update: int = 5,
                 target_distribution: str = 'uniform',
                 num_tasks: int = 1):
        super().__init__()
        self.first_n_states = first_n_states
        self.weight = weight
        self.disc_grad_norm = disc_grad_norm
        self.gen_grad_norm = gen_grad_norm
        self._encoder = encoder
        self.steps_per_update = steps_per_update
        self._projection = FeedForward(
            input_dim=encoder.get_output_dim(),
            hidden_dims=num_tasks,
            num_layers=1,
            activations=Activation.by_name("relu")(),
        )
        self.log_softmax = torch.nn.LogSoftmax()
        self._d_loss = torch.nn.NLLLoss(reduction='mean')
        self._g_loss = None
        if target_distribution == 'uniform':
            kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
            self._g_loss = lambda inputs: kl_loss(
                inputs,
                inputs.new_ones(inputs.size()) / inputs.size(1)
            )

    def get_hidden_states(self, generator, tasks):
        def get_hidden_states(task):
            hidden_states = []
            masks = []
            labels = []
            device = next(generator.parameters()).device
            for inputs in task[:self.first_n_states]:
                inputs = move_to_device(inputs, device)
                output_dict = generator(**inputs)
                hidden_states.append(output_dict["hidden_state"])
                mask = output_dict["mask"]
                masks.append(mask)
                labels.append(inputs["langs"])
            return pad_batched_tensors(hidden_states), \
                torch.cat(labels, dim=0), \
                pad_batched_tensors(masks)
        hidden_states, labels, masks = zip(*map(get_hidden_states, tasks))
        return hidden_states, labels, masks

    def forward(self, task_hidden_states, labels, masks=None, detach=False):
        task_hidden_states = pad_batched_tensors(task_hidden_states)
        labels = torch.cat(labels, dim=0)
        masks = pad_batched_tensors(masks)[:,1:]

        if detach:
            task_hidden_states = task_hidden_states.detach()
        task_representations = self._encoder(task_hidden_states, masks)
        task_logits = self.log_softmax(self._projection(task_representations))
        acc = (task_logits.max(dim=-1)[1] == labels).float().mean()
        d_loss = self._d_loss(task_logits, labels)
        if self._g_loss is not None:
            g_loss = self._g_loss(task_logits)
        else:
            g_loss = -d_loss
        return d_loss, g_loss, acc

    def get_alpha(self, batch_num, num_batches):
        p = float(batch_num) / num_batches
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        return alpha

