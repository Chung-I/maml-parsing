from itertools import chain

import torch
import torch.nn as nn

from allennlp.common import Params
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules import FeedForward
from allennlp.training.optimizers import Optimizer

class TaskAdvNet(nn.Module):
    def __init__(self,
                 encoder: Seq2VecEncoder,
                 projection: FeedForward,
                 optimizer: torch.nn.Optimizer,
                 first_n_states: int = 1,
                 weight: float = 0.0):
        self.first_n_states = first_n_states
        self.weight = weight
        self._encoder = encoder
        self._projection = projection
        self._loss_func = torch.nn.NLLLoss(reduction='mean')

    def forward(self, task_hidden_states, masks, labels):
        task_hidden_states = torch.cat(task_hidden_states, dim=0)
        masks = torch.cat(masks, dim=0)
        labels = task_hidden_states.new_tensor(labels).long()
        task_representations = self._encoder(task_hidden_states.detach(), masks)
        task_logits = self._projection(task_representations)
        loss = self._loss_func(task_logits, labels)
        return loss

    @classmethod
    def from_params(cls,
                    params: Params,
                    **extras):
        encoder = Seq2VecEncoder.from_params(params.pop("encoder"))
        projection = FeedForward.from_params(params.pop("projection"))
        first_n_states = params.pop_int("first_n_states", 1)
        weight = params.pop_float("weight", 0.0)
        parameters = [[n, p] for n, p in chain(encoder.named_parameters(),
                                               projection.named_parameters())
                      if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))
        return cls(
            encoder,
            projection,
            optimizer,
            first_n_states,
            weight,
        )



