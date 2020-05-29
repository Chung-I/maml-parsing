__author__ = 'max'

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

def crf_loss(energy, target_heads, mask=None):
    '''

    Args:
        heads: Tensor
            the head input tensor with shape = [batch, length, model_dim]
        children: Tensor
            the child input tensor with shape = [batch, length, model_dim]
        target_heads: Tensor
            the tensor of target labels with shape [batch, length]
        mask:Tensor or None
            the mask tensor with shape = [batch, length]

    Returns: Tensor
            A 1D tensor for minus log likelihood loss
    '''
    batch, length = target_heads.size()
    # [batch, length, length]
    A = torch.exp(energy.double())
    # mask out invalid positions
    if mask is not None:
        mask = mask.double()
        A = A * mask.unsqueeze(2) * mask.unsqueeze(1)

    # set diagonal elements to 0
    diag_mask = 1.0 - torch.eye(length).unsqueeze(0).type_as(energy)
    A = A * diag_mask
    energy = energy * diag_mask

    # get D [batch, length]
    D = A.sum(dim=2)
    rtol = 1e-4
    atol = 1e-6
    D += atol
    if mask is not None:
        D = D * mask

    # [batch, length, length]
    D = torch.diag_embed(D)

    # compute laplacian matrix
    # [batch, length, length]
    L = D - A

    if mask is not None:
        L = L + torch.diag_embed(1. - mask)

    # compute partition Z(x) [batch]
    L = L[:, 1:, 1:]
    z = torch.logdet(L)

    return z.float()
