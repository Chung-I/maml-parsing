import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

from unittest import TestCase
from src.modules.vib import ContinuousVIB

class BaseModel(nn.Module):
    def __init__(self, linear):
        super(BaseModel, self).__init__()
        self.linear = linear

    def forward(self, inputs, return_metric=False):
        output_dict = {}
        outputs = self.linear(inputs)
        output_dict["loss"] = torch.sum(outputs ** 2)
        output_dict["metric"] = 0
        return output_dict

class ContinuousVIBTest(TestCase):
    def setUp(self):
        tag_dim = 5
        embedding_dim = 10
        encoder_output_dim = 10
        activation = "leaky_relu"
        type_token_reg = False
        self.VIB = ContinuousVIB(
            tag_dim,
            embedding_dim,
            encoder_output_dim,
            activation,
            type_token_reg=type_token_reg,
        )
        self.mean1 = torch.tensor([[[-0.1, 0.5],
                                    [0.7, -0.4]],
                                   [[0.3, -0.9],
                                    [-0.6, 0.2]]])
        self.mean2 = torch.tensor([[[0.8, -0.0],
                                    [-0.4, 0.5]],
                                   [[-0.3, 0.7],
                                    [0.2, -0.1]]])
        self.std1 = torch.tensor([[[0.3, 0.1],
                                    [0.2, 0.4]],
                                   [[0.6, 0.4],
                                    [0.9, 0.2]]])
        self.cov1 = self.std1 ** 2
        self.std2 = torch.tensor([[[0.2, 0.3],
                                    [0.3, 0.8]],
                                   [[0.6, 0.5],
                                    [0.3, 0.4]]])
        self.cov2 = self.std2 ** 2
        self.mask = torch.tensor([[1, 1], [1, 0]])

    def test_kldiv(self):
        kldiv = self.VIB.kl_div((self.mean1, self.cov1), (self.mean2, self.cov2), self.mask)
        np.testing.assert_array_almost_equal(kldiv, 8.617202)
