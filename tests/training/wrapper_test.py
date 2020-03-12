import torch
import torch.nn as nn
import numpy as np

from unittest import TestCase
from src.training.wrapper import Wrapper, BaseWrapper, MultiWrapper, ReptileWrapper, FOMAMLWrapper

class BaseModel(nn.Module):
    def __init__(self, linear):
        super(BaseModel, self).__init__()
        self.linear = linear

    def forward(self, inputs):
        output_dict = {}
        outputs = self.linear(inputs)
        output_dict["loss"] = torch.sum(outputs ** 2)
        return output_dict

class MultiWrapperTest(TestCase):
    def setUp(self):
        linear = nn.Linear(2, 2, bias=False)
        self.model = BaseModel(linear)
        self.model.linear.weight = nn.Parameter(torch.Tensor([[0.1, 0.3],[0.2, 0.4]]))
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1.0)
        #self.optimizer_cls = 'SGD'
        #self.optimizer_kwargs = {'lr': 1.0}
        self.tasks = [
            [{"inputs": torch.tensor([-0.1, 0.1], requires_grad=True)},
             {"inputs": torch.tensor([0.1, -0.1], requires_grad=True)}],
            [{"inputs": torch.tensor([0.1, -0.1], requires_grad=True)},
             {"inputs": torch.tensor([-0.1, 0.1], requires_grad=True)}],
        ]
        self.cuda_device = -1
        self.wrapper = MultiWrapper(self.model, self.optimizer)

    def test_outer_loop(self):
        self.optimizer.zero_grad()
        expected_grad = torch.tensor([[-0.004, 0.004],[-0.004, 0.004]])
        expected_weight = torch.tensor([[0.104, 0.296],[0.204, 0.396]])
        loss = self.wrapper(self.tasks)
        self.optimizer.step()
        np.testing.assert_array_almost_equal(loss, 0.0008)
        np.testing.assert_array_almost_equal(self.wrapper.model.linear.weight.grad, expected_grad)
        np.testing.assert_array_almost_equal(self.wrapper.model.linear.weight.data, expected_weight)

class ReptileWrapperTest(TestCase):
    def setUp(self):
        linear = nn.Linear(2, 2, bias=False)
        self.model = BaseModel(linear)
        self.model.linear.weight = nn.Parameter(torch.Tensor([[0.1, 0.3],[0.2, 0.4]]))
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1.0)
        optimizer_cls = 'SGD'
        optimizer_kwargs = {'lr': 1.0}
        self.tasks = [
            [{"inputs": torch.tensor([-0.1, 0.1], requires_grad=True)},
             {"inputs": torch.tensor([0.1, -0.1], requires_grad=True)}],
            [{"inputs": torch.tensor([0.1, -0.1], requires_grad=True)},
             {"inputs": torch.tensor([-0.1, 0.1], requires_grad=True)}],
        ]
        self.cuda_device = -1
        self.wrapper = ReptileWrapper(self.model, self.optimizer, optimizer_cls,
                                      optimizer_kwargs)

    def test_outer_loop(self):
        self.optimizer.zero_grad()
        expected_grad = torch.tensor([[-0.00392, 0.00392],[-0.00392, 0.00392]])
        expected_weight = torch.tensor([[0.10392, 0.29608],[0.20392, 0.39608]])
        loss = self.wrapper(self.tasks)
        self.optimizer.step()
        np.testing.assert_array_almost_equal(loss, 0.00076864)
        np.testing.assert_array_almost_equal(self.wrapper.model.linear.weight.grad, expected_grad)
        np.testing.assert_array_almost_equal(self.wrapper.model.linear.weight.data, expected_weight)

class FOMAMLWrapperTest(TestCase):
    def setUp(self):
        linear = nn.Linear(2, 2, bias=False)
        self.model = BaseModel(linear)
        self.model.linear.weight = nn.Parameter(torch.Tensor([[0.1, 0.3],[0.2, 0.4]]))
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1.0)
        optimizer_cls = 'SGD'
        optimizer_kwargs = {'lr': 1.0}
        self.tasks = [
            [{"inputs": torch.tensor([-0.1, 0.1], requires_grad=True)},
             {"inputs": torch.tensor([0.1, -0.1], requires_grad=True)}],
            [{"inputs": torch.tensor([0.1, -0.1], requires_grad=True)},
             {"inputs": torch.tensor([-0.1, 0.1], requires_grad=True)}],
        ]
        self.cuda_device = -1
        self.wrapper = FOMAMLWrapper(self.model, self.optimizer, optimizer_cls,
                                      optimizer_kwargs)

    def test_outer_loop(self):
        self.optimizer.zero_grad()
        expected_grad = torch.tensor([[-0.00384, 0.00384],[-0.00384, 0.00384]])
        expected_weight = torch.tensor([[0.10384, 0.29616],[0.20384, 0.39616]])
        loss = self.wrapper(self.tasks)
        self.optimizer.step()
        np.testing.assert_array_almost_equal(loss, 0.00076864)
        np.testing.assert_array_almost_equal(self.wrapper.model.linear.weight.grad, expected_grad)
        np.testing.assert_array_almost_equal(self.wrapper.model.linear.weight.data, expected_weight)