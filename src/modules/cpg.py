import torch
import torch.nn as nn
from copy import deepcopy

def get_submodule(module, submodule_name, delim="."):
    parts = submodule_name.split(delim)
    prefix = parts[0]
    suffix = delim.join(parts[1:])
    if suffix:
        submodule = getattr(module, prefix)
        return get_submodule(submodule, suffix)
    else:
        return module, getattr(module, prefix), prefix

class CPG(nn.Module):
    def __init__(self, input_size, base_module, filter_func=(lambda  n_w: n_w[-1].requires_grad)):
        super().__init__()
        self.cpg_funcs = dict()
        self.base_module = base_module
        self.param_names = [name for name, w in filter(filter_func, self.base_module.named_parameters())]
        total_numels = sum([w.numel() for name, w in filter(filter_func, self.base_module.named_parameters())])

        for name in self.param_names:
            module, weight, weight_name = get_submodule(self.base_module, name)
            del module._parameters[weight_name]
            setattr(module, weight_name, weight.data)

        self.cpg_weight = torch.nn.Linear(input_size, total_numels, bias=False)

    def forward(self, cpg_inputs):
        all_weights = self.cpg_weight(cpg_inputs)
        start = 0
        for name in self.param_names:
            module, ori_weight, weight_name = get_submodule(self.base_module, name)
            numel = ori_weight.numel()
            cpg_weight = torch.narrow(all_weights, -1, start, numel).reshape(ori_weight.shape)
            start = start + numel
            setattr(module, weight_name, cpg_weight.data)

            
            



