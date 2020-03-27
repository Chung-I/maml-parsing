from typing import Type
from itertools import product

import torch
from torch.nn import Parameter

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, _Seq2SeqWrapper
def weight_drop_factory(module_class):
    class WeightDrop(module_class):
        def __init__(self, module_args, weights, wdrop=0, variational=False):
            super(WeightDrop, self).__init__(**module_args)
            self.weights = weights
            self.wdrop = wdrop
            self.variational = variational
            self._setup()
    
        def widget_demagnetizer_y2k_edition(*args, **kwargs):
            # We need to replace flatten_parameters with a nothing function
            # It must be a function rather than a lambda as otherwise pickling explodes
            # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
            # (╯°□°）╯︵ ┻━┻
            return
    
        def _setup(self):
            # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
            if issubclass(type(self), torch.nn.RNNBase):
                self.flatten_parameters = self.widget_demagnetizer_y2k_edition
    
            for name_w in self.weights:
                print('Applying weight drop of {} to {}'.format(self.wdrop, name_w))
                w = getattr(self, name_w)
                del self._parameters[name_w]
                self.register_parameter(name_w + '_raw', Parameter(w.data))
    
        def _setweights(self):
            for name_w in self.weights:
                raw_w = getattr(self, name_w + '_raw')
                w = None
                if self.variational:
                    mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                    if raw_w.is_cuda: mask = mask.cuda()
                    mask = torch.nn.functional.dropout(mask, p=self.wdrop, training=self.training)
                    w = mask.expand_as(raw_w) * raw_w
                else:
                    w = torch.nn.functional.dropout(raw_w, p=self.wdrop, training=self.training)
                setattr(self, name_w, w)
    
        def forward(self, *args):
            self._setweights()
            return super().forward(*args)

    return WeightDrop

class _WeightDropSeq2SeqWrapper:
    PYTORCH_MODELS = [torch.nn.GRU, torch.nn.LSTM, torch.nn.RNN]

    def __init__(self, module_class: Type[torch.nn.modules.RNNBase]) -> None:
        self._module_class = module_class

    def __call__(self, **kwargs) -> PytorchSeq2SeqWrapper:
        return self.from_params(Params(kwargs))

    def from_params(self, params: Params, **extras) -> PytorchSeq2SeqWrapper:
        if not params.pop_bool('batch_first', True):
            raise ConfigurationError("Our encoder semantics assumes batch is always first!")
        if self._module_class in self.PYTORCH_MODELS:
            params['batch_first'] = True
        stateful = params.pop_bool('stateful', False)
        weight_dropout = params.pop_float('weight_dropout', 0.0)
        variational = params.pop_float('variational', True)
        num_layers = params.get('num_layers', 1)
        bidirectional = params.get('bidirectional', False)
        all_recurrent_weights = [f"weight_hh_l{layer}{suffix}" for layer, suffix in 
            product(range(num_layers), [""] + ["_reverse"] * (1 if bidirectional else 0))]

        if weight_dropout > 0.0:
            module = weight_drop_factory(self._module_class)(
                module_args=params.as_dict(infer_type_and_cast=True),
                weights=all_recurrent_weights,
                wdrop=weight_dropout,
                variational=variational,
            )
        else:
            module = self._module_class(**params.as_dict(infer_type_and_cast=True))

        return PytorchSeq2SeqWrapper(module, stateful=stateful)

Seq2SeqEncoder.register("weightdrop_gru")(_WeightDropSeq2SeqWrapper(torch.nn.GRU))
Seq2SeqEncoder.register("weightdrop_lstm")(_WeightDropSeq2SeqWrapper(torch.nn.LSTM))

if __name__ == '__main__':
    import torch

    # Input is (seq, batch, input)
    x = torch.autograd.Variable(torch.randn(2, 1, 10)).cuda()
    h0 = None

    ###

    print('Testing WeightDrop')
    print('=-=-=-=-=-=-=-=-=-=')

    ###

    print('Testing WeightDrop with Linear')

    lin = weight_drop_factory(torch.nn.Linear)({"in_features": 10, "out_features": 10},
                                                 ['weight'], wdrop=0.9)
    lin.cuda()
    run1 = [x.sum() for x in lin(x).data]
    run2 = [x.sum() for x in lin(x).data]

    print('All items should be different')
    print('Run 1:', run1)
    print('Run 2:', run2)

    assert run1[0] != run2[0]
    assert run1[1] != run2[1]

    print('---')

    ###

    print('Testing WeightDrop with LSTM')

    wdrnn = weight_drop_factory(torch.nn.LSTM)({"input_size": 10, "hidden_size": 10},
                                               ['weight_hh_l0'], wdrop=0.9, variational=True)
    wdrnn.cuda()

    run1 = [x.sum() for x in wdrnn(x, h0)[0].data]
    run2 = [x.sum() for x in wdrnn(x, h0)[0].data]

    print('First timesteps should be equal, all others should differ')
    print('Run 1:', run1)
    print('Run 2:', run2)

    # First time step, not influenced by hidden to hidden weights, should be equal
    assert run1[0] == run2[0]
    # Second step should not
    assert run1[1] != run2[1]

    print('---')
