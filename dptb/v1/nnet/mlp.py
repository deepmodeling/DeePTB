import torch
import torch.nn as nn

import torch.nn.functional as F
from torch import Tensor
from dptb.utils.tools import _get_activation_fn
from typing import Optional, Any, Union, Callable, OrderedDict

class MLP(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, if_batch_normalized=False, device='cpu', dtype=torch.float32):
        super(MLP, self).__init__()
        self.in_layer = nn.Linear(in_features=n_in, out_features=n_hidden, device=device, dtype=dtype)
        self.out_layer = nn.Linear(in_features=n_hidden, out_features=n_out, device=device, dtype=dtype)

        if if_batch_normalized:
            self.bn1 = nn.BatchNorm1d(n_hidden)
            self.bn2 = nn.BatchNorm1d(n_out)
        self.if_batch_normalized = if_batch_normalized
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(MLP, self).__setstate__(state)

    def forward(self, x):
        x = self.in_layer(x)
        if self.if_batch_normalized:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.out_layer(x)
        if self.if_batch_normalized:
            x = self.bn2(x)

        return x

class FFN(nn.Module):
    def __init__(self, config, activation, if_batch_normalized=False, device='cpu', dtype=torch.float32):
        super(FFN, self).__init__()
        self.layers = nn.ModuleList([])
        for kk in range(len(config)-1):
            self.layers.append(MLP(**config[kk], if_batch_normalized=if_batch_normalized, activation=activation, device=device, dtype=dtype))
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

        if config[-1].get('n_hidden') is None:
            self.out_layer = nn.Linear(in_features=config[-1]['n_in'], out_features=config[-1]['n_out'], device=device, dtype=dtype)
            # nn.init.normal_(self.out_layer.weight, mean=0, std=1e-3)
            # nn.init.normal_(self.out_layer.bias, mean=0, std=1e-3)
        else:
            self.out_layer = MLP(**config[-1],  if_batch_normalized=False, activation=activation, device=device, dtype=dtype)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        return self.out_layer(x)

if __name__ == '__main__':
    config = [
        {'n_in':3, 'n_hidden':4, 'n_out':8},
        {'n_in': 8, 'n_hidden': 6, 'n_out': 4}
    ]
    net = FFN(config, activation='relu', if_batch_normalized=True)

    a = torch.randn(100, 3)

    print(net(a).size())
