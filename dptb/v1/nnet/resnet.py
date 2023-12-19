import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dptb.nnet.mlp import MLP
from dptb.utils.tools import _get_activation_fn
from typing import Optional, Any, Union, Callable




class ResBlock(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, if_batch_normalized=False, device='cpu', dtype=torch.float32):
        super(ResBlock, self).__init__()
        self.layer = MLP(n_in, n_hidden, n_out, if_batch_normalized=if_batch_normalized, device=device, dtype=dtype, activation=activation)
        self.n_out = n_out
        self.n_in = n_in
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        pass
        # super(ResBlock, self).__setstate__(state)

    def forward(self, x):
        out = self.layer(x)
        if self.n_in < self.n_out:
            out = nn.functional.interpolate(x.unsqueeze(1), size=[self.n_out]).squeeze(1) + out
        elif self.n_in == self.n_out:
            out = x + out
        else:
            out = nn.functional.adaptive_avg_pool1d(input=x, output_size=self.n_out) + out

        out = self.activation(out)

        return out

class ResNet(nn.Module):
    def __init__(self, config, activation, if_batch_normalized=False, device='cpu', dtype=torch.float32):
        super(ResNet, self).__init__()
        self.layers = nn.ModuleList([])
        for kk in range(len(config)-1):
            self.layers.append(ResBlock(**config[kk], if_batch_normalized=if_batch_normalized, activation=activation, device=device, dtype=dtype))
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
        {'n_in': 3, 'n_hidden': 4, 'n_out': 8},
        {'n_in': 8, 'n_hidden': 6, 'n_out': 4}
    ]
    net = ResNet(config, activation='relu', if_batch_normalized=True)

    a = torch.randn(100, 3)

    print(net(a).size())