from torch.nn import Linear
import torch
from dptb.data import AtomicDataDict
from typing import Optional, Any, Union, Callable, OrderedDict
from torch import Tensor
from dptb.utils.tools import _get_activation_fn
import torch.nn.functional as F

class AtomicLinear(torch.nn.Module):
    def init(
            self, 
            in_features: int,
            out_features: int, 
            field = AtomicDataDict.NODE_FEATURES_KEY,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu")
            ):
        self.linear = Linear(in_features, out_features, dtype=dtype, device=device)
        self.field = field
    
    def forward(self, data: AtomicDataDict.Type):
        data[self.field] = self.linear(data[self.field])
        return data
    
class AtomicMLP(torch.nn.Module):
    def __init__(
            self,
            in_feature, 
            hidden_feature, 
            out_feature,
            field = AtomicDataDict.NODE_FEATURES_KEY,
            activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, 
            if_batch_normalized: bool = False, 
            device: Union[str, torch.dvice] = torch.device('cpu'), 
            dtype: Union[str, torch.dtype] = torch.float32
            ):
        super(AtomicMLP, self).__init__()
        self.in_layer = AtomicLinear(
            in_features=in_feature, 
            out_features=hidden_feature, 
            field = field,
            device=device, 
            dtype=dtype)
        self.out_layer = AtomicLinear(
            in_features=hidden_feature, 
            out_features=out_feature, 
            field=field,
            device=device, 
            dtype=dtype)

        if if_batch_normalized:
            self.bn1 = torch.nn.BatchNorm1d(hidden_feature)
            self.bn2 = torch.nn.BatchNorm1d(out_feature)
        self.if_batch_normalized = if_batch_normalized
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

        self.field = field

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(AtomicMLP, self).__setstate__(state)

    def forward(self, data: AtomicDataDict.Type):
        data = self.in_layer(data)
        if self.if_batch_normalized:
            data[self.field] = self.bn1(data[self.field])
        data[self.field] = self.activation(data[self.field])
        data = self.out_layer(data)
        if self.if_batch_normalized:
            data[self.field] = self.bn2(data[self.field])

        return data
    

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