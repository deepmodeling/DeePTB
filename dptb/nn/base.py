from torch.nn import Linear
import torch
from dptb.data import AtomicDataDict
from typing import Optional, Any, Union, Callable, OrderedDict, List
from torch import Tensor
from dptb.utils.tools import _get_activation_fn
import torch.nn.functional as F
import torch.nn as nn

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
    
class AtomicFFN(torch.nn.Module):
    def __init__(
        self, 
        config: List[dict],
        field: AtomicDataDict.NODE_FEATURES_KEY,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        if_batch_normalized: bool = False, 
        device: Union[str, torch.dvice] = torch.device('cpu'), 
        dtype: Union[str, torch.dtype] = torch.float32
        ):
        super(AtomicFFN, self).__init__()
        self.layers = torch.nn.ModuleList([])
        for kk in range(len(config)-1):
            self.layers.append(
                AtomicResBlock(
                    **config[kk], 
                    field=field,
                    if_batch_normalized=if_batch_normalized, 
                    activation=activation, 
                    device=device, 
                    dtype=dtype
                    )
                )
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

        if config[-1].get('hidden_feature') is None:
            self.out_layer = AtomicLinear(in_features=config[-1]['in_feature'], out_features=config[-1]['out_feature'], field=field, device=device, dtype=dtype)
        else:
            self.out_layer = AtomicMLP(**config[-1], field=field,  if_batch_normalized=False, activation=activation, device=device, dtype=dtype)

    def forward(self, data: AtomicDataDict.Type):
        for layer in self.layers:
            data = layer(data)
            data[self.field] = self.activation(data[self.field])

        return self.out_layer(data)
    

class AtomicResBlock(torch.nn.Module):
    def __init__(self, 
                 in_feature: int, 
                 hidden_feature: int, 
                 out_feature: int, 
                 field = AtomicDataDict.NODE_FEATURES_KEY,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, 
                 if_batch_normalized: bool=False, 
                 device: Union[str, torch.dvice] = torch.device('cpu'), 
                 dtype: Union[str, torch.dtype] = torch.float32
            ):
        super(AtomicResBlock, self).__init__()
        self.layer = AtomicMLP(in_feature, hidden_feature, out_feature, field=field, if_batch_normalized=if_batch_normalized, device=device, dtype=dtype, activation=activation)
        self.out_feature = out_feature
        self.in_feature = in_feature
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

        self.field = field

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(AtomicResBlock, self).__setstate__(state)

    def forward(self, data: AtomicDataDict.Type):
        if self.in_feature < self.out_feature:
            res = F.interpolate(data[self.field].unsqueeze(1), size=[self.out_feature]).squeeze(1)
        elif self.in_feature == self.out_feature:
            res =  data[self.field]
        else:
            res = F.adaptive_avg_pool1d(input=data[self.field], output_size=self.n_out)

        data = self.layer(data)
        data[self.field] = data[self.field] + res

        data[self.field] = self.activation(data[self.field])

        return data

# The ResNet class is a neural network model that consists of multiple residual blocks and a final
# output layer, with options for activation functions and batch normalization.

class AtomicResNet(torch.nn.Module):
    def __init__(
            self, 
            config: List[dict],
            field: AtomicDataDict.NODE_FEATURES_KEY,
            activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
            if_batch_normalized: bool = False, 
            device: Union[str, torch.dvice] = torch.device('cpu'), 
            dtype: Union[str, torch.dtype] = torch.float32,
            **kwargs,
            ):
        """_summary_

        Parameters
        ----------
        config : list
            ep: config = [
                {'in_feature': 3, 'hidden_feature': 4, 'out_feature': 8},
                {'in_feature': 8, 'hidden_feature': 6, 'out_feature': 4}
            ]
        activation : _type_
            _description_
        if_batch_normalized : bool, optional
            _description_, by default False
        device : str, optional
            _description_, by default 'cpu'
        dtype : _type_, optional
            _description_, by default torch.float32
        """
        super(AtomicResNet, self).__init__()
        self.layers = torch.nn.ModuleList([])
        for kk in range(len(config)-1):
            self.layers.append(
                AtomicResBlock(
                    **config[kk], 
                    field=field,
                    if_batch_normalized=if_batch_normalized, 
                    activation=activation, 
                    device=device, 
                    dtype=dtype
                    )
                )
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation


        if config[-1].get('n_hidden') is None:
            self.out_layer = AtomicLinear(in_features=config[-1]['in_feature'], out_features=config[-1]['out_feature'], field=field, device=device, dtype=dtype)
        else:
            self.out_layer = AtomicMLP(**config[-1],  if_batch_normalized=False, field=field, activation=activation, device=device, dtype=dtype)

        self.field = field
    def forward(self, data: AtomicDataDict.Type):
        for layer in self.layers:
            data = layer(data)
            data[self.field] = self.activation(data[self.field])

        return self.out_layer(data)
    
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
    

class ResBlock(torch.nn.Module):
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

class ResNet(torch.nn.Module):
    def __init__(self, config, activation, if_batch_normalized=False, device='cpu', dtype=torch.float32, **kwargs):
        super(ResNet, self).__init__()
        self.layers = torch.nn.ModuleList([])
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