from torch.nn import Linear
import torch
from dptb.data import AtomicDataDict
from typing import Optional, Any, Union, Callable, OrderedDict, List
from torch import Tensor
from dptb.utils.constants import dtype_dict
from dptb.utils.tools import _get_activation_fn
from e3nn.util.codegen import CodeGenMixin
from e3nn.math import normalize2mom
import torch.nn.functional as F
import math
from torch import fx
import torch.nn as nn

class AtomicLinear(torch.nn.Module):
    def __init__(
            self, 
            in_features: int,
            out_features: int, 
            in_field = AtomicDataDict.NODE_FEATURES_KEY,
            out_field = AtomicDataDict.NODE_FEATURES_KEY,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu")
            ):
        super(AtomicLinear, self).__init__()
        if isinstance(device, str):
            device = torch.device(device)
        if isinstance(dtype, str):
            dtype = dtype_dict[dtype]
        self.linear = Linear(in_features, out_features, dtype=dtype, device=device)
        self.in_field = in_field
        self.out_field = out_field
    
    def forward(self, data: AtomicDataDict.Type):
        data[self.out_field] = self.linear(data[self.in_field])
        return data
    
class Identity(torch.nn.Module):
    def __init__(
        self,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
        **kwargs,
    ):
        super(Identity, self).__init__()

    def forward(self, data: AtomicDataDict) -> AtomicDataDict:
        return data

    
class AtomicMLP(torch.nn.Module):
    def __init__(
            self,
            in_features, 
            hidden_features, 
            out_features,
            in_field = AtomicDataDict.NODE_FEATURES_KEY,
            out_field = AtomicDataDict.NODE_FEATURES_KEY,
            activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, 
            if_batch_normalized: bool = False, 
            device: Union[str, torch.device] = torch.device('cpu'), 
            dtype: Union[str, torch.dtype] = torch.float32
            ):
        super(AtomicMLP, self).__init__()
        if isinstance(device, str):
            device = torch.device(device)
        if isinstance(dtype, str):
            dtype = dtype_dict[dtype]
        self.in_layer = Linear(
            in_features=in_features, 
            out_features=hidden_features, 
            device=device, 
            dtype=dtype)
        
        self.out_layer = Linear(
            in_features=hidden_features, 
            out_features=out_features,
            device=device, 
            dtype=dtype)

        if if_batch_normalized:
            self.bn1 = torch.nn.BatchNorm1d(hidden_features)
            self.bn2 = torch.nn.BatchNorm1d(out_features)
        self.if_batch_normalized = if_batch_normalized
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

        self.in_field = in_field
        self.out_field = out_field

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(AtomicMLP, self).__setstate__(state)

    def forward(self, data: AtomicDataDict.Type):
        x = self.in_layer(data[self.in_field])
        if self.if_batch_normalized:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.out_layer(x)
        if self.if_batch_normalized:
            x = self.bn2(x)
        data[self.out_field] = x

        return data
    
class AtomicFFN(torch.nn.Module):
    def __init__(
        self,
        config: List[dict],
        in_field: AtomicDataDict.NODE_FEATURES_KEY,
        out_field: AtomicDataDict.NODE_FEATURES_KEY,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        if_batch_normalized: bool = False,
        device: Union[str, torch.device] = torch.device('cpu'), 
        dtype: Union[str, torch.dtype] = torch.float32,
        **kwargs
        ):
        super(AtomicFFN, self).__init__()
        self.layers = torch.nn.ModuleList([])
        for kk in range(len(config)-1):
            if kk == 0:
                self.layers.append(
                    AtomicMLP(
                        **config[kk], 
                        in_field=in_field,
                        out_field=out_field,
                        if_batch_normalized=if_batch_normalized, 
                        activation=activation, 
                        device=device, 
                        dtype=dtype
                        )
                    )
            else:
                self.layers.append(
                    AtomicMLP(
                        **config[kk], 
                        in_field=out_field,
                        out_field=out_field,
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

        if config[-1].get('hidden_features') is None:
            self.out_layer = AtomicLinear(in_features=config[-1]['in_features'], out_features=config[-1]['out_features'], in_field=out_field, out_field=out_field, device=device, dtype=dtype)
        else:
            self.out_layer = AtomicMLP(**config[-1], in_field=out_field, out_field=out_field,  if_batch_normalized=False, activation=activation, device=device, dtype=dtype)
        self.out_field = out_field
        self.in_field = in_field
        # self.out_norm = nn.LayerNorm(config[-1]['out_features'], elementwise_affine=True)

    def forward(self, data: AtomicDataDict.Type):
        out_scale = self.out_scale(data[self.in_field])
        out_shift = self.out_shift(data[self.in_field])
        for layer in self.layers:
            data = layer(data)
            data[self.out_field] = self.activation(data[self.out_field])

        data = self.out_layer(data)
        # data[self.out_field] = self.out_norm(data[self.out_field])
        return data
    

class AtomicResBlock(torch.nn.Module):
    def __init__(self, 
                 in_features: int, 
                 hidden_features: int, 
                 out_features: int, 
                 in_field = AtomicDataDict.NODE_FEATURES_KEY,
                 out_field = AtomicDataDict.NODE_FEATURES_KEY,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, 
                 if_batch_normalized: bool=False, 
                 device: Union[str, torch.device] = torch.device('cpu'), 
                 dtype: Union[str, torch.dtype] = torch.float32
            ):
        
        super(AtomicResBlock, self).__init__()
        self.in_field = in_field
        self.out_field = out_field
        self.layer = AtomicMLP(in_features, hidden_features, out_features, in_field=in_field, out_field=out_field, if_batch_normalized=if_batch_normalized, device=device, dtype=dtype, activation=activation)
        self.out_features = out_features
        self.in_features = in_features
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(AtomicResBlock, self).__setstate__(state)

    def forward(self, data: AtomicDataDict.Type):
        if self.in_features < self.out_features:
            res = F.interpolate(data[self.in_field].unsqueeze(1), size=[self.out_features]).squeeze(1)
        elif self.in_features == self.out_features:
            res =  data[self.in_field]
        else:
            res = F.adaptive_avg_pool1d(input=data[self.in_field], output_size=self.out_features)

        data = self.layer(data)
        data[self.out_field] = data[self.out_field] + res

        data[self.out_field] = self.activation(data[self.out_field])

        return data

# The ResNet class is a neural network model that consists of multiple residual blocks and a final
# output layer, with options for activation functions and batch normalization.

class AtomicResNet(torch.nn.Module):
    def __init__(
            self, 
            config: List[dict],
            in_field: AtomicDataDict.NODE_FEATURES_KEY,
            out_field: AtomicDataDict.NODE_FEATURES_KEY,
            activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
            if_batch_normalized: bool = False, 
            device: Union[str, torch.device] = torch.device('cpu'), 
            dtype: Union[str, torch.dtype] = torch.float32,
            **kwargs,
            ):
        """_summary_

        Parameters
        ----------
        config : list
            ep: config = [
                {'in_features': 3, 'hidden_features': 4, 'out_features': 8},
                {'in_features': 8, 'hidden_features': 6, 'out_features': 4}
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
        self.in_field = in_field
        self.out_field = out_field
        self.layers = torch.nn.ModuleList([])
        for kk in range(len(config)-1):
            # the first layer will take the in_field as key to take `data[in_field]` and output the out_field, data[out_field] = layer(data[in_field])
            # the rest of the layers will take the out_field as key to take `data[out_field]` and output the out_field, data[out_field] = layer(data[out_field])
            # That why we need to set the in_field and out_field for 1st layer and the rest of the layers.
            if kk == 0:
                self.layers.append(
                    AtomicResBlock(
                        **config[kk], 
                        in_field=in_field,
                        out_field=out_field,
                        if_batch_normalized=if_batch_normalized, 
                        activation=activation, 
                        device=device, 
                        dtype=dtype
                        )
                    )
            else:
                self.layers.append(
                    AtomicResBlock(
                        **config[kk], 
                        in_field=out_field,
                        out_field=out_field,
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
            self.out_layer = AtomicLinear(in_features=config[-1]['in_features'], out_features=config[-1]['out_features'], in_field=out_field, out_field=out_field, device=device, dtype=dtype)
            nn.init.normal_(self.out_layer.linear.weight, mean=0, std=1e-3)
            nn.init.normal_(self.out_layer.linear.bias, mean=0, std=1e-3)
        else:
            self.out_layer = AtomicMLP(**config[-1],  if_batch_normalized=False, in_field=in_field, out_field=out_field, activation=activation, device=device, dtype=dtype)
            nn.init.normal_(self.out_layer.out_layer.weight, mean=0, std=1e-3)
            nn.init.normal_(self.out_layer.out_layer.bias, mean=0, std=1e-3)
        # self.out_norm = nn.LayerNorm(config[-1]['out_features'], elementwise_affine=True)
        

    def forward(self, data: AtomicDataDict.Type):

        for layer in self.layers:
            data = layer(data)
            data[self.out_field] = self.activation(data[self.out_field])
        data = self.out_layer(data)
        # data[self.out_field] = self.out_norm(data[self.out_field])
        return data
    
class MLP(nn.Module):
    def __init__(
            self, 
            in_features, 
            hidden_features, 
            out_features, 
            activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, 
            if_batch_normalized=False, 
            device: Union[str, torch.device]=torch.device('cpu'), 
            dtype: Union[str, torch.dtype] = torch.float32,
            ):
        super(MLP, self).__init__()

        if isinstance(device, str):
            device = torch.device(device)
        if isinstance(dtype, str):
            dtype = dtype_dict[dtype]

        self.in_layer = nn.Linear(in_features=in_features, out_features=hidden_features, device=device, dtype=dtype)
        self.out_layer = nn.Linear(in_features=hidden_features, out_features=out_features, device=device, dtype=dtype)

        if if_batch_normalized:
            self.bn1 = nn.BatchNorm1d(hidden_features)
            self.bn2 = nn.BatchNorm1d(out_features)
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
    def __init__(
            self, 
            config, 
            activation, 
            if_batch_normalized=False, 
            device: Union[str, torch.device]=torch.device('cpu'), 
            dtype: Union[str, torch.dtype] = torch.float32,
            **kwargs
            ):
        super(FFN, self).__init__()
        if isinstance(device, str):
            device = torch.device(device)
        if isinstance(dtype, str):
            dtype = dtype_dict[dtype]
            
        self.layers = nn.ModuleList([])
        for kk in range(len(config)-1):
            self.layers.append(MLP(**config[kk], if_batch_normalized=if_batch_normalized, activation=activation, device=device, dtype=dtype))
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

        if config[-1].get('hidden_features') is None:
            self.out_layer = nn.Linear(in_features=config[-1]['in_features'], out_features=config[-1]['out_features'], device=device, dtype=dtype)
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
    def __init__(
            self, 
            in_features, 
            hidden_features, 
            out_features, 
            activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, 
            if_batch_normalized=False, 
            device: Union[str, torch.device]=torch.device('cpu'), 
            dtype: Union[str, torch.dtype] = torch.float32,
            ):
        super(ResBlock, self).__init__()
        if isinstance(device, str):
            device = torch.device(device)
        if isinstance(dtype, str):
            dtype = dtype_dict[dtype]

        self.layer = MLP(in_features, hidden_features, out_features, if_batch_normalized=if_batch_normalized, device=device, dtype=dtype, activation=activation)
        self.out_features = out_features
        self.in_features = in_features
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        pass
        # super(ResBlock, self).__setstate__(state)

    def forward(self, x):
        out = self.layer(x)
        if self.in_features < self.out_features:
            out = nn.functional.interpolate(x.unsqueeze(1), size=[self.out_features]).squeeze(1) + out
        elif self.in_features == self.out_features:
            out = x + out
        else:
            out = nn.functional.adaptive_avg_pool1d(input=x, output_size=self.out_features) + out

        out = self.activation(out)

        return out

class ResNet(torch.nn.Module):
    def __init__(
            self, 
            config, 
            activation, 
            if_batch_normalized=False, 
            device: Union[str, torch.device]=torch.device('cpu'), 
            dtype: Union[str, torch.dtype] = torch.float32,
            **kwargs
            ):
        super(ResNet, self).__init__()
        if isinstance(device, str):
            device = torch.device(device)
        if isinstance(dtype, str):
            dtype = dtype_dict[dtype]

        self.layers = torch.nn.ModuleList([])
        for kk in range(len(config)-1):
            self.layers.append(ResBlock(**config[kk], if_batch_normalized=if_batch_normalized, activation=activation, device=device, dtype=dtype))
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation


        if config[-1].get('hidden_features') is None:
            self.out_layer = nn.Linear(in_features=config[-1]['in_features'], out_features=config[-1]['out_features'], device=device, dtype=dtype)
            # nn.init.normal_(self.out_layer.weight, mean=0, std=1e-3)
            # nn.init.normal_(self.out_layer.bias, mean=0, std=1e-3)
        else:
            self.out_layer = MLP(**config[-1],  if_batch_normalized=False, activation=activation, device=device, dtype=dtype)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        return self.out_layer(x)

class ScalarMLPFunction(CodeGenMixin, torch.nn.Module):
    """Module implementing an MLP according to provided options."""

    in_features: int
    out_features: int

    def __init__(
        self,
        mlp_input_dimension: Optional[int],
        mlp_latent_dimensions: List[int],
        mlp_output_dimension: Optional[int],
        mlp_nonlinearity: Optional[str] = "silu",
        mlp_initialization: str = "normal",
        mlp_dropout_p: float = 0.0,
        mlp_batchnorm: bool = False,
    ):
        super().__init__()
        nonlinearity = {
            None: None,
            "silu": torch.nn.functional.silu,
            "ssp": ShiftedSoftPlus,
        }[mlp_nonlinearity]
        if nonlinearity is not None:
            nonlin_const = normalize2mom(nonlinearity).cst
        else:
            nonlin_const = 1.0

        dimensions = (
            ([mlp_input_dimension] if mlp_input_dimension is not None else [])
            + mlp_latent_dimensions
            + ([mlp_output_dimension] if mlp_output_dimension is not None else [])
        )
        assert len(dimensions) >= 2  # Must have input and output dim
        num_layers = len(dimensions) - 1

        self.in_features = dimensions[0]
        self.out_features = dimensions[-1]

        # Code
        params = {}
        graph = fx.Graph()
        tracer = fx.proxy.GraphAppendingTracer(graph)

        def Proxy(n):
            return fx.Proxy(n, tracer=tracer)

        features = Proxy(graph.placeholder("x"))
        norm_from_last: float = 1.0

        base = torch.nn.Module()

        for layer, (h_in, h_out) in enumerate(zip(dimensions, dimensions[1:])):
            # do dropout
            if mlp_dropout_p > 0:
                # only dropout if it will do something
                # dropout before linear projection- https://stats.stackexchange.com/a/245137
                features = Proxy(graph.call_module("_dropout", (features.node,)))

            # make weights
            w = torch.empty(h_in, h_out)

            if mlp_initialization == "normal":
                w.normal_()
            elif mlp_initialization == "uniform":
                # these values give < x^2 > = 1
                w.uniform_(-math.sqrt(3), math.sqrt(3))
            elif mlp_initialization == "orthogonal":
                # this rescaling gives < x^2 > = 1
                torch.nn.init.orthogonal_(w, gain=math.sqrt(max(w.shape)))
            else:
                raise NotImplementedError(
                    f"Invalid mlp_initialization {mlp_initialization}"
                )

            # generate code
            params[f"_weight_{layer}"] = w
            w = Proxy(graph.get_attr(f"_weight_{layer}"))
            w = w * (
                norm_from_last / math.sqrt(float(h_in))
            )  # include any nonlinearity normalization from previous layers
            features = torch.matmul(features, w)

            if mlp_batchnorm:
                # if we call batchnorm, do it after the nonlinearity
                features = Proxy(graph.call_module(f"_bn_{layer}", (features.node,)))
                setattr(base, f"_bn_{layer}", torch.nn.BatchNorm1d(h_out))

            # generate nonlinearity code
            if nonlinearity is not None and layer < num_layers - 1:
                features = nonlinearity(features)
                # add the normalization const in next layer
                norm_from_last = nonlin_const

        graph.output(features.node)

        for pname, p in params.items():
            setattr(base, pname, torch.nn.Parameter(p))

        if mlp_dropout_p > 0:
            # with normal dropout everything blows up
            base._dropout = torch.nn.AlphaDropout(p=mlp_dropout_p)

        self._codegen_register({"_forward": fx.GraphModule(base, graph)})

    def forward(self, x):
        return self._forward(x)

@torch.jit.script
def ShiftedSoftPlus(x: torch.Tensor):
    return torch.nn.functional.softplus(x) - math.log(2.0)