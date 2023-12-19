from torch_geometric.nn.conv import MessagePassing
import torch
from typing import Optional, Tuple, Union
from dptb.data import AtomicDataDict
from dptb.nn.embedding.emb import Embedding
from ..base import ResNet, FFN
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
from dptb.utils.constants import dtype_dict
from ..type_encode.one_hot import OneHotAtomEncoding
from ..cutoff import polynomial_cutoff
from ..radial_basis import BesselBasis
from torch_runstats.scatter import scatter

def get_neuron_config(nl):
    n = len(nl)
    if n % 2 == 0:
        d_out = nl[-1]
        nl = nl[:-1]
    config = []
    for i in range(1,len(nl)-1, 2):
        config.append({'in_features': nl[i-1], 'hidden_features': nl[i], 'out_features': nl[i+1]})

    if n % 2 == 0:
        config.append({'in_features': nl[-1], 'out_features': d_out})

    return config

@Embedding.register("mpnn")
class MPNN(torch.nn.Module):
    def __init__(
            self,
            r_max:Union[float, torch.Tensor],
            p:Union[int, torch.LongTensor],
            n_basis: Union[int, torch.LongTensor, None]=None,
            n_node: Union[int, torch.LongTensor, None]=None,
            n_edge: Union[int, torch.LongTensor, None]=None,
            n_atom: int=1,
            n_layer: int=1,
            node_net: dict={},
            edge_net: dict={},
            if_exp: bool=False,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu")):
        
        super(MPNN, self).__init__()

        self.n_node = n_node
        self.n_edge = n_edge
        if isinstance(r_max, float):
            self.r_max = torch.tensor(r_max, dtype=dtype, device=device)
        else:
            self.r_max = r_max

        self.p = p
        self.layers = torch.nn.ModuleList([])
        for _ in range(n_layer):
            self.layers.append(
                CGConvLayer(
                    r_max=self.r_max, 
                    p=p, 
                    n_edge=n_edge, 
                    n_node=n_node, 
                    node_net=node_net, 
                    edge_net=edge_net, 
                    dtype=dtype,
                    device=device,
                    if_exp=if_exp,
                    )
                    )
            
        self.onehot = OneHotAtomEncoding(num_types=n_atom, set_features=False)
        self.node_emb = torch.nn.Linear(n_atom, n_node)
        edge_net["config"] = get_neuron_config([2*n_node+n_basis]+edge_net["neurons"]+[n_edge])
        self.edge_emb = ResNet(**edge_net, device=device, dtype=dtype)
        self.bessel = BesselBasis(r_max=r_max, num_basis=n_basis, trainable=True)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = self.onehot(data)
        data = AtomicDataDict.with_env_vectors(data, with_lengths=True)
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)

        node_features = self.node_emb(data[AtomicDataDict.NODE_ATTRS_KEY])
        env_features = self.edge_emb(torch.cat([node_features[data[AtomicDataDict.ENV_INDEX_KEY][0]], node_features[data[AtomicDataDict.ENV_INDEX_KEY][1]], self.bessel(data[AtomicDataDict.ENV_LENGTH_KEY])], dim=-1))
        edge_features = self.edge_emb(torch.cat([node_features[data[AtomicDataDict.EDGE_INDEX_KEY][0]], node_features[data[AtomicDataDict.EDGE_INDEX_KEY][1]], self.bessel(data[AtomicDataDict.EDGE_LENGTH_KEY])], dim=-1))

        for layer in self.layers:
            node_features, env_features, edge_features = layer(
                env_index=data[AtomicDataDict.ENV_INDEX_KEY],
                edge_index=data[AtomicDataDict.EDGE_INDEX_KEY],
                env_emb=env_features,
                edge_emb=edge_features,
                node_emb=node_features,
                env_length=data[AtomicDataDict.ENV_LENGTH_KEY],
                edge_length=data[AtomicDataDict.EDGE_LENGTH_KEY],
            )
        data[AtomicDataDict.NODE_FEATURES_KEY] = node_features

        data[AtomicDataDict.EDGE_FEATURES_KEY] = edge_features

        return data
    
    @property
    def out_edge_dim(self):
        return self.n_edge

    @property
    def out_node_dim(self):
        return self.n_node
    

class MPNNLayer(MessagePassing):
    def __init__(
            self, 
            r_max:Union[float, torch.Tensor],
            p:Union[int, torch.LongTensor],
            n_edge: int,
            n_node: int,
            node_net: dict={},
            edge_net: dict={},
            aggr="mean",
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"), **kwargs):
        
        super(MPNNLayer, self).__init__(aggr=aggr, **kwargs)

        if isinstance(device, str):
            device = torch.device(device)
        if isinstance(dtype, str):
            dtype = dtype_dict[dtype]

        if isinstance(r_max, float):
            self.r_max = torch.tensor(r_max, dtype=dtype, device=device)
        else:
            self.r_max = r_max

        self.p = p

        edge_net["config"] = get_neuron_config([2*n_node+n_edge]+edge_net["neurons"]+[n_edge])
        self.mlp_edge = ResNet(**edge_net, device=device, dtype=dtype)
        node_net["config"] = get_neuron_config([2*n_node+n_edge]+node_net["neurons"]+[n_node])
        self.mlp_node = ResNet(**node_net, dtype=dtype, device=device)

        self.node_layer_norm = torch.nn.LayerNorm(n_node, elementwise_affine=True)

        self.device = device
        self.dtype = dtype

    def forward(self, edge_index, env_index, node_emb, env_emb, edge_emb):
        
        z_ik = torch.cat([node_emb[env_index[0]], node_emb[env_index[1]], env_emb], dim=-1)
        node_emb = node_emb + self.propagate(env_index, z_ik=z_ik)

        env_emb = self.mlp_edge(torch.cat([node_emb[env_index[0]], env_emb, node_emb[env_index[1]]], dim=-1))
        edge_emb = self.mlp_edge(torch.cat([node_emb[edge_index[0]], edge_emb, node_emb[edge_index[1]]], dim=-1))

        return node_emb, env_emb, edge_emb

    def message(self, z_ik):
        
        return self.mlp_node(z_ik)

    def update(self, aggr_out):
        """_summary_

        Parameters
        ----------
        aggr_out : The output of the aggregation layer, which is the mean of the message vectors as size [N, D, 3]

        Returns
        -------
        _type_
            _description_
        """

        aggr_out = aggr_out.reshape(aggr_out.shape[0], -1)
        return self.node_layer_norm(aggr_out) # [N, D*D]

class CGConvLayer(MessagePassing):
    def __init__(
            self, 
            r_max:Union[float, torch.Tensor],
            p:Union[int, torch.LongTensor],
            n_edge: int,
            n_node: int,
            aggr="add",
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
            if_exp: bool=False,
            **kwargs):
        
        super(CGConvLayer, self).__init__(aggr=aggr, **kwargs)

        if isinstance(device, str):
            device = torch.device(device)
        if isinstance(dtype, str):
            dtype = dtype_dict[dtype]

        if isinstance(r_max, float):
            self.r_max = torch.tensor(r_max, dtype=dtype, device=device)
        else:
            self.r_max = r_max

        self.p = p
        self.if_exp = if_exp

        self.lin_edge_f = Linear(2*n_node+n_edge, n_edge, device=device, dtype=dtype)
        self.lin_edge_s = Linear(2*n_node+n_edge, n_edge, device=device, dtype=dtype)
        self.lin_node_f = Linear(2*n_node+n_edge, n_node, dtype=dtype, device=device)
        self.lin_node_s = Linear(2*n_node+n_edge, n_node, dtype=dtype, device=device)

        self.node_layer_norm = torch.nn.LayerNorm(n_node, elementwise_affine=True)

        self.device = device
        self.dtype = dtype

    def forward(self, edge_index, env_index, node_emb, env_emb, edge_emb, env_length, edge_length):
        z_ik = torch.cat([node_emb[env_index[0]], node_emb[env_index[1]], env_emb], dim=-1)
        node_emb = node_emb + self.propagate(env_index, z_ik=z_ik, env_length=env_length)

        env_feature_in = torch.cat([node_emb[env_index[0]], env_emb, node_emb[env_index[1]]], dim=-1)
        env_emb = self.lin_edge_f(env_feature_in).sigmoid() * \
            F.softplus(self.lin_edge_s(env_feature_in))
        if self.if_exp:
            sigma = 3
            n = 2
            env_emb = env_emb * torch.exp(-env_length ** n / sigma ** n / 2).view(-1, 1)
        
        edge_feature_in = torch.cat([node_emb[edge_index[0]], edge_emb, node_emb[edge_index[1]]], dim=-1)
        edge_emb = self.lin_edge_f(edge_feature_in).sigmoid() * \
            F.softplus(self.lin_edge_s(edge_feature_in))
        if self.if_exp:
            sigma = 3
            n = 2
            edge_emb = edge_emb * torch.exp(-edge_length ** n / sigma ** n / 2).view(-1, 1)

        return node_emb, env_emb, edge_emb


    def message(self, z_ik, env_length) -> torch.Tensor:
        out = self.lin_node_f(z_ik).sigmoid() * F.softplus(self.lin_node_s(z_ik))
        if self.if_exp:
            sigma = 3
            n = 2
            out = out * torch.exp(-env_length ** n / sigma ** n / 2).view(-1, 1)
        return self.node_layer_norm(out)

    

# class CGConv(MessagePassing):
#     def __init__(self, channels: Union[int, Tuple[int, int]], dim: int = 0,
#                  aggr: str = 'add', normalization: str = None,
#                  bias: bool = True, if_exp: bool = False, **kwargs):
#         super(CGConv, self).__init__(aggr=aggr, flow="source_to_target", **kwargs)
#         self.channels = channels
#         self.dim = dim
#         self.normalization = normalization
#         self.if_exp = if_exp

#         if isinstance(channels, int):
#             channels = (channels, channels)

#         self.lin_f = nn.Linear(sum(channels) + dim, channels[1], bias=bias)
#         self.lin_s = nn.Linear(sum(channels) + dim, channels[1], bias=bias)
#         if self.normalization == 'BatchNorm':
#             self.bn = nn.BatchNorm1d(channels[1], track_running_stats=True)
#         elif self.normalization == 'LayerNorm':
#             self.ln = LayerNorm(channels[1])
#         elif self.normalization == 'PairNorm':
#             self.pn = PairNorm(channels[1])
#         elif self.normalization == 'InstanceNorm':
#             self.instance_norm = InstanceNorm(channels[1])
#         elif self.normalization is None:
#             pass
#         else:
#             raise ValueError('Unknown normalization function: {}'.format(normalization))

#         self.reset_parameters()

#     def reset_parameters(self):
#         self.lin_f.reset_parameters()
#         self.lin_s.reset_parameters()
#         if self.normalization == 'BatchNorm':
#             self.bn.reset_parameters()

#     def forward(self, x: Union[torch.Tensor, PairTensor], edge_index: Adj,
#                 edge_attr: OptTensor, env_index, env_attr, batch, distance, size: Size = None) -> torch.Tensor:
#         """"""
#         if isinstance(x, torch.Tensor):
#             x: PairTensor = (x, x)

#         # propagate_type: (x: PairTensor, edge_attr: OptTensor)
#         out = self.propagate(edge_index, x=x, edge_attr=edge_attr, distance=distance, size=size)
#         if self.normalization == 'BatchNorm':
#             out = self.bn(out)
#         elif self.normalization == 'LayerNorm':
#             out = self.ln(out, batch)
#         elif self.normalization == 'PairNorm':
#             out = self.pn(out, batch)
#         elif self.normalization == 'InstanceNorm':
#             out = self.instance_norm(out, batch)
#         out += x[1]
#         return out

#     def message(self, x_i, x_j, edge_attr: OptTensor, distance) -> torch.Tensor:
#         z = torch.cat([x_i, x_j, edge_attr], dim=-1)
#         out = self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))
#         if self.if_exp:
#             sigma = 3
#             n = 2
#             out = out * torch.exp(-distance ** n / sigma ** n / 2).view(-1, 1)
#         return out

#     def __repr__(self):
#         return '{}({}, dim={})'.format(self.__class__.__name__, self.channels, self.dim)