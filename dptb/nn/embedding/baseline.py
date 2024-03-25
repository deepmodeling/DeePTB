from torch_geometric.nn import MessagePassing
from torch_geometric.nn import Aggregation
import torch
from typing import Optional, Tuple, Union
from dptb.data import AtomicDataDict
from dptb.nn.embedding.emb import Embedding
from ..base import ResNet, FFN
from torch.nn import Linear
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

@Embedding.register("baseline")
class BASELINE(torch.nn.Module):
    def __init__(
            self,
            rc:Union[float, torch.Tensor],
            p:Union[int, torch.LongTensor],
            n_axis: Union[int, torch.LongTensor, None]=None,
            n_basis: Union[int, torch.LongTensor, None]=None,
            n_radial: Union[int, torch.LongTensor, None]=None,
            n_sqrt_radial: Union[int, torch.LongTensor, None]=None,
            n_atom: int=1,
            n_layer: int=1,
            radial_net: dict={},
            hidden_net: dict={},
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
            **kwargs,):
        
        super(BASELINE, self).__init__()

        assert n_axis <= n_sqrt_radial
        self.n_radial = n_radial
        self.n_sqrt_radial = n_sqrt_radial
        self.n_axis = n_axis

        if isinstance(rc, float):
            self.rc = torch.tensor(rc, dtype=dtype, device=device)
        else:
            self.rc = rc

        self.p = p
        self.node_emb_layer = _NODE_EMB(rc=self.rc, p=p, n_axis=n_axis, n_basis=n_basis, n_radial=n_radial, n_sqrt_radial=n_sqrt_radial, n_atom=n_atom, radial_net=radial_net, dtype=dtype, device=device)
        self.layers = torch.nn.ModuleList([])
        for _ in range(n_layer):
            self.layers.append(BaselineLayer(n_atom=n_atom, rc=self.rc, p=p, n_radial=n_radial, n_sqrt_radial=n_sqrt_radial, n_axis=n_axis, n_hidden=n_axis*n_sqrt_radial, hidden_net=hidden_net, radial_net=radial_net, dtype=dtype, device=device))
        self.onehot = OneHotAtomEncoding(num_types=n_atom, set_features=False)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = self.onehot(data)
        data = AtomicDataDict.with_env_vectors(data, with_lengths=True)
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)

        env_radial, edge_radial, node_emb, env_hidden, edge_hidden = self.node_emb_layer(
            env_vectors=data[AtomicDataDict.ENV_VECTORS_KEY], 
            atom_attr=data[AtomicDataDict.NODE_ATTRS_KEY], 
            env_index=data[AtomicDataDict.ENV_INDEX_KEY], 
            edge_index=data[AtomicDataDict.EDGE_INDEX_KEY],
            env_length=data[AtomicDataDict.ENV_LENGTH_KEY],
            edge_length=data[AtomicDataDict.EDGE_LENGTH_KEY],
        )


        for layer in self.layers:
            env_radial, env_hidden, edge_radial, edge_hidden, node_emb = layer(
                env_length=data[AtomicDataDict.ENV_LENGTH_KEY], 
                edge_length=data[AtomicDataDict.EDGE_LENGTH_KEY],
                env_index=data[AtomicDataDict.ENV_INDEX_KEY],
                edge_index=data[AtomicDataDict.EDGE_INDEX_KEY],
                env_radial=env_radial,
                edge_radial=edge_radial,
                node_emb=node_emb, 
                env_hidden=env_hidden,
                edge_hidden=edge_hidden,
            )
        
        # env_length = data[AtomicDataDict.ENV_LENGTH_KEY]
        # data[AtomicDataDict.NODE_FEATURES_KEY] = \
        #     scatter(src=polynomial_cutoff(x=env_length, r_max=self.rc, p=self.p).reshape(-1, 1) * env_radial, index=data[AtomicDataDict.ENV_INDEX_KEY][0], dim=0, reduce="sum")
        data[AtomicDataDict.NODE_FEATURES_KEY] = node_emb

        data[AtomicDataDict.EDGE_FEATURES_KEY] = edge_radial

        return data
    
    @property
    def out_edge_dim(self):
        return self.n_radial

    @property
    def out_node_dim(self):
        return self.n_sqrt_radial * self.n_axis
    
class SE2Aggregation(Aggregation):
    def forward(self, x: torch.Tensor, index: torch.LongTensor, **kwargs):
        """_summary_

        Parameters
        ----------
        x : tensor of size (N, d), where d dimension looks like (emb(s(r)), hat{x}, hat{y}, hat{z})
            The is the embedding of the env_vectors
        index : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        direct_vec = x[:, -3:]
        x = x[:,:-3].unsqueeze(-1) * direct_vec.unsqueeze(1) # [N_env, D, 3]
        return self.reduce(x, index, reduce="mean", dim=0) # [N_atom, D, 3] following the orders of atom index.


class _NODE_EMB(MessagePassing):
    def __init__(
            self, 
            rc:Union[float, torch.Tensor],
            p:Union[int, torch.LongTensor],
            n_axis: Union[int, torch.LongTensor, None]=None,
            n_basis: Union[int, torch.LongTensor, None]=None,
            n_sqrt_radial: Union[int, torch.LongTensor, None]=None,
            n_radial: Union[int, torch.LongTensor, None]=None,
            aggr: SE2Aggregation=SE2Aggregation(),
            radial_net: dict={},
            n_atom: int=1,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"), **kwargs):
        
        super(_NODE_EMB, self).__init__(aggr=aggr, **kwargs)

        if isinstance(device, str):
            device = torch.device(device)
        if isinstance(dtype, str):
            dtype = dtype_dict[dtype]

        if n_axis == None:
            self.n_axis = n_sqrt_radial
        else:
            self.n_axis = n_axis

        radial_net["config"] = get_neuron_config([2*n_atom+n_basis]+radial_net["neurons"]+[n_radial])
        self.mlp_radial = FFN(**radial_net, device=device, dtype=dtype)
        radial_net["config"] = get_neuron_config([2*n_atom+n_basis]+radial_net["neurons"]+[n_sqrt_radial])
        self.mlp_sqrt_radial = FFN(**radial_net, device=device, dtype=dtype)
        self.mlp_emb = Linear(n_radial, self.n_axis*n_sqrt_radial, device=device, dtype=dtype)
        if isinstance(rc, float):
            self.rc = torch.tensor(rc, dtype=dtype, device=device)
        else:
            self.rc = rc
        
        self.p = p

        self.n_axis = n_axis
        self.device = device
        self.dtype = dtype
        
        self.n_out = self.n_axis * n_sqrt_radial

        self.bessel = BesselBasis(r_max=rc, num_basis=n_basis, trainable=True)
        self.node_layer_norm = torch.nn.LayerNorm(self.n_out, elementwise_affine=True)
        self.edge_layer_norm = torch.nn.LayerNorm(n_radial, elementwise_affine=True)

    def forward(self, env_vectors, atom_attr, env_index, edge_index, env_length, edge_length):
        n_env = env_index.shape[1]
        n_edge = edge_index.shape[1]
        env_attr = atom_attr[env_index].transpose(1,0).reshape(n_env,-1)
        edge_attr = atom_attr[edge_index].transpose(1,0).reshape(n_edge,-1)
        ud_env = polynomial_cutoff(x=env_length, r_max=self.rc, p=self.p).reshape(-1, 1)
        ud_edge = polynomial_cutoff(x=edge_length, r_max=self.rc, p=self.p).reshape(-1, 1)
        
        env_sqrt_radial = self.mlp_sqrt_radial(torch.cat([env_attr, ud_env * self.bessel(env_length)], dim=-1)) * ud_env
        
        env_radial = self.edge_layer_norm(self.mlp_radial(torch.cat([env_attr, ud_env * self.bessel(env_length)], dim=-1))) * ud_env
        edge_radial = self.edge_layer_norm(self.mlp_radial(torch.cat([edge_attr, ud_edge * self.bessel(edge_length)], dim=-1))) * ud_edge
        
        node_emb = self.propagate(env_index, env_vectors=env_vectors, env_length=env_length, ud=ud_env, env_sqrt_radial=env_sqrt_radial) # [N_atom, D, 3]
        env_hidden = self.mlp_emb(env_radial) * (node_emb[env_index[1]]+node_emb[env_index[0]]) * 0.5
        edge_hidden = self.mlp_emb(edge_radial) * (node_emb[edge_index[1]]+node_emb[edge_index[0]]) * 0.5
        
        return env_radial, edge_radial, node_emb, env_hidden, edge_hidden

    def message(self, env_vectors, env_length, env_sqrt_radial, ud):
        snorm = env_length.unsqueeze(-1) * ud
        env_vectors = snorm * env_vectors / env_length.unsqueeze(-1)
        return torch.cat([env_sqrt_radial, env_vectors], dim=-1) # [N_env, D_emb + 3]

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
        out = torch.bmm(aggr_out, aggr_out.transpose(1, 2))[:,:,:self.n_axis].flatten(start_dim=1, end_dim=2)
        
        
        return self.node_layer_norm(out) # [N, D*D]
    

class BaselineLayer(MessagePassing):
    def __init__(
            self, 
            rc:Union[float, torch.Tensor],
            p:Union[int, torch.LongTensor],
            n_radial: int,
            n_sqrt_radial: int,
            n_axis: int,
            n_atom: int,
            n_hidden: int,
            radial_net: dict={},
            hidden_net: dict={},
            aggr="mean",
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"), **kwargs):
        
        super(BaselineLayer, self).__init__(aggr=aggr, **kwargs)

        if isinstance(device, str):
            device = torch.device(device)
        if isinstance(dtype, str):
            dtype = dtype_dict[dtype]

        if isinstance(rc, float):
            self.rc = torch.tensor(rc, dtype=dtype, device=device)
        else:
            self.rc = rc
        
        self.p = p

        self.mlp_emb = Linear(n_radial, n_axis*n_sqrt_radial, device=device, dtype=dtype)
        hidden_net["config"] = get_neuron_config([n_axis*n_sqrt_radial+n_hidden]+hidden_net["neurons"]+[n_hidden])
        self.mlp_hid = FFN(**hidden_net, device=device, dtype=dtype)
        radial_net["config"] = get_neuron_config([n_radial+n_hidden]+radial_net["neurons"]+[n_radial])
        self.mlp_radial = ResNet(**radial_net, dtype=dtype, device=device)

        self.node_layer_norm = torch.nn.LayerNorm(n_axis*n_sqrt_radial, elementwise_affine=True)
        self.edge_layer_norm = torch.nn.LayerNorm(n_radial, elementwise_affine=True)

        self.device = device
        self.dtype = dtype

    def forward(self, env_length, edge_length, edge_index, env_index, env_radial, edge_radial, node_emb, env_hidden, edge_hidden):
        # n_env = env_index.shape[1]
        # n_edge = edge_index.shape[1]
        # env_attr = atom_attr[env_index].transpose(1,0).reshape(n_env,-1)
        # edge_attr = atom_attr[edge_index].transpose(1,0).reshape(n_edge,-1)
        
        env_weight = self.mlp_emb(env_radial)
        # node_emb can descripe the node very well
        node_emb = 0.89442719 * node_emb + 0.4472 * self.propagate(env_index, node_emb=node_emb[env_index[1]], env_weight=env_weight) # [N_atom, D, 3]
        # import matplotlib.pyplot as plt
        # fig = plt.figure(figsize=(15,4))
        # plt.plot(node_emb.detach().T)
        # plt.title("node_emb")
        # plt.show()

        # env_hidden 长得太像了
        env_hidden = self.mlp_hid(torch.cat([node_emb[env_index[0]], env_hidden], dim=-1))
        edge_hidden = self.mlp_hid(torch.cat([node_emb[edge_index[0]], edge_hidden], dim=-1))
        # node_emb = _node_emb + node_emb

        # import matplotlib.pyplot as plt
        # fig = plt.figure(figsize=(15,4))
        # plt.plot(edge_hidden.detach().T)
        # plt.title("edge_hidden")
        # plt.show()

        ud_env = polynomial_cutoff(x=env_length, r_max=self.rc, p=self.p).reshape(-1, 1)
        ud_edge = polynomial_cutoff(x=edge_length, r_max=self.rc, p=self.p).reshape(-1, 1)
        env_radial = 0.89442719 * env_radial + 0.4472 * ud_env * self.edge_layer_norm(self.mlp_radial(torch.cat([env_radial, env_hidden], dim=-1)))
        edge_radial = 0.89442719 * edge_radial + 0.4472 * ud_edge * self.edge_layer_norm(self.mlp_radial(torch.cat([edge_radial, edge_hidden], dim=-1)))

        return env_radial, env_hidden, edge_radial, edge_hidden, node_emb

    def message(self, node_emb, env_weight):
        
        return env_weight * node_emb

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