from torch_geometric.nn import MessagePassing
from torch_geometric.nn import Aggregation
import torch
from typing import Optional, Tuple, Union
from dptb.data import AtomicDataDict
from dptb.nn.embedding.emb import Embedding
from ..base import ResNet, FFN
from dptb.utils.constants import dtype_dict
from ..type_encode.one_hot import OneHotAtomEncoding
from ..cutoff import polynomial_cutoff

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

class SE2Aggregation(Aggregation):
    def forward(self, x: torch.Tensor, index: torch.LongTensor, **kwargs):
        """_summary_

        Parameters
        ----------
        x : tensor of size (N, d), where d dimension looks like (emb(s(r)), \hat{x}, \hat{y}, \hat{z})
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


class _SE2Descriptor(MessagePassing):
    def __init__(
            self, 
            rc: Union[float, torch.Tensor], 
            p: int,
            n_axis: Union[int, torch.LongTensor, None]=None,
            aggr: SE2Aggregation=SE2Aggregation(),
            radial_embedding: dict={},
            n_atom: int=1,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"), **kwargs):
        
        super(_SE2Descriptor, self).__init__(aggr=aggr, **kwargs)

        if isinstance(device, str):
            device = torch.device(device)
        if isinstance(dtype, str):
            dtype = dtype_dict[dtype]


        radial_embedding["config"] = get_neuron_config([2*n_atom+radial_embedding["n_basis"]]+radial_embedding["neurons"])
        
        self.env_embedding = FFN(**radial_embedding, device=device, dtype=dtype)
        if isinstance(rc, float):
            self.rc = torch.tensor(rc, dtype=dtype, device=device)
        else:
            self.rc = rc
        self.p = p

        assert len(self.rc.flatten()) == 1
        self.device = device
        self.dtype = dtype

        self.n_out = None

    def forward(self, env_vectors, env_length, atom_attr, env_index, edge_index, edge_length):
        n_env = env_index.shape[1]
        # initilize the node and env embeddings
        env_attr = atom_attr[env_index].transpose(1,0).reshape(n_env,-1)
        out_node = self.propagate(env_index, env_vectors=env_vectors, env_attr=env_attr) # [N_atom, D, 3]
        out_edge = self.edge_updater(edge_index, node_descriptor=out_node, edge_length=edge_length) # [N_edge, D*D]

        return out_node, out_edge

    def message(self, env_vectors, env_attr):
        rij = env_vectors.norm(dim=-1, keepdim=True)
        snorm = self.smooth(rij, self.rs, self.rc)
        env_vectors = snorm * env_vectors / rij
        return torch.cat([self.embedding_net(torch.cat([snorm, env_attr], dim=-1)), env_vectors], dim=-1) # [N_env, D_emb + 3]

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
        out = out - out.mean(1, keepdim=True)
        out = out / out.norm(dim=1, keepdim=True)
        return out # [N, D*D]
    
    def edge_update(self, edge_index, node_descriptor, edge_length):
        return torch.cat([node_descriptor[edge_index[0]] + node_descriptor[edge_index[1]], 1/edge_length.reshape(-1,1)], dim=-1) # [N_edge, D*D]
    
    def smooth(self, r: torch.Tensor, rs: torch.Tensor, rc: torch.Tensor):
        r_ = torch.zeros_like(r)
        r_[r<rs] = 1/r[r<rs]
        x = (r - rc) / (rs - rc)
        mid_mask = (rs<=r) * (r < rc)
        r_[mid_mask] = 1/r[mid_mask] * (x[mid_mask]**3 * (10 + x[mid_mask] * (-15 + 6 * x[mid_mask])) + 1)

        return r_
    
