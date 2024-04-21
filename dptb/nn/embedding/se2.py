from torch_geometric.nn import MessagePassing
from torch_geometric.nn import Aggregation
import torch
from typing import Optional, Tuple, Union
from dptb.data import AtomicDataDict
from dptb.nn.embedding.emb import Embedding
from ..base import ResNet
from dptb.utils.constants import dtype_dict
from ..type_encode.one_hot import OneHotAtomEncoding

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

@Embedding.register("se2")
class SE2Descriptor(torch.nn.Module):
    def __init__(
            self, 
            rs: Union[float, torch.Tensor], 
            rc:Union[float, torch.Tensor],
            n_axis: Union[int, torch.LongTensor, None]=None,
            n_atom: int=1,
            radial_net: dict={},
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
            **kwargs,
            ) -> None:
        """
        a demo input
        se2_config = {
                    "rs": 3.0, 
                    "rc": 4.0,
                    "n_axis": 4,
                    "n_atom": 2,
                    "radial_embedding": {
                        "neurons": [10,20,30],
                        "activation": "tanh",
                        "if_batch_normalized": False
                    },
                    "dtype": "float32",
                    "device": "cpu"
                }
        """
        
        super(SE2Descriptor, self).__init__()
        self.onehot = OneHotAtomEncoding(num_types=n_atom, set_features=False)
        self.descriptor = _SE2Descriptor(rs=rs, rc=rc, n_atom=n_atom, radial_net=radial_net, n_axis=n_axis, dtype=dtype, device=device)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """_summary_

        Parameters
        ----------
        data : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        data = self.onehot(data)
        data = AtomicDataDict.with_env_vectors(data, with_lengths=True)
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)

        data[AtomicDataDict.NODE_FEATURES_KEY], data[AtomicDataDict.EDGE_FEATURES_KEY] = self.descriptor(
            data[AtomicDataDict.ENV_VECTORS_KEY],
            data[AtomicDataDict.NODE_ATTRS_KEY],
            data[AtomicDataDict.ENV_INDEX_KEY], 
            data[AtomicDataDict.EDGE_INDEX_KEY],
            data[AtomicDataDict.EDGE_LENGTH_KEY],
            )
        
        return data
    
    @property
    def out_edge_dim(self):
        return self.descriptor.n_out + 1

    @property
    def out_node_dim(self):
        return self.descriptor.n_out



    
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
        direct_vec = x[:, -4:]
        x = x[:,:-4].unsqueeze(-1) * direct_vec.unsqueeze(1) # [N_env, D, 4]
        return self.reduce(x, index, reduce="mean", dim=0) # [N_atom, D, 4] following the orders of atom index.


class _SE2Descriptor(MessagePassing):
    def __init__(
            self, 
            rs: Union[float, torch.Tensor], 
            rc:Union[float, torch.Tensor],
            n_axis: Union[int, torch.LongTensor, None]=None,
            aggr: SE2Aggregation=SE2Aggregation(),
            radial_net: dict={},
            n_atom: int=1,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"), **kwargs):
        
        super(_SE2Descriptor, self).__init__(aggr=aggr, **kwargs, flow="target_to_source")

        if isinstance(device, str):
            device = torch.device(device)
        if isinstance(dtype, str):
            dtype = dtype_dict[dtype]


        # radial_net["config"] = get_neuron_config([2*n_atom+1]+radial_net["neurons"])
        # self.embedding_net = ResNet(**radial_net, device=device, dtype=dtype)
        self.embedding_net = ResNet(
            **radial_net, 
            config=get_neuron_config([2*n_atom+1]+radial_net["neurons"]), 
            device=device, 
            dtype=dtype
            )
        
        if isinstance(rs, float):
            self.rs = torch.tensor(rs, dtype=dtype, device=device)
        else:
            self.rs = rs
        if isinstance(rc, float):
            self.rc = torch.tensor(rc, dtype=dtype, device=device)
        else:
            self.rc = rc

        assert len(self.rc.flatten()) == 1 and len(self.rs.flatten()) == 1
        assert self.rs < self.rc
        self.n_axis = n_axis
        self.device = device
        self.dtype = dtype
        if n_axis == None:
            self.n_axis = radial_net["neurons"][-1]
        self.n_out = self.n_axis * radial_net["neurons"][-1]

    def forward(self, env_vectors, atom_attr, env_index, edge_index, edge_length):
        n_env = env_index.shape[1]
        env_attr = atom_attr[env_index].transpose(1,0).reshape(n_env,-1)
        out_node = self.propagate(env_index, env_vectors=env_vectors, env_attr=env_attr) # [N_atom, D, 3]
        out_edge = self.edge_updater(edge_index, node_descriptor=out_node, edge_length=edge_length) # [N_edge, D*D]

        return out_node, out_edge

    def message(self, env_vectors, env_attr):
        rij = env_vectors.norm(dim=-1, keepdim=True)
        snorm = self.smooth(rij, self.rs, self.rc)
        env_vectors = torch.cat([snorm, snorm * env_vectors / rij], dim=-1)
        return torch.cat([self.embedding_net(torch.cat([snorm, env_attr], dim=-1)), env_vectors], dim=-1) # [N_env, D_emb + 4]
      

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
        out = out / (out.norm(dim=1, keepdim=True))
        return out # [N, D*D]
    
    def edge_update(self, edge_index, node_descriptor, edge_length):
        return torch.cat([node_descriptor[edge_index[0]] + node_descriptor[edge_index[1]], 1/edge_length.reshape(-1,1)], dim=-1) # [N_edge, D*D]
    
    def smooth(self, r: torch.Tensor, rs: torch.Tensor, rc: torch.Tensor):
        assert rs<rc, f"rs={rs} should be smaller than rc={rc}"
        r_ = torch.zeros_like(r)
        r_[r<rs] = 1/r[r<rs]
        x = (r - rs) / (rc - rs)
        mid_mask = (rs<=r) * (r < rc)
        r_[mid_mask] = 1/r[mid_mask] * (x[mid_mask]**3 * (-10 + x[mid_mask] * (15 - 6 * x[mid_mask])) + 1)

        return r_
