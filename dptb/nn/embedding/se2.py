from torch_geometric.nn import MessagePassing
from torch_geometric.nn import Aggregation
import torch
from typing import Optional, Tuple, Union
from dptb.data import AtomicDataDict
from dptb.nn.embedding.emb import Embedding
from ..base import ResNet
from dptb.utils.tools import get_neuron_config
from ..type_encode.one_hot import OneHotAtomEncoding

@Embedding.register("se2")
class SE2Descriptor(torch.nn.Module):
    def __init__(
            self, 
            rs: Union[float, torch.Tensor], 
            rc:Union[float, torch.Tensor],
            n_axis: Union[int, torch.LongTensor, None]=None,
            n_atom: int=1,
            radial_embedding: dict={},
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu")
            ) -> None:
        
        super(SE2Descriptor, self).__init__()
        self.onehot = OneHotAtomEncoding(num_types=n_atom, set_features=False)
        self.descriptor = _SE2Descriptor(rs=rs, rc=rc, n_atom=n_atom, radial_embedding=radial_embedding, n_axis=n_axis, dtype=dtype, device=device)

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

        data[AtomicDataDict.NODE_FEATURES_KEY], data[AtomicDataDict.EDGE_FEATURES_KEY] = self.descriptor(
            data[AtomicDataDict.ENV_VECTORS_KEY],
            data[AtomicDataDict.NODE_ATTRS_KEY],
            data[AtomicDataDict.ENV_INDEX_KEY], 
            data[AtomicDataDict.EDGE_INDEX_KEY]
            )
        
        return data
    
    @property
    def out_edge_dim(self):
        return self.descriptor.n_out

    @property
    def out_note_dim(self):
        return self.descriptor.n_out



    
class SE2Aggregation(Aggregation):
    def forward(self, x: torch.Tensor, env_index: torch.LongTensor):
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
        return self.reduce(x, env_index, reduce="mean", dim=0) # [N_atom, D, 3] following the orders of atom index.


class _SE2Descriptor(MessagePassing):
    def __init__(
            self, 
            rs: Union[float, torch.Tensor], 
            rc:Union[float, torch.Tensor],
            n_axis: Union[int, torch.LongTensor, None]=None,
            aggr: SE2Aggregation=SE2Aggregation(),
            radial_embedding: dict={},
            n_atom: int=1,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"), **kwargs):
        
        super(_SE2Descriptor, self).__init__(aggr=aggr, **kwargs)


        radial_embedding["config"] = get_neuron_config([2*n_atom+1]+radial_embedding["neurons"])
        self.embedding_net = ResNet(**radial_embedding, device=device, dtype=dtype)
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
            self.n_axis = radial_embedding["neurons"][-1]
        self.n_out = self.n_axis * radial_embedding["neurons"][-1]

    def forward(self, env_vectors, atom_attr, env_index, edge_index):
        n_env = env_vectors.shape[1]
        out_node = self.propagate(env_index, env_vectors=env_vectors, env_attr=atom_attr[env_index.T.flatten()].reshape(n_env,2,-1).squeeze(1)) # [N_atom, D, 3]
        out_edge = self.edge_updater(out_node, edge_index) # [N_edge, D*D]

        return out_node, out_edge

    def message(self, env_vectors, env_attr):
        snorm = self.smooth(env_vectors.norm(-1, keepdim=True), self.rs, self.rc)
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

        return torch.bmm(aggr_out, aggr_out.transpose(1, 2))[:,:,:self.n_axis].flatten(start_dim=1, end_dim=2) # [N, D*D]
    
    def edge_update(self, node_descriptor, edge_index):
        
        return node_descriptor[edge_index[0]] + node_descriptor[edge_index[1]] # [N_edge, D*D]
    
    def smooth(self, r: torch.Tensor, rs: torch.Tensor, rc: torch.Tensor):
        if r < rs:
            return 1/r
        elif rs <= r and r < rc:
            x = (r - rc) / (rs - rc)
            return 1/r * (x**3 * (10 + x * (-15 + 6 * x)) + 1)
        else:
            return torch.zeros_like(r, dtype=r.dtype, device=r.device)

