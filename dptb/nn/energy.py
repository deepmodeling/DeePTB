"""
The quantities module of GNN, with AtomicDataDict.Type as input and output the same class. Unlike the other, this module can act on 
    one field and get features of an other field. E.p, the energy model should act on NODE_FEATURES or EDGE_FEATURES to get NODE or EDGE
    ENERGY. Then it will be summed up to graph level features TOTOL_ENERGY.
"""
import torch
import numpy as np
import torch.nn as nn
from dptb.nn.hr2hk import HR2HK
from typing import Union, Optional, Dict, List
from dptb.data.transforms import OrbitalMapper
from dptb.data import AtomicDataDict

class Eigenvalues(nn.Module):
    def __init__(
            self,
            idp: Union[OrbitalMapper, None]=None,
            h_edge_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
            h_node_field: str = AtomicDataDict.NODE_FEATURES_KEY,
            h_out_field: str = AtomicDataDict.HAMILTONIAN_KEY,
            out_field: str = AtomicDataDict.ENERGY_EIGENVALUE_KEY,
            s_edge_field: str = None,
            s_node_field: str = None,
            s_out_field: str = None,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu")):
        super(Eigenvalues, self).__init__()

        self.h2k = HR2HK(idp=idp, edge_field=h_edge_field, node_field=h_node_field, out_field=h_out_field, dtype=dtype, device=device)
        if s_edge_field is not None:
            self.s2k = HR2HK(idp=idp, overlap=True, edge_field=s_edge_field, node_field=s_node_field, out_field=s_out_field, dtype=dtype, device=device)
            self.overlap = True
        else:
            self.overlap = False
        self.out_field = out_field
        self.h_out_field = h_out_field
        self.s_out_field = s_out_field


    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = self.h2k(data)
        if self.overlap:
            data = self.s2k(data)
            chklowt = torch.linalg.cholesky(data[self.s_out_field])
            chklowtinv = torch.linalg.inv(chklowt)
            Heff = (chklowtinv @ data[self.h_out_field] @ torch.transpose(chklowtinv,dim0=1,dim1=2).conj())
        else:
            Heff = data[self.h_out_field]
        
        data[self.out_field] = torch.linalg.eigvalsh(Heff)

        return data
