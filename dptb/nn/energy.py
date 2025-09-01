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
import logging
log = logging.getLogger(__name__)

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

        self.h2k = HR2HK(
            idp=idp, 
            edge_field=h_edge_field, 
            node_field=h_node_field, 
            out_field=h_out_field, 
            dtype=dtype, 
            device=device,
            )
        
        if s_edge_field is not None:
            self.s2k = HR2HK(
                idp=idp, 
                overlap=True, 
                edge_field=s_edge_field, 
                node_field=s_node_field, 
                out_field=s_out_field, 
                dtype=dtype, 
                device=device,
                )
            
            self.overlap = True
        else:
            self.overlap = False

        self.out_field = out_field
        self.h_out_field = h_out_field
        self.s_out_field = s_out_field


    def forward(self, 
                data: AtomicDataDict.Type, 
                nk: Optional[int]=None,
                eig_solver: str='torch') -> AtomicDataDict.Type:

        if eig_solver is None:
            eig_solver = 'torch'
            log.warning("eig_solver is not set, using default 'torch'.")
        if eig_solver not in ['torch', 'numpy']:
            log.error(f"eig_solver should be 'torch' or 'numpy', but got {eig_solver}.")
            raise ValueError        

        kpoints = data[AtomicDataDict.KPOINT_KEY]
        if kpoints.is_nested:
            nested = True
            assert kpoints.size(0) == 1
            kpoints = kpoints[0]
        else:
            nested = False
        num_k = kpoints.shape[0]
        eigvals = []
        if nk is None:
            nk = num_k
        for i in range(int(np.ceil(num_k / nk))):
            data[AtomicDataDict.KPOINT_KEY] = kpoints[i*nk:(i+1)*nk]
            data = self.h2k(data)
            if self.overlap:
                data = self.s2k(data)
                if eig_solver == 'torch':
                    chklowt = torch.linalg.cholesky(data[self.s_out_field])
                    chklowtinv = torch.linalg.inv(chklowt)
                    data[self.h_out_field] = (chklowtinv @ data[self.h_out_field] @ torch.transpose(chklowtinv,dim0=1,dim1=2).conj())
                elif eig_solver == 'numpy':
                    chklowt = np.linalg.cholesky(data[self.s_out_field].detach().numpy())
                    chklowtinv = np.linalg.inv(chklowt)
                    data[self.h_out_field] = (chklowtinv @ data[self.h_out_field].detach().numpy() @ np.transpose(chklowtinv,(0,2,1)).conj())
            else:
                data[self.h_out_field] = data[self.h_out_field]
            
            if eig_solver == 'torch':
                eigvals.append(torch.linalg.eigvalsh(data[self.h_out_field]))
            elif eig_solver == 'numpy':
                eigvals.append(torch.from_numpy(np.linalg.eigvalsh(a=data[self.h_out_field])))

        data[self.out_field] = torch.nested.as_nested_tensor([torch.cat(eigvals, dim=0)])
        if nested:
            data[AtomicDataDict.KPOINT_KEY] = torch.nested.as_nested_tensor([kpoints])
        else:
            data[AtomicDataDict.KPOINT_KEY] = kpoints

        return data
    
class Eigh(nn.Module):
    def __init__(
            self,
            idp: Union[OrbitalMapper, None]=None,
            h_edge_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
            h_node_field: str = AtomicDataDict.NODE_FEATURES_KEY,
            h_out_field: str = AtomicDataDict.HAMILTONIAN_KEY,
            eigval_field: str = AtomicDataDict.ENERGY_EIGENVALUE_KEY,
            eigvec_field: str = AtomicDataDict.EIGENVECTOR_KEY,
            s_edge_field: str = None,
            s_node_field: str = None,
            s_out_field: str = None,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu")):
        super(Eigh, self).__init__()

        self.h2k = HR2HK(
            idp=idp, 
            edge_field=h_edge_field, 
            node_field=h_node_field, 
            out_field=h_out_field, 
            dtype=dtype, 
            device=device,
            )
        
        if s_edge_field is not None:
            self.s2k = HR2HK(
                idp=idp, 
                overlap=True, 
                edge_field=s_edge_field, 
                node_field=s_node_field, 
                out_field=s_out_field, 
                dtype=dtype, 
                device=device,
                )
            
            self.overlap = True
        else:
            self.overlap = False

        self.eigval_field = eigval_field
        self.eigvec_field = eigvec_field
        self.h_out_field = h_out_field
        self.s_out_field = s_out_field


    def forward(self, data: AtomicDataDict.Type, nk: Optional[int]=None) -> AtomicDataDict.Type:
        kpoints = data[AtomicDataDict.KPOINT_KEY]
        if kpoints.is_nested:
            nested = True
            assert kpoints.size(0) == 1
            kpoints = kpoints[0]
        else:
            nested = False
        num_k = kpoints.shape[0]
        eigvals = []
        eigvecs = []
        if nk is None:
            nk = num_k
        for i in range(int(np.ceil(num_k / nk))):
            data[AtomicDataDict.KPOINT_KEY] = kpoints[i*nk:(i+1)*nk]
            data = self.h2k(data)
            if self.overlap:
                data = self.s2k(data)
                chklowt = torch.linalg.cholesky(data[self.s_out_field])
                chklowtinv = torch.linalg.inv(chklowt)
                data[self.h_out_field] = (
                    chklowtinv @ data[self.h_out_field] @ torch.transpose(chklowtinv,dim0=1,dim1=2).conj()
                )

            eigval, eigvec = torch.linalg.eigh(data[self.h_out_field])
            if self.overlap:
                eigvec = torch.transpose(
                    torch.transpose(chklowtinv,dim0=1,dim1=2).conj() @ eigvec,
                    dim0=1,dim1=2)

            eigvecs.append(eigvec)
            eigvals.append(eigval)

        data[self.eigval_field] = torch.nested.as_nested_tensor([torch.cat(eigvals, dim=0)])
        data[self.eigvec_field] = torch.cat(eigvecs, dim=0)

        if nested:
            data[AtomicDataDict.KPOINT_KEY] = torch.nested.as_nested_tensor([kpoints])
        else:
            data[AtomicDataDict.KPOINT_KEY] = kpoints

        return data