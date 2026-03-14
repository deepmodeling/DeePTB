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
                eig_solver: str='torch',
                ill_threshold: Optional[float]=1e-5) -> AtomicDataDict.Type:

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
            h_transformed_np = None

            batch_eigvals_torch = None
            batch_eigvals_np = None

            if self.overlap:
                data = self.s2k(data)
                if eig_solver == 'torch':
                    if ill_threshold is None:
                        chklowt = torch.linalg.cholesky(data[self.s_out_field])
                        chklowtinv = torch.linalg.inv(chklowt)
                        data[self.h_out_field] = (chklowtinv @ data[self.h_out_field] @ torch.transpose(chklowtinv,dim0=1,dim1=2).conj())
                    else:
                        S_k = data[self.s_out_field]
                        H_k = data[self.h_out_field]
                        egval_S, egvec_S = torch.linalg.eigh(S_k)
                        B = S_k.shape[0]
                        num_orbitals = H_k.shape[-1]
                        real_dtype = torch.float32 if H_k.dtype in [torch.complex64, torch.float32] else torch.float64

                        processed_eigvals_list = []
                        for k_idx in range(B):
                            healthy_mask = egval_S[k_idx] > ill_threshold
                            n_healthy = int(healthy_mask.sum().item())

                            if n_healthy == 0:
                                egval = torch.full((num_orbitals,), 1e4, dtype=real_dtype, device=H_k.device)
                                processed_eigvals_list.append(egval)
                                continue

                            if healthy_mask.all():
                                L = torch.linalg.cholesky(S_k[k_idx])
                                L_inv = torch.linalg.inv(L)
                                H_transformed = L_inv @ H_k[k_idx] @ L_inv.conj().T
                                egval = torch.linalg.eigvalsh(H_transformed)
                                processed_eigvals_list.append(egval)
                                continue

                            U_sel = egvec_S[k_idx, :, healthy_mask]
                            eval_sel = egval_S[k_idx, healthy_mask]

                            H_proj = U_sel.conj().T @ H_k[k_idx] @ U_sel
                            S_proj = torch.diag(eval_sel).to(dtype=H_proj.dtype, device=H_proj.device)

                            L = torch.linalg.cholesky(S_proj)
                            L_inv = torch.linalg.inv(L)
                            H_transformed = L_inv @ H_proj @ L_inv.conj().T
                            egval_proj = torch.linalg.eigvalsh(H_transformed)

                            num_projected_out = num_orbitals - egval_proj.shape[0]
                            if num_projected_out > 0:
                                padding = torch.full((num_projected_out,), 1e4, dtype=egval_proj.dtype, device=egval_proj.device)
                                egval = torch.cat([egval_proj, padding], dim=0)
                            else:
                                egval = egval_proj

                            processed_eigvals_list.append(egval)
                        batch_eigvals_torch = torch.stack(processed_eigvals_list, dim=0)

                elif eig_solver == 'numpy':
                    if ill_threshold is None:
                        s_np = data[self.s_out_field].detach().cpu().numpy()
                        h_np = data[self.h_out_field].detach().cpu().numpy()
                        chklowt = np.linalg.cholesky(s_np)
                        chklowtinv = np.linalg.inv(chklowt)
                        h_transformed_np = chklowtinv @ h_np @ np.transpose(chklowtinv,(0,2,1)).conj()
                    else:
                        s_np = data[self.s_out_field].detach().cpu().numpy()
                        h_np = data[self.h_out_field].detach().cpu().numpy()
                        egval_S, egvec_S = np.linalg.eigh(s_np)
                        B = s_np.shape[0]
                        num_orbitals = h_np.shape[-1]
                        real_dtype = np.float32 if h_np.dtype in [np.complex64, np.float32] else np.float64

                        processed_eigvals_list = []
                        for k_idx in range(B):
                            healthy_mask = egval_S[k_idx] > ill_threshold
                            n_healthy = int(healthy_mask.sum())

                            if n_healthy == 0:
                                egval = np.full((num_orbitals,), 1e4, dtype=real_dtype)
                                processed_eigvals_list.append(egval)
                                continue

                            if healthy_mask.all():
                                L = np.linalg.cholesky(s_np[k_idx])
                                L_inv = np.linalg.inv(L)
                                H_transformed = L_inv @ h_np[k_idx] @ L_inv.conj().T
                                egval = np.linalg.eigvalsh(H_transformed)
                                processed_eigvals_list.append(egval)
                                continue

                            U_sel = egvec_S[k_idx, :, healthy_mask]
                            eval_sel = egval_S[k_idx, healthy_mask]

                            H_proj = U_sel.conj().T @ h_np[k_idx] @ U_sel
                            S_proj = np.diag(eval_sel).astype(H_proj.dtype)

                            L = np.linalg.cholesky(S_proj)
                            L_inv = np.linalg.inv(L)
                            H_transformed = L_inv @ H_proj @ L_inv.conj().T
                            egval_proj = np.linalg.eigvalsh(H_transformed)

                            num_projected_out = num_orbitals - egval_proj.shape[0]
                            if num_projected_out > 0:
                                padding = np.full((num_projected_out,), 1e4, dtype=egval_proj.dtype)
                                egval = np.concatenate([egval_proj, padding], axis=0)
                            else:
                                egval = egval_proj

                            processed_eigvals_list.append(egval)
                        batch_eigvals_np = np.stack(processed_eigvals_list, axis=0)

            if eig_solver == 'torch':
                if batch_eigvals_torch is not None:
                    eigvals.append(batch_eigvals_torch)
                else:
                    eigvals.append(torch.linalg.eigvalsh(data[self.h_out_field]))
            elif eig_solver == 'numpy':
                if batch_eigvals_np is not None:
                    eigvals_np = batch_eigvals_np
                else:
                    if h_transformed_np is None:
                        h_transformed_np = data[self.h_out_field].detach().cpu().numpy()
                    eigvals_np = np.linalg.eigvalsh(a=h_transformed_np)
                # Preserve dtype by converting to the Hamiltonian's original dtype
                eigvals.append(torch.from_numpy(eigvals_np).to(dtype=self.h2k.dtype, device=self.h2k.device))

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