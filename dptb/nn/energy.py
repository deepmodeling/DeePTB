"""
The quantities module of GNN, with AtomicDataDict.Type as input and output the same class.

This version:
  - Keeps SOC (Full H) + scalar overlap (S is NxN) compatibility by expanding S -> blockdiag(S,S) implicitly.
  - Adds ill-conditioned overlap fallback via ill_threshold projection (migrated from the second script).
"""

import os  # kept for compatibility with your previous iterations (even if unused)

import torch
import numpy as np
import torch.nn as nn
from dptb.nn.hr2hk import HR2HK
from typing import Union, Optional
from dptb.data.transforms import OrbitalMapper
from dptb.data import AtomicDataDict


def _blockdiag_dup(mat: torch.Tensor) -> torch.Tensor:
    """
    Build block diagonal duplication:
      mat: [B, N, N] -> out: [B, 2N, 2N] = [[mat,0],[0,mat]]
    """
    zeros = torch.zeros_like(mat)
    row1 = torch.cat([mat, zeros], dim=-1)
    row2 = torch.cat([zeros, mat], dim=-1)
    return torch.cat([row1, row2], dim=-2)


def _blockdiag_dup_vecs(vecs: torch.Tensor) -> torch.Tensor:
    """
    Build block diagonal duplication for eigenvectors:
      vecs: [B, N, N] -> out: [B, 2N, 2N] = [[vecs,0],[0,vecs]]
    """
    B, N, _ = vecs.shape
    out = torch.zeros((B, 2 * N, 2 * N), dtype=vecs.dtype, device=vecs.device)
    out[:, :N, :N] = vecs
    out[:, N:, N:] = vecs
    return out


class Eigenvalues(nn.Module):
    def __init__(
            self,
            idp: Union[OrbitalMapper, None] = None,
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

    def forward(
            self,
            data: AtomicDataDict.Type,
            nk: Optional[int] = None,
            ill_threshold: Optional[float] = 2e-3
    ) -> AtomicDataDict.Type:
        """
        Compute eigenvalues along k-points.

        ill_threshold:
          - None: legacy behavior (pure Cholesky reduction of generalized eigenproblem)
          - float: robust projection for ill-conditioned overlap S
        """
        kpoints = data[AtomicDataDict.KPOINT_KEY]
        if kpoints.is_nested:
            nested = True
            assert kpoints.size(0) == 1
            kpoints0 = kpoints[0]
        else:
            nested = False
            kpoints0 = kpoints

        num_k = kpoints0.shape[0]
        eigvals_chunks = []
        if nk is None:
            nk = num_k

        for i in range(int(np.ceil(num_k / nk))):
            data[AtomicDataDict.KPOINT_KEY] = kpoints0[i * nk:(i + 1) * nk]
            data = self.h2k(data)

            H_k = data[self.h_out_field]  # [B, dimH, dimH]

            if not self.overlap:
                batch_eigvals = torch.linalg.eigvalsh(H_k)
                eigvals_chunks.append(batch_eigvals)
                continue

            # overlap branch
            data = self.s2k(data)
            S_k = data[self.s_out_field]  # [B, dimS, dimS]

            # SOC mismatch detection: H is 2N but S is N (scalar overlap)
            soc_mismatch = (H_k.shape[-1] == 2 * S_k.shape[-1])

            # print(ill_threshold)

            # --------
            # Case A: no ill fallback (legacy cholesky)
            # --------
            if ill_threshold is None:
                L = torch.linalg.cholesky(S_k)
                L_inv = torch.linalg.inv(L)  # [B,N,N]

                if soc_mismatch:
                    # Expand L_inv -> blockdiag(L_inv, L_inv), then H' = Linv_big H Linv_big^H
                    L_inv_big = _blockdiag_dup(L_inv)
                    H_k_transformed = (L_inv_big @ H_k @ L_inv_big.mH)
                else:
                    H_k_transformed = (L_inv @ H_k @ L_inv.mH)

                batch_eigvals = torch.linalg.eigvalsh(H_k_transformed)
                eigvals_chunks.append(batch_eigvals)
                continue

            # --------
            # Case B: ill-conditioned overlap fallback (projection)
            # --------
            # Eigen-decompose S in its native dimension (N×N). For SOC mismatch, we later duplicate to 2N implicitly.
            egval_S, egvec_S = torch.linalg.eigh(S_k)  # egval_S:[B,N], egvec_S:[B,N,N]
            processed_eigvals_list = []

            if soc_mismatch:
                num_orbitals = H_k.shape[-1]  # 2N
                N = S_k.shape[-1]
            else:
                num_orbitals = H_k.shape[-1]  # N
                N = S_k.shape[-1]

            B = S_k.shape[0]
            for k_idx in range(B):
                # healthy indices in S-eigen space (size N)
                healthy_mask = egval_S[k_idx] > ill_threshold
                n_healthy = int(healthy_mask.sum().item())

                if n_healthy == 0:
                    # Extreme case: everything ill -> return all padded
                    egval = torch.full(
                        (num_orbitals,),
                        1e4,
                        dtype=H_k.dtype.real_dtype if H_k.is_complex() else H_k.dtype,
                        device=H_k.device,
                    )
                    processed_eigvals_list.append(egval)
                    continue

                if healthy_mask.all():
                    # well-conditioned -> solve full generalized eig (with SOC mismatch handled)
                    S_i = S_k[k_idx]
                    H_i = H_k[k_idx]

                    L = torch.linalg.cholesky(S_i)
                    L_inv = torch.linalg.inv(L)

                    if soc_mismatch:
                        L_inv_big = _blockdiag_dup(L_inv)
                        H_transformed = L_inv_big @ H_i @ L_inv_big.mH
                    else:
                        H_transformed = L_inv @ H_i @ L_inv.mH

                    egval = torch.linalg.eigvalsh(H_transformed)
                    processed_eigvals_list.append(egval)
                    continue

                # ill-conditioned -> project
                U = egvec_S[k_idx]  # [N,N]
                evals = egval_S[k_idx]  # [N]

                U_sel = U[:, healthy_mask]  # [N, M]
                eval_sel = evals[healthy_mask]  # [M]

                if soc_mismatch:
                    # Build V = blockdiag(U_sel, U_sel): [2N, 2M]
                    # and S_proj = diag([eval_sel, eval_sel])
                    V = torch.zeros((2 * N, 2 * n_healthy), dtype=U.dtype, device=U.device)
                    V[:N, :n_healthy] = U_sel
                    V[N:, n_healthy:] = U_sel

                    # Project H: [2M,2M]
                    H_proj = V.mH @ H_k[k_idx] @ V

                    # S_proj is diagonal in this eigenbasis
                    eval_dup = torch.cat([eval_sel, eval_sel], dim=0)  # [2M]
                    S_proj = torch.diag(eval_dup).to(dtype=H_proj.dtype, device=H_proj.device)

                    L = torch.linalg.cholesky(S_proj)
                    L_inv = torch.linalg.inv(L)
                    H_transformed = L_inv @ H_proj @ L_inv.mH
                    egval_proj = torch.linalg.eigvalsh(H_transformed)

                    # pad to 2N
                    num_projected_out = num_orbitals - egval_proj.shape[0]
                    if num_projected_out > 0:
                        padding = torch.full(
                            (num_projected_out,),
                            1e4,
                            dtype=egval_proj.dtype,
                            device=egval_proj.device
                        )
                        egval = torch.cat([egval_proj, padding], dim=0)
                    else:
                        egval = egval_proj

                    processed_eigvals_list.append(egval)
                else:
                    # Non-SOC: V = U_sel: [N,M], S_proj = diag(eval_sel)
                    V = U_sel  # [N,M]
                    H_proj = V.mH @ H_k[k_idx] @ V
                    S_proj = torch.diag(eval_sel).to(dtype=H_proj.dtype, device=H_proj.device)

                    L = torch.linalg.cholesky(S_proj)
                    L_inv = torch.linalg.inv(L)
                    H_transformed = L_inv @ H_proj @ L_inv.mH
                    egval_proj = torch.linalg.eigvalsh(H_transformed)

                    # pad to N
                    num_projected_out = num_orbitals - egval_proj.shape[0]
                    if num_projected_out > 0:
                        padding = torch.full(
                            (num_projected_out,),
                            1e4,
                            dtype=egval_proj.dtype,
                            device=egval_proj.device
                        )
                        egval = torch.cat([egval_proj, padding], dim=0)
                    else:
                        egval = egval_proj

                    processed_eigvals_list.append(egval)

            batch_eigvals = torch.stack(processed_eigvals_list, dim=0)
            eigvals_chunks.append(batch_eigvals)

        # concat all chunks
        final_eigvals = torch.cat(eigvals_chunks, dim=0)

        # IMPORTANT:
        # Keep historical behavior used by ElecStruCal: store eigenvalues as nested tensor with one item.
        data[self.out_field] = torch.nested.as_nested_tensor([final_eigvals])

        # restore kpoints
        if nested:
            data[AtomicDataDict.KPOINT_KEY] = torch.nested.as_nested_tensor([kpoints0])
        else:
            data[AtomicDataDict.KPOINT_KEY] = kpoints0

        return data


class Eigh(nn.Module):
    """
    Keep your first-script SOC scalar-overlap (S NxN, H 2N×2N) compatibility.
    Note: ill_threshold projection is NOT migrated here (same as your second script), because it would require
          careful eigenvector padding / reconstruction semantics. If you want, I can extend Eigh similarly.
    """
    def __init__(
            self,
            idp: Union[OrbitalMapper, None] = None,
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

    def forward(self, data: AtomicDataDict.Type, nk: Optional[int] = None) -> AtomicDataDict.Type:
        kpoints = data[AtomicDataDict.KPOINT_KEY]
        if kpoints.is_nested:
            nested = True
            assert kpoints.size(0) == 1
            kpoints0 = kpoints[0]
        else:
            nested = False
            kpoints0 = kpoints

        num_k = kpoints0.shape[0]
        eigvals = []
        eigvecs = []
        if nk is None:
            nk = num_k

        for i in range(int(np.ceil(num_k / nk))):
            data[AtomicDataDict.KPOINT_KEY] = kpoints0[i * nk:(i + 1) * nk]
            data = self.h2k(data)

            chklowtinv_final = None

            if self.overlap:
                data = self.s2k(data)

                H = data[self.h_out_field]  # [B, dimH, dimH]
                S = data[self.s_out_field]  # [B, dimS, dimS]

                L = torch.linalg.cholesky(S)
                L_inv = torch.linalg.inv(L)  # [B, dimS, dimS]

                # SOC mismatch: H is 2N while S is N
                if H.shape[-1] == 2 * S.shape[-1]:
                    L_inv_big = _blockdiag_dup(L_inv)
                    chklowtinv_final = L_inv_big
                    data[self.h_out_field] = (L_inv_big @ H @ L_inv_big.mH)
                else:
                    chklowtinv_final = L_inv
                    data[self.h_out_field] = (L_inv @ H @ L_inv.mH)

            # eig
            eigval, eigvec = torch.linalg.eigh(data[self.h_out_field])

            # restore eigenvectors to AO basis if overlap
            if self.overlap:
                # x = L^{-H} y  (but we stored L_inv already)
                raw_eigvec = chklowtinv_final.mH @ eigvec
                eigvecs.append(raw_eigvec.mH)  # [B, nband, norb]
            else:
                eigvecs.append(eigvec.mH)

            eigvals.append(eigval)

        data[self.eigval_field] = torch.nested.as_nested_tensor([torch.cat(eigvals, dim=0)])
        data[self.eigvec_field] = torch.cat(eigvecs, dim=0)

        if nested:
            data[AtomicDataDict.KPOINT_KEY] = torch.nested.as_nested_tensor([kpoints0])
        else:
            data[AtomicDataDict.KPOINT_KEY] = kpoints0

        return data