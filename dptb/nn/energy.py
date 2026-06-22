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
try:
    from dptb.utils.pardiso_wrapper import PyPardisoSolver
    from dptb.utils.feast_wrapper import FeastSolver
    from scipy.sparse.linalg import eigsh, LinearOperator
except ImportError:
    PyPardisoSolver = None
    FeastSolver = None
    eigsh = None
    LinearOperator = None

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
                ill_threshold: Optional[float]=None,
                ill_pad_value: float=1e4) -> AtomicDataDict.Type:

        if eig_solver is None:
            eig_solver = 'torch'
            log.warning("eig_solver is not set, using default 'torch'.")
        if eig_solver not in ['torch', 'numpy']:
            msg = f"eig_solver should be 'torch' or 'numpy', but got {eig_solver}."
            log.error(msg)
            raise ValueError(msg)

        kpoints = data[AtomicDataDict.KPOINT_KEY]
        if kpoints.is_nested:
            nested = True
            assert kpoints.size(0) == 1
            kpoints = kpoints[0]
        else:
            nested = False
        num_k = kpoints.shape[0]
        eigvals = []
        eigval_masks = []
        if nk is None:
            nk = num_k

        for i in range(int(np.ceil(num_k / nk))):
            data[AtomicDataDict.KPOINT_KEY] = kpoints[i*nk:(i+1)*nk]
            data = self.h2k(data)
            h_transformed_np = None

            batch_eigvals_torch = None
            batch_eigvals_np = None
            batch_mask = None

            if self.overlap:
                data = self.s2k(data)
                if eig_solver == 'torch':
                    if ill_threshold is None:
                        chklowt = torch.linalg.cholesky(data[self.s_out_field])
                        chklowtinv = torch.linalg.inv(chklowt)
                        h_transformed = (
                            chklowtinv
                            @ data[self.h_out_field]
                            @ torch.transpose(chklowtinv, dim0=1, dim1=2).conj()
                        )
                    else:
                        S_k = data[self.s_out_field]
                        H_k = data[self.h_out_field]
                        egval_S, egvec_S = torch.linalg.eigh(S_k)
                        B = S_k.shape[0]
                        num_orbitals = H_k.shape[-1]
                        real_dtype = torch.float32 if H_k.dtype in [torch.complex64, torch.float32] else torch.float64

                        processed_eigvals_list = []
                        processed_mask_list = []
                        for k_idx in range(B):
                            healthy_mask = egval_S[k_idx] > ill_threshold
                            n_healthy = int(healthy_mask.sum().item())
                            abs_k_idx = i * nk + k_idx
                            min_s = float(egval_S[k_idx].min().detach().cpu().item())

                            if n_healthy == 0:
                                log.warning(
                                    "All overlap eigenvalues are below ill_threshold=%s at k-point %s "
                                    "(min eig(S)=%s); returning padded eigenvalues.",
                                    ill_threshold, abs_k_idx, min_s,
                                )
                                egval = torch.full((num_orbitals,), ill_pad_value, dtype=real_dtype, device=H_k.device)
                                processed_eigvals_list.append(egval)
                                processed_mask_list.append(torch.zeros((num_orbitals,), dtype=torch.bool, device=H_k.device))
                                continue

                            if healthy_mask.all():
                                log.debug(
                                    "All overlap eigenvalues are healthy at k-point %s (ill_threshold=%s); using standard Cholesky and validity mask all ones.",
                                    abs_k_idx, ill_threshold,
                                )
                                L = torch.linalg.cholesky(S_k[k_idx])
                                L_inv = torch.linalg.inv(L)
                                H_transformed = L_inv @ H_k[k_idx] @ L_inv.conj().T
                                egval = torch.linalg.eigvalsh(H_transformed)
                                processed_eigvals_list.append(egval)
                                processed_mask_list.append(torch.ones((num_orbitals,), dtype=torch.bool, device=H_k.device))
                                continue

                            log.warning(
                                "Projecting out %s/%s overlap modes at k-point %s "
                                "(min eig(S)=%s, ill_threshold=%s).",
                                num_orbitals - n_healthy, num_orbitals, abs_k_idx, min_s, ill_threshold,
                            )
                            U_sel = egvec_S[k_idx][:, healthy_mask]
                            eval_sel = egval_S[k_idx, healthy_mask]

                            H_proj = U_sel.conj().T @ H_k[k_idx] @ U_sel
                            S_proj = torch.diag(eval_sel).to(dtype=H_proj.dtype, device=H_proj.device)

                            L = torch.linalg.cholesky(S_proj)
                            L_inv = torch.linalg.inv(L)
                            H_transformed = L_inv @ H_proj @ L_inv.conj().T
                            egval_proj = torch.linalg.eigvalsh(H_transformed)

                            num_projected_out = num_orbitals - egval_proj.shape[0]
                            if num_projected_out > 0:
                                padding = torch.full((num_projected_out,), ill_pad_value, dtype=egval_proj.dtype, device=egval_proj.device)
                                egval = torch.cat([egval_proj, padding], dim=0)
                                mask = torch.cat([
                                    torch.ones((egval_proj.shape[0],), dtype=torch.bool, device=H_k.device),
                                    torch.zeros((num_projected_out,), dtype=torch.bool, device=H_k.device),
                                ], dim=0)
                            else:
                                egval = egval_proj
                                mask = torch.ones((num_orbitals,), dtype=torch.bool, device=H_k.device)

                            processed_eigvals_list.append(egval)
                            processed_mask_list.append(mask)
                        batch_eigvals_torch = torch.stack(processed_eigvals_list, dim=0)
                        batch_mask = torch.stack(processed_mask_list, dim=0)

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
                        processed_mask_list = []
                        for k_idx in range(B):
                            healthy_mask = egval_S[k_idx] > ill_threshold
                            n_healthy = int(healthy_mask.sum())
                            abs_k_idx = i * nk + k_idx
                            min_s = float(egval_S[k_idx].min())

                            if n_healthy == 0:
                                log.warning(
                                    "All overlap eigenvalues are below ill_threshold=%s at k-point %s "
                                    "(min eig(S)=%s); returning padded eigenvalues.",
                                    ill_threshold, abs_k_idx, min_s,
                                )
                                egval = np.full((num_orbitals,), ill_pad_value, dtype=real_dtype)
                                processed_eigvals_list.append(egval)
                                processed_mask_list.append(np.zeros((num_orbitals,), dtype=bool))
                                continue

                            if healthy_mask.all():
                                log.debug(
                                    "All overlap eigenvalues are healthy at k-point %s (ill_threshold=%s); using standard Cholesky and validity mask all ones.",
                                    abs_k_idx, ill_threshold,
                                )
                                L = np.linalg.cholesky(s_np[k_idx])
                                L_inv = np.linalg.inv(L)
                                H_transformed = L_inv @ h_np[k_idx] @ L_inv.conj().T
                                egval = np.linalg.eigvalsh(H_transformed)
                                processed_eigvals_list.append(egval)
                                processed_mask_list.append(np.ones((num_orbitals,), dtype=bool))
                                continue

                            log.warning(
                                "Projecting out %s/%s overlap modes at k-point %s "
                                "(min eig(S)=%s, ill_threshold=%s).",
                                num_orbitals - n_healthy, num_orbitals, abs_k_idx, min_s, ill_threshold,
                            )
                            U_sel = egvec_S[k_idx][:, healthy_mask]
                            eval_sel = egval_S[k_idx, healthy_mask]

                            H_proj = U_sel.conj().T @ h_np[k_idx] @ U_sel
                            S_proj = np.diag(eval_sel).astype(H_proj.dtype)

                            L = np.linalg.cholesky(S_proj)
                            L_inv = np.linalg.inv(L)
                            H_transformed = L_inv @ H_proj @ L_inv.conj().T
                            egval_proj = np.linalg.eigvalsh(H_transformed)

                            num_projected_out = num_orbitals - egval_proj.shape[0]
                            if num_projected_out > 0:
                                padding = np.full((num_projected_out,), ill_pad_value, dtype=egval_proj.dtype)
                                egval = np.concatenate([egval_proj, padding], axis=0)
                                mask = np.concatenate([
                                    np.ones((egval_proj.shape[0],), dtype=bool),
                                    np.zeros((num_projected_out,), dtype=bool),
                                ], axis=0)
                            else:
                                egval = egval_proj
                                mask = np.ones((num_orbitals,), dtype=bool)

                            processed_eigvals_list.append(egval)
                            processed_mask_list.append(mask)
                        batch_eigvals_np = np.stack(processed_eigvals_list, axis=0)
                        batch_mask = torch.from_numpy(np.stack(processed_mask_list, axis=0)).to(device=self.h2k.device)

            else:
                h_transformed = data[self.h_out_field]
                if ill_threshold is not None:
                    abs_k_start = i * nk
                    log.debug(
                        "ill_threshold=%s provided but ignored for k-point batch starting at %s because overlap is disabled.",
                        ill_threshold, abs_k_start,
                    )

            if eig_solver == 'torch':
                if batch_eigvals_torch is not None:
                    eigvals.append(batch_eigvals_torch)
                else:
                    eigvals.append(torch.linalg.eigvalsh(h_transformed))
            elif eig_solver == 'numpy':
                if batch_eigvals_np is not None:
                    eigvals_np = batch_eigvals_np
                else:
                    if h_transformed_np is None:
                        h_transformed_np = data[self.h_out_field].detach().cpu().numpy()
                    eigvals_np = np.linalg.eigvalsh(a=h_transformed_np)
                # Preserve dtype by converting to the Hamiltonian's original dtype
                eigvals.append(torch.from_numpy(eigvals_np).to(dtype=self.h2k.dtype, device=self.h2k.device))
            if batch_mask is not None:
                eigval_masks.append(batch_mask)

        data[self.out_field] = torch.nested.as_nested_tensor([torch.cat(eigvals, dim=0)])
        if eigval_masks:
            data[AtomicDataDict.EIGENVALUE_VALID_MASK_KEY] = torch.nested.as_nested_tensor([torch.cat(eigval_masks, dim=0)])
        if nested:
            data[AtomicDataDict.KPOINT_KEY] = torch.nested.as_nested_tensor([kpoints])
        else:
            data[AtomicDataDict.KPOINT_KEY] = kpoints

        return data

class PardisoEig:
    def __init__(self, sigma: float = 0.0, neig: int = 10, mode: str = 'normal'):
        """
        Solver using Pardiso for shift-invert eigenvalue problems.

        Args:
            sigma: Shift value (target energy).
            neig: Number of eigenvalues to solve for.
            mode: Eigsh mode ('normal', 'buckling', 'cayley').
        """
        if PyPardisoSolver is None or eigsh is None:
            raise ImportError("PardisoEig requires MKL (pypardiso) and scipy.sparse.linalg")

        self.sigma = sigma
        self.neig = neig
        self.mode = mode


    def solve(self, h_container, s_container, kpoints:  Union[list, torch.Tensor, np.ndarray], return_eigenvectors: bool = False):
        """
        Solve eigenvalues for given k-points.

        Args:
            h_container: vbcsr.ImageContainer for Hamiltonian.
            s_container: vbcsr.ImageContainer for Overlap (can be None).
            kpoints: Array of k-points (Nk, 3).
            return_eigenvectors: If True, return (eigenvalues, eigenvectors). Default False.

        Returns:
            list of eigenvalues arrays (and eigenvectors arrays if return_eigenvectors=True).
        """

        # Ensure kpoints is numpy array
        if isinstance(kpoints, torch.Tensor):
            kpoints = kpoints.cpu().numpy()

        eigvals_list = []
        eigvecs_list = []

        # Create a single solver instance and reuse it across k-points.
        # PARDISO stores LU factors in opaque `pt` handles; creating a new
        # solver per k-point without calling free_memory() leaks all that
        # internal MKL memory.
        solver = PyPardisoSolver(mtype=13)

        try:
            for k in kpoints:
                hk = h_container.sample_k(k, symm=True)

                if s_container is not None:
                    sk = s_container.sample_k(k, symm=True)
                    hk -= self.sigma * sk
                    A = hk.to_scipy(format="csr")
                    M = sk
                else:
                    hk.shift(-self.sigma)
                    A = hk.to_scipy(format="csr")
                    M = None

                A.sort_indices()
                A.sum_duplicates()
                N = A.shape[0]

                solver.factorize(A)

                def matvec(b):
                    return solver.solve(A, b)

                Op = LinearOperator((N, N), matvec=matvec, dtype=A.dtype)

                try:
                    # Use larger NCV to help convergence, especially for clustered eigenvalues
                    ncv =  max(2*self.neig + 1, 20)
                    vals, vecs = eigsh(A=hk, M=M, k=self.neig, sigma=0.0, OPinv=Op, mode=self.mode, which="LM", ncv=ncv)
                except Exception:
                    # Retry with larger NCV if ARPACK fails (e.g. error 3: No shifts could be applied)
                    # This often happens when eigenvalues are clustered near the shift
                    ncv =  max(5*self.neig, 50)
                    vals, vecs = eigsh(A=hk, M=M, k=self.neig, sigma=0.0, OPinv=Op, mode=self.mode, which="LM", ncv=ncv)

                # Release PARDISO's internal LU factorization memory for this
                # k-point before moving to the next one.  Without this, the
                # factorization buffers from *every* k-point accumulate.
                solver.free_memory()
                del Op

                eigvals_list.append(vals + self.sigma)
                if return_eigenvectors:
                    eigvecs_list.append(vecs)
        finally:
            # Guarantee all internal PARDISO memory is released even on error.
            solver.free_memory(everything=True)

        if return_eigenvectors:
            return eigvals_list, eigvecs_list
        else:
            return eigvals_list

class FEASTEig:
    def __init__(self, emin: float = -1.0, emax: float = 1.0, m0: Optional[int] = None,
                 max_refinement: int = 3, uplo: str = 'U', extract_triangular: bool = True):
        """
        Solver using FEAST algorithm for finding eigenvalues in a given interval.

        Args:
            emin, emax: Energy interval [emin, emax].
            m0: Initial subspace size estimate.
            max_refinement: Number of refinements if subspace is too small.
            uplo: 'U' (Upper) or 'L' (Lower) triangular part to use.
            extract_triangular: Whether to extract triangular part automatically.
        """

        if FeastSolver is None:
            raise ImportError("FEAST solver not available")

        self.emin = emin
        self.emax = emax
        self.m0 = m0
        self.max_refinement = max_refinement
        self.uplo = uplo
        self.extract_triangular = extract_triangular

        # Initialize solver to check availability
        try:
            self.solver = FeastSolver()
        except ImportError as e:
            raise ImportError(f"FEAST solver not available: {e}") from e
        except Exception as e:
             raise RuntimeError(f"Failed to initialize FeastSolver: {e}") from e

    def solve(self, h_container, s_container, kpoints: Union[list, torch.Tensor, np.ndarray], return_eigenvectors: bool = False):
        """
        Solve eigenvalues for given k-points using FEAST.

        Args:
            h_container: Container for Hamiltonian (must support sample_k().to_scipy()).
            s_container: Container for Overlap (can be None).
            kpoints: Array of k-points.
            return_eigenvectors: If True, return (eigenvalues, eigenvectors). Default False.

        Returns:
            list of eigenvalues arrays (and eigenvectors arrays if return_eigenvectors=True).
        """
        if isinstance(kpoints, torch.Tensor):
            kpoints = kpoints.cpu().numpy()

        eigvals_list = []
        eigvecs_list = []

        for k in kpoints:
            # Get Hamiltonian and Overlap at k
            # Assuming h_container.sample_k returns object with .to_scipy()
            hk_obj = h_container.sample_k(k, symm=True)
            if hasattr(hk_obj, 'to_scipy'):
                 hk = hk_obj.to_scipy(format="csr")
            else:
                 # Fallback if it checks sparse type
                 hk = hk_obj

            if s_container is not None:
                sk_obj = s_container.sample_k(k, symm=True)
                if hasattr(sk_obj, 'to_scipy'):
                     sk = sk_obj.to_scipy(format="csr")
                else:
                     sk = sk_obj
            else:
                sk = None

            # Solve
            evals, vecs = self.solver.solve(
                hk, M=sk, emin=self.emin, emax=self.emax,
                m0=self.m0, max_refinement=self.max_refinement,
                uplo=self.uplo, extract_triangular=self.extract_triangular
            )

            eigvals_list.append(evals)
            if return_eigenvectors:
                eigvecs_list.append(vecs)

        if return_eigenvectors:
            return eigvals_list, eigvecs_list
        else:
            return eigvals_list


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
        eigvecs = []
        if nk is None:
            nk = num_k
        for i in range(int(np.ceil(num_k / nk))):
            data[AtomicDataDict.KPOINT_KEY] = kpoints[i*nk:(i+1)*nk]
            data = self.h2k(data)
            chklowtinv = None
            chklowtinv_np = None
            h_transformed_np = None
            if self.overlap:
                data = self.s2k(data)
                if eig_solver == 'torch':
                    chklowt = torch.linalg.cholesky(data[self.s_out_field])
                    chklowtinv = torch.linalg.inv(chklowt)
                    h_transformed = (
                        chklowtinv @ data[self.h_out_field] @ torch.transpose(chklowtinv,dim0=1,dim1=2).conj()
                    )
                elif eig_solver == 'numpy':
                    s_np = data[self.s_out_field].detach().cpu().numpy()
                    h_np = data[self.h_out_field].detach().cpu().numpy()
                    chklowt_np = np.linalg.cholesky(s_np)
                    chklowtinv_np = np.linalg.inv(chklowt_np)
                    h_transformed_np = chklowtinv_np @ h_np @ np.transpose(chklowtinv_np,(0,2,1)).conj()
            else:
                h_transformed = data[self.h_out_field]

            if eig_solver == 'torch':
                eigval, eigvec = torch.linalg.eigh(h_transformed)
                if self.overlap:
                    eigvec = torch.transpose(chklowtinv, dim0=1, dim1=2).conj() @ eigvec
            elif eig_solver == 'numpy':
                if h_transformed_np is None:
                    h_transformed_np = data[self.h_out_field].detach().cpu().numpy()
                eigval_np, eigvec_np = np.linalg.eigh(h_transformed_np)
                if self.overlap:
                    eigvec_np = np.transpose(chklowtinv_np,(0,2,1)).conj() @ eigvec_np
                eigval = torch.from_numpy(eigval_np).to(dtype=self.h2k.dtype, device=self.h2k.device)
                eigvec = torch.from_numpy(eigvec_np).to(dtype=data[self.h_out_field].dtype, device=self.h2k.device)

            eigvecs.append(eigvec)
            eigvals.append(eigval)

        data[self.eigval_field] = torch.nested.as_nested_tensor([torch.cat(eigvals, dim=0)])
        data[self.eigvec_field] = torch.cat(eigvecs, dim=0)

        if nested:
            data[AtomicDataDict.KPOINT_KEY] = torch.nested.as_nested_tensor([kpoints])
        else:
            data[AtomicDataDict.KPOINT_KEY] = kpoints

        return data
