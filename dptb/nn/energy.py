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
            h_transformed_np = None
            if self.overlap:
                data = self.s2k(data)
                if eig_solver == 'torch':
                    chklowt = torch.linalg.cholesky(data[self.s_out_field])
                    chklowtinv = torch.linalg.inv(chklowt)
                    data[self.h_out_field] = (chklowtinv @ data[self.h_out_field] @ torch.transpose(chklowtinv,dim0=1,dim1=2).conj())
                elif eig_solver == 'numpy':
                    s_np = data[self.s_out_field].detach().cpu().numpy()
                    h_np = data[self.h_out_field].detach().cpu().numpy()
                    chklowt = np.linalg.cholesky(s_np)
                    chklowtinv = np.linalg.inv(chklowt)
                    h_transformed_np = chklowtinv @ h_np @ np.transpose(chklowtinv,(0,2,1)).conj()

            if eig_solver == 'torch':
                eigvals.append(torch.linalg.eigvalsh(data[self.h_out_field]))
            elif eig_solver == 'numpy':
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
            
            # Try PARDISO first, fall back to scipy SuperLU if PARDISO fails
            # (MKL PARDISO has a known bug with certain block-structured patterns)
            solver = PyPardisoSolver(mtype=13)
            solver.factorize(A)
            
            def matvec(b):
                return solver.solve(A, b)
                
            Op = LinearOperator((N, N), matvec=matvec, dtype=A.dtype)
            
            try:
                # Use larger NCV to help convergence, especially for clustered eigenvalues
                ncv =  max(2*self.neig + 1, 20)
                vals, vecs = eigsh(A=hk, M=M, k=self.neig, sigma=self.sigma, OPinv=Op, mode=self.mode, which="LM", ncv=ncv)
            except Exception:
                # Retry with larger NCV if ARPACK fails (e.g. error 3: No shifts could be applied)
                # This often happens when eigenvalues are clustered near the shift
                ncv =  max(5*self.neig, 50)
                vals, vecs = eigsh(A=hk, M=M, k=self.neig, sigma=self.sigma, OPinv=Op, mode=self.mode, which="LM", ncv=ncv)
            
            eigvals_list.append(vals)
            if return_eigenvectors:
                eigvecs_list.append(vecs)
            
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
            raise ImportError(f"FEAST solver not available: {e}")
        except Exception as e:
             raise RuntimeError(f"Failed to initialize FeastSolver: {e}")

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