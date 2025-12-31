import torch
import numpy as np
import logging
from typing import Optional, Union, List
from dptb.postprocess.unified.utils import calculate_fermi_level

from dptb.data import AtomicDataDict

log = logging.getLogger(__name__)

class ACAccessor:
    def __init__(self, system):
        self._system = system
        
        # Check if model has overlap
        self.overlap = hasattr(system.calculator, 'overlap') and system.calculator.overlap
        
    def compute(self,
                omegas: Union[np.ndarray, torch.Tensor],
                kmesh: List[int],
                eta: float = 0.05, # Broadening
                broadening: str = 'gaussian', # 'gaussian' or 'lorentzian'
                temperature: float = 300.0,
                direction: str = 'xx',
                return_components: bool = False,
                method: str = 'vectorized'
                ):
        """
        Compute optical conductivity. (Real part, absorption).
        Uses Gauge 2 (Intra-cell position gauge).
        
        Args:
            omegas: Frequency grid (eV)
            kmesh: k-point mesh [nx, ny, nz]
            eta: Broadening parameter (eV)
            broadening: 'gaussian' or 'lorentzian'
            temperature: Temperature (K)
            direction: Direction string, e.g., 'xx', 'xy', etc.
            return_components: If True, return additional components
            method: Calculation method ('vectorized', 'loop', or 'jit')
        
        Returns:
            Complex optical conductivity tensor element.
            For Lorentzian: both real and imaginary parts are physical.
            Real part: absorption coefficient
            Imaginary part: related to refractive index (Kramers-Kronig)
        """
        assert len(direction) == 2
        dir_map = {'x': 0, 'y': 1, 'z': 2}
        idx_alpha = dir_map[direction[0]]
        idx_beta = dir_map[direction[1]]
        
        # K-Point Sampling
        from dptb.utils.make_kpoints import kmesh_sampling
        kpoints = kmesh_sampling(kmesh, is_gamma_center=True)
        weights = torch.ones(kpoints.shape[0]) / kpoints.shape[0]
        
        batch_size = 200 # Smaller batch due to dense matrices
        nk_total = kpoints.shape[0]
        
        device = self._system.calculator.device
        sigma_total = torch.zeros(len(omegas), dtype=torch.complex128, device=device)
        omegas_t = torch.as_tensor(omegas, device=device, dtype=torch.float64)
        
        data_template = self._system._atomic_data.copy()
        
        # Calculate Volume from cell
        cell = data_template[AtomicDataDict.CELL_KEY]
        if cell.dim() == 3:
            cell = cell[0]
        volume = torch.det(cell).item()
        
        for i_start in range(0, nk_total, batch_size):
            i_end = min(i_start + batch_size, nk_total)
            log.info(f"Optical: Processing k-points {i_start}-{i_end}/{nk_total}")
            
            k_batch = torch.as_tensor(kpoints[i_start:i_end], device=device, dtype=self._system.calculator.dtype)
            w_batch = torch.as_tensor(weights[i_start:i_end], device=device, dtype=torch.float64)
            
            data_batch = data_template.copy()
            
            # 1. Compute H(k), S(k) and derivatives using native calculator method
            # get_hk returns (Hk, dHdk, Sk, dSdk) when with_derivative=True
            Hk, dHdk, Sk, dSdk = self._system.calculator.get_hk(
                data_batch, 
                k_points=k_batch, 
                with_derivative=True
            )
            
            # Hk: [Nk, N, N]
            # dHdk: [Nk, N, N, 3]
                
            # 3. Solve Eigenvalues
            # If Overlap, solve generalized: H c = E S c.
            if self.overlap:
                try:
                    # Cholesky decomposition of S
                    L = torch.linalg.cholesky(Sk)
                    L_inv = torch.linalg.inv(L)    
                    # H_orth = L_inv @ Hk @ L_inv^H
                    H_orth = L_inv @ Hk @ torch.transpose(L_inv.conj(), 1, 2)
                except Exception as e:
                    log.error(f"Cholesky failed: {e}. S matrix might not be positive definite.")
                    raise e
                    
                eigs, vecs_orth = torch.linalg.eigh(H_orth)
                # vecs = L_inv^H @ vecs_orth
                vecs = torch.transpose(L_inv.conj(), 1, 2) @ vecs_orth
            else:
                eigs, vecs = torch.linalg.eigh(Hk)
                
            # 4. Matrix Elements
            def get_matrix_elem(Op):
                # <n | Op | m> = C^H @ Op @ C
                return torch.transpose(vecs.conj(), 1, 2) @ Op @ vecs

            dH_alpha = dHdk[..., idx_alpha]
            v_alpha = get_matrix_elem(dH_alpha)
            
            dH_beta = dHdk[..., idx_beta]
            v_beta = get_matrix_elem(dH_beta)
            
            if self.overlap:
                dS_alpha = dSdk[..., idx_alpha]
                dS_beta = dSdk[..., idx_beta]
                s_alpha_elem = get_matrix_elem(dS_alpha)
                s_beta_elem = get_matrix_elem(dS_beta)
                
                # v_nm = <n|dH|m> - (En+Em)/2 <n|dS|m>
                E_n = eigs.unsqueeze(2)
                E_m = eigs.unsqueeze(1)
                E_sym = 0.5 * (E_n + E_m)
                
                v_alpha = v_alpha - E_sym * s_alpha_elem
                v_beta = v_beta - E_sym * s_beta_elem
                
            # 5. Kubo Sum
            # Fermi
            efermi = self._system.efermi
            beta_T = 1.0 / (8.617e-5 * temperature)
            f = 1.0 / (1.0 + torch.exp(beta_T * (eigs - efermi)))
            
            f_n = f.unsqueeze(2) # [Nk, N, 1]
            f_m = f.unsqueeze(1) # [Nk, 1, N]
            E_n = eigs.unsqueeze(2)
            E_m = eigs.unsqueeze(1)
            
            E_mn = E_m - E_n
            f_mn = f_n - f_m
            
            # Matrix Element Product
            # M_nm = v_alpha * v_beta^* for general direction
            M_nm = v_alpha * v_beta.transpose(1, 2)
            
            mask_deg = torch.abs(E_mn) < 1e-6
            
            # Avoid division by zero
            E_mn_safe = E_mn.clone()
            E_mn_safe[mask_deg] = 1.0
            
            T_nm = M_nm * f_mn / E_mn_safe
            T_nm[mask_deg] = 0.0 
            
            # Use selected method
            # Flatten for efficiency (common prep)
            E_flat = E_mn.flatten()
            T_weighted_flat = (T_nm * w_batch.reshape(-1, 1, 1)).flatten()
            
            if method == 'jit':
                term = accumulate_sigma_jit(E_flat, T_weighted_flat, omegas_t, eta, broadening)
            elif method == 'vectorized':
                term = self._accumulate_vectorized(E_flat, T_weighted_flat, omegas_t, eta, broadening)
            elif method == 'loop':
                 term = self._accumulate_loop(E_flat, T_weighted_flat, omegas_t, eta, broadening)
            else:
                raise ValueError(f"Unknown method: {method}")
                
            sigma_total += term
                
        # Units
        spin_factor = 2.0 
        factor = np.pi * spin_factor / volume
        return sigma_total * factor

    def _accumulate_loop(self, E_flat, T_flat, omegas, eta, broadening):
        """
        Loop-based accumulation (often fastest due to memory efficiency).
        
        For Lorentzian: uses complex form 1/(E - ω + iη)
        """
        sigma_contr = torch.zeros_like(omegas, dtype=torch.complex128)
        sqrt_2pi = np.sqrt(2 * np.pi)
        pi = np.pi
        
        for i, w in enumerate(omegas):
             if w < 1e-4: 
                 continue
             
             if broadening == 'gaussian':
                 arg = (E_flat - w) / eta
                 delta = torch.exp(-0.5 * arg**2) / (eta * sqrt_2pi)
             else:
                 # Lorentzian (complex form): 1/(E - w + iη)
                 diff = E_flat - w
                 #delta = (eta / np.pi) / (diff**2 + eta**2)
                 delta = -1.0j/(diff - 1.0j * eta) / pi
                 
             sigma_contr[i] = torch.sum(T_flat * delta)
        return sigma_contr

@torch.jit.script
def accumulate_sigma_jit(E_flat: torch.Tensor, T_flat: torch.Tensor, omegas: torch.Tensor, eta: float, broadening: str) -> torch.Tensor:
    """
    JIT-compiled kernel for accumulating conductivity contributions.
    
    For Lorentzian: uses complex form 1/(E - ω + iη) to get both real and imaginary parts.
    Real part: absorption coefficient
    Imaginary part: related to refractive index
    """
    sigma_contr = torch.zeros_like(omegas, dtype=torch.complex128)
    # Constants
    sqrt_2pi = 2.506628274631
    pi = 3.14159265359
    
    for i in range(len(omegas)):
        w = omegas[i]
        if w < 1e-4:
            continue
            
        # Vectorized over bands/k-points (E_flat)
        if broadening == 'gaussian':
            # Gaussian: delta = exp(-0.5*((E-w)/eta)^2) / (eta * sqrt(2pi))
            arg = (E_flat - w) / eta
            delta = torch.exp(-0.5 * arg * arg) / (eta * sqrt_2pi)
        elif broadening == 'lorentzian':
            # Lorentzian (complex form): 1/(E - w + iη)
            # = [(E-w) - iη] / [(E-w)² + η²]
            diff = E_flat - w
            delta = -1.0j/(diff - 1.0j * eta) / pi
        else:
            raise ValueError(f"Unknown broadening type {broadening}, should be 'gaussian' or 'lorentzian'")
            
        val = torch.sum(T_flat * delta)
        sigma_contr[i] = val
        
    return sigma_contr

