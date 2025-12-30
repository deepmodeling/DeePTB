import torch
import numpy as np
import logging
from typing import Optional, Union, List
from dptb.postprocess.unified.utils import calculate_fermi_level

from dptb.data import AtomicDataDict

log = logging.getLogger(__name__)

class OpticalAccessor:
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
                return_components: bool = False
                ):
        """
        Compute optical conductivity. (Real part, absorption).
        Uses Gauge 2 (Intra-cell position gauge).
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
            # M_nm = v_nm * v_mn. 
            # If v is Hermitian: v_mn = v_nm*. So M_nm = |v_nm|^2 (if alpha=beta).
            # v_alpha[n,m] * v_beta[m,n]
            M_nm = v_alpha * v_beta.transpose(1, 2)
            
            # T_nm = (f_n - f_m) / E_mn * M_nm
            # Handle degenerate safely
            mask_deg = torch.abs(E_mn) < 1e-6
            
            T_nm = M_nm * f_mn / E_mn
            T_nm[mask_deg] = 0.0 # Exclude intraband/degenerate from interband
            
            # Broadening
            def get_delta(diff, w, gamma):
                if broadening == 'gaussian':
                    return torch.exp(-0.5 * ((diff - w)/gamma)**2) / (gamma * np.sqrt(2 * np.pi))
                else: 
                     return (gamma/np.pi) / ((diff - w)**2 + gamma**2)
            
            for iw, w_val in enumerate(omegas_t):
                if w_val < 1e-4: continue
                delta = get_delta(E_mn, w_val, eta)
                
                # Sum over n, m, k
                # T_nm already contains (f_n - f_m)/E_mn approx (f_n - f_m)/w
                # So we sum T_nm * delta.
                term = torch.sum(T_nm * delta * w_batch.reshape(-1, 1, 1), dim=(0, 1, 2))
                sigma_total[iw] += term
                
        # Units
        # Theoretical prefactor: pi * (2 * g_s) / V
        # g_s = 1 for spin-polarized, 2 for non-polarized (usually).
        # We assume g_s = 1 per band in the model, but usually models are spin-degenerate 
        # unless specified. Reference uses 2 * g_s.
        # Let's assume the model is non-spin-polarized closed shell -> factor 2.
        # And if spin is explicit, it handles it.
        # Default to factor 2 for consistency with reference which assumes g_s=2.
        spin_factor = 2.0 
        factor = np.pi * spin_factor / volume
        return sigma_total * factor

