import logging
import os
import sys
import torch
from typing import TYPE_CHECKING, Optional, Union, List
from dptb.postprocess.common import is_gui_available
import numpy as np
import matplotlib.pyplot as plt
from dptb.data import AtomicDataDict
from dptb.utils.make_kpoints import kmesh_sampling

if TYPE_CHECKING:
    from dptb.postprocess.unified.system import TBSystem

log = logging.getLogger(__name__)

class DosData:
    """
    Data class for Density of States (DOS) results.
    """
    def __init__(self, energy_grid: np.ndarray, total_dos: np.ndarray, 
                 pdos: Optional[np.ndarray] = None, pdos_labels: Optional[List[str]] = None,
                 fermi_level: float = 0.0):
        self.energy_grid = energy_grid
        self.total_dos = total_dos
        self.pdos = pdos # Shape [N_energy, N_orbitals]
        self.pdos_labels = pdos_labels
        self.fermi_level = fermi_level

    def plot(self, filename: Optional[str] = 'dos.png', show: Optional[bool] = None, 
             xlim: Optional[List[float]] = None, plot_pdos: bool = False, selected_orbitals: Union[List[int], List[str]] = None):
        """
        Plot the DOS.
        
        Args:
            plot_pdos: Whether to plot PDOS (if available).
            selected_orbitals: List of indices or labels of orbitals to plot. If None, plots all if plot_pdos=True.
        """
        fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
        
        # Plot Total DOS
        ax.plot(self.energy_grid - self.fermi_level, self.total_dos, color='k', lw=1, label='Total')
        
        # Plot PDOS
        if plot_pdos and self.pdos is not None:
            if selected_orbitals is None:
                # Plot all (might be messy if too many)
                indices = range(self.pdos.shape[1])
            else:
                # Resolve indices
                indices = []
                for item in selected_orbitals:
                    if isinstance(item, int):
                         indices.append(item)
                    elif isinstance(item, str) and self.pdos_labels:
                         try:
                             indices.append(self.pdos_labels.index(item))
                         except ValueError:
                             log.warning(f"Orbital label {item} not found.")

            for i in indices:
                label = self.pdos_labels[i] if self.pdos_labels else f"Orbital {i}"
                ax.plot(self.energy_grid - self.fermi_level, self.pdos[:, i], lw=0.8, alpha=0.8, label=label)
            
            ax.legend(fontsize=6)

        ax.set_xlabel('Energy - $E_F$ (eV)')
        ax.set_ylabel('DOS (states/eV)')
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        
        if xlim:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(self.energy_grid.min() - self.fermi_level, self.energy_grid.max() - self.fermi_level)
            
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300)
            log.info(f"DOS plot saved to {filename}")

        if show is None:
            should_show = is_gui_available()
        else:
            should_show = show
            
        if should_show:
            plt.show()
        else:
            plt.close()

    def export(self, path: str):
        """Save DOS data to file."""
        data_dict = {
            "energy": self.energy_grid, 
            "dos": self.total_dos, 
            "fermi_level": self.fermi_level
        }
        if self.pdos is not None:
            data_dict["pdos"] = self.pdos
            if self.pdos_labels:
                data_dict["pdos_labels"] = self.pdos_labels
                
        np.savez(path, **data_dict)

class DosAccessor:
    """
    Accessor for DOS functionality on a TBSystem.
    """
    def __init__(self, system: 'TBSystem'):
        self._system = system
        self._config = {}
        self._dos_data = None 
        self._k_points = None

    def set_kpoints(self, kmesh: List[int], is_gamma_center: bool = True):
        """
        Set K-point sampling for DOS calculation.
        Strategies:
        1. Generate k-points.
        2. Update data with k-points.
        
        Args:
            kmesh: [nkx, nky, nkz] grid.
            is_gamma_center: Whether to shift k-points to Gamma center.
        """        
        # Eager generation
        self._k_points = kmesh_sampling(kmesh, is_gamma_center=is_gamma_center)
        self._num_k = self._k_points.shape[0]
        
        # Prepare Data for Model
        k_tensor = torch.as_tensor(self._k_points, dtype=self._system._calculator.dtype, device=self._system._calculator.device)
        self._system._atomic_data[AtomicDataDict.KPOINT_KEY] = torch.nested.as_nested_tensor([k_tensor])

    @property
    def kpoints(self):
        return self._k_points

    def set_dos_config(self, erange, npts, efermi=0.0, smearing='gaussian', sigma=0.05, pdos=False, **kwargs):
        # Update processing config
        assert smearing in ['gaussian','lorentzian'], "The smearing should be either 'gaussian' or 'lorentzian' !"
        self._config.update({
            "erange": erange,
            "npts": npts,
            "efermi":efermi,
            "smearing": smearing,
            "sigma": sigma,
            "pdos": pdos,
            **kwargs
        })
        
    def compute(self):
        """
        Calculate DOS based on the stored configuration.
        """
        if not self._config:
            raise RuntimeError("DOS config not set. Call set_dos_config first.")
        if self._k_points is None:
            raise RuntimeError("The kpoints not set. Call set_kpoints first.")
        
        data = self._system._atomic_data
        num_k = self._num_k
                
        # 3. Calculate Eigenvalues/Vectors
        calc_pdos = self._config.get('pdos', False)
        
        if calc_pdos:
            data, eigs, vecs = self._system.calculator.get_eigenstates(data)
            # vecs: [Nk, Norb, Norb] (assuming 1 batch)
            # eigs: [Nk, Norb]
            
            # Retrieve Overlap if present
            sk = None
            if self._system.calculator.overlap:
                # s_out_field is typically 'overlap' or 'edge_overlap' transformed?
                # data should contain Sk if eigh_solver.s2k was called
                # Check calculator implementation: it calls s2k.
                # Eigh stores it in s_out_field ('overlap')
                sk = data.get(AtomicDataDict.OVERLAP_KEY)
                # FIX: dptb.nn.energy.Eigh transposes eigenvectors when overlap is present [State, Basis]
                # We need standardized [Basis, State]
                if vecs is not None:
                    vecs = vecs.transpose(-2, -1)
                    
        else:
            data, eigs = self._system.calculator.get_eigenvalues(data)
            vecs = None
        
        eigenvalues = eigs.detach().cpu().numpy() # [Nk, Nb]
        eigenvalues_flat = eigenvalues.flatten()
        
        # 4. Compute Weights for DOS/PDOS
        # Total DOS: weight = 1 for each state
        # PDOS: weight = projected char
        
        erange = self._config['erange']
        npts = self._config['npts']
        efermi = self._config['efermi']
        sigma = self._config['sigma']
        smearing = self._config['smearing']

        # energy range w.r.t E-fermi
        energy_grid = np.linspace(erange[0] + efermi, erange[1] + efermi, npts)
        
        # Broadening Function
        def broadening(E_grid, E_vals, sigma, method):
            # E_grid: [N_E, 1]
            # E_vals: [1, N_states]
            delta = E_grid[:, None] - E_vals[None, :]
            if method == 'gaussian':
                return np.exp(-0.5 * (delta / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)
            elif method == 'lorentzian':
                return (1 / np.pi) * (sigma / (delta**2 + sigma**2))
            else:
                return 0

        # Calculate Total DOS
        broadened = broadening(energy_grid, eigenvalues_flat, sigma, smearing) # [Npts, Nk*Nb]
        total_dos = np.sum(broadened, axis=1) / num_k
        
        pdos = None
        pdos_labels = None
        
        if calc_pdos and vecs is not None:
            # vecs shape: [Nk, Norb(basis), Nb(bands)] ??
            # Usually eig returns [..., M, M], columns are eigenvectors.
            # vecs[k, i, j] : i-th component of j-th eigenvector at k
            
            vecs = vecs.detach().cpu().numpy() # [Nk, Norb, Nb]
            
            # Weights calculation w_{k, b, alpha}
            # alpha: orbital index
            # If orthogonal: |C_{alpha, b}(k)|^2
            # If overlap: Re(C_{alpha, b}^* (S C)_{alpha, b}) = Re(C^*_{a,b} \sum_g S_{ag} C_{g,b})
            
            if sk is not None:
                sk = sk.detach().cpu().numpy() # [Nk, Norb, Norb]
                # SC = S @ C -> [Nk, Norb, Nb]
                sc = np.einsum('kij,kjb->kib', sk, vecs)
                weights = np.real(np.conj(vecs) * sc) # [Nk, Norb, Nb]
            else:
                weights = np.abs(vecs)**2
                
            # Flatten k and b: [Norb, Nk*Nb]
            # Transpose to [Norb, Nk, Nb] -> reshape
            nk, norb, nb = weights.shape
            weights = weights.transpose(1, 0, 2).reshape(norb, -1) # [Norb, N_states]
            
            # Apply broadening
            # dos_alpha(E) = sum_states weight_{alpha, state} * delta(E - E_state)
            # We already have `broadened` matrix [Npts, N_states]
            # PDOS[alpha, E] = weights[alpha, :] @ broadened.T ?? 
            # No, broadened is [Npts, N_states]
            # PDOS[E, alpha] = broadened @ weights.T
            # broadened: [Npts, N_states]
            # weights.T: [N_states, Norb]
            
            pdos = np.dot(broadened, weights.T) / num_k # [Npts, Norb]
            
            # Labels
            if hasattr(self._system, 'atom_orbs'):
                pdos_labels = self._system.atom_orbs
            else:
                 # Generic fallback
                 pdos_labels = [f"Orbital {i}" for i in range(pdos.shape[1])]

        self._dos_data = DosData(energy_grid, total_dos, pdos=pdos, pdos_labels=pdos_labels)
        self._system.has_dos = True
        return self._dos_data
        
    @property
    def dos_data(self):
        if self._dos_data is None:
             raise RuntimeError("DOS not calculated. Call calculate_dos() first.")
        return self._dos_data

    def plot(self, **kwargs):
        return self.dos_data.plot(**kwargs)
        
    def save(self, path: str):
        return self.dos_data.export(path)

        
