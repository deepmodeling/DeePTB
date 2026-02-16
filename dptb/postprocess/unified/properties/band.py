import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import logging
import os
import sys
import torch
from dptb.data import AtomicDataDict
from dptb.utils.make_kpoints import ase_kpath, abacus_kpath, vasp_kpath
from typing import TYPE_CHECKING, Optional, Union, List
from dptb.postprocess.common import is_gui_available

if TYPE_CHECKING:
    from dptb.postprocess.unified.system import TBSystem

log = logging.getLogger(__name__)

class BandStructureData:
    """
    Data class for Band Structure results.
    
    Attributes:
        eigenvalues (np.ndarray): Shape [Nk, Nb]
        kpoints (np.ndarray): Shape [Nk, 3]
        xlist (np.ndarray): Shape [Nk], coordinates for plotting along the path.
        labels (List[str]): Labels for high-symmetry points.
        high_sym_kpoints (np.ndarray): X-coordinates of high-symmetry points.
        fermi_level (float): The Fermi level energy.
    """
    
    def __init__(
        self,
        eigenvalues: np.ndarray,
        xlist: np.ndarray,
        high_sym_kpoints: np.ndarray,
        labels: List[str],
        fermi_level: float = 0.0,
        kpoints: Optional[np.ndarray] = None
    ):
        self.eigenvalues = eigenvalues
        self.xlist = xlist
        self.high_sym_kpoints = high_sym_kpoints
        self.labels = labels
        self.fermi_level = fermi_level
        self.kpoints = kpoints

    def plot(
        self,
        filename: Optional[str] = 'band.png',
        emin: Optional[float] = None,
        emax: Optional[float] = None,
        show: Optional[bool] = None,
        ref_bands: Optional[Union[str, np.ndarray]] = None
    ):
        """
        Plot the band structure.
        
        Args:
            filename: Output filename for saving the plot. If None, plot won't be saved.
            emin: Minimum energy for y-axis limits.
            emax: Maximum energy for y-axis limits.
            show: Whether to display the plot. If None, automatically detects GUI availability.
            ref_bands: Reference bands data for comparison.
        """
        import matplotlib
        matplotlib.rcParams['font.size'] = 7
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
        matplotlib.rcParams['axes.linewidth'] = 0.5
        
        fig = plt.figure(figsize=(3.2, 2.8), dpi=200)
        ax = fig.add_subplot(111)
        
        # Plot Reference if provided
        if ref_bands is not None:
            if isinstance(ref_bands, str):
                ref_data = np.load(ref_bands)
            else:
                ref_data = ref_bands
            
            # Simple realignment logic (shift to align mins)
            # Note: A more robust alignment might be needed in future
            shift = np.min(self.eigenvalues) - np.min(ref_data)
            ref_data = ref_data + shift
            
            # Downsample for scatter plot clarity
            nintp = max(1, len(self.xlist) // 25)
            ax.plot(self.xlist[::nintp], ref_data[::nintp] - self.fermi_level, 
                    'o', ms=2, color='#5d5d5d', alpha=0.95, label="Ref")

        # Plot predicted bands
        ax.plot(self.xlist, self.eigenvalues - self.fermi_level, 
                color="tab:red", lw=0.5, alpha=0.95, label="DeePTB")
        
        # High symmetry lines
        if self.high_sym_kpoints is not None:
            for x in self.high_sym_kpoints[1:-1]:
                ax.axvline(x, color='gray', lw=0.3, ls='--')
                
        # Ticks and Labels
        ax.set_xticks(self.high_sym_kpoints)
        ax.set_xticklabels(self.labels)
        
        # Limits
        ax.set_xlim(self.xlist.min(), self.xlist.max())
        if emin is not None and emax is not None:
            ax.set_ylim(emin, emax)
            
        ax.set_ylabel('E - EF (eV)', fontsize=8)
        
        ax.tick_params(direction='in')
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300)
            log.info(f"Band structure plot saved to {filename}")
            
        # Determine whether to show the plot based on GUI availability
        if show is None:
            # Auto-detect GUI availability
            should_show = is_gui_available()
            if should_show:
                log.info("GUI detected, displaying plot")
            else:
                log.info("No GUI detected, closing plot")
        else:
            should_show = show
            
        if should_show:
            plt.show()
        else:
            plt.close()
            
    def export(self, path: str):
        """Save raw data to .npz or .npy"""
        np.savez(
            path, 
            eigenvalues=self.eigenvalues, 
            xlist=self.xlist,
            labels=self.labels,
            high_sym_kpoints=self.high_sym_kpoints,
            fermi_level=self.fermi_level
        )

class BandAccessor:
    """
    Accessor for Band Structure functionality on a TBSystem.
    Allows syntax like: system.band.set_kpath(...)
    """
    """
    Accessor for Band Structure functionality on a TBSystem.
    Allows syntax like: system.band.set_kpath(...)
    """
    def __init__(self, system: 'TBSystem'):
        self._system = system
        self._k_points = None
        self._x_list = None
        self._high_sym_kpoints = None
        self._k_labels = None
        self._band_data = None
        
    @property
    def klist(self):
        """Get the list of K-points."""
        return self._k_points
    
    @property
    def labels(self):
        """Get the K-point labels."""
        return self._k_labels
        
    @property
    def xlist(self):
        """Get the x-axis coordinates for plotting."""
        return self._x_list
        
    def set_kpath(self, method: str, **kwargs):
        """
        Configure the K-path for band structure calculations.
        """
        self._system._kpath_config = {"method": method, **kwargs}
        
        if method == 'ase':
            pathstr = kwargs.get('pathstr', kwargs.get('kpath', 'GXWLGK'))
            npoints = kwargs.get('total_nkpoints', kwargs.get('nkpoints', 100))
            self._k_points, self._x_list, self._high_sym_kpoints, self._k_labels = ase_kpath(
                self._system.atoms, pathstr, npoints
            )
            
        elif method == 'abacus':
            kpath_def = kwargs.get('kpath')
            self._k_labels = kwargs.get('klabels')
            self._k_points, self._x_list, self._high_sym_kpoints = abacus_kpath(
                self._system.atoms, kpath_def
            )
            
        elif method == 'vasp':
            pathstr = kwargs.get('pathstr', kwargs.get('kpath'))
            hs_dict = kwargs.get('high_sym_kpoints_dict', kwargs.get('high_sym_kpoints'))
            num_in_line = kwargs.get('number_in_line', 20)
            self._k_points, self._x_list, self._high_sym_kpoints, self._k_labels = vasp_kpath(
                self._system.atoms, pathstr, hs_dict, num_in_line
            )
            
        elif method == 'array':
            self._k_points = kwargs['kpath']
            self._k_labels = kwargs.get('labels')
            self._x_list = kwargs.get('xlist')
            self._high_sym_kpoints = kwargs.get('high_sym_kpoints')
            
        else:
            raise ValueError(f"Unknown kpath method: {method}")
            
        log.info(f"K-path configured using {method}. Total k-points: {len(self._k_points)}")

        # Prepare Data with K-points immediately
        # Shallow copy is enough for dict if we only add a key        
        # Create NestedTensor for K-points as expected by model (Batched)
        # Assuming batch size 1 for single structure
        k_tensor = torch.as_tensor(self._k_points, dtype=self._system._calculator.dtype, device=self._system._calculator.device)
        self._system._atomic_data[AtomicDataDict.KPOINT_KEY] = torch.nested.as_nested_tensor([k_tensor])
        
    def compute(self):
        """
        Compute the band structure using the configured K-path and store result in system.
        """
        if self._k_points is None:
            raise RuntimeError("K-path not set. Call system.band.set_kpath() first.")
            
        # Use system state data
        data = self._system._atomic_data
        
        # Calculate
        data, eigs = self._system.calculator.get_eigenvalues(data)
        
        # Extract results
        eigenvalues = eigs.detach().cpu().numpy() # [Nk, Nb]
        
        if self._system._efermi is None:
            efermi = 0.0
            log.info('The efermi is not unknown, set it to 0.0!')
        else:
            efermi = self._system._efermi
        # Create Data Object
        self._band_data = BandStructureData(
            eigenvalues=eigenvalues,
            kpoints=self._k_points,
            xlist=self._x_list,
            labels=self._k_labels,
            high_sym_kpoints=self._high_sym_kpoints,
            fermi_level = efermi
        )
        self._system.has_bands = True
        return self._band_data

    @property
    def band_data(self):
        """Get the computed band structure data."""
        if self._band_data is None:
             raise RuntimeError("Band structure not computed. Call system.band.compute() first.")
        return self._band_data

    def plot(self, **kwargs):
        """Plot the computed band structure."""
        return self.band_data.plot(**kwargs)

    def save(self, path: str):
        """Save the computed band structure."""
        return self.band_data.export(path)

