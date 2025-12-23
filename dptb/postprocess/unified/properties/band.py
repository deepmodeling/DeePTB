import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import logging
import os
import sys
from typing import Optional, Union, List
from dptb.postprocess.common import is_gui_available
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
