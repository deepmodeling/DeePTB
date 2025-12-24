
import logging
import torch
from typing import Optional, Union, TYPE_CHECKING
import numpy as np
from dptb.postprocess.topybinding import ToPybinding
from dptb.postprocess.totbplas import TBPLaS
from dptb.postprocess.interfaces import ToWannier90, ToPythTB

if TYPE_CHECKING:
    from dptb.postprocess.unified.system import TBSystem

log = logging.getLogger(__name__)

class ExportAccessor:
    """
    Accessor for exporting TBSystem to third-party software formats.
    """
    def __init__(self, system: 'TBSystem'):
        self._system = system

    def to_pythtb(self):
        """
        Export to PythTB model.
        
        Returns:
            pythtb.tb_model: The PythTB model object.
        """
        exporter = ToPythTB(model=self._system.model, device=self._system.calculator.device)
        return exporter.get_model(data=self._system._atomic_data)

    def to_pybinding(self, results_path: Optional[str] = None):
        """
        Export to Pybinding Lattice.
        
        Args:
            results_path: Optional path to save results.
            
        Returns:
            pybinding.Lattice: The Pybinding lattice object.
        """
        # Overlap check is handled in ToPybinding
        exporter = ToPybinding(
            model=self._system.model, 
            results_path=results_path, 
            overlap=False, # Pybinding doesn't support overlap usually
            device=self._system.calculator.device
        )
        return exporter.get_lattice(self._system._atomic_data)

    def to_tbplas(self, results_path: Optional[str] = None):
        """
        Export to TBPLaS PrimitiveCell.
        
        Args:
            results_path: Optional path to save results.
            
        Returns:
            tuple: (tbplas.PrimitiveCell, e_fermi)
        """
        exporter = TBPLaS(
            model=self._system.model,
            results_path=results_path,
            overlap=self._system.calculator.overlap,
            device=self._system.calculator.device
        )
        # Assuming effer from system if available, else 0.0
        e_fermi = self._system.efermi if self._system._efermi is not None else 0.0
        return exporter.get_cell(self._system._atomic_data, e_fermi=e_fermi)

    def to_wannier90(self, filename_prefix: str = "wannier90"):
        """
        Export to Wannier90 input files (_hr.dat, .win, _centres.xyz).
        
        Args:
            filename_prefix: Prefix for output files.
        """
        exporter = ToWannier90(self._system.model, device=self._system.calculator.device)
        
        e_fermi = self._system.efermi if self._system._efermi is not None else 0.0
        
        exporter.write_hr(
            self._system._atomic_data, 
            filename=f"{filename_prefix}_hr.dat", 
            e_fermi=e_fermi
        )
        exporter.write_win(filename=f"{filename_prefix}.win")
        exporter.write_centres(filename=f"{filename_prefix}_centres.xyz")
        log.info(f"Exported Wannier90 files with prefix '{filename_prefix}'")
