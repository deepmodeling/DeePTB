from typing import Protocol, Union, Tuple, Optional, Any
import torch
import numpy as np
from dptb.data import AtomicData, AtomicDataDict
from dptb.utils.argcheck import get_cutoffs_from_model_options
from dptb.nn.energy import Eigenvalues
import logging

log = logging.getLogger(__name__)

class HamiltonianCalculator(Protocol):
    """Protocol defining the interface for a Hamiltonian calculator."""
    
    device: torch.device
    dtype: torch.dtype

    def get_hamiltonian(self, atomic_data: AtomicDataDict) -> dict:
        """
        Calculate the Hamiltonian and Overlap (if applicable) for the given atomic data.
        
        Args:
            atomic_data: The input atomic data dictionary containing structure and k-points.
        
        Returns:
            The atomic data dictionary updated with Hamiltonian/Overlap blocks.
        """
        ...
        
    def get_eigenvalues(self, atomic_data: AtomicDataDict) -> Tuple[dict, torch.Tensor]:
        """
        Calculate eigenvalues for the given atomic data.
        
        Args:
            atomic_data: The input atomic data dictionary.
            
        Returns:
            A tuple containing the updated atomic data and the eigenvalues tensor.
        """
        ...
    
    def get_orbital_info(self) -> dict:
        """Return information about the orbitals/basis set."""
        ...

class DeePTBAdapter:
    """Adapter for DeePTB PyTorch models to match HamiltonianCalculator protocol."""
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.device = model.device
        self.dtype = model.dtype
        self.model.eval()
        
        # Check model capabilities
        self.overlap = hasattr(model, 'overlap')
        if not self.model.transform:
            raise RuntimeError('The model.transform is not True, please check the model.')
            
        # Initialize Eigenvalues solver helper
        if self.overlap:
            self.eigv_solver = Eigenvalues(
                idp=model.idp,
                device=self.device,
                s_edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
                s_node_field=AtomicDataDict.NODE_OVERLAP_KEY,
                s_out_field=AtomicDataDict.OVERLAP_KEY,
                dtype=model.dtype,
            )
        else:
            self.eigv_solver = Eigenvalues(
                idp=model.idp,
                device=self.device,
                dtype=model.dtype,
            )
            
        # Cutoffs
        r_max, er_max, oer_max = get_cutoffs_from_model_options(model.model_options)
        self.cutoffs = {'r_max': r_max, 'er_max': er_max, 'oer_max': oer_max}
        if self.cutoffs['r_max'] is None:
            log.error('The r_max is not provided in model_options, please provide it in AtomicData_options.')
            raise RuntimeError('The r_max is not provided in model_options, please provide it in AtomicData_options.')
        
    def get_hamiltonian(self, atomic_data: AtomicDataDict) -> AtomicDataDict:
        # Run model forward pass to get H/S blocks
        return self.model(atomic_data)

    def get_eigenvalues(self, atomic_data: AtomicDataDict) -> Tuple[AtomicDataDict, torch.Tensor]:
        # 1. Get Hamiltonian
        data = self.get_hamiltonian(atomic_data)
        
        # 2. Verify Overlap logic
        if self.overlap:
             if data.get(AtomicDataDict.EDGE_OVERLAP_KEY) is None:
                 raise RuntimeError("Overlap model but no overlap in output.")
                 
        # 3. Solve Eigenvalues
        data = self.eigv_solver(data)
        
        eigs = data[AtomicDataDict.ENERGY_EIGENVALUE_KEY][0] # atomic_data is usually batched, take 0
        return data, eigs

    def get_orbital_info(self) -> dict:
        return {
            "type_names": self.model.idp.type_names,
            # Add more metadata as needed
        }
