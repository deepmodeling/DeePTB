from typing import Union, Tuple, Optional, Any
from abc import ABC, abstractmethod
import torch
import numpy as np
from dptb.data import AtomicData, AtomicDataDict
from dptb.utils.argcheck import get_cutoffs_from_model_options
from dptb.nn.energy import Eigenvalues
import logging
from dptb.data.interfaces.ham_to_feature import feature_to_block

log = logging.getLogger(__name__)

class HamiltonianCalculator(ABC):
    """Abstract Base Class defining the interface for a Hamiltonian calculator."""
    
    device: torch.device
    dtype: torch.dtype

    @abstractmethod
    def model_forward(self, atomic_data: dict) -> dict:
        """
        Run the model forward pass to update atomic data with Hamiltonian inputs/features.
        
        Args:
            atomic_data: The input atomic data dictionary.
        
        Returns:
            The atomic data dictionary updated with Hamiltonian/Overlap blocks/features.
        """
        pass

    @abstractmethod
    def get_hamiltonian_blocks(self, atomic_data: dict) -> Tuple[Any, Any]:
        """
        Get the Hamiltonian (and Overlap) blocks from the atomic data.
        
        Args:
             atomic_data: The input atomic data.
             
        Returns:
             Tuple of (H_blocks, S_blocks). S_blocks can be None.
        """
        pass
        
    @abstractmethod
    def get_eigenvalues(self, atomic_data: dict) -> Tuple[dict, torch.Tensor]:
        """
        Calculate eigenvalues for the given atomic data.
        
        Args:
            atomic_data: The input atomic data dictionary.
            
        Returns:
            A tuple containing the updated atomic data and the eigenvalues tensor.
        """
        pass
    
    @abstractmethod
    def get_orbital_info(self) -> dict:
        """Return information about the orbitals/basis set."""
        pass

class DeePTBAdapter(HamiltonianCalculator):
    """Adapter for DeePTB PyTorch models to match HamiltonianCalculator interface."""
    
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
        
    def model_forward(self, atomic_data: dict) -> dict:
        # Check for override overlap in input
        # If overlap is present before model run, we treat it as an override to be preserved
        override_edge = atomic_data.get(AtomicDataDict.EDGE_OVERLAP_KEY)
        override_node = atomic_data.get(AtomicDataDict.NODE_OVERLAP_KEY)

        # Run model forward pass to get H/S blocks
        atomic_data = self.model(atomic_data)
        
        # Restore overlap if it was an override
        # We only need to do this if the model actually has overlap capability (and thus might have overwritten it)
        if self.overlap and override_edge is not None:
            atomic_data[AtomicDataDict.EDGE_OVERLAP_KEY] = override_edge
            if override_node is not None:
                atomic_data[AtomicDataDict.NODE_OVERLAP_KEY] = override_node          
        return atomic_data

    def get_hamiltonian_blocks(self, atomic_data):
        atomic_data = self.model_forward(atomic_data)
        Hblocks = feature_to_block(atomic_data, idp=self.model.idp)
        if self.overlap:
            Sblocks = feature_to_block(atomic_data, idp=self.model.idp, overlap=True)
        else:
            Sblocks = None

        return Hblocks, Sblocks 
    
    def get_eigenvalues(self, 
                        atomic_data: dict, 
                        nk: Optional[int]=None,
                        solver: Optional[str]=None) -> Tuple[dict, torch.Tensor]:
        # 1. Get Hamiltonian
        atomic_data = self.model_forward(atomic_data)
        
        # 2. Verify Overlap logic
        if self.overlap:
             if atomic_data.get(AtomicDataDict.EDGE_OVERLAP_KEY) is None:
                 raise RuntimeError("Overlap model but no overlap in output.")
                 
        # 3. Solve Eigenvalues
        atomic_data = self.eigv_solver(data=atomic_data,nk=nk, eig_solver=solver)
        
        eigs = atomic_data[AtomicDataDict.ENERGY_EIGENVALUE_KEY][0] # atomic_data is usually batched, take 0
        return atomic_data, eigs

    def get_orbital_info(self) -> dict:
        return {
            "type_names": self.model.idp.type_names,
            # Add more metadata as needed
        }
