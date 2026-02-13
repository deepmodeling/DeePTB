from typing import Union, Tuple, Optional, Any
from abc import ABC, abstractmethod
import torch
import numpy as np
from dptb.data import AtomicData, AtomicDataDict
from dptb.utils.argcheck import get_cutoffs_from_model_options
from dptb.nn.energy import Eigenvalues, Eigh
from dptb.data.interfaces.ham_to_feature import feature_to_block
from dptb.nn.hr2hk import HR2HK
from dptb.nn.hr2hR import Hr2HR

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
    def get_hr(self, atomic_data: dict) -> Tuple[Any, Any]:
        """
        Get the Hamiltonian (and Overlap) blocks from the atomic data.
        
        Args:
             atomic_data: The input atomic data.
             
        Returns:
             Tuple of (H_blocks, S_blocks). S_blocks can be None.
        """
        pass

    @abstractmethod
    def get_hR(self, atomic_data: dict) -> Tuple[Any, Any]:
        """
        Get the Hamiltonian (and Overlap) as vbcsr.ImageContainer.
        
        Args:
             atomic_data: The input atomic data.
             
        Returns:
             Tuple of (H_container, S_container). S_container can be None.
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
    def get_eigenstates(self, atomic_data: dict) -> Tuple[dict, torch.Tensor, torch.Tensor]:
        """
        Calculate eigenvalues and eigenvectors.
        
        Args:
            atomic_data: The input atomic data dictionary.
            
        Returns:
            Tuple of (updated atomic_data, eigenvalues, eigenvectors).
            Eigenvectors shape: [Batch, Nk, Norb, Norb] (if batched) or [Nk, Norb, Norb]
        """
        pass

    @abstractmethod
    def get_hk(self, atomic_data: dict, k_points: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Calculate H(k) and S(k).
        
        Args:
            atomic_data: The input atomic data.
            k_points: Optional k-points to evaluate at. If provided, they override any k-points in atomic_data.
            
        Returns:
            Tuple of (H(k), S(k)). S(k) is None if no overlap.
        """
        pass

    @abstractmethod
    def get_orbital_info(self) -> dict:
        """Return information about the orbitals/basis set."""
        pass

class DeePTBAdapter(HamiltonianCalculator):
    """Adapter for DeePTB PyTorch models to match HamiltonianCalculator interface."""
    
    def __init__(self, model: torch.nn.Module, override_overlap: str = None):
        self.model = model
        self.device = model.device
        self.dtype = model.dtype
        self.model.eval()
        
        # Check model capabilities
        self.overlap = hasattr(model, 'overlap') or (override_overlap != None)
        if not self.model.transform:
            raise RuntimeError('The model.transform is not True, please check the model.')
            
        # Initialize Solvers
        # 1. Eigenvalues (only values, lighter)
        # 2. Eigh (values + vectors, heavier)
        
        solver_kwargs = {
            "idp": model.idp,
            "device": self.device,
            "dtype": model.dtype
        }
        
        if self.overlap:
            overlap_kwargs = {
                "s_edge_field": AtomicDataDict.EDGE_OVERLAP_KEY,
                "s_node_field": AtomicDataDict.NODE_OVERLAP_KEY,
                "s_out_field": AtomicDataDict.OVERLAP_KEY,
            }
            solver_kwargs.update(overlap_kwargs)
            
        self.eigv_solver = Eigenvalues(**solver_kwargs)
        self.eigh_solver = Eigh(**solver_kwargs)
            
        # Cutoffs
        r_max, er_max, oer_max = get_cutoffs_from_model_options(model.model_options)
        self.cutoffs = {'r_max': r_max, 'er_max': er_max, 'oer_max': oer_max}
        if self.cutoffs['r_max'] is None:
            # log.error('The r_max is not provided in model_options, please provide it in AtomicData_options.') # Removed log import
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

    def get_hr(self, atomic_data):
        atomic_data = self.model_forward(atomic_data)
        Hblocks = feature_to_block(atomic_data, idp=self.model.idp)
        if self.overlap:
            Sblocks = feature_to_block(atomic_data, idp=self.model.idp, overlap=True)
        else:
            Sblocks = None

        return Hblocks, Sblocks 
    
    def get_hR(self, atomic_data):
                # Initialize hR converters
        h2R = Hr2HR(
            idp=self.model.idp,
            edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
            node_field=AtomicDataDict.NODE_FEATURES_KEY,
            overlap=False,
            dtype=self.model.dtype,
            device=self.device
        )
        if self.overlap:
            s2R = Hr2HR(
                idp=self.model.idp,
                edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
                node_field=AtomicDataDict.NODE_OVERLAP_KEY,
                overlap=True,
                dtype=self.model.dtype,
                device=self.device
            )
        atomic_data = self.model_forward(atomic_data)
        h_container = h2R(atomic_data)
        if self.overlap:
            s_container = s2R(atomic_data)
        else:
            s_container = None
        return h_container, s_container
    
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

    def get_eigenstates(self, atomic_data: dict, nk: Optional[int]=None) -> Tuple[dict, torch.Tensor, torch.Tensor]:
        # 1. Get Hamiltonian
        atomic_data = self.model_forward(atomic_data)
        
        # 2. Verify Overlap logic
        if self.overlap:
             if atomic_data.get(AtomicDataDict.EDGE_OVERLAP_KEY) is None:
                 raise RuntimeError("Overlap model but no overlap in output.")

        # 3. Solve Eigenvalues + Eigenvectors
        atomic_data = self.eigh_solver(data=atomic_data, nk=nk)
        
        eigs = atomic_data[AtomicDataDict.ENERGY_EIGENVALUE_KEY][0]
        vecs = atomic_data[AtomicDataDict.EIGENVECTOR_KEY] # Usually not nested? Check nn/energy.py 
        # In Eigh.forward: data[self.eigvec_field] = torch.cat(eigvecs, dim=0) -> [Summary_Nk, Norb, Norb]
        # It is NOT nested in current implementation of Eigh.
        
        return atomic_data, eigs, vecs
    
    def get_hk(self, atomic_data: dict, k_points: Optional[Union[torch.Tensor, np.ndarray, list]] = None, with_derivative: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # init_h2k, s2k
        h2k = HR2HK(
            idp=self.model.idp, 
            edge_field=AtomicDataDict.EDGE_FEATURES_KEY, 
            node_field=AtomicDataDict.NODE_FEATURES_KEY, 
            out_field=AtomicDataDict.HAMILTONIAN_KEY, 
            derivative=with_derivative,
            out_derivative_field=AtomicDataDict.HAMILTONIAN_DERIV_KEY,
            dtype=self.model.dtype, 
            device=self.device,
            )
        if self.overlap:
            s2k = HR2HK(
                idp=self.model.idp, 
                overlap=True, 
                edge_field=AtomicDataDict.EDGE_OVERLAP_KEY, 
                node_field=AtomicDataDict.NODE_OVERLAP_KEY, 
                out_field=AtomicDataDict.OVERLAP_KEY, 
                derivative=with_derivative,
                out_derivative_field=AtomicDataDict.OVERLAP_DERIV_KEY,
                dtype=self.model.dtype, 
                device=self.device,
                )
        
         # Inject k_points if provided

        if k_points is not None:
             # Create lightweight copy and inject k_points
             atomic_data = atomic_data.copy()
             
             # Convert to tensor if needed
             if not isinstance(k_points, torch.Tensor):
                  k_points = torch.as_tensor(k_points, dtype=self.dtype, device=self.device)
             else:
                  k_points = k_points.to(device=self.device, dtype=self.dtype)
             
             # Create NestedTensor: [1, Nk, 3] assuming single structure
             if k_points.dim() == 2:
                  k_nested = torch.nested.as_nested_tensor([k_points])
                  atomic_data[AtomicDataDict.KPOINT_KEY] = k_nested
             else:
                  # If already nested or batched, trust usage
                  atomic_data[AtomicDataDict.KPOINT_KEY] = k_points
        else:
            assert atomic_data.get(AtomicDataDict.KPOINT_KEY) is not None, "No kpoints found in atomic_data. pls provide kpoints."

        # 1. Forward pass
        atomic_data = self.model_forward(atomic_data)
        
        # 2. H(R) -> H(k)
        atomic_data = h2k(atomic_data)
        hk = atomic_data[AtomicDataDict.HAMILTONIAN_KEY]
        
        # 3. S(R) -> S(k)
        sk = None
        if self.overlap:
            atomic_data = s2k(atomic_data)
            sk = atomic_data[AtomicDataDict.OVERLAP_KEY]
        
        # 4. Return with or without derivatives
        if with_derivative:
            hk_deriv = atomic_data[AtomicDataDict.HAMILTONIAN_DERIV_KEY]
            sk_deriv = atomic_data.get(AtomicDataDict.OVERLAP_DERIV_KEY)
            return hk, hk_deriv, sk, sk_deriv
        else:
            return hk, sk

    def get_orbital_info(self) -> dict:
        
        orbs_per_type = {}
        for atomtype, orb_dict in self.model.idp.basis.items():
            orb_list = []
            for o in orb_dict:
                if "s" in o: orb_list.append(o)
                elif "p" in o: orb_list.extend([o+"_y", o+"_z", o+"_x"]) # Standard Wannier90 p-order usually z,x,y or similar? keeping dptb order
                elif "d" in o: orb_list.extend([o+"_xy", o+"_yz", o+"_z2", o+"_xz", o+"_x2-y2"])
            orbs_per_type[atomtype] = orb_list

        return orbs_per_type
