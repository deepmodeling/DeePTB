import numpy as np
import torch
from typing import Union, Optional, List, Dict
import ase
from ase import Atoms
from ase.io import read
import logging
from dptb.postprocess.unified.calculator import HamiltonianCalculator, DeePTBAdapter
from dptb.data import AtomicData, AtomicDataDict
from dptb.utils.make_kpoints import ase_kpath, abacus_kpath, vasp_kpath, kmesh_sampling
from dptb.nn.build import build_model
from dptb.utils.argcheck import get_cutoffs_from_model_options

log = logging.getLogger(__name__)

class TBSystem:
    """
    Central class representing a Tight-Binding System (Structure + Model).
    
    Attributes:
        atoms (ase.Atoms): The atomic structure.
        calculator (HamiltonianCalculator): The model calculator.
    """
    
    def __init__(
                 self,
                 data: Union[AtomicData, ase.Atoms, str],
                 calculator: Union[HamiltonianCalculator, torch.nn.Module, str],
                 override_overlap: Optional[str] = None,
                 device: Optional[Union[str, torch.device]]= torch.device("cpu")
                 ):
        # Initialize Calculator/Model
        if isinstance(calculator, str):
            # Load from checkpoint path
            log.info(f"Loading model from checkpoint: {calculator}")
            _model = build_model(checkpoint=calculator, common_options={'device': device})
            self._calculator = DeePTBAdapter(_model)
        elif isinstance(calculator, torch.nn.Module):
            self._calculator = DeePTBAdapter(calculator)
        elif isinstance(calculator, HamiltonianCalculator) or hasattr(calculator, 'get_eigenvalues'):
            # Allow objects that look like the protocol
            self._calculator = calculator
        else:
            raise ValueError("calculator must be a path string or a torch.nn.Module object or HamiltonianCalculator.")
        
        # Initialize state properties
        self._bands = None
        self._dos = None
        self.has_bands = False
        self.has_dos = False
        
        self._atomic_data = self.set_atoms(data, override_overlap)

    @property
    def calculator(self) -> HamiltonianCalculator:
        """Access the calculator."""
        return self._calculator

    @property
    def model(self) -> torch.nn.Module:
        """Access the model."""
        return self._calculator.model

    def set_atoms(self,struct: Optional[Union[AtomicData, ase.Atoms, str]] = None, override_overlap: Optional[str] = None) -> AtomicDataDict:
        """Set the atomic structure."""
        if struct is None:
            return self._atomic_data
        
        # Reset state flags
        self.has_bands=False
        self.has_dos=False
        
        atomic_options = self._calculator.cutoffs        
        if isinstance(struct, str):
            structase = read(struct)
            data_obj = AtomicData.from_ase(structase, **atomic_options)
        elif isinstance(struct, ase.Atoms):
            data_obj = AtomicData.from_ase(struct, **atomic_options)
        elif isinstance(struct, AtomicData):
            log.info('The data is already an instance of AtomicData. Then the data is used directly.')
            data_obj = struct
        else:
            raise ValueError('data should be either a string, ase.Atoms, or AtomicData')
        
        # Handle Overlap Override
        overlap_flag = hasattr(self._calculator.model, 'overlap')
        
        if overlap_flag and isinstance(override_overlap, str):
            assert os.path.exists(override_overlap), "Overlap file not found."
            with h5py.File(override_overlap, "r") as overlap_blocks:
                if len(overlap_blocks) != 1:
                    log.info('Overlap file contains more than one overlap matrix, only first will be used.')
                if overlap_flag:
                    log.warning('override_overlap is enabled while model contains overlap, override_overlap will be used.')
                    
                if "0" in overlap_blocks:
                    overlaps = overlap_blocks["0"]
                else:
                    overlaps = overlap_blocks["1"]
                    
                block_to_feature(data_obj, self._calculator.model.idp, blocks=False, overlap_blocks=overlaps)
        
        data_obj = AtomicData.to_AtomicDataDict(data_obj.to(self._calculator.device))
        self._atomic_data = data_obj
        return data_obj

    def get_bands(self, kpath_config: Optional[dict] = None):
        # 计算能带，返回 bands
        # bands 应该是一个类，也有属性。bands.kpoints, bands.eigenvalues, bands.klabels, bands.kticks, 也有函数 bands.plot()
        if self.has_bands:
            return self._bands
        else:
            assert kpath_config is not None, "kpath_config must be provided if bands not calculated."
            self._bands = BandAccessor(self)
            self._bands.set_kpath(**kpath_config)
            self._bands.compute()
            self.has_bands = True
            return self._bands


    @property
    def band(self) -> 'BandAccessor':
        """Access band structure functionality (Lazy initialization)."""
        if self._bands is None:
            self._bands = BandAccessor(self)
        return self._bands

    @property
    def bands(self):
        """Deprecated alias or strictly for result access."""
        assert self.has_bands, "Bands have not been calculated. Please call get_bands() or use sys.band.compute() first."
        return self._bands
    
    def get_dos(self, kmesh: Optional[Union[list,np.ndarray]] = None, erange: Optional[Union[list,np.ndarray]] = None, 
                    npts: Optional[int] = None, smearing: Optional[str] = 'gaussian', sigma: Optional[float] = 0.05, **kwargs):
        """
        docstring, to be added!
        """
        # 计算态密度，返回 dos
        # dos 应该是一个类，也有属性。dos.kmesh, dos.eigenvalues, dos.klabels, dos.kticks, 也有函数 dos.plot()
        if self.has_dos:
            return  self._dos
        else:
            assert kmesh is not None, "kmesh must be provided."
            self._dos = DosAccessor(self)
            self._dos.set_dos_config(kmesh, erange, npts, smearing, sigma, **kwargs)
            self._dos.calculate_dos()
            self.has_dos = True
            return self._dos

    @property
    def dos(self):
        assert self.has_dos, "DOS have not been calculated. Please call get_dos() first."
        return self._dos

class DosAccessor:
    """
    Accessor for DOS functionality on a TBSystem.
    """
    def __init__(self, system: 'TBSystem'):
        self._system = system
        self._config = {}
        self._data = None # Placeholder for DOS data object

    def set_dos_config(self, kmesh, erange, npts, smearing, sigma, **kwargs):
        self._config = {
            "kmesh": kmesh,
            "erange": erange,
            "npts": npts,
            "smearing": smearing,
            "sigma": sigma,
            **kwargs
        }

    def calculate_dos(self):
        # Placeholder for actual DOS calculation logic
        # Would involve:
        # 1. generating K-mesh
        # 2. _system._prepare_kpoint_data
        # 3. calculator.get_eigenvalues
        # 4. smearing/integration
        log.warning("DOS calculation logic is a placeholder.")
        self._data = {"dos": None} # Placeholder

        
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
        
        # Create Data Object
        from .properties.band import BandStructureData
        
        self._band_data = BandStructureData(
            eigenvalues=eigenvalues,
            kpoints=self._k_points,
            xlist=self._x_list,
            labels=self._k_labels,
            high_sym_kpoints=self._high_sym_kpoints,
            fermi_level=0.0 # TODO: Calculate Fermi level via calculator or system
        )
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

    def get_hamiltonian_at_k(self, k_points):
        """Get H(k) and S(k) at specific k-points."""
        data = self._system._atomic_data[np.asarray(k_points)]
        data = self._system._calculator.get_hamiltonian(data)
        # Depending on implementation, might need to extract H/S from datadict
        # Left for Phase 3
        return data
