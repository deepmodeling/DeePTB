import numpy as np
import torch
from typing import Union, Optional, List, Dict
from ase import Atoms
from ase.io import read
import logging

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
                 data: Union[AtomicData, ase.Atoms, str],
                 calculator: Union[HamiltonianCalculator, torch.nn.Module, str],
                 override_overlap: Optional[str] = None,
                 device: Optional[Union[str, torch.device]]= torch.device("cpu")
                 ):
        # Initialize Calculator/Model
        if isinstance(calculator, str):
            # Load from checkpoint path
            log.info(f"Loading model from checkpoint: {model}")
            _model = build_model(checkpoint=model, common_options={'device': device})
            self._calculator = HamiltonianCalculator(_model)
        elif isinstance(calculator, torch.nn.Module):
            self._calculator = HamiltonianCalculator(calculator)
        elif isinstance(calculator, HamiltonianCalculator):
            self._calculator = calculator
        else:
            raise ValueError("calculator must be a path string or a torch.nn.Module object or HamiltonianCalculator.")
        
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
        
        # set 一些 flag, 当结构改变时，需要重新计算一些属性，所以这些属性的 flag 需要重新设置
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
            assert kpath_config is not None, "kpath_config must be provided."
            self._bands = BandAccessor(self)
            self._bands.set_kpath(**kpath_config)
            self._bands.calculate_bands()
            self.has_bands = True
            return self._bands

    @property
    def bands(self):
        assert self.has_bands, "Bands have not been calculated. Please call get_bands() first."
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

        
class BandAccessor:
    """
    Accessor for Band Structure functionality on a TBSystem.
    Allows syntax like: system.band.set_kpath(...)
    """
    def __init__(self, system: 'TBSystem'):
        self._system = system
        
    def set_kpath(self, method: str, **kwargs):
        """
        Configure the K-path for band structure calculations.
        """
        self._system._kpath_config = {"method": method, **kwargs}
        
        if method == 'ase':
            pathstr = kwargs.get('pathstr', kwargs.get('kpath', 'GXWLGK'))
            npoints = kwargs.get('total_nkpoints', kwargs.get('nkpoints', 100))
            self._system._k_points, self._system._x_list, self._system._high_sym_kpoints, self._system._k_labels = ase_kpath(
                self._system.atoms, pathstr, npoints
            )
            
        elif method == 'abacus':
            kpath_def = kwargs.get('kpath')
            self._system._k_labels = kwargs.get('klabels')
            self._system._k_points, self._system._x_list, self._system._high_sym_kpoints = abacus_kpath(
                self._system.atoms, kpath_def
            )
            
        elif method == 'vasp':
            pathstr = kwargs.get('pathstr', kwargs.get('kpath'))
            hs_dict = kwargs.get('high_sym_kpoints_dict', kwargs.get('high_sym_kpoints'))
            num_in_line = kwargs.get('number_in_line', 20)
            self._system._k_points, self._system._x_list, self._system._high_sym_kpoints, self._system._k_labels = vasp_kpath(
                self._system.atoms, pathstr, hs_dict, num_in_line
            )
            
        elif method == 'array':
            self._system._k_points = kwargs['kpath']
            self._system._k_labels = kwargs.get('labels')
            self._system._x_list = kwargs.get('xlist')
            self._system._high_sym_kpoints = kwargs.get('high_sym_kpoints')
            
        else:
            raise ValueError(f"Unknown kpath method: {method}")
            
        log.info(f"K-path configured using {method}. Total k-points: {len(self._system._k_points)}")

    def compute(self):
        """
        Compute the band structure using the configured K-path and store result in system.
        """
        if self._system._k_points is None:
            raise RuntimeError("K-path not set. Call system.band.set_kpath() first.")
            
        # Prepare Data
        data = self._system._get_atomic_data(self._system._k_points)
        
        # Calculate
        data, eigs = self._system.calculator.get_eigenvalues(data)
        
        # Extract results
        eigenvalues = eigs.detach().cpu().numpy() # [Nk, Nb]
        
        # Create Data Object
        from .properties.band import BandStructureData
        
        self._system._band_data = BandStructureData(
            eigenvalues=eigenvalues,
            kpoints=self._system._k_points,
            xlist=self._system._x_list,
            labels=self._system._k_labels,
            high_sym_kpoints=self._system._high_sym_kpoints,
            fermi_level=0.0 # TODO: Calculate Fermi level via calculator or system
        )
        return self._system._band_data

    @property
    def data(self):
        """Get the computed band structure data."""
        if self._system._band_data is None:
             raise RuntimeError("Band structure not computed. Call system.band.compute() first.")
        return self._system._band_data

    def plot(self, **kwargs):
        """Plot the computed band structure."""
        return self.data.plot(**kwargs)

    def save(self, path: str):
        """Save the computed band structure."""
        return self.data.export(path)

    def get_hamiltonian_at_k(self, k_points):
        """Get H(k) and S(k) at specific k-points."""
        data = self._get_atomic_data(np.asarray(k_points))
        data = self.calculator.get_hamiltonian(data)
        # Depending on implementation, might need to extract H/S from datadict
        # Left for Phase 3
        return data
