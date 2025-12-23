import numpy as np
import torch
import os
import h5py
from typing import Union, Optional, List, Dict
import ase
from ase import Atoms
from ase.io import read
import logging
from dptb.postprocess.unified.calculator import HamiltonianCalculator, DeePTBAdapter
from dptb.data import AtomicData, AtomicDataDict, block_to_feature
from dptb.utils.make_kpoints import ase_kpath, abacus_kpath, vasp_kpath, kmesh_sampling
from dptb.nn.build import build_model
from dptb.postprocess.unified.properties.band import BandAccessor
from dptb.postprocess.unified.properties.dos import DosAccessor

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

    @property
    def data(self) -> AtomicDataDict:
        """Access the atomic data."""
        return self._atomic_data
    
    @property
    def atoms(self) -> ase.Atoms:
        """Return the ASE Atoms object representing the system."""
        return self._atoms

    def set_atoms(self,struct: Optional[Union[AtomicData, ase.Atoms, str]] = None, override_overlap: Optional[str] = None) -> AtomicDataDict:
        """Set the atomic structure."""
        if struct is None:
            return self._atomic_data
        
        # Reset state flags
        self.has_bands=False
        self.has_dos=False
        
        atomic_options = self._calculator.cutoffs        
        if isinstance(struct, str):
            self._atoms = read(struct)
            data_obj = AtomicData.from_ase(self._atoms, **atomic_options)
        elif isinstance(struct, ase.Atoms):
            self._atoms = struct
            data_obj = AtomicData.from_ase(struct, **atomic_options)
        elif isinstance(struct, AtomicData):
            log.info('The data is already an instance of AtomicData. Then the data is used directly.')
            data_obj = struct
            self._atoms = struct.to("cpu").to_ase()
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
    def band_data(self):
        """Deprecated alias or strictly for result access."""
        assert self.has_bands, "Bands have not been calculated. Please call get_bands() or use sys.band.compute() first."
        return self._bands.band_data
    
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
