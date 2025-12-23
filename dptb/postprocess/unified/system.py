import numpy as np
import torch
import os
import h5py
from typing import Union, Optional, List, Dict
import ase
from ase.io import read
import logging
from dptb.postprocess.unified.calculator import HamiltonianCalculator, DeePTBAdapter
from dptb.data import AtomicData, AtomicDataDict, block_to_feature
from dptb.nn.build import build_model
from dptb.postprocess.unified.properties.band import BandAccessor
from dptb.postprocess.unified.properties.dos import DosAccessor
from dptb.utils.constants import atomic_num_dict_r

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
    
    @property
    def atom_orbs(self):
        return self._atom_orbs
    
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
    
    @property
    def dos(self):
        if self._dos is None:
        # assert self.has_dos, "DOS have not been calculated. Please call get_dos() first."
            self._dos = DosAccessor(self)
        return self._dos
    
    @property
    def dos_data(self):
        assert self.has_dos, "DOS have not been calculated. please call get_dos() or use sys.dos.compute() first."
        return self._dos.dos_data

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
        self._atomic_data = self._calculator.model.idp(data_obj)
        self._atom_orbs = self.get_atom_orbs()
        
        return self._atomic_data

    def get_atom_orbs(self):
        orbs_per_type = self.calculator.get_orbital_info()
        atomic_numbers = self.model.idp.untransform(self._atomic_data['atom_types']).numpy().flatten()
        atomic_symbols = [atomic_num_dict_r[i] for i in atomic_numbers]
        atom_orbs=[]
        for i in range(len(atomic_symbols)):
            iatype=atomic_symbols[i]
            for iorb in orbs_per_type[iatype]:
                atom_orbs.append(f"{i}-{iatype}-{iorb}")
        return atom_orbs
    
    def get_bands(self, kpath_config: Optional[dict] = None, reuse: Optional[bool]=True, **kwargs):
        # 计算能带，返回 bands
        # bands 应该是一个类，也有属性。bands.kpoints, bands.eigenvalues, bands.klabels, bands.kticks, 也有函数 bands.plot()
        if self.has_bands and reuse:
            return self._bands
        else:
            assert kpath_config is not None, "kpath_config must be provided if bands not calculated."
            self._bands = BandAccessor(self)
            self._bands.set_kpath(**kpath_config)
            self._bands.compute()
            self.has_bands = True
            return self._bands

    
    def get_dos(self, kmesh: Optional[Union[list,np.ndarray]] = None, is_gamma_center: Optional[bool] = True, erange: Optional[Union[list,np.ndarray]] = None, 
                    npts: Optional[int] = 100, efermi: Optional[Union[int, float]]=0.0, 
                    smearing: Optional[str] = 'gaussian', sigma: Optional[float] = 0.05, pdos: Optional[bool]=False, reuse: Optional[bool]=True, **kwargs):
        """
        docstring, to be added!
        """
        # 计算态密度，返回 dos
        # dos 应该是一个类，也有属性。dos.kmesh, dos.eigenvalues, dos.klabels, dos.kticks, 也有函数 dos.plot()
        if self.has_dos and reuse:
            return  self._dos
        else:
            assert kmesh is not None, "kmesh must be provided."
            self._dos = DosAccessor(self)
            self._dos.set_kpoints(kmesh=kmesh,is_gamma_center=is_gamma_center)
            self._dos.set_dos_config(erange=erange, npts=npts, efermi=efermi, smearing=smearing, sigma=sigma, pdos=pdos, **kwargs)
            self._dos.compute()
            self.has_dos = True
            return self._dos