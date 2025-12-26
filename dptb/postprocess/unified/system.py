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
from dptb.postprocess.unified.utils import calculate_fermi_level
from dptb.utils.make_kpoints import kmesh_sampling
from dptb.postprocess.unified.properties.export import ExportAccessor

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
        self._export = None
        self._total_electrons = None
        self._efermi = None
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
    def atomic_symbols(self):
        return self._atomic_symbols

    @property
    def total_electrons(self):
        if self._total_electrons is None:
            print('Please call set_electrons first!')
        else:
            return self._total_electrons
    

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
    
    @property
    def efermi(self):
        assert self._efermi is not None, "The efermi is not set! call get_efermi to calcualted the efermi, or set_efermi to set to a custom value." 
        return self._efermi

    @property
    def export(self) -> 'ExportAccessor':
        """Access export interfaces (Lazy initialization)."""
        if self._export is None:
            self._export = ExportAccessor(self)
        return self._export


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
        self._atom_orbs, self._atomic_symbols = self.get_atom_orbs()
        
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
        return atom_orbs, atomic_symbols
    
    def set_electrons(self, nel_atom: Dict[str, int]) -> float:
        """
        Calculate the total number of valence electrons in the system.

        Parameters
        ----------
        nel_atom : Dict[str, int]
            Dictionary mapping element symbols to number of valence electrons.
            Example: {'Si': 4, 'H': 1}

        Returns
        -------
        float
            Total number of valence electrons.
        """
        if nel_atom is None:
             raise ValueError("nel_atom dictionary is required to calculate total electrons.")
        
        try:
            self._total_electrons = np.array([nel_atom[s] for s in self.atomic_symbols]).sum()
        except KeyError as e:
            raise KeyError(f"Element {e} found in system but not in nel_atom dictionary: {nel_atom}")

    def set_efermi(self, efermi):
        log.info(f'efermi is setted to {efermi}')
        self._efermi = efermi


    def get_efermi(self, kmesh: List[int],
                         is_gamma_center: bool = True,
                         temperature: float = 300,
                         smearing_method: str = 'FD',
                         q_tol: float = 1e-5,
                         **kwargs):
        # get efermi from scratch.
        kpoints = kmesh_sampling(kmesh, is_gamma_center=is_gamma_center)
        k_tensor = torch.as_tensor(kpoints, 
                                   dtype=self.calculator.dtype, 
                                   device=self.calculator.device)
        data = self._atomic_data.copy()             
        data[AtomicDataDict.KPOINT_KEY] = torch.nested.as_nested_tensor([k_tensor])
        data, eigs = self.calculator.get_eigenvalues(data)

        calculated_efermi = self.estimate_efermi_e(
                        eigenvalues=eigs.detach().numpy(),
                        temperature = temperature,
                        smearing_method=smearing_method,
                        q_tol  = q_tol, **kwargs)
        
        self.set_efermi(efermi = calculated_efermi)

        return calculated_efermi
    
    def estimate_efermi_e(self, eigenvalues,
                         k_weights=None,
                         temperature: float = 300,
                         smearing_method: str = 'FD',
                         q_tol: float = 1e-5) -> float:
        """
        Calculate Fermi level from eigenvalues. 
        This is give a freedom that parse eigenvlues from outside calculation, such as band and dos calculations
        
        Parameters
        ----------
        eigenvalues : np.ndarray
            Eigenvalues array.
        weights : np.ndarray, optional
            Weights for K-points.
        temperature : float
            Smearing temperature in Kelvin.
        smearing_method : str
            'FD' or 'Gaussian'.
        q_tol : float
            Charge convergence tolerance.

        Returns
        -------
        float
            Fermi Energy (eV).
        """      
                    
        # SOC detection: if model has 'soc_param', spindeg=1, else 2
        spindeg = 1 if hasattr(self.model, 'soc_param') else 2
        
        return calculate_fermi_level(
            eigenvalues=eigenvalues,
            total_electrons=self.total_electrons,
            spindeg=spindeg,
            weights=k_weights,
            temperature=temperature,
            smearing_method=smearing_method,
            q_tol=q_tol
        )    
        

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
                    npts: Optional[int] = 100, smearing: Optional[str] = 'gaussian', sigma: Optional[float] = 0.05, pdos: Optional[bool]=False, reuse: Optional[bool]=True, **kwargs):
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
            self._dos.set_dos_config(erange=erange, npts=npts, smearing=smearing, sigma=sigma, pdos=pdos, **kwargs)
            self._dos.compute()
            self.has_dos = True
            return self._dos