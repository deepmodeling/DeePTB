import numpy as np
import torch
import os
import h5py
import re
from dataclasses import dataclass
from typing import Any, Union, Optional, List, Dict, Tuple
import ase
from ase.io import read
import logging
from dptb.postprocess.unified.calculator import HamiltonianCalculator, DeePTBAdapter
from dptb.data import AtomicData, AtomicDataDict, block_to_feature
from dptb.nn.build import build_model
from dptb.postprocess.unified.properties.band import BandAccessor
from dptb.postprocess.unified.properties.dos import DosAccessor
from dptb.postprocess.unified.properties.optical_conductivity import ACAccessor
from dptb.utils.constants import atomic_num_dict_r, anglrMId
from dptb.postprocess.unified.utils import calculate_fermi_level
from dptb.utils.make_kpoints import kmesh_sampling
from dptb.postprocess.unified.properties.export import ExportAccessor

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SCCState:
    is_converged: bool
    scc_shift: torch.Tensor
    mulliken_charge: Optional[np.ndarray]
    delta_charge: Optional[np.ndarray]
    E_fermi: Any
    elec_totE: Any

    def __getitem__(self, key: str):
        return getattr(self, key)


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
            self._calculator = DeePTBAdapter(_model,override_overlap)
        elif isinstance(calculator, torch.nn.Module):
            self._calculator = DeePTBAdapter(calculator,override_overlap)
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
        self._scc = None
        self._scc_run_options = None
        self._scc_state = None
        self._use_scc = False
        
        self._atomic_data = self.set_atoms(data, override_overlap)

    @staticmethod
    def _pop_solver_kwargs(kwargs: dict, context: str) -> dict:
        solver = kwargs.pop("solver", None)
        eig_solver = kwargs.pop("eig_solver", None)
        if solver is not None and eig_solver is not None and solver != eig_solver:
            raise ValueError(
                f"solver and eig_solver were both provided to {context} with different values: "
                f"{solver!r} != {eig_solver!r}."
            )
        if solver is None and eig_solver is not None:
            log.warning("eig_solver is accepted for compatibility in %s; prefer solver.", context)
            solver = eig_solver

        solver_kwargs = {}
        if solver is not None:
            solver_kwargs["solver"] = solver
        for key in ("nk", "ill_threshold", "ill_pad_value"):
            if key in kwargs:
                solver_kwargs[key] = kwargs.pop(key)
        return solver_kwargs

    @staticmethod
    def _eigen_cache_key(solver_kwargs: dict) -> dict:
        return {key: value for key, value in solver_kwargs.items() if key != "nk"}

    @classmethod
    def _normalize_cache_value(cls, value):
        if isinstance(value, np.ndarray):
            return cls._normalize_cache_value(value.tolist())
        if isinstance(value, dict):
            return tuple(
                (key, cls._normalize_cache_value(item))
                for key, item in sorted(value.items())
            )
        if isinstance(value, (list, tuple)):
            return tuple(cls._normalize_cache_value(item) for item in value)
        return value

    @staticmethod
    def _cache_matches(last_key: Optional[dict], requested_key: dict) -> bool:
        if last_key is None:
            return False
        return all(last_key.get(key) == value for key, value in requested_key.items())

    @classmethod
    def _band_cache_key(cls, solver_kwargs: dict, kpath_config: Optional[dict] = None) -> dict:
        cache_key = {"solver": cls._eigen_cache_key(solver_kwargs)}
        if kpath_config is not None:
            cache_key["kpath_config"] = cls._normalize_cache_value(kpath_config)
        return cache_key

    @classmethod
    def _dos_cache_key(
        cls,
        solver_kwargs: dict,
        kmesh: Optional[Union[list, np.ndarray]] = None,
        is_gamma_center: Optional[bool] = True,
        erange: Optional[Union[list, np.ndarray]] = None,
        npts: Optional[int] = 100,
        smearing: Optional[str] = 'gaussian',
        sigma: Optional[float] = 0.05,
        pdos: Optional[bool] = False,
        extra_kwargs: Optional[dict] = None,
    ) -> dict:
        return {
            "solver": cls._eigen_cache_key(solver_kwargs),
            "dos_config": cls._normalize_cache_value({
                "kmesh": kmesh,
                "is_gamma_center": is_gamma_center,
                "erange": erange,
                "npts": npts,
                "smearing": smearing,
                "sigma": sigma,
                "pdos": pdos,
                "extra_kwargs": extra_kwargs or {},
            }),
        }

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

    @property
    def accond(self) -> 'ACAccessor':
        """Access optical conductivity properties (Lazy initialization)."""
        if not hasattr(self, '_optical_conductivity') or self._optical_conductivity is None:
            self._optical_conductivity = ACAccessor(self)
        return self._optical_conductivity

    @property
    def scc(self):
        """Access the configured SCC engine, if any."""
        return self._scc

    @property
    def scc_state(self) -> Optional[SCCState]:
        """Access cached SCC convergence state."""
        return self._scc_state

    @property
    def has_scc(self) -> bool:
        """Whether SCC has been configured on this system."""
        return self._scc is not None


    def set_atoms(self,struct: Optional[Union[AtomicData, ase.Atoms, str]] = None, override_overlap: Optional[str] = None) -> AtomicDataDict:
        """Set the atomic structure."""
        if struct is None:
            return self._atomic_data
        
        # Reset state flags
        self.has_bands=False
        self.has_dos=False
        self._scc_state = None
        
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
        
        if isinstance(override_overlap, str):
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

    def enable_scc(self,
                   params,
                   nel_atom: Dict[str, int],
                   overlap: Optional[bool] = None,
                   scc_dtype: torch.dtype = torch.float64,
                   **run_options):
        """
        Configure SCC as an optional electronic-state refinement for this system.

        TBSystem remains a scheduler: the SCC implementation is delegated to
        SKSCC, while TBSystem stores the converged SCC state for downstream
        band/DOS/export/electron-phonon style workflows.
        """
        from dptb.nn.dftb.dftb_scc import SKSCC
        from dptb.nn.dftb.scc_params import SCCParams

        if not isinstance(params, SCCParams):
            raise TypeError("params must be an SCCParams instance.")
        if nel_atom is None:
            raise ValueError("nel_atom is required to enable SCC.")

        self._scc = SKSCC(
            model=self.model,
            params=params,
            overlap=overlap,
            scc_dtype=scc_dtype,
        )
        self._scc_run_options = {
            "nel_atom": nel_atom,
            **run_options,
        }
        self._scc_state = None
        self._use_scc = True
        return self

    def disable_scc(self):
        """Disable SCC for subsequent TBSystem calculations without discarding the engine."""
        self._use_scc = False
        return self

    def run_scc(self, **run_options):
        """Run SCC iterations and cache the converged charge/Hamiltonian state."""
        if self._scc is None or self._scc_run_options is None:
            raise RuntimeError("SCC is not configured. Call enable_scc(...) first.")

        options = {**self._scc_run_options, **run_options}
        options.setdefault("AtomicData_options", {"r_max": self._scc.r_max})

        self._scc.run_iters(data=self.atoms, **options)
        self._scc_state = SCCState(
            is_converged=self._scc.is_converged,
            scc_shift=self._scc.scc_shift,
            mulliken_charge=None if self._scc.mulliken.mul_charge is None else self._scc.mulliken.mul_charge.copy(),
            delta_charge=None if self._scc.mulliken.delta_charge is None else self._scc.mulliken.delta_charge.copy(),
            E_fermi=self._scc.E_fermi,
            elec_totE=self._scc.elec_totE,
        )
        if self._scc.E_fermi is not None:
            self.set_efermi(self._scc.E_fermi)
        return self._scc_state

    def _resolve_use_scc(self, use_scc: Optional[bool]) -> bool:
        if use_scc is not None:
            return use_scc
        return self._use_scc and self._scc_state is not None

    def _require_scc_state(self) -> SCCState:
        if self._scc is None:
            raise RuntimeError("SCC is not configured. Call enable_scc(...) first.")
        if self._scc_state is None:
            raise RuntimeError("SCC has not been run. Call run_scc(...) first.")
        return self._scc_state

    def _prepare_kpoint_data(self,
                             atomic_data: Optional[dict] = None,
                             k_points: Optional[Union[torch.Tensor, np.ndarray, list]] = None) -> AtomicDataDict:
        data = (self._atomic_data if atomic_data is None else atomic_data).copy()
        if k_points is not None:
            if not isinstance(k_points, torch.Tensor):
                k_points = torch.as_tensor(k_points, dtype=self.calculator.dtype, device=self.calculator.device)
            else:
                k_points = k_points.to(dtype=self.calculator.dtype, device=self.calculator.device)
            if k_points.dim() == 2:
                data[AtomicDataDict.KPOINT_KEY] = torch.nested.as_nested_tensor([k_points])
            else:
                data[AtomicDataDict.KPOINT_KEY] = k_points
        assert data.get(AtomicDataDict.KPOINT_KEY) is not None, "No kpoints found. Please provide k_points."
        return data

    def _get_scc_hk_data(self,
                         atomic_data: Optional[dict] = None,
                         k_points: Optional[Union[torch.Tensor, np.ndarray, list]] = None) -> Tuple[dict, torch.Tensor, Optional[torch.Tensor]]:
        scc_state = self._require_scc_state()
        data = self._prepare_kpoint_data(atomic_data=atomic_data, k_points=k_points)
        data = self.model(data)
        data = self._scc.h2k(data)
        sk = None
        if self._scc.overlap:
            data = self._scc.s2k(data)
            sk = data[AtomicDataDict.OVERLAP_KEY]
        scc_hk = self._scc.cal_scc_hk(
            data=data,
            per_atom_indices=self._scc.mulliken.per_atom_indices,
            scc_shift=scc_state.scc_shift,
        )
        hk = data[AtomicDataDict.HAMILTONIAN_KEY].clone().to(dtype=self._scc.scc_cdtype) + scc_hk
        data[AtomicDataDict.HAMILTONIAN_KEY] = hk
        return data, hk, sk

    def get_hk(self,
               atomic_data: Optional[dict] = None,
               k_points: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
               use_scc: Optional[bool] = None,
               with_derivative: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Return the H(k), optionally including the converged SCC shift.

        ``use_scc=None`` follows the TBSystem state: once SCC is enabled and run,
        downstream calls use the SCC-corrected Hamiltonian by default. Pass
        ``use_scc=True`` or ``False`` to force either branch explicitly.
        """
        if not self._resolve_use_scc(use_scc):
            data = self._atomic_data if atomic_data is None else atomic_data
            return self.calculator.get_hk(data, k_points=k_points, with_derivative=with_derivative)

        if with_derivative:
            raise NotImplementedError("SCC-corrected H(k) derivatives are not implemented in TBSystem.")
        _, hk, sk = self._get_scc_hk_data(atomic_data=atomic_data, k_points=k_points)
        return hk, sk

    def get_eigenvalues(self,
                        atomic_data: Optional[dict] = None,
                        k_points: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
                        use_scc: Optional[bool] = None,
                        nk: Optional[int] = None,
                        solver: Optional[str] = None,
                        ill_threshold: Optional[float] = None,
                        ill_pad_value: float = 1e4) -> Tuple[dict, torch.Tensor]:
        """Calculate eigenvalues using either bare-model or SCC-corrected H(k)."""
        if not self._resolve_use_scc(use_scc):
            data = self._atomic_data if atomic_data is None else atomic_data
            if k_points is not None:
                data = self._prepare_kpoint_data(data, k_points)
            return self.calculator.get_eigenvalues(
                data,
                nk=nk,
                solver=solver,
                ill_threshold=ill_threshold,
                ill_pad_value=ill_pad_value,
            )

        if ill_threshold is not None:
            log.warning("ill_threshold is ignored for SCC eigenvalues because SCC fallback projection is not implemented.")

        data, hk, sk = self._get_scc_hk_data(atomic_data=atomic_data, k_points=k_points)
        eigvecs, eigvals = self._scc.eigh_solver(
            H_mat=hk,
            overlap=self._scc.overlap,
            overlap_mat=sk,
        )
        eigval = torch.cat(eigvals, dim=0)
        data[AtomicDataDict.ENERGY_EIGENVALUE_KEY] = torch.nested.as_nested_tensor([eigval])
        return data, eigval

    def get_eigenstates(self,
                        atomic_data: Optional[dict] = None,
                        k_points: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
                        use_scc: Optional[bool] = None,
                        nk: Optional[int] = None,
                        solver: Optional[str] = None) -> Tuple[dict, torch.Tensor, torch.Tensor]:
        """Calculate eigenvalues/eigenvectors using bare or SCC-corrected H(k)."""
        if not self._resolve_use_scc(use_scc):
            data = self._atomic_data if atomic_data is None else atomic_data
            if k_points is not None:
                data = self._prepare_kpoint_data(data, k_points)
            return self.calculator.get_eigenstates(data, nk=nk, solver=solver)

        data, hk, sk = self._get_scc_hk_data(atomic_data=atomic_data, k_points=k_points)
        eigvecs, eigvals = self._scc.eigh_solver(
            H_mat=hk,
            overlap=self._scc.overlap,
            overlap_mat=sk,
        )
        eigval = torch.cat(eigvals, dim=0)
        eigvec = torch.cat(eigvecs, dim=0)
        data[AtomicDataDict.ENERGY_EIGENVALUE_KEY] = torch.nested.as_nested_tensor([eigval])
        data[AtomicDataDict.EIGENVECTOR_KEY] = eigvec
        return data, eigval, eigvec

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
        solver_kwargs = self._pop_solver_kwargs(kwargs, "TBSystem.get_efermi")
        efermi_kwargs = {
            key: kwargs.pop(key)
            for key in ("k_weights",)
            if key in kwargs
        }
        if kwargs:
            log.warning(f"Ignoring unsupported get_efermi options: {sorted(kwargs)}")

        kpoints = kmesh_sampling(kmesh, is_gamma_center=is_gamma_center)
        k_tensor = torch.as_tensor(kpoints, 
                                   dtype=self.calculator.dtype, 
                                   device=self.calculator.device)
        data = self._atomic_data.copy()
        data[AtomicDataDict.KPOINT_KEY] = torch.nested.as_nested_tensor([k_tensor])
        data, eigs = self.get_eigenvalues(data, **solver_kwargs)

        eigs_valid_mask = data.get(AtomicDataDict.EIGENVALUE_VALID_MASK_KEY)
        if eigs_valid_mask is not None:
            eigs_valid_mask = eigs_valid_mask[0].detach().cpu().numpy()

        calculated_efermi = self.estimate_efermi_e(
                        eigenvalues=eigs.detach().numpy(),
                        temperature = temperature,
                        smearing_method=smearing_method,
                        q_tol=q_tol,
                        eigenvalue_valid_mask=eigs_valid_mask,
                        **efermi_kwargs)
        
        self.set_efermi(efermi = calculated_efermi)

        return calculated_efermi
    
    def estimate_efermi_e(self, eigenvalues,
                         k_weights=None,
                         temperature: float = 300,
                         smearing_method: str = 'FD',
                         q_tol: float = 1e-5,
                         eigenvalue_valid_mask: np.ndarray = None) -> float:
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
            q_tol=q_tol,
            eigenvalue_valid_mask=eigenvalue_valid_mask,
        )    
        

    def get_bands(self, kpath_config: Optional[dict] = None, reuse: Optional[bool]=True, **kwargs):
        # 计算能带，返回 bands
        # bands 应该是一个类，也有属性。bands.kpoints, bands.eigenvalues, bands.klabels, bands.kticks, 也有函数 bands.plot()
        solver_kwargs = self._pop_solver_kwargs(kwargs, "TBSystem.get_bands")
        cache_key = self._band_cache_key(solver_kwargs, kpath_config)
        if kwargs:
            log.warning(f"Ignoring unsupported get_bands options: {sorted(kwargs)}")

        if (
            self.has_bands
            and reuse
            and self._cache_matches(getattr(self, "_last_band_cache_key", None), cache_key)
        ):
            return self._bands
        else:
            if not (self.has_bands and reuse):
                assert kpath_config is not None, "kpath_config must be provided if bands not calculated."
                self._bands = BandAccessor(self)
                self._bands.set_kpath(**kpath_config)
            elif kpath_config is not None:
                self._bands.set_kpath(**kpath_config)
            self._bands.compute(**solver_kwargs)
            self._last_band_cache_key = self._band_cache_key(solver_kwargs, kpath_config)
            self.has_bands = True
            return self._bands

    
    def get_dos(self, kmesh: Optional[Union[list,np.ndarray]] = None, is_gamma_center: Optional[bool] = True, erange: Optional[Union[list,np.ndarray]] = None, 
                    npts: Optional[int] = 100, smearing: Optional[str] = 'gaussian', sigma: Optional[float] = 0.05, pdos: Optional[bool]=False, reuse: Optional[bool]=True, **kwargs):
        """
        docstring, to be added!
        """
        # 计算态密度，返回 dos
        # dos 应该是一个类，也有属性。dos.kmesh, dos.eigenvalues, dos.klabels, dos.kticks, 也有函数 dos.plot()
        solver_kwargs = self._pop_solver_kwargs(kwargs, "TBSystem.get_dos")
        cache_key = self._dos_cache_key(
            solver_kwargs=solver_kwargs,
            kmesh=kmesh,
            is_gamma_center=is_gamma_center,
            erange=erange,
            npts=npts,
            smearing=smearing,
            sigma=sigma,
            pdos=pdos,
            extra_kwargs=kwargs,
        )
        if (
            self.has_dos
            and reuse
            and self._cache_matches(getattr(self, "_last_dos_cache_key", None), cache_key)
        ):
            return  self._dos
        else:
            assert kmesh is not None, "kmesh must be provided."
            self._dos = DosAccessor(self)
            self._dos.set_kpoints(kmesh=kmesh,is_gamma_center=is_gamma_center)
            self._dos.set_dos_config(erange=erange, npts=npts, smearing=smearing, sigma=sigma, pdos=pdos, **kwargs)
            self._dos.compute(**solver_kwargs)
            self._last_dos_cache_key = self._dos_cache_key(
                solver_kwargs=solver_kwargs,
                kmesh=kmesh,
                is_gamma_center=is_gamma_center,
                erange=erange,
                npts=npts,
                smearing=smearing,
                sigma=sigma,
                pdos=pdos,
                extra_kwargs=kwargs,
            )
            self.has_dos = True
            return self._dos

    def _add_scc_to_realspace_features(self, data: AtomicDataDict, scc_shift: torch.Tensor) -> AtomicDataDict:
        node_features = data[AtomicDataDict.NODE_FEATURES_KEY].clone()
        edge_features = data[AtomicDataDict.EDGE_FEATURES_KEY].clone()
        shift = scc_shift.to(device=node_features.device, dtype=node_features.dtype)

        if self._scc.overlap:
            if AtomicDataDict.NODE_OVERLAP_KEY not in data or AtomicDataDict.EDGE_OVERLAP_KEY not in data:
                raise RuntimeError("Overlap SCC requires node and edge overlap features to build H(R).")
            node_overlap = data[AtomicDataDict.NODE_OVERLAP_KEY].to(device=node_features.device, dtype=node_features.dtype)
            edge_overlap = data[AtomicDataDict.EDGE_OVERLAP_KEY].to(device=edge_features.device, dtype=edge_features.dtype)
            edge_shift = scc_shift.to(device=edge_features.device, dtype=edge_features.dtype)
            edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]
            edge_coeff = 0.5 * (edge_shift[edge_index[0]] + edge_shift[edge_index[1]])
            data[AtomicDataDict.NODE_FEATURES_KEY] = node_features + node_overlap * shift[:, None]
            data[AtomicDataDict.EDGE_FEATURES_KEY] = edge_features + edge_overlap * edge_coeff[:, None]
            return data

        idp = self.model.idp
        idp.get_orbpair_maps()
        for iorb in idp.full_basis:
            lval = anglrMId[re.findall(r"[a-zA-Z]+", iorb)[0]]
            norb = 2 * lval + 1
            orbpair = f"{iorb}-{iorb}"
            orb_slice = idp.orbpair_maps[orbpair]
            diag = torch.eye(norb, device=node_features.device, dtype=node_features.dtype).reshape(-1)
            node_features[:, orb_slice] = node_features[:, orb_slice] + shift[:, None] * diag[None, :]
        data[AtomicDataDict.NODE_FEATURES_KEY] = node_features
        data[AtomicDataDict.EDGE_FEATURES_KEY] = edge_features
        return data

    def _get_scc_hR_data(self):
        scc_state = self._require_scc_state()
        data = self._atomic_data.copy()
        data = self.calculator.model_forward(data)
        data = self._add_scc_to_realspace_features(data, scc_state.scc_shift)
        return self.calculator.features_to_hR(data)

    def get_hR(self, use_scc: Optional[bool] = None):
        """
        Get the Hamiltonian (and Overlap) as vbcsr.ImageContainer.

        Returns:
             Tuple of (H_container, S_container). S_container can be None.
        """
        if not self._resolve_use_scc(use_scc):
            return self._calculator.get_hR(self._atomic_data)
        return self._get_scc_hR_data()


    def to_pardiso(self, output_dir: Optional[str] = "pardiso_input"):
        """
        Export system data for Pardiso/Julia band structure calculation.

        The following files will be generated in the output directory:
        - predicted_hamiltonians.h5: Hamiltonian matrix elements.
        - predicted_overlaps.h5: Overlap matrix elements (if applicable).
        - atomic_numbers.dat: Atomic numbers of the system.
        - positions.dat: Atomic positions (Cartesian).
        - cell.dat: Unit cell vectors.
        - basis.dat: Basis set information.

        Parameters
        ----------
        output_dir : str, optional
            Output directory path. Default is "pardiso_input".

        Returns
        -------
        None
        """
        os.makedirs(output_dir, exist_ok=True)
        log.info(f"Exporting Pardiso data to: {os.path.abspath(output_dir)}")
        
        # Calculate Hr and Sr
        hr, sr = self.calculator.get_hr(self.data)
        hr = self._symmetrize_hamiltonian(hr)
        if sr is not None:
            sr = self._symmetrize_hamiltonian(sr)
            
        # Save HDF5
        self._save_h5([hr], "predicted_hamiltonians.h5", output_dir)
        if sr is not None:
            self._save_h5([sr], "predicted_overlaps.h5", output_dir)
            
        # Save auxiliary files
        with open(os.path.join(output_dir, "atomic_numbers.dat"), "w") as f:
            for z in self.atoms.get_atomic_numbers():
                f.write(f"{z}\n")

        with open(os.path.join(output_dir, "positions.dat"), "w") as f:
            for pos in self.atoms.get_positions():
                f.write(f"{pos[0]} {pos[1]} {pos[2]}\n")

        with open(os.path.join(output_dir, "cell.dat"), "w") as f:
            for vec in self.atoms.get_cell():
                f.write(f"{vec[0]} {vec[1]} {vec[2]}\n")
        
        # basis.dat
        basis_str_dict = {}
        # Access basis info from model
        basis_info = self.model.idp.basis
        
        for elem, orbitals in basis_info.items():
            counts = {'s': 0, 'p': 0, 'd': 0, 'f': 0, 'g': 0}
            for o in orbitals:
                for orb_type in "spdfg" :
                    if orb_type in o:
                        counts[orb_type] += 1
                        break
            
            compressed = ""
            for orb_type in "spdfg":
                if counts[orb_type] > 0:
                    compressed += f"{counts[orb_type]}{orb_type}"
            
            basis_str_dict[elem] = compressed
            
        with open(os.path.join(output_dir, "basis.dat"), "w") as f:
            f.write(str(basis_str_dict))
            
        log.info("Successfully saved all Pardiso data.")

    def to_pardiso_debug(self, output_dir: Optional[str] = "pardiso_input"):
        """
        Export legacy Pardiso text/HDF5 inputs for debugging and backwards compatibility.
        """
        self.to_pardiso(output_dir=output_dir)

    def to_pardiso_json(self, output_dir: Optional[str] = "pardiso_input"):
        """
        Export system data for the modular Pardiso/Julia backend.

        The following files will be generated in the output directory:
        - predicted_hamiltonians.h5: Hamiltonian matrix elements.
        - predicted_overlaps.h5: Overlap matrix elements (if applicable).
        - structure.json: Structure and basis information.
        """
        import json
        import dptb

        os.makedirs(output_dir, exist_ok=True)
        log.info(f"Exporting Pardiso JSON data to: {os.path.abspath(output_dir)}")

        hr, sr = self.calculator.get_hr(self.data)
        hr = self._symmetrize_hamiltonian(hr)
        if sr is not None:
            sr = self._symmetrize_hamiltonian(sr)

        self._save_h5([hr], "predicted_hamiltonians.h5", output_dir)
        if sr is not None:
            self._save_h5([sr], "predicted_overlaps.h5", output_dir)

        symbols = self.atoms.get_chemical_symbols()
        structure = {
            "cell": self.atoms.get_cell().array.tolist(),
            "pbc": self.atoms.get_pbc().tolist(),
            "nsites": len(self.atoms),
            "symbols": symbols,
            "chemical_formula": self.atoms.get_chemical_formula(mode="reduce"),
            "positions": self.atoms.get_positions().tolist(),
        }

        basis = self.model.idp.basis
        l_map = {"s": 1, "p": 3, "d": 5, "f": 7, "g": 9}
        orbital_counts = {}
        for elem, orbs in basis.items():
            norb = 0
            for orb in orbs:
                for orb_type, count in l_map.items():
                    if orb_type in orb:
                        norb += count
                        break
            orbital_counts[elem] = norb

        spinful = hasattr(self.model, "soc_param")
        spin_multiplier = 2 if spinful else 1
        site_norbits = [orbital_counts[s] * spin_multiplier for s in symbols]
        basis_info = {
            "basis": basis,
            "orbital_counts": orbital_counts,
            "site_norbits": site_norbits,
            "total_orbitals": int(np.sum(site_norbits)),
            "spinful": spinful,
        }

        structure_data = {
            "basis_info": basis_info,
            "structure": structure,
            "meta": {
                "version": "1.0",
                "generator": f"DeePTB {dptb.__version__}",
            },
        }

        json_path = os.path.join(output_dir, "structure.json")
        with open(json_path, "w") as f:
            json.dump(structure_data, f, indent=2)

        log.info("Successfully saved all Pardiso JSON data.")

    def _save_h5(self, h_dict, fname, output_dir):
        """
        Save dictionary of matrices to HDF5 file.

        """
        path = os.path.join(output_dir, fname)
        
        # Ensure input is a list for the loop
        if isinstance(h_dict, dict):
            ham = [h_dict]
        else:
            ham = h_dict

        with h5py.File(path, 'w') as f:
            for i in range(len(ham)):
                grp = f.create_group(str(i))
                for key, block in ham[i].items():
                    data = block.detach().numpy() if isinstance(block, torch.Tensor) else block
                    grp.create_dataset(key, data=data)
        log.info(f"Saved {fname} ({len(ham)} blocks)")

    

    def _symmetrize_hamiltonian(self, h_dict):
        """
        Ensure that for every block H_ij(R), the conjugate block H_ji(-R) exists.
        Key format: "src_dst_rx_ry_rz"

        Parameters
        ----------
        h_dict : dict
            Dictionary of Hamiltonian blocks.

        Returns
        -------
        dict
            Symmetrized dictionary containing all conjugate blocks.
        """
        keys = list(h_dict.keys())
        for key in keys:
            parts = key.split('_')
            src, dst = int(parts[0]), int(parts[1])
            rx, ry, rz = int(parts[2]), int(parts[3]), int(parts[4])
            
            rev_key = f"{dst}_{src}_{-rx}_{-ry}_{-rz}"
            
            if rev_key not in h_dict:
                block = h_dict[key]
                if isinstance(block, torch.Tensor):
                    h_dict[rev_key] = block.t().conj()
                else:
                    h_dict[rev_key] = block.T.conj()
        return h_dict
