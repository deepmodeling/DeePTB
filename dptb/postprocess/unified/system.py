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
from dptb.postprocess.unified.properties.optical_conductivity import ACAccessor
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
        self._atoms: Optional[ase.Atoms] = None
        self._atomic_data: Optional[AtomicDataDict] = None 
        self._atom_orbs = None
        self._atomic_symbols = None
        
        self.set_atoms(data, override_overlap)

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


    def set_atoms(self, atoms: Optional[Union[AtomicData, ase.Atoms, str]] = None, override_overlap: Optional[str] = None) -> AtomicDataDict:
        """
        Set the atomic structure and generate inputs.
        """
        if atoms is None:
            return self._atomic_data
        
        # Reset state
        self.has_bands=False
        self.has_dos=False
        
        # 1. Establish Physical Structure (ASE Atoms)
        if isinstance(atoms, str):
            self._atoms = read(atoms)
        elif isinstance(atoms, ase.Atoms):
            self._atoms = atoms
        elif isinstance(atoms, AtomicData):
            log.info('The data is already an instance of AtomicData. Reconstructing Atoms from it.')
            self._atoms = atoms.to("cpu").to_ase() 
        else:
            raise ValueError('data should be either a string, ase.Atoms, or AtomicData')

        log.info("Generating AtomicData inputs...")
        
        # 2. Generate AtomicData
        atomic_options = self._calculator.cutoffs
        data_obj = AtomicData.from_ase(self._atoms, **atomic_options)
        
        # 3. Handle Overlap Override
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
        
        # 4. Finalize
        data_obj = AtomicData.to_AtomicDataDict(data_obj.to(self._calculator.device))
        self._atomic_data = self._calculator.model.idp(data_obj)
        
        self._atomic_symbols = self._atoms.get_chemical_symbols()
        self._atom_orbs = self.get_atom_orbs()
        
        return self._atomic_data

    @property
    def atomic_symbols(self):
        return self._atomic_symbols

    @property
    def atom_orbs(self):
        return self._atom_orbs

    def get_atom_orbs(self):
        """
        Get flattened list of all orbitals in the system.
        Delegates orbital expansion rules to the calculator.
        """
        orbs_per_type = self.calculator.get_orbital_info()
        atom_orbs = []
        
        for i, symbol in enumerate(self.atomic_symbols):
            if symbol in orbs_per_type:
                for orb in orbs_per_type[symbol]:
                     atom_orbs.append(f"{symbol}{i+1}_{orb}")
            else:
                log.warning(f"Atom {symbol} not found in model basis/orbital info.")
                
        return atom_orbs
    
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


    def to_pardiso_debug(self, output_dir: Optional[str] = "pardiso_input"):
        """
        Export system data for Pardiso/Julia (Legacy Text Format for Debugging).

        The following files will be generated in the output directory:
        - predicted_hamiltonians.h5: Hamiltonian matrix elements.
        - predicted_overlaps.h5: Overlap matrix elements (if applicable).
        - structure.json: Atomic structure and basis information.
        - *.dat: Legacy text files (positions.dat, basis.dat, etc.)

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
            
        # --- Legacy Export (for sparse_calc_npy_print.jl) ---
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

        log.info("Successfully saved all Pardiso data (Legacy Text Format).")

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

    def to_pardiso(self, output_dir: Optional[str] = "pardiso_input"):
        """
        Export system data for Pardiso/Julia (Standard JSON Format).

        The following files will be generated in the output directory:
        - predicted_hamiltonians.h5: Hamiltonian matrix elements (DFT-compatible format).
        - predicted_overlaps.h5: Overlap matrix elements (if applicable).
        - structure.json: Structure and basis information (pre-computed for Julia).

        The JSON file contains pre-computed data that Julia can directly use:
        - site_norbits: Number of orbitals per atom (computed from model.idp.atom_norb)
        - norbits: Total number of orbitals
        - All structure information in standard format

        This eliminates the need for Julia to:
        - Parse text files
        - Convert atomic numbers to symbols
        - Count orbitals from basis strings

        Parameters
        ----------
        output_dir : str, optional
            Output directory path. Default is "pardiso_input".

        Returns
        -------
        None
        """
        import json

        os.makedirs(output_dir, exist_ok=True)
        log.info(f"Exporting Pardiso data (NEW format) to: {os.path.abspath(output_dir)}")

        # Calculate Hr and Sr
        hr, sr = self.calculator.get_hr(self.data)
        hr = self._symmetrize_hamiltonian(hr)
        if sr is not None:
            sr = self._symmetrize_hamiltonian(sr)

        # Save HDF5 (keep DFT-compatible format)
        self._save_h5([hr], "predicted_hamiltonians.h5", output_dir)
        if sr is not None:
            self._save_h5([sr], "predicted_overlaps.h5", output_dir)

        # Save structure information in JSON format (DeePTB Schema v1.0)
        
        # 1. Structure (Geometry)
        import dptb
        symbols = self.atoms.get_chemical_symbols()
        
        structure = {
            'cell': self.atoms.get_cell().array.tolist(),
            'pbc': self.atoms.get_pbc().tolist(),
            'nsites': len(self.atoms),
            'chemical_formula': self.atoms.get_chemical_formula(mode='reduce'),
            'positions': self.atoms.get_positions().tolist(),
        }
        
        # 2. Basis Info (Model/Physics)
        basis = self.model.idp.basis
        l_map = {'s': 1, 'p': 3, 'd': 5, 'f': 7}
        orbital_counts = {}
        
        # Pre-calculate orbitals per element
        for elem, orbs in basis.items():
            norb = 0
            for orb in orbs:
                orb_type = orb[-1] 
                for t in l_map:
                    if t in orb:
                        norb += l_map[t]
                        break
            orbital_counts[elem] = norb

        # Compute total orbitals
        total_orbitals = 0
        for s in symbols:
             if s in orbital_counts:
                 total_orbitals += orbital_counts[s]
            
        basis_info = {
            'basis': basis,
            'orbital_counts': orbital_counts,
            'total_orbitals': total_orbitals,
            'spinful': hasattr(self.model, 'soc_param'),
        }

        structure_data = {
            'basis_info': basis_info,
            'structure': structure,
            'meta': {
                'version': '1.0',
                'generator': f"DeePTB {dptb.__version__}"
            }
        }

        json_path = os.path.join(output_dir, "structure.json")
        self._save_formatted_json(structure_data, json_path)

        log.info(f"Successfully saved all Pardiso data (NEW format).")
        log.info(f"  - Hamiltonian blocks: {len(hr)}")
        log.info(f"  - Structure: {len(self.atoms)} atoms, {basis_info['total_orbitals']} orbitals")
        log.info(f"  - Files: predicted_hamiltonians.h5, predicted_overlaps.h5, structure.json")

    def _save_formatted_json(self, data, path):
        """
        Save JSON with compact formatting for numeric arrays.
        Collapses innermost lists (like vectors) to single lines.
        """
        import json
        import re
        
        text = json.dumps(data, indent=2)
        
        # Regex to collapse short lists (innermost lists not containing other lists/dicts)
        # Collapse lists (arrays)
        def collapse_list(match):
            content = match.group(1)
            tokens = [token.strip() for token in content.split(',') if token.strip()]
            
            # Try to format as aligned floats
            try:
                if not tokens: return "[]"
                # Check if numbers
                floats = [float(t) for t in tokens]
                # Use fixed width for alignment (e.g. positions matrix)
                # {:>18.12f} aligns decimal points
                formatted = [f"{x:>18.12f}" for x in floats]
                compact = "[" + ", ".join(formatted) + "]"
                return compact
            except ValueError:
                # Fallback for non-numbers (e.g. strings)
                compact = "[" + ", ".join(tokens) + "]"
                return compact

        text = re.sub(r'\[([^\[\]\{\}]*)\]', collapse_list, text)

        # Collapse simple dictionaries (like orbital_counts)
        # Matches { ... } where ... contains no { } [ ]
        def collapse_dict(match):
             content = match.group(1)
             # Basic token cleaning
             items = [token.strip() for token in content.split(',') if token.strip()]
             compact = "{" + ", ".join(items) + "}"
             return compact
             
        text = re.sub(r'\{([^{}\[\]]*)\}', collapse_dict, text)
        
        with open(path, 'w') as f:
            f.write(text)

