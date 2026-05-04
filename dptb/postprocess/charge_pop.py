from ase.io import read
import ase
from dptb.data import AtomicData, AtomicDataDict
from dptb.postprocess.elec_struc_cal import ElecStruCal
import torch
from typing import Optional, Union, List
import logging
import numpy as np
from dptb.utils.ksampling import sample as ksampling

log = logging.getLogger(__name__)

# Try to import numba for accelerated bincount summation
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    log.debug("Numba not available, falling back to NumPy implementation")


# Numba-accelerated bincount summation kernel
if NUMBA_AVAILABLE:
    @njit(parallel=True, cache=True)
    def _bincount_sum_numba(diag_vals: np.ndarray, orb2atom: np.ndarray, natoms: int) -> np.ndarray:
        """
        Parallel bincount summation over k-points using Numba.

        Args:
            diag_vals: (nk, norb) array of diagonal values (real, float64)
            orb2atom: (norb,) array mapping orbital index to atom index (int64)
            natoms: number of atoms

        Returns:
            trace_vals: (nk, natoms) array of summed values per atom
        """
        nk, norb = diag_vals.shape
        trace_vals = np.zeros((nk, natoms), dtype=np.float64)

        for k in prange(nk):  # Parallel loop over k-points
            for iorb in range(norb):
                iatom = orb2atom[iorb]
                trace_vals[k, iatom] += diag_vals[k, iorb]

        return trace_vals

    @njit(parallel=True, cache=True)
    def _direct_diag_rhos_numba(V: np.ndarray, SV: np.ndarray, occ: np.ndarray) -> np.ndarray:
        """
        Compute diagonal of Rho_S = DM @ S directly using Numba.

        Mathematical derivation:
            diag(Rho_S)_i = [DM @ S]_{i,i}
                          = Σ_n occ_n * V*_{i,n} * [S@V]_{i,n}

        Note: occ already includes the spin degeneracy factor (spindeg * f_n),
        where f_n is the occupation probability in [0, 1].

        Args:
            V: (nk, norb, nstate) eigenvector matrix (complex128)
            SV: (nk, norb, nstate) overlap @ V matrix (complex128)
            occ: (nk, nstate) occupation numbers including spindeg factor (float64)

        Returns:
            diag_vals: (nk, norb) diagonal values of Rho_S (float64)
        """
        nk, norb, nstate = V.shape
        diag_vals = np.zeros((nk, norb), dtype=np.float64)

        for k in prange(nk):  # Parallel over k-points
            for i in range(norb):
                val = 0.0
                for n in range(nstate):
                    # V*_{i,n} * SV_{i,n} * occ_n (occ includes spindeg)
                    val += (np.conj(V[k, i, n]) * SV[k, i, n] * occ[k, n]).real
                diag_vals[k, i] = val

        return diag_vals


def _bincount_sum_numpy(diag_vals: np.ndarray, orb2atom: np.ndarray, natoms: int) -> np.ndarray:
    """
    NumPy fallback for bincount summation using dense projection matrix.

    Args:
        diag_vals: (nk, norb) array of diagonal values
        orb2atom: (norb,) array mapping orbital index to atom index
        natoms: number of atoms

    Returns:
        trace_vals: (nk, natoms) array of summed values per atom
    """
    norb = diag_vals.shape[1]
    # Build projection matrix P (natoms, norb) where P[i,j]=1 if orb j belongs to atom i
    P = np.zeros((natoms, norb), dtype=np.float64)
    P[orb2atom, np.arange(norb)] = 1.0
    # Vectorized matrix multiplication: (nk, norb) @ (norb, natoms) -> (nk, natoms)
    return diag_vals @ P.T


def _direct_diag_rhos_numpy(V: np.ndarray, SV: np.ndarray, occ: np.ndarray) -> np.ndarray:
    """
    NumPy fallback for direct diagonal computation using einsum.

    Note: occ already includes the spin degeneracy factor (spindeg * f_n).

    Args:
        V: (nk, norb, nstate) eigenvector matrix (complex128)
        SV: (nk, norb, nstate) overlap @ V matrix (complex128)
        occ: (nk, nstate) occupation numbers including spindeg factor (float64)

    Returns:
        diag_vals: (nk, norb) diagonal values of Rho_S (float64)
    """
    return np.real(np.einsum('kin,kin,kn->ki', np.conj(V), SV, occ))


def direct_diag_rhos(V: np.ndarray, SV: np.ndarray, occ: np.ndarray) -> np.ndarray:
    """
    Compute diagonal of Rho_S = DM @ S directly.

    Uses Numba parallel kernel if available, otherwise falls back to NumPy einsum.

    Note: occ already includes the spin degeneracy factor (spindeg * f_n),
    where f_n is the occupation probability in [0, 1].

    Args:
        V: (nk, norb, nstate) eigenvector matrix (complex128)
        SV: (nk, norb, nstate) overlap @ V matrix (complex128)
        occ: (nk, nstate) occupation numbers including spindeg factor (float64)

    Returns:
        diag_vals: (nk, norb) diagonal values of Rho_S (float64)
    """
    if NUMBA_AVAILABLE:
        # Ensure correct dtypes and memory layout for Numba
        V_c128 = np.ascontiguousarray(V, dtype=np.complex128)
        SV_c128 = np.ascontiguousarray(SV, dtype=np.complex128)
        occ_f64 = np.ascontiguousarray(occ, dtype=np.float64)
        return _direct_diag_rhos_numba(V_c128, SV_c128, occ_f64)
    else:
        return _direct_diag_rhos_numpy(V, SV, occ)


def bincount_sum(diag_vals: np.ndarray, orb2atom: np.ndarray, natoms: int) -> np.ndarray:
    """
    Efficient bincount summation over k-points.

    Uses Numba parallel kernel if available, otherwise falls back to NumPy.

    Args:
        diag_vals: (nk, norb) array of diagonal values
        orb2atom: (norb,) array mapping orbital index to atom index
        natoms: number of atoms

    Returns:
        trace_vals: (nk, natoms) array of summed values per atom
    """
    if NUMBA_AVAILABLE:
        # Ensure correct dtypes for Numba
        diag_vals_f64 = np.ascontiguousarray(diag_vals, dtype=np.float64)
        orb2atom_i64 = np.ascontiguousarray(orb2atom, dtype=np.int64)
        return _bincount_sum_numba(diag_vals_f64, orb2atom_i64, natoms)
    else:
        return _bincount_sum_numpy(diag_vals, orb2atom, natoms)


class Mulliken(ElecStruCal):
    """
    Class for calculating Mulliken charges from electronic structure data.
    Args:
        model (torch.nn.Module): The DeePTB model used for electronic structure prediction.
        results_path (str, optional): Path to store or retrieve calculation results. Defaults to None.
        device (str, optional): Device to run calculations on ('cpu' or 'cuda'). Defaults to 'cpu'.
        eig_method (str, optional): Method for eigenvalue decomposition. Defaults to 'eigh'.
    Attributes:
        results_path (str): Path for results.
        structase (ase.Atoms or None): ASE atoms object representing the structure.
        pbc (array-like): Periodic boundary conditions of the structure.
        mul_charge (np.ndarray): Calculated Mulliken charges per atom.
        delta_charge (np.ndarray): Charge fluctuations per atom.
        per_atom_norbs (list): Number of orbitals per atom.
        per_atom_charge (list): Number of electrons per atom.
        per_atom_indices (np.ndarray): Indices for slicing orbitals per atom.
    Methods:
        get_mulcharge(data, kmeshgrid, nel_atom, Temp=50.0, AtomicData_options=None):
            Calculates Mulliken charges for the given structure and electronic data.
            Args:
                data (Union[AtomicData, ase.Atoms, str]): Input structure or electronic data.
                kmeshgrid (List[int]): K-point mesh grid for sampling.
                nel_atom (dict): Dictionary mapping atom types to number of electrons.
                Temp (float, optional): Temperature in Kelvin for Fermi-Dirac smearing. Defaults to 50.0.
                AtomicData_options (dict, optional): Additional options for AtomicData. Defaults to None.
            Returns:
                data: Updated data object with calculated properties.
    """
    def __init__(self,
                 model:torch.nn.Module,
                 results_path: str=None,
                 device: str='cpu',
                 eig_method: str = 'eigh'):

        super().__init__(model=model,
                         device=device,
                         eig_method=eig_method)
        self.results_path = results_path
        self.smearing_method = None  # Default smearing method
        self.structase = None  # Initialize structase to None
        self.klist = None  # Initialize klist to None
        self.wk = None  # Initialize wk to None
        self.pbc = None  # Initialize pbc to None
        self.estimated_E_fermi = None # Initialize estimated_E_fermi to None
        self.mul_charge = None
        self.delta_charge = None
        self.per_atom_norbs = None
        self.per_atom_charge = None
        self.per_atom_indices = None

    def reset(self) -> None:
        """Reset per-calculation state variables.

        This method clears all state variables that are specific to a single
        structure calculation, allowing the Mulliken instance to be reused for
        different structures without interference from previous calculations.

        This is called by DFTBSCC.reset() to ensure proper state cleanup when
        reusing DFTBSCC instances across multiple structures (e.g., in parallel
        Bayesian optimization).

        The following attributes are reset to None:
        - structase: ASE Atoms object for the structure
        - klist, wk: K-point list and weights
        - pbc: Periodic boundary conditions
        - estimated_E_fermi: Estimated Fermi energy
        - elec_bandE: Electronic band energy
        - mul_charge, delta_charge: Mulliken charges
        - per_atom_norbs, per_atom_charge, per_atom_indices: Per-atom orbital info
        - smearing_method: Smearing method for occupation
        """
        self.structase = None
        self.klist = None
        self.wk = None
        self.pbc = None
        self.estimated_E_fermi = None
        self.elec_bandE = None
        self.mul_charge = None
        self.delta_charge = None
        self.per_atom_norbs = None
        self.per_atom_charge = None
        self.per_atom_indices = None
        self.smearing_method = None

    def get_mulcharge(self, 
                      data: Union[AtomicData, ase.Atoms, str], 
                      kmeshgrid: Optional[List[int]] = None,
                      kmeshspacing: Optional[List[float]] = None,
                      kgamma_center: Optional[bool] = True,
                      krotational_symmetry: Optional[bool] = True,
                      ktime_inversion_symmetry: Optional[bool] = True,
                      smearing_method: str = "Fermi-Dirac",
                      nel_atom: dict = None,
                      Temp: float = 50.0,
                      AtomicData_options:dict=None,
                      ): 
        
        if nel_atom is None:
            log.error("nel_atom must be provided to calculate Mulliken charges.")
            raise ValueError("nel_atom must be provided to calculate Mulliken charges.")
        
        assert isinstance(nel_atom, dict), "nel_atom should be a dictionary with atom types as keys and number of electrons as values."

        # get the ase structure
        if self.structase is None:
            if isinstance(data, str):
                structase = read(data)
            elif isinstance(data, ase.Atoms):
                structase = data
            elif isinstance(data, AtomicData):
                structase = data.to("cpu").to_ase()
            self.structase = structase
        assert isinstance(self.structase, ase.Atoms), "structase should be an ase.Atoms object."
        self.pbc = self.structase.pbc

        self.smearing_method = smearing_method

        # generate kmesh for charge calculation
        if self.klist is None:
            klist, wk = ksampling(self.structase,
                                  meshgrid=kmeshgrid,
                                  meshspacing=kmeshspacing,
                                  gamma_centered=kgamma_center,
                                  rotational_symmetry=krotational_symmetry,
                                  time_inversion_symmetry=ktime_inversion_symmetry)
            log.info(f'KPOINTS  kmesh sampling: {klist.shape[0]} kpoints')
            self.klist = klist
            self.wk = wk

        # get_fermi_level would calculate the eigvalues, eigvectors and fermi level
        # When data[AtomicDataDict.ENERGY_EIGENVALUE_KEY] is not None, it will be used directly
        data, estimated_E_fermi, elec_bandE, occ = self.get_fermi_level(  data=data,
                                                                    nel_atom=nel_atom,
                                                                    klist = self.klist,
                                                                    wk = self.wk,
                                                                    pbc=self.pbc,
                                                                    AtomicData_options=AtomicData_options,
                                                                    temp=Temp,
                                                                    smearing_method=self.smearing_method)
        
        # calculate number of orbitals, valence electrons for each atom and the cumulative orbital indices
        per_atom_norbs, per_atom_charge, per_atom_indices = self.per_atom_norb(nel_atom)

        # calculate Mulliken charge
        mul_charge = self.cal_mul_charge(   per_atom_norbs = per_atom_norbs,
                                            per_atom_indices = per_atom_indices,
                                            eigenvectors = data[AtomicDataDict.EIGENVECTOR_KEY].detach().cpu().to(torch.complex128).numpy(),
                                            overlap_np = data[AtomicDataDict.OVERLAP_KEY].detach().cpu().to(torch.complex128).numpy(),
                                            occ = occ,
                                            wk = self.wk)

        
        delta_charge = mul_charge - np.array(per_atom_charge)
        
        self.mul_charge = mul_charge
        self.delta_charge = delta_charge
        self.per_atom_norbs = per_atom_norbs
        self.per_atom_charge = per_atom_charge
        self.per_atom_indices = per_atom_indices
        self.estimated_E_fermi = estimated_E_fermi
        self.elec_bandE = elec_bandE

        # assert abs(sum(mul_charge)-sum(per_atom_charge)) < 1e-3, f"Charge conservation check failed: mulliken charge {sum(mul_charge)} != total valence electrons {sum(per_atom_charge)}."
        # assert abs(np.sum(delta_charge)) < 1e-3, "Charge conservation check failed: sum of charge fluctuations is not zero."

        if abs(np.sum(mul_charge) - sum(per_atom_charge)) > 5e-2:
            log.warning(f"Charge conservation check failed: mulliken charge {sum(mul_charge)} != total valence electrons {sum(per_atom_charge)}.")
        if abs(np.sum(delta_charge)) > 5e-2:
            log.warning("Charge conservation check failed: sum of charge fluctuations is not zero.")

        return data


    def per_atom_norb(self,nel_atom: dict) -> List[int]:
        """
        Calculates the number of orbitals, valence electrons for each atom and the cumulative orbital indices.
        Args:
            nel_atom (dict): A dictionary mapping chemical symbols to the number of valence electrons for each atom.
        Returns:
            Tuple[List[int], List[int], np.ndarray]:
                - per_atom_norbs: List of the number of orbitals for each atom.
                - per_atom_charge: List of valence electrons for each atom.
                - per_atom_indices: Numpy array of cumulative orbital indices for slicing per-atom orbital data.
        """      
        symbols = [atom.symbol for atom in self.structase]
        per_atom_norbs = [int(self.model.idp.atom_norb[self.model.idp.chemical_symbol_to_type[s]]) for s in symbols]
        per_atom_charge = [nel_atom[s] for s in symbols]
        per_atom_indices = np.insert(np.cumsum(per_atom_norbs), 0, 0)# Insert 0 at the beginning for slicing

        return per_atom_norbs, per_atom_charge, per_atom_indices
    
    def cal_mul_charge(self,
                       per_atom_norbs: List[int],
                       per_atom_indices: np.ndarray,
                       eigenvectors: np.ndarray,
                       overlap_np: np.ndarray,
                       occ: np.ndarray,
                       wk: np.ndarray) -> np.ndarray:
        """
        Calculates the Mulliken charge population for each atom in the system.

        This method computes the Mulliken charges by constructing the density matrix using the eigenvectors
        and occupation numbers, and projecting onto the atomic orbital basis using the overlap matrix.
        The calculation is performed for each k-point and summed over.

        Args:
            per_atom_norbs (List[int]): Number of orbitals per atom.
            per_atom_indices (np.ndarray): Indices marking the start and end of orbitals for each atom.
            eigenvectors (np.ndarray): Eigenvectors for each k-point, shape (nk, nstate, norb).
            overlap_np (np.ndarray): Overlap matrices for each k-point (numpy array, complex128).
            occ (np.ndarray): Occupation numbers for each k-point and state, shape (nk, nstate).
                Note: occ already includes the spin degeneracy factor (spindeg * f_n),
                where f_n is the occupation probability in [0, 1].
            wk (np.ndarray): Weights for each k-point.

        Returns:
            np.ndarray: Mulliken charge population for each atom.
        """
        # Optimized: compute only diagonal elements of Rho_S = DM @ S
        # Mathematical derivation:
        #   diag(Rho_S)_i = [DM @ S]_{i,i} = Σ_l DM[i,l] * S[l,i]
        #                 = Σ_n occ_n * Σ_l C_{i,n} * C*_{l,n} * S[l,i]
        #                 = Σ_n occ_n * C*_{i,n} * [S @ C]_{i,n}
        # where occ_n = spindeg * f_n (includes spin degeneracy factor)
        # This avoids computing full (nk, norb, norb) DM and Rho_S matrices

        V = eigenvectors.transpose(0, 2, 1)  # (nk, norb, nstate)
        SV = overlap_np @ V                   # (nk, norb, nstate) - S @ C

        # Direct diagonal computation using Numba JIT if available
        # Uses parallel kernel over k-points for acceleration
        diag_vals = direct_diag_rhos(V, SV, occ)

        # Efficient summation over atoms using parallel bincount
        # Uses Numba JIT if available, otherwise falls back to vectorized NumPy
        natoms = len(per_atom_norbs)
        orb2atom = np.repeat(np.arange(natoms), np.diff(per_atom_indices))
        trace_vals = bincount_sum(diag_vals, orb2atom, natoms)  # (nk, natoms)
        mul_charge = wk @ trace_vals

        return mul_charge
