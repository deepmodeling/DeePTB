from dptb.postprocess.charge_pop import Mulliken
from dptb.nn.dftb.sk_param import SKParam
from dptb.nn.dftb.scc_params import SCCParams
from dptb.data import  AtomicDataDict,AtomicData
from dptb.nn.dftbsk import DFTBSK
from ase.io import read
import ase
import numpy as np
import torch
import logging
from typing import Union, Optional, List, Dict, Any
from dptb.nn.hr2hk import HR2HK
from dptb.nn.dftb.gamma import get_expgamma, get_inv_r, get_Gamma
# from dptb.utils.constants import Bohr2Ang, Harte2eV
from dptb.nn.dftb.interp import calculate_atomic_rep
from dptb.nn.dftb.scc_mixer import SCCMixer, get_mixer

log = logging.getLogger(__name__)


class SKSCC(object):
    '''Generic SK self-consistent charge (SCC) engine.'''

    def __init__(self,
                 model,
                 params: SCCParams,
                 overlap: bool = True,
                 scc_dtype: torch.dtype = torch.float64) -> None:
        if model is None:
            raise ValueError("SKSCC requires a pre-initialized SK model.")
        if params is None:
            raise ValueError("SKSCC requires prepared SCCParams.")
        if not isinstance(params, SCCParams):
            raise TypeError("params must be an SCCParams instance.")
        if params.r_max is None:
            raise ValueError("SCCParams.r_max is required for SCC graph construction.")
        if overlap and not getattr(model, "overlap", False):
            raise ValueError("SKSCC requires a model initialized with overlap=True when overlap=True.")

        self.scc_dtype = scc_dtype
        self.scc_cdtype = torch.complex128 if scc_dtype == torch.float64 else torch.complex64
        self.model = model
        self.r_max = params.r_max
        self.skp = params
        self.scc_params = params
        self.mulliken = Mulliken(model=model,
                                 device=model.device,
                                 eig_method='eigh')
        self.overlap = overlap

        self.reset()

        self.h2k = HR2HK(
            idp = model.idp,
            edge_field = AtomicDataDict.EDGE_FEATURES_KEY,
            node_field = AtomicDataDict.NODE_FEATURES_KEY,
            out_field = AtomicDataDict.HAMILTONIAN_KEY,
            dtype = self.scc_dtype,
            device = model.device,
            )

        self.s2k = HR2HK(
            idp = model.idp,
            overlap = True,
            edge_field = AtomicDataDict.EDGE_OVERLAP_KEY,
            node_field = AtomicDataDict.NODE_OVERLAP_KEY,
            out_field = AtomicDataDict.OVERLAP_KEY,
            dtype = self.scc_dtype,
            device = model.device,
            )

    def reset(self) -> None:
        '''
        Reset per-calculation state variables.

        This method clears all state variables that are specific to a single
        structure calculation, allowing the DFTBSCC instance to be reused for
        different structures without interference from previous calculations.

        This is useful for performance optimization when processing multiple
        structures, as it avoids the overhead of reinitializing the expensive
        model, SK parameters, and transformers for each structure.

        The following attributes are reset to None:
        - atomic_numbers: Atomic numbers of the current structure
        - elec_totE, elec_H0_bandE, elec_bandE: Electronic energies
        - E_fermi: Fermi energy
        - mulcharge_old: Previous iteration Mulliken charges
        - scc_shift, scc_shift_energy: SCC corrections
        - data: Atomic data dictionary
        - expGamma, expGamma_onsite: Exponential gamma functions
        - inv_r: Inverse distance matrix
        - Gamma: Coulomb interaction matrix
        - total_energy, total_rep_energy: Total energies

        Also resets the Mulliken calculator state via mulliken.reset().
        '''
        self.atomic_numbers = None
        self.elec_totE = None
        self.elec_H0_bandE = None
        self.elec_bandE = None
        self.E_fermi = None
        self.mulcharge_old = None
        self.scc_shift = None
        self.scc_shift_energy = None
        self.data = None
        self.expGamma = None
        self.expGamma_onsite = None
        self.inv_r = None
        self.Gamma = None
        self.total_energy = None
        self.total_rep_energy = None
        self.is_converged = False  # Track SCC convergence status
        # Also reset the Mulliken calculator state
        self.mulliken.reset()

    def run_iters(self,
                  data: Union[ase.Atoms, str],
                  nel_atom: dict = None,
                  kmeshgrid: Optional[List[int]] = None,
                  kmeshspacing: Optional[List[float]] = None,
                  kgamma_center: bool = True,
                  krotational_symmetry: bool = True,
                  ktime_inversion_symmetry: bool = True,
                  Temp: float = 300,
                  AtomicData_options: dict = None,
                  smearing_method: str = 'Fermi-Dirac',
                  mixer: Union[str, SCCMixer] = 'simple',
                  tol: float = 1e-6,
                  mix_rate: float = 0.25,
                  max_iter: int = 50,
                  mixer_options: Dict[str, Any] = None) -> AtomicDataDict:
        '''
        Run self-consistent charge iterations for DFTB-SCC calculations.

        This method performs the iterative self-consistent charge (SCC) cycle
        to achieve charge convergence in the DFTB calculation. In each iteration:
        1. Mulliken charges are calculated from the current wavefunctions
        2. Charge differences from reference are computed
        3. SCC potential shifts are calculated via Coulomb interactions
        4. Hamiltonian is updated with SCC corrections
        5. New eigenvalues/eigenvectors are obtained
        6. Process repeats until charge convergence

        The iteration uses charge mixing to improve convergence stability.

        Parameters
        ----------
        data : Union[ase.Atoms, str]
            Input atomic structure. Can be either:
            - An ASE Atoms object containing the structure
            - A file path (str) to a structure file readable by ASE
              (e.g., CIF, POSCAR, XYZ formats)
        nel_atom : dict
            Number of valence electrons for each element type.
            Example: {'C': 4, 'H': 1, 'O': 6}
        kmeshgrid : Optional[List[int]], optional
            K-point mesh grid for Brillouin zone sampling, specified as
            [nk1, nk2, nk3]. Default is None. Either kmeshgrid or kmeshspacing
            must be provided for k-point generation.
        kmeshspacing : Optional[List[float]], optional
            K-point mesh spacing for automatic grid generation. Specified as
            [dk1, dk2, dk3] in reciprocal space units. Default is None.
            Alternative to kmeshgrid for specifying k-point density.
        kgamma_center : bool, optional
            Whether to use Gamma-centered k-point mesh. Default is True.
        krotational_symmetry : bool, optional
            Whether to apply rotational symmetry to reduce k-points. Default is True.
        ktime_inversion_symmetry : bool, optional
            Whether to apply time-inversion symmetry to reduce k-points. Default is True.
        Temp : float, optional
            Electronic temperature in Kelvin for Fermi-Dirac distribution
            in the occupation calculation. Higher temperatures provide better
            convergence for metallic systems. Default is 300 K.
        AtomicData_options : dict, optional
            Additional options passed to AtomicData constructor. Useful options
            include 'r_max' for cutoff radius. If None, uses default r_max
            from Slater-Koster parameters.
        mixer : Union[str, SCCMixer], optional
            Charge mixing method for SCC convergence. Can be either a string
            specifying the mixer type or an SCCMixer instance. Available types:
            - 'simple' or 'linear': Simple linear mixing (default)
            - 'anderson' or 'pulay': Anderson/Pulay mixing with history
            - 'broyden': Modified Broyden mixing (Johnson, PRB 38, 12807)
            - 'diis': Direct Inversion in the Iterative Subspace
            Default is 'simple'.
        tol : float, optional
            Convergence tolerance for maximum charge difference between
            iterations (in units of elementary charge e). Default is 1e-6.
            Iteration stops when max|Δq| < tol.
        mix_rate : float, optional
            Mixing parameter for charge density updates. Only used when mixer
            is 'simple' and mixer_options is None. Uses linear mixing:
            q_new = mix_rate * q_current + (1 - mix_rate) * q_old.
            Smaller values (0.1-0.3) improve stability but slow convergence.
            Default is 0.25.
        max_iter : int, optional
            Maximum number of SCC iterations allowed. If convergence is not
            reached within this limit, a warning is issued. Default is 50.
        mixer_options : Dict[str, Any], optional
            Options passed to the mixer constructor when mixer is a string.
            If None, uses default options. Common options include:
            - For SimpleMixer: {'mix_param': 0.2}
            - For AndersonMixer: {'n_generations': 6, 'mix_param': 0.2}
            - For BroydenMixer: {'max_iter': 100, 'mix_param': 0.2}
            - For DIISMixer: {'n_generations': 6, 'init_mix_param': 0.2}
            Default is None.

        Returns
        -------
        None
            Results are stored in instance attributes:
            - self.data: AtomicDataDict with updated Hamiltonian and eigenvalues
            - self.elec_totE: Total electronic energy (eV)
            - self.E_fermi: Fermi energy (eV)
            - self.mulliken.mul_charge: Final Mulliken charges

        '''

        # Reset per-calculation state to allow instance reuse across structures
        self.reset()

        is_converged = False
        iteration = 0

        # Prepare data
        if isinstance(data, str):
            structase = read(data)
        elif isinstance(data, ase.Atoms):
            structase = data
        assert isinstance(structase, ase.Atoms), "Input data must be an ASE Atoms object or a valid file path."
        self.atomic_numbers = structase.get_atomic_numbers()

        atomic_elements = [ase.data.chemical_symbols[num] for num in self.atomic_numbers]
        assert nel_atom is not None, "nel_atom dictionary must be provided."
        for el in atomic_elements:
            assert el in nel_atom, f"Element {el} not found in nel_atom dictionary."


        if AtomicData_options is None:
            AtomicData_options = {}
            AtomicData_options['r_max'] = self.r_max


        # Initialize the charge mixer
        if isinstance(mixer, str):
            # Create mixer from string type
            if mixer_options is None:
                # Use default options with mix_rate for backward compatibility
                if mixer.lower() in ('simple', 'linear'):
                    mixer_options = {'mix_param': mix_rate}
                elif mixer.lower() in ('anderson', 'pulay'):
                    mixer_options = {'mix_param': mix_rate, 'n_generations': 6}
                elif mixer.lower() == 'broyden':
                    mixer_options = {'mix_param': mix_rate, 'max_iter': max_iter}
                elif mixer.lower() == 'diis':
                    mixer_options = {'init_mix_param': mix_rate, 'n_generations': 6}
                else:
                    mixer_options = {}
            scc_mixer = get_mixer(mixer, **mixer_options)
            log.debug(f"Using {mixer} mixer for SCC iterations with options: {mixer_options}")
        elif isinstance(mixer, SCCMixer):
            scc_mixer = mixer
            log.debug(f"Using provided {type(mixer).__name__} for SCC iterations")
        else:
            raise TypeError(f"mixer must be a string or SCCMixer instance, got {type(mixer)}")


        # SCC iteration loops
        # Following DFTBplus convention:
        # - mulcharge_old stores the MIXED input charge used for the current iteration
        # - q_diff = q_out - q_inp (output minus the actual input)
        # - q_new = mixer.mix(q_inp, q_diff)
        # - Convergence is checked on ||q_diff|| = ||q_out - q_inp||
        while not is_converged and iteration <= max_iter:
            if iteration == max_iter:
                log.warning("Maximum number of iterations reached without convergence.")
                break

            # Calculate Mulliken charges (this gives q_out)
            data = self.mulliken.get_mulcharge(data=data,
                                            kmeshgrid=kmeshgrid,
                                            kmeshspacing=kmeshspacing,
                                            kgamma_center=kgamma_center,
                                            krotational_symmetry=krotational_symmetry,
                                            ktime_inversion_symmetry=ktime_inversion_symmetry,
                                            smearing_method=smearing_method,
                                            nel_atom=nel_atom,
                                            Temp=Temp,
                                            AtomicData_options=AtomicData_options)
            
            if iteration == 0:
                self.elec_H0_bandE = torch.tensor([self.mulliken.elec_bandE], dtype=self.scc_dtype)
                # Initialize the mixer
                scc_mixer.reset(n_elem=len(self.mulliken.mul_charge))
                # First iteration: use output as input for next iteration
                self.mulcharge_old = self.mulliken.mul_charge.copy()
                log.debug(f'Starting self-consistent charge iterations, tolerance is {tol}.')
                iteration += 1

            else: # iteration > 0
                assert self.mulcharge_old is not None, "mulcharge_old should not be None after first iteration."
                # q_out = current Mulliken output
                # q_inp = mulcharge_old (the MIXED input that was used)
                # q_diff = q_out - q_inp (DFTBplus convention)
                q_out = self.mulliken.mul_charge.copy()
                q_inp = self.mulcharge_old
                q_diff = q_out - q_inp

                # Apply the mixer: q_new = q_inp + mixParam * q_diff (DFTBplus convention)
                # The mixed charge becomes the input for the next iteration
                self.mulliken.mul_charge = scc_mixer.mix(q_inp, q_diff)
                self.mulliken.delta_charge = self.mulliken.mul_charge - np.array(self.mulliken.per_atom_charge)

                # Store the MIXED charge as input for next iteration (DFTBplus convention)
                self.mulcharge_old = self.mulliken.mul_charge.copy()

                # Convergence criterion: ||q_diff|| = ||q_out - q_inp|| (DFTBplus convention)
                diff = np.abs(q_diff)
                log.debug(f' ITERATION: {int(iteration)}  Max charge difference: {diff.max().item():.15f} e')
                iteration += 1

                if diff.max() < tol: # Converged and not exiting due to max_iter
                    assert self.scc_shift_energy is not None, "scc_shift_energy should not be None when converged."
                    assert self.elec_H0_bandE is not None, "elec_H0_bandE should not be None when converged."
                    log.debug(f'Convergence reached after {int(iteration)} iters.')
                    is_converged = True
                    self.is_converged = True  # Mark as converged

            
            # Prepare Gamma and inv_r only once
            if self.expGamma is None or self.expGamma_onsite is None:
                self.expGamma, self.expGamma_onsite = get_expgamma(data=data, idp=self.model.idp, skp=self.skp)
            if self.inv_r is None:
                self.inv_r = get_inv_r(data=data)
            if self.Gamma is None:
                self.Gamma = get_Gamma( data=data,
                                        expGamma=self.expGamma, 
                                        expGamma_onsite=self.expGamma_onsite, 
                                        inv_r=self.inv_r)

            # Get scc_shift
            self.Gamma = self.Gamma.to(dtype=self.scc_dtype)
            delta_charge_tensor = torch.from_numpy(self.mulliken.delta_charge).to(dtype=self.scc_dtype)
            self.scc_shift = self.Gamma @ delta_charge_tensor
            self.scc_shift_energy = 0.5 * delta_charge_tensor @ self.scc_shift
            scc_HK = self.cal_scc_hk(data=data,
                                     per_atom_indices=self.mulliken.per_atom_indices,
                                     scc_shift=self.scc_shift)
            
            # Use specified precision for Hamiltonian iteration
            H_iter = data[AtomicDataDict.HAMILTONIAN_KEY].clone().to(dtype=self.scc_cdtype) + scc_HK

            # Calculate eigenvalues and eigenvectors
            # Also convert overlap matrix to specified precision for consistent eigenvalue solver
            overlap_mat_dp = data[AtomicDataDict.OVERLAP_KEY].to(dtype=self.scc_cdtype) if self.overlap else None
            eigvecs, eigvals = self.eigh_solver(H_mat=H_iter,
                                                overlap=self.overlap,
                                                overlap_mat=overlap_mat_dp)

            # Update eigenvalues and eigenvectors in the data dictionary
            data[AtomicDataDict.ENERGY_EIGENVALUE_KEY] = torch.nested.as_nested_tensor([torch.cat(eigvals, dim=0)])
            data[AtomicDataDict.EIGENVECTOR_KEY] = torch.cat(eigvecs, dim=0)
        
        self.data = data
        self.E_fermi = self.mulliken.estimated_E_fermi
        self.elec_bandE = self.mulliken.elec_bandE
        self.elec_totE = self.elec_H0_bandE + self.scc_shift_energy

        log.debug(f'Total electronic energy: {self.elec_totE.item()} eV.')


    def get_total_energy(self,
                         data: Union[ase.Atoms, str],
                         nel_atom: dict,
                         sigma_rep: dict = None,
                         kmeshgrid: Optional[List[int]] = None,
                         kmeshspacing: Optional[List[float]] = None,
                         kgamma_center: bool = True,
                         krotational_symmetry: bool = True,
                         ktime_inversion_symmetry: bool = True,
                         tol: float = 1e-6,
                         mix_rate: float = 0.25,
                         max_iter: int = 50,
                         Temp: float = 300,
                         AtomicData_options: dict = None,
                         smearing_method: str = 'Fermi-Dirac',
                         mixer: Union[str, SCCMixer] = 'simple',
                         mixer_options: Dict[str, Any] = None) -> float:
        '''
        Get the DFTB total energy for specific structures.

        The workflow is as follows:
        1. Read the structure from the provided file path using ASE.
        2. Run the self-consistent charge (SCC) iterations to compute the electronic structure.
        3. Run `calculate_atomic_rep` to compute the repulsive energy.
        4. Sum the total electronic energy and repulsive energy to get the total energy.

        Parameters
        ----------
        data : Union[ase.Atoms, str]
            Input atomic structure. Can be either:
            - An ASE Atoms object containing the structure
            - A file path (str) to a structure file readable by ASE
              (e.g., CIF, POSCAR, XYZ formats)
        nel_atom : dict
            Number of valence electrons for each element type.
            Example: {'C': 4, 'H': 1, 'O': 6}
        sigma_rep : dict
            Dictionary of atomic sigma representations for repulsive energy calculation,
            e.g. {'B': 0.5, 'N': 0.5} in Angstrom units.
        kmeshgrid : Optional[List[int]], optional
            K-point mesh grid for Brillouin zone sampling, specified as
            [nk1, nk2, nk3]. Default is None. Either kmeshgrid or kmeshspacing
            must be provided for k-point generation.
        kmeshspacing : Optional[List[float]], optional
            K-point mesh spacing for automatic grid generation. Specified as
            [dk1, dk2, dk3] in reciprocal space units. Default is None.
            Alternative to kmeshgrid for specifying k-point density.
        kgamma_center : bool, optional
            Whether to use Gamma-centered k-point mesh. Default is True.
        krotational_symmetry : bool, optional
            Whether to apply rotational symmetry to reduce k-points. Default is True.
        ktime_inversion_symmetry : bool, optional
            Whether to apply time-inversion symmetry to reduce k-points. Default is True.
        tol : float, optional
            Convergence tolerance for maximum charge difference between
            iterations. Default is 1e-6.
        mix_rate : float, optional
            Mixing parameter for charge density updates. Default is 0.25.
        max_iter : int, optional
            Maximum number of SCC iterations allowed. Default is 50.
        Temp : float, optional
            Electronic temperature in Kelvin. Default is 300 K.
        AtomicData_options : dict, optional
            Additional options passed to AtomicData constructor.
            It includes 'r_max' for cutoff radius. If None, uses default r_max
            from Slater-Koster parameters.
        mixer : Union[str, SCCMixer], optional
            Charge mixing method for SCC convergence. Can be either a string
            specifying the mixer type or an SCCMixer instance. Available types:
            - 'simple' or 'linear': Simple linear mixing (default)
            - 'anderson' or 'pulay': Anderson/Pulay mixing with history
            - 'broyden': Modified Broyden mixing (Johnson, PRB 38, 12807)
            - 'diis': Direct Inversion in the Iterative Subspace
            Default is 'simple'.
        mixer_options : Dict[str, Any], optional
            Options passed to the mixer constructor when mixer is a string.
            If None, uses default options.

        Returns
        -------
        float
            Total energy including electronic and repulsive contributions in eV.

        Examples
        --------
        >>> from dptb.nn.dftb.dftb_scc import DFTBSCC
        >>>
        >>> # Initialize DFTBSCC with basis and SK parameter path
        >>> basis = {'B': ['2s', '2p'], 'N': ['2s', '2p']}
        >>> sk_path = 'path/to/slakos'
        >>> dftbscc = DFTBSCC(basis=basis, sk_path=sk_path, overlap=True)
        >>>
        >>> # Calculate total energy for a structure with default simple mixer
        >>> total_energy = dftbscc.get_total_energy(
        ...     data='structure.vasp',
        ...     nel_atom={'B': 3, 'N': 5},
        ...     sigma_rep={'B': 0.5, 'N': 0.5},
        ...     kmeshgrid=[20, 20, 1],
        ...     tol=1e-8,
        ...     mix_rate=0.2,
        ...     max_iter=1000
        ... )
        >>> print(f'Total energy: {total_energy:.4f} eV')
        Total energy: -100.5968 eV
        >>>
        >>> # Using Anderson mixer for better convergence
        >>> total_energy = dftbscc.get_total_energy(
        ...     data='structure.vasp',
        ...     nel_atom={'B': 3, 'N': 5},
        ...     sigma_rep={'B': 0.5, 'N': 0.5},
        ...     kmeshgrid=[20, 20, 1],
        ...     mixer='anderson',
        ...     mixer_options={'n_generations': 8, 'mix_param': 0.3}
        ... )
        '''

        # Step 1 & 2: Read structure and run SCC iterations
        self.run_iters(
            data=data,
            nel_atom=nel_atom,
            kmeshgrid=kmeshgrid,
            kmeshspacing=kmeshspacing,
            kgamma_center=kgamma_center,
            krotational_symmetry=krotational_symmetry,
            ktime_inversion_symmetry=ktime_inversion_symmetry,
            tol=tol,
            mix_rate=mix_rate,
            max_iter=max_iter,
            Temp=Temp,
            AtomicData_options=AtomicData_options,
            smearing_method=smearing_method,
            mixer=mixer,
            mixer_options=mixer_options
        )

        if sigma_rep is None:
            repulsive = getattr(self.scc_params, "repulsive", None)
            if isinstance(repulsive, dict):
                sigma_rep = repulsive.get("sigma_rep")
        if sigma_rep is None:
            raise ValueError("Repulsive parameters are required for total energy. run_iters() can be used for electronic SCC without repulsive parameters.")

        # Step 3: Calculate repulsive energy
        _, _, total_rep_energy = calculate_atomic_rep(
            data=self.data,
            idp_sk=self.model.idp_sk,
            sigma_rep=sigma_rep
        )
        self.total_rep_energy = total_rep_energy
        # Step 4: Sum electronic and repulsive energies
        total_energy = self.elec_totE + total_rep_energy
        self.total_energy = total_energy

        return total_energy.item()
        

    def cal_scc_hk(self,
                   data: AtomicDataDict,
                   per_atom_indices: np.ndarray,
                   scc_shift: torch.Tensor) -> torch.Tensor:
        '''
        Calculate self-consistent charge correction to the Hamiltonian in k-space.

        This method computes the SCC contribution to the Hamiltonian matrix
        H_SCC, which accounts for the electrostatic interaction between charge
        fluctuations. The correction is applied element-wise based on the
        atomic charge shifts and orbital overlaps.

        For matrix elements in assignment process:
        - Diagonal (on-site): H_SCC[i,i] = 0.5 * shift[atom_i]
        - Off-diagonal (same atom): H_SCC[i,j] = S[i,j] * shift[atom_i]
        - Off-diagonal (different atoms):
          H_SCC[i,j] = 0.5 * (shift[atom_i] + shift[atom_j]) * S[i,j]
        And then symmetrized to ensure Hermiticity:
        H_SCC = H_SCC + H_SCC†

        where shift is the SCC potential and S is the overlap matrix.

        Parameters
        ----------
        data : AtomicDataDict
            Atomic data dictionary containing Hamiltonian and overlap matrices.
            Must include keys:
            - AtomicDataDict.HAMILTONIAN_KEY: k-space Hamiltonian
            - AtomicDataDict.OVERLAP_KEY: k-space overlap matrix
        per_atom_indices : np.ndarray
            Cumulative indices marking orbital boundaries for each atom.
            Shape: (n_atoms + 1,). For example, if atoms have [2, 3, 4]
            orbitals, per_atom_indices = [0, 2, 5, 9].
        scc_shift : torch.Tensor
            Self-consistent charge potential shift for each atom.
            Shape: (n_atoms,). Units: eV or Hartree depending on parameters.

        Returns
        -------
        torch.Tensor
            SCC correction matrix to add to the Hamiltonian. Shape matches
            the input Hamiltonian: (n_kpts, n_orbitals, n_orbitals).
            The matrix is Hermitian (H_SCC = H_SCC†).
        '''

        assert AtomicDataDict.HAMILTONIAN_KEY in data, "Hamiltonian key not found in data."
        assert AtomicDataDict.OVERLAP_KEY in data, "Overlap key not found in data."
        assert per_atom_indices is not None, "per_atom_indices must be provided."
        assert scc_shift is not None, "scc_shift must be provided."

        nk = data[AtomicDataDict.HAMILTONIAN_KEY].shape[0]
        norb = data[AtomicDataDict.HAMILTONIAN_KEY].shape[1]
        n_atoms = len(per_atom_indices) - 1
        device = scc_shift.device

        # Build orbital-to-atom mapping: orb2atom[orb_idx] = atom_idx
        # Use cached version if provided, otherwise compute
        orb2atom_np = np.repeat(np.arange(n_atoms), np.diff(per_atom_indices))
        orb2atom = torch.from_numpy(orb2atom_np).to(device)

        # Get shift value for each orbital based on its parent atom
        shift_per_orb = scc_shift[orb2atom]  # (norb,)

        # Get overlap matrix in specified precision for numerical accuracy
        overlap_data = data[AtomicDataDict.OVERLAP_KEY].to(dtype=self.scc_cdtype)

        # Create index arrays for vectorized operations
        idx = torch.arange(norb, device=device)

        # Atom assignments for each orbital pair (i, j)
        atom_i = orb2atom[:, None]  # (norb, 1)
        atom_j = orb2atom[None, :]  # (1, norb)

        # Boolean masks for different cases
        same_atom = (atom_i == atom_j)  # (norb, norb)
        diag_mask = (idx[:, None] == idx[None, :])  # (norb, norb)
        upper_tri_off_diag = (idx[:, None] < idx[None, :])  # strict upper triangle

        # Build coefficient matrix for upper triangle (including diagonal)
        # shift_i and shift_j are the shifts for orbitals i and j
        shift_i = shift_per_orb[:, None]  # (norb, 1)
        shift_j = shift_per_orb[None, :]  # (1, norb)

        # Default coefficient: 0.5 * (shift_i + shift_j) for different atoms
        coeff = 0.5 * (shift_i + shift_j)

        # For same atom off-diagonal: use shift_i only (not averaged)
        coeff = torch.where(same_atom & ~diag_mask, shift_i.expand(norb, norb), coeff)

        # For diagonal: 0.5 * shift_i (will be doubled by symmetrization to get 1.0 * shift)
        coeff = torch.where(diag_mask, 0.5 * shift_i.expand(norb, norb), coeff)

        # Convert coefficient to complex dtype to match scc_HK dtype
        coeff = coeff.to(dtype=self.scc_cdtype)

        # Initialize scc_HK for upper triangle computation
        scc_HK = torch.zeros((nk, norb, norb), dtype=self.scc_cdtype, device=device)

        # Fill diagonal elements (coefficient only, no overlap multiplication)
        scc_HK[:, idx, idx] = coeff[idx, idx]

        # Fill off-diagonal upper triangle elements (coefficient * overlap)
        # Get row and column indices for strict upper triangle
        row_idx, col_idx = torch.where(upper_tri_off_diag)
        scc_HK[:, row_idx, col_idx] = overlap_data[:, row_idx, col_idx] * coeff[row_idx, col_idx]

        # Symmetrize: add conjugate transpose to make Hermitian
        # This doubles the diagonal and fills the lower triangle
        scc_HK = scc_HK + scc_HK.transpose(1, 2).conj()
        scc_HK = scc_HK.contiguous()

        return scc_HK
    
    @staticmethod
    def eigh_solver(H_mat: torch.Tensor,
                    overlap: bool = False,
                    overlap_mat: torch.Tensor = None) -> tuple:
        '''
        Solve the generalized or standard eigenvalue problem for the Hamiltonian.

        This method solves either the standard eigenvalue problem (H|ψ⟩ = E|ψ⟩)
        or the generalized eigenvalue problem (H|ψ⟩ = ES|ψ⟩) where H is the
        Hamiltonian, S is the overlap matrix, E are eigenvalues, and |ψ⟩ are
        eigenvectors.

        For the generalized problem, the Cholesky orthogonalization method is
        used:
        1. Cholesky decomposition: S = L L†
        2. Transform Hamiltonian: H' = L⁻¹ H L⁻†
        3. Solve standard problem: H'|ψ'⟩ = E|ψ'⟩
        4. Back-transform eigenvectors: |ψ⟩ = L⁻†|ψ'⟩

        Parameters
        ----------
        H_mat : torch.Tensor
            Hamiltonian matrix in k-space. Shape: (n_kpts, n_orbitals, n_orbitals).
            Must be Hermitian (H = H†). Can be complex-valued.
        overlap : bool, optional
            Whether to solve generalized eigenvalue problem with overlap matrix.
            If False, solves standard eigenvalue problem H|ψ⟩ = E|ψ⟩.
            Default is False.
        overlap_mat : torch.Tensor, optional
            Overlap matrix S. Required when overlap=True. Shape must match H_mat.
            Must be positive definite and Hermitian. Default is None.

        Returns
        -------
        tuple of (list, list)
            eigvecs : list of torch.Tensor
                List containing one tensor of eigenvectors. Shape of tensor:
                (n_kpts, n_orbitals, n_orbitals). Each column is an eigenvector.
                Eigenvectors are orthonormal: ⟨ψ_i|ψ_j⟩ = δ_ij (or ⟨ψ_i|S|ψ_j⟩ = δ_ij
                for generalized case).
            eigvals : list of torch.Tensor
                List containing one tensor of eigenvalues. Shape: (n_kpts, n_orbitals).
                Eigenvalues are sorted in ascending order. Units: eV or Hartree.


        '''

        if overlap and overlap_mat is None:
            raise ValueError("Overlap matrix must be provided when overlap is True.")

        eigvecs, eigvals = [],[]
        if overlap:
            chklowt = torch.linalg.cholesky(overlap_mat)
            chklowtinv = torch.linalg.inv(chklowt)
            H_mat = (
                chklowtinv @ H_mat @ torch.transpose(chklowtinv,dim0=1,dim1=2).conj()
                )
        eigval, eigvec = torch.linalg.eigh(H_mat)
        if overlap:
            eigvec = torch.transpose(
                torch.transpose(chklowtinv,dim0=1,dim1=2).conj() @ eigvec,
                dim0=1,dim1=2)
                        
        eigvecs.append(eigvec)
        eigvals.append(eigval)

        return eigvecs, eigvals

    # @staticmethod
    # def cal_ovp_r0(data: AtomicDataDict,
    #                model: DFTBSK) -> AtomicDataDict:
        
    #     from dptb.data.interfaces.ham_to_feature import feature_to_block
    #     ovp_R_dict = feature_to_block(data=data, idp=model.idp, overlap=True)       
        
    #     atom_id_to_indices = {}
    #     ist = 0
    #     for idx in data[AtomicDataDict.ATOM_TYPE_KEY].flatten():
    #         onsite_idx  = str(idx.item()) + '_' + str(idx.item()) + '_0_0_0'
    #         assert onsite_idx in ovp_R_dict, f"Onsite block {onsite_idx} not found in ovp_R_dict."
    #         atom_id_to_indices[idx.item()] = slice(ist, ist+ ovp_R_dict[onsite_idx].shape[0])
    #         ist += ovp_R_dict[onsite_idx].shape[0]

    #     ovp_R0_all = torch.zeros((ist, ist), dtype=torch.complex64)

    #     for iatom_id in atom_id_to_indices.keys():
    #         for jatom_id in atom_id_to_indices.keys():

    #             block_idx =  '_'.join(map(str, map(int, [iatom_id, jatom_id] \
    #                             + list(torch.zeros_like(data[AtomicDataDict.EDGE_CELL_SHIFT_KEY][0]).cpu().numpy()))))
    #             slice_i = atom_id_to_indices[int(block_idx.split('_')[0])]
    #             slice_j = atom_id_to_indices[int(block_idx.split('_')[1])] 

    #             if block_idx not in ovp_R_dict.keys():
    #                 block_idx =  '_'.join(map(str, map(int, [jatom_id, iatom_id] \
    #                             + list(torch.zeros_like(data[AtomicDataDict.EDGE_CELL_SHIFT_KEY][0]).cpu().numpy()))))
    #                 assert block_idx in ovp_R_dict.keys(), f"Block {block_idx} not found in ovp_R."
    #                 ovp_R0_all[slice_i, slice_j] = ovp_R_dict[block_idx].T.conj()
    #             else:
    #                 ovp_R0_all[slice_i, slice_j] = ovp_R_dict[block_idx]


class DFTBSCC(SKSCC):
    def __init__(self,
                 basis: dict,
                 sk_path: str,
                 smooth_ski: bool = False,
                 overlap: bool = True,
                 scc_dtype: torch.dtype = torch.float64) -> None:
        log.warning("DFTBSCC(basis=..., sk_path=...) is a compatibility wrapper; use SKSCC(model=..., params=...) for new code.")
        model = DFTBSK(basis=basis,
                       skdata=sk_path,
                       overlap=overlap,
                       smooth_ski=smooth_ski,
                       dtype=scc_dtype)
        skp = SKParam(basis=basis, skdata=sk_path, cal_rcuts=True, dtype=scc_dtype)
        params = SCCParams.from_skparam(skp)
        super().__init__(model=model,
                         params=params,
                         overlap=overlap,
                         scc_dtype=scc_dtype)
