import numpy as np
from ase.io import read
import ase
from typing import Union
import torch
from typing import Optional
import logging
log = logging.getLogger(__name__)
from dptb.data import AtomicData, AtomicDataDict
from dptb.nn.energy import Eigenvalues, Eigh
from dptb.utils.constants import kB_eV_per_K
from dptb.utils.occupy import calculate_fermi_level, ffd, dffd, fgau
from dptb.utils.ksampling import sample as ksampling
from dptb.postprocess.common import load_data_for_model

# This class `ElecStruCal`  is designed to calculate electronic structure properties such as
# eigenvalues and Fermi energy based on provided input data and model.
# It serve as a basic post-processing class to load data and provide Fermi energy.

class ElecStruCal(object):
    def __init__ (
            self,
            model: torch.nn.Module,
            device: Union[str, torch.device]=None,
            eig_method: str='eigvalsh' # 'eigh' or 'eigvalsh'
            ):
        '''It initializes ElecStruCal object with a neural network model, optional results path, GUI
        usage flag, and device information, and sets up eigenvalues  based on model properties.

        Parameters
        ----------
        model : torch.nn.Module
            The `model` parameter is expected to be an instance of `torch.nn.Module` that you want to load.
        device : Union[str, torch.device]
            The `device` parameter in the `__init__` function is used to specify the device on which the model
        will be loaded and run. It can be either a string representing the device (e.g., 'cpu' or 'cuda') or
        a torch.device object.
        eig_method : str
            The `eig_method` parameter in the `__init__` function specifies the method to be used for solving
        the eigenvalue problem. It can take two possible values: 'eigvalsh' or 'eigh'.  'eigvalsh' is used for
        calculating eigenvalues only, while 'eigh' is used for calculating both eigenvalues and eigenvectors.
        The default value is 'eigvalsh'.

        '''
        if  device is None:
            device = model.device
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        if eig_method not in ['eigvalsh', 'eigh']:
            raise ValueError(f'Unknown eig_method: {eig_method}, should be either "eigvalsh" or "eigh".')
        self.eig_method = eig_method

        self.model = model
        self.model.eval()
        self.overlap = hasattr(model, 'overlap')

        if not self.model.transform:
            log.error('The model.transform is not True, please check the model.')
            raise RuntimeError('The model.transform is not True, please check the model.')

        self.eig_solver = self._make_eig_solver(with_overlap=self.overlap)

    def _make_eig_solver(self, with_overlap: bool = False):
        solver_cls = {
            'eigvalsh': Eigenvalues,
            'eigh': Eigh,
        }[self.eig_method]
        solver_kwargs = {
            'idp': self.model.idp,
            'device': self.device,
            'dtype': self.model.dtype,
        }
        if with_overlap:
            solver_kwargs.update({
                's_edge_field': AtomicDataDict.EDGE_OVERLAP_KEY,
                's_node_field': AtomicDataDict.NODE_OVERLAP_KEY,
                's_out_field': AtomicDataDict.OVERLAP_KEY,
            })
        return solver_cls(**solver_kwargs)

    def get_data(self,
                 data: Union[AtomicData, ase.Atoms, str],
                 pbc:Union[bool,list]=None,
                 device: Union[str, torch.device]=None,
                 AtomicData_options:dict=None,
                 override_overlap:Optional[str]=None):
        '''The function `get_data` takes input data in the form of a string, ase.Atoms object, or AtomicData
        object, processes it accordingly, and returns the AtomicData class.

        Parameters
        ----------
        data : Union[AtomicData, ase.Atoms, str]
            The `data` parameter in the `get_data` function can be one of the following types:
        string, ase.Atoms object, or AtomicData object.
        AtomicData_options : dict
            The `AtomicData_options` parameter is a dictionary that contains options or configurations for
        creating an `AtomicData` object from an `ase.Atoms` object.
        device : Union[str, torch.device]
            The `device` parameter in the `get_data` function is used to specify the device on which the data
        should be processed. If no device is provided, it defaults to `self.device`.
        override_overlap : the path for overlap.h5 to use and override overlap matrix from model.

        Returns
        -------
            the loaded AtomicData object.

        '''
        return load_data_for_model(
            data=data,
            model=self.model,
            device=device if device else self.device,
            pbc=pbc,
            AtomicData_options=AtomicData_options,
            override_overlap=override_overlap
        )


    def get_eigs(self,
                 data: Union[AtomicData, ase.Atoms, str],
                 klist: np.ndarray,
                 pbc:Union[bool,list]=None,
                 AtomicData_options:dict=None,
                 override_overlap:Optional[str]=None,
                 eig_solver:Optional[str]=None,
                 ill_threshold:Optional[float]=None,
                 ill_pad_value:float=1e4):
        '''This function calculates eigenvalues for Hk at specified k-points.

        Parameters
        ----------
        data : Union[AtomicData, ase.Atoms, str]
            The `data` parameter in the `get_eigs` function can be of type `AtomicData`, `ase.Atoms`, or `str`.
        klist : np.ndarray
            The `klist` parameter in the `get_eigs` function is expected to be a numpy array containing a list
        of k-points. These k-points are used to calculate the eigenvalues of the system.
        AtomicData_options : dict
            The `AtomicData_options` parameter is a dictionary that contains options for configuring the
        `AtomicData` object.
        override_overlap : the path for overlap.h5 to use and override overlap matrix from model.
        ill_threshold : optional float
            If set, project out overlap modes whose eigenvalues are below this threshold.
        ill_pad_value : float
            Padding value used for projected-out eigenvalues to preserve the dense band shape.

        Returns
        -------
            The function `get_eigs` returns the loaded data and the energy eigenvalues as a numpy array.

        '''

        data  = self.get_data(data=data, pbc=pbc, device=self.device,AtomicData_options=AtomicData_options, override_overlap=override_overlap)
        for key, value in list(data.items()):
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                data[key] = value.to(dtype=self.model.dtype)
        # set the kpoint of the AtomicData
        data[AtomicDataDict.KPOINT_KEY] = \
            torch.nested.as_nested_tensor([torch.as_tensor(klist, dtype=self.model.dtype, device=self.device)])
        if isinstance(override_overlap, str):
            override_overlap_edge = data[AtomicDataDict.EDGE_OVERLAP_KEY]
            override_overlap_node = data[AtomicDataDict.NODE_OVERLAP_KEY]
        # get the eigenvalues
        data = self.model(data)
        if isinstance(override_overlap, str):
            data[AtomicDataDict.EDGE_OVERLAP_KEY] = override_overlap_edge
            data[AtomicDataDict.NODE_OVERLAP_KEY] = override_overlap_node
        if self.overlap or isinstance(override_overlap, str):
            assert data.get(AtomicDataDict.EDGE_OVERLAP_KEY) is not None
        eig_solver_obj = self.eig_solver
        if override_overlap is not None and not self.overlap:
            eig_solver_obj = self._make_eig_solver(with_overlap=True)
        if isinstance(eig_solver_obj, Eigenvalues):
            data = eig_solver_obj(
                data,
                eig_solver=eig_solver,
                ill_threshold=ill_threshold,
                ill_pad_value=ill_pad_value,
            )
        else:
            if ill_threshold is not None:
                log.warning("ill_threshold is ignored when eig_method='eigh'.")
            data = eig_solver_obj(data, eig_solver=eig_solver)

        # if self.eig_method == 'eigh', the eigenvectors are calculated.
        # The eigenvectors are stored in data[AtomicDataDict.EIGENVECTOR_KEY].detach().cpu().numpy()
        return data, data[AtomicDataDict.ENERGY_EIGENVALUE_KEY][0].detach().cpu().numpy()


    def get_fermi_level(self,
                        data: Union[AtomicData, ase.Atoms, str],
                        nel_atom: dict,
                        meshgrid: list = None,
                        klist: np.ndarray=None,
                        wk: np.ndarray=None,
                        pbc:Union[bool,list]=None,
                        AtomicData_options:dict=None,
                        q_tol:float=1e-6,
                        smearing_method:str='FD',
                        temp:float=300,
                        eig_solver:Optional[str]=None,
                        ill_threshold:Optional[float]=None,
                        ill_pad_value:float=1e4):
        '''This function calculates the Fermi level based on provided data with iteration method, electron counts per atom, and
        optional parameters like specific k-points and eigenvalues.

        Parameters
        ----------
        data : Union[AtomicData, ase.Atoms, str]
            The `data` parameter in the `get_fermi_level` method can accept an instance of `AtomicData`,
        `ase.Atoms`, or a string.
        nel_atom : dict
            The `nel_atom` parameter is a dictionary that contains the number of valence electrons for each
        atom type in your system. It is used to calculate the Fermi level based on the total number of
        valence electrons specified for each atom type.
        kmesh : list
            The `kmesh` parameter is used to specify the k-point mesh for sampling in the Brillouin zone. It is
        a list that defines the mesh grid for k-point sampling. If `klist` is not provided, the k-points
        will be generated based on this mesh.
        klist : np.ndarray
            The `klist` parameter is a numpy array that contains a list of k-points in the Brillouin zone. It
        is used in the calculation of the Fermi level in the provided function `get_fermi_level`.
        Note that if `klist` and kmesh are both provided, the `klist` parameter will be used to calculate the Fermi level.
        wk : np.ndarray
            The `wk` parameter is a numpy array that contains the k-point weights corresponding to `klist`.
        These weights are used for proper Brillouin zone integration, especially when k-point symmetry
        reduction is applied. If `wk` is None and `klist` is provided, uniform weights (1/nk) will be used.
        AtomicData_options : dict
            The `AtomicData_options` parameter in the `get_fermi_level` method is a dictionary that allows you
        to pass additional options or settings related to Atomicdata processing.
        eigenvalues : np.ndarray
            The `eigenvalues` parameter in the `get_fermi_level` method is an optional parameter that allows
        you to provide pre-calculated eigenvalues for the system. If `eigenvalues` is provided, the method
        will use these provided eigenvalues directly. Otherwise, the eigenvalues will be calculated from the model
        on the specified k-points (from kmesh or klist).
        q_tol: float
            The `q_tol` parameter in the `get_fermi_level` function represents the tolerance level for the
        calculated charge compared to the total number of electrons.
        smearing_method : str
            The `smearing_method` parameter in the `get_fermi_level` function is used to specify the method of
        smearing to be used in the calculation of the Fermi energy. The default method is 'FD' (Fermi-Dirac).
        Other possible methods include 'Gaussian'.
        temp : float
            The `temp` parameter in the `get_fermi_level` function represents the temperature for smearing in the
        calculation of the Fermi energy.

        Returns
        -------
            The function `get_fermi_level` returns two values: `data` and `E_fermi`.

        '''


        assert meshgrid is not None or klist is not None, 'kmesh or klist should be provided.'
        assert isinstance(nel_atom, dict)

        # klist would be used if provided, otherwise kmesh would be used to generate klist
        if klist is None:
            # Extract structure from data for k-sampling
            if isinstance(data, str):
                structase = read(data)
            elif isinstance(data, ase.Atoms):
                structase = data
            elif isinstance(data, AtomicData):
                structase = data.to("cpu").to_ase()
            else:
                raise ValueError('data should be either a string, ase.Atoms, or AtomicData')

            klist, wk = ksampling(structase,
                                  meshgrid=meshgrid,
                                  gamma_centered=True,
                                  rotational_symmetry=False,
                                  time_inversion_symmetry=True)
            log.info(f'KPOINTS  kmesh sampling: {klist.shape[0]} kpoints')
        else:
            if wk is None:
                wk = np.ones(klist.shape[0])/klist.shape[0]
            log.info(f'KPOINTS  klist: {klist.shape[0]} kpoints')

        # eigenvalues would be used if provided, otherwise the eigenvalues would be calculated from the model on the specified k-points
        if not AtomicDataDict.ENERGY_EIGENVALUE_KEY in data:
            data, eigs = self.get_eigs(data=data, klist=klist, pbc=pbc,
                                       AtomicData_options=AtomicData_options,
                                       eig_solver=eig_solver,
                                       ill_threshold=ill_threshold,
                                       ill_pad_value=ill_pad_value)
            log.info('Getting eigenvalues from the model.')
        else:
            log.info('The eigenvalues are already in data. will use them.')
            eigs = data[AtomicDataDict.ENERGY_EIGENVALUE_KEY][0].detach().cpu().numpy()
        eigs_valid_mask = data.get(AtomicDataDict.EIGENVALUE_VALID_MASK_KEY)
        if eigs_valid_mask is not None:
            eigs_valid_mask = eigs_valid_mask[0].detach().cpu().numpy()

        if nel_atom is not None:
            atomtype_list = data[AtomicDataDict.ATOM_TYPE_KEY].flatten().tolist()
            atomtype_symbols = np.asarray(self.model.idp.type_names)[atomtype_list].tolist()
            total_nel = np.array([nel_atom[s] for s in atomtype_symbols]).sum()
            if hasattr(self.model,'soc_param'):
                spindeg = 1 # SOC case
            else:
                spindeg = 2 # spin-degenerate
            if eigs_valid_mask is not None:
                eigs = np.where(eigs_valid_mask, eigs, ill_pad_value)
            E_fermi, occ, diff_ne, eband, eband_free = self.cal_E_fermi_advanced(eigs,
                                                                                total_nel,
                                                                                spindeg,
                                                                                wk,
                                                                                q_tol= q_tol,
                                                                                smearing_method = smearing_method,
                                                                                temp=temp)
            log.info(f'Estimated E_fermi: {E_fermi} based on the valence electrons setting nel_atom : {nel_atom} .')

        else:
            E_fermi = None
            raise RuntimeError('nel_atom should be provided to calculate Fermi energy.')


        if self.eig_method == 'eigh':
            return data, E_fermi, eband, occ
        return data, E_fermi



    @classmethod
    def cal_E_fermi(cls,eigenvalues: np.ndarray, total_electrons: int, spindeg: int=2,wk: np.ndarray=None,
                    q_tol:float=1e-10,smearing_method:str='FD',temp:float=300,
                    eigenvalue_valid_mask: np.ndarray=None):
        '''This  function calculates the Fermi energy using iteration algorithm.

            In this version, the function calculates the Fermi energy in the case of spin-degeneracy.
        The smearing method here is to ensure the convergence of the Fermi energy calculation especially in metal systems.
        The detailed description of the smearing methods can be found in dos Santos, F. J. and N. Marzari (2023). "Fermi energy
        determination for advanced smearing techniques." Physical Review B 107(19): 195122.

        Parameters
        ----------
        eigenvalues : np.ndarray
            The `eigenvalues` parameter is expected to be a NumPy array containing the eigenvalues of the system.
        total_electrons : int
            The `total_electrons` parameter represents the total number of electrons in the system. It is used
        in the calculation of the Fermi energy.
        spindeg : int, optional
            The `spindeg` parameter in the `cal_E_fermi` method represents the spin degeneracy factor, which is
        typically equal to 2 for systems with spin-degeneracy.
        wk : np.ndarray
            The `wk` parameter in the `cal_E_fermi` function represents the weights assigned to each kpoints
        in the calculation. If `wk` is not provided by the user, the function assigns equal weight to each
        kpoint for the calculation of the Fermi energy.
        q_tol: float
            The `q_tol` parameter in the `cal_E_fermi` function represents the tolerance level for the
        calculated charge compared to the total number of electrons.
        smearing_method : str
            The `smearing_method` parameter in the `cal_E_fermi` function is used to specify the method of
        smearing to be used in the calculation of the Fermi energy. The default method is 'FD' (Fermi-Dirac).
        Other possible methods include 'Gaussian'.
        temp : float
            The `temp` parameter in the `cal_E_fermi` function represents the temperature for smearing in the
        calculation of the Fermi energy.

        Returns
        -------
            The Fermi energy `Ef`

        '''

        nextafter = np.nextafter
        total_electrons = total_electrons / spindeg # This version is for the case of spin-degeneracy
        log.info('Calculating Fermi energy in the case of spin-degeneracy.')


        if eigenvalue_valid_mask is not None:
            eigenvalue_valid_mask = np.asarray(eigenvalue_valid_mask, dtype=bool)
            if eigenvalue_valid_mask.shape != eigenvalues.shape:
                raise ValueError(
                    "eigenvalue_valid_mask should have the same shape as eigenvalues, "
                    f"got {eigenvalue_valid_mask.shape} and {eigenvalues.shape}."
                )
            valid_eigenvalues = eigenvalues[eigenvalue_valid_mask]
            if valid_eigenvalues.size == 0:
                raise ValueError("No valid eigenvalues are available to calculate Fermi energy.")
            occupation_mask = eigenvalue_valid_mask.astype(eigenvalues.dtype)
        else:
            valid_eigenvalues = eigenvalues
            occupation_mask = 1.0

        # calculate boundaries
        min_Ef, max_Ef = valid_eigenvalues.min(), valid_eigenvalues.max()
        kT = kB_eV_per_K * temp
        drange = kT*np.sqrt(-np.log(q_tol*1e-2))
        min_Ef = min_Ef - drange
        max_Ef = max_Ef + drange

        Ef = (min_Ef + max_Ef) * 0.5

        if wk is None:
            wk = np.ones(eigenvalues.shape[0]) / eigenvalues.shape[0]
            log.info('wk is not provided, using equal weight for kpoints.')

        icounter = 0
        while nextafter(min_Ef, max_Ef) < max_Ef:
        # while icounter <= 150:
            icounter += 1
            # Calculate guessed charge
            wk = wk.reshape(-1,1)
            if smearing_method == "Fermi-Dirac" or smearing_method == "FD":
                q_cal = (wk * cls.fermi_dirac_smearing(eigenvalues,kT=kT, mu=Ef) * occupation_mask).sum()
            elif smearing_method == "Gaussian" or smearing_method == "G":
                q_cal = (wk * cls.Gaussian_smearing(eigenvalues,sigma = kT, mu=Ef) * occupation_mask).sum()
            else:
                raise ValueError(f'Unknown smearing method: {smearing_method}')

            if abs(q_cal - total_electrons) < q_tol:
                log.info(f'Fermi energy converged after {icounter} iterations.')
                log.info(f'q_cal: {q_cal*spindeg}, total_electrons: {total_electrons*spindeg}, diff q: {abs(q_cal - total_electrons)*spindeg}')
                return Ef

            if q_cal >= total_electrons:
                max_Ef = Ef
            else:
                min_Ef = Ef
            Ef = (min_Ef + max_Ef) * 0.5

        log.warning(f'Fermi level bisection did not converge under tolerance {q_tol} after {icounter} iterations.')
        log.info(f'q_cal: {q_cal*spindeg}, total_electrons: {total_electrons*spindeg}, diff q: {abs(q_cal - total_electrons)*spindeg}')
        return Ef



    @classmethod
    def cal_E_fermi_advanced(cls,
                            eigenvalues: np.ndarray,
                            total_electrons: int,
                            spindeg: int=2,
                            wk: np.ndarray=None,
                            q_tol: float=1e-6,
                            smearing_method:str='FD',
                            temp:float=300):
        '''Calculate Fermi energy using scipy.optimize.brent with optional refinement.

        This is an advanced implementation that combines scipy's Brent method with
        Newton-Raphson polishing and occ rescaling to achieve highly accurate
        Fermi energy and occupation numbers.

        The method implements a multi-stage approach:
        1. Initial Fermi energy search using scipy.optimize.brent (via
           calculate_fermi_level from dptb.utils.occupy)
        2. If charge error exceeds q_tol, Newton-Raphson polishing refines Ef
           (Fermi-Dirac only) for rapid quadratic convergence
        3. Final rescaling of occupation numbers ensures exact electron count

        This stage-2 and stage-3 algorithm is inspired by DFTB+ (src/dftbp/dftb/etemp.F90).
        For smearing technique details, see dos Santos, F. J. and N. Marzari (2023).
        "Fermi energy determination for advanced smearing techniques."
        Physical Review B 107(19): 195122.

        Parameters
        ----------
        eigenvalues : np.ndarray
            Eigenvalues with shape (nk, nbands). The eigenvalues of the system
            at each k-point.
        total_electrons : int
            Total number of electrons in the system. Used to determine occupation.
        spindeg : int, optional
            Spin degeneracy factor: 2 for spin-degenerate systems (each band holds
            2 electrons), 1 for SOC systems (each band holds 1 electron).
            Default is 2.
        wk : np.ndarray, optional
            K-point weights with shape (nk,). Must sum to 1. If not provided,
            equal weights are assigned to all k-points. Default is None.
        q_tol : float, optional
            Tolerance for charge convergence. Default is 1e-6.
        smearing_method : str, optional
            Smearing method to use. Supported: 'FD'/'Fermi-Dirac' for Fermi-Dirac,
            'Gaussian'/'G' for Gaussian smearing. Default is 'FD'.
        temp : float, optional
            Temperature in Kelvin for smearing width calculation (sigma = kT).
            Default is 300.

        Returns
        -------
        Ef : float
            Fermi energy in eV. After Newton-Raphson polishing (if applicable)
            and rescaling, this produces an occupation that matches the target
            electron count.
        occ : np.ndarray
            Occupation numbers with shape (nk, nbands). These include the spin
            degeneracy factor, so values are in the range [0, spindeg]. They are
            rescaled to ensure the k-point-weighted sum exactly equals
            total_electrons.
        diff_ne : float
            Electron count difference before rescaling: |N_calc - N_target|.
            This represents the convergence quality of the Fermi energy search.
            A value exceeding q_tol triggers Newton-Raphson refinement.
        eband : float
            Band energy (sum of occupied eigenvalues weighted by occupation and
            k-point weights) in eV. Computed as sum(eigenvalues * occ * wk).
            This is the total electronic kinetic energy contribution.
        eband_free : float
            Free band energy (entropy-corrected band energy) in eV. For finite
            temperature calculations, this includes the -TS term where S is the
            electronic entropy from the smearing function. At T=0, eband_free
            equals eband.

        Examples
        --------
        >>> from dptb.postprocess.elec_struc_cal import ElecStruCal
        >>>
        >>> # Create eigenvalues for 8 k-points, 10 bands each
        >>> eigs = np.random.uniform(-5, 5, (8, 10))
        >>>
        >>> # Calculate Fermi level for a system with 12 electrons
        >>> ef = ElecStruCal.cal_E_fermi_advanced(
        ...     eigenvalues=eigs,
        ...     total_electrons=12,
        ...     spindeg=2,  # spin-degenerate system
        ...     smearing_method='FD',
        ...     temp=300
        ... )
        >>> print(f"Fermi energy: {ef:.4f} eV")

        With custom k-point weights and Gaussian smearing:

        >>> # Non-uniform k-point weights (must sum to 1)
        >>> wk = np.array([0.125, 0.125, 0.125, 0.125, 0.25, 0.125, 0.0625, 0.0625])
        >>>
        >>> ef = ElecStruCal.cal_E_fermi_advanced(
        ...     eigenvalues=eigs,
        ...     total_electrons=12,
        ...     spindeg=2,
        ...     wk=wk,
        ...     smearing_method='Gaussian',
        ...     temp=500  # Higher temperature
        ... )

        For SOC (spin-orbit coupling) systems:

        >>> # SOC system: spindeg=1 (no spin degeneracy, each band holds 1 electron)
        >>> eigs_soc = np.random.uniform(-4, 4, (10, 15))
        >>>
        >>> ef_soc = ElecStruCal.cal_E_fermi_advanced(
        ...     eigenvalues=eigs_soc,
        ...     total_electrons=15,  # fills 15 bands
        ...     spindeg=1,  # SOC case
        ...     smearing_method='FD',
        ...     temp=300
        ... )

        Integration with ElecStruCal for full band structure calculations:

        >>> from dptb.nn import build_model
        >>>
        >>> # Load trained model
        >>> model = build_model(checkpoint='path/to/model.pth')
        >>> nel_atom = {"Si": 4}  # Silicon with 4 valence electrons
        >>>
        >>> # Create ElecStruCal instance
        >>> elec_cal = ElecStruCal(model=model, device='cpu')
        >>>
        >>> # Calculate Fermi level for a structure
        >>> data, efermi, _ = elec_cal.get_fermi_level(
        ...     data='POSCAR',
        ...     nel_atom=nel_atom,
        ...     meshgrid=[12, 12, 12],
        ...     smearing_method='FD',
        ...     temp=300
        ... )
        >>> # cal_E_fermi_advanced is called internally by get_fermi_level

        Notes
        -----
        **Algorithm Overview:**
        - This is a wrapper around dptb.utils.occupy.calculate_fermi_level
        - Uses scipy.optimize.brent instead of bisection for faster convergence
        - Produces identical results to cal_E_fermi for both spindeg=1 and spindeg=2

        **Internal Implementation:**
        - calculate_fermi_level uses nspin=1 with internal g=2 factor
        - For spindeg=2: ne = total_electrons (matches internal g=2)
        - For spindeg=1: ne = 2*total_electrons (compensates for internal g=2)
        - The q_tol parameter triggers Newton-Raphson refinement when exceeded

        **Occupation convention:**
        The returned `occ` includes the spin degeneracy factor, i.e., fully
        occupied bands have occ=spindeg. This is consistent with
        `dptb.utils.occupy.calculate_fermi_level` and expected by downstream
        consumers like `dptb.postprocess.charge_pop`.

        **References:**
        - DFTB+ src/dftbp/dftb/etemp.F90 for NR polishing and rescaling

        See Also
        --------
        cal_E_fermi : Original bisection-based implementation
        dptb.utils.occupy.calculate_fermi_level : Underlying implementation
        '''


        #TODO: refactor and test spindeg=1 case, now it is assumed to be correct.

        # Input validation
        assert eigenvalues.ndim == 2, \
            f"Eigenvalues must be 2D (nk, nbands), got shape {eigenvalues.shape}"
        assert spindeg in [1, 2], \
            f"spindeg must be 1 or 2, got {spindeg}"
        assert total_electrons > 0, \
            f"total_electrons must be positive, got {total_electrons}"

        nk, nbands = eigenvalues.shape

        # Set up k-point weights
        if wk is None:
            wk = np.ones(nk) / nk
            log.info('wk is not provided, using equal weight for kpoints.')
        else:
            assert wk.shape == (nk,), \
                f"wk shape mismatch: expected ({nk},), got {wk.shape}"

        # Reshape eigenvalues from (nk, nbands) to (nspin, nk, nbands)
        # For both spin-degenerate (spindeg=2) and SOC (spindeg=1) cases,
        # we use nspin=1 because:
        # - Spin-degenerate: spin-up and spin-down are identical
        # - SOC: spin is not a good quantum number, single set includes SO effects
        nspin = 1 #TODO: consider nspin=2 for non-collinear cases in future
        eigs_3d = eigenvalues.reshape(nspin, nk, nbands)

        sigma = kB_eV_per_K * temp  # kT in eV

        # Map smearing method names to calculate_fermi_level convention
        method_map = {
            'FD': 'fd',
            'Fermi-Dirac': 'fd',
            'fermi-dirac': 'fd',
            'fd': 'fd',
            'Gaussian': 'gaussian',
            'G': 'gaussian',
            'gaussian': 'gaussian',
            'gau': 'gaussian'
        }

        method = method_map.get(smearing_method)
        if method is None:
            raise ValueError(
                f'Unknown smearing method: {smearing_method}. '
                f'Supported methods: FD, Fermi-Dirac, Gaussian, G'
            )

        # Adjust electron count for calculate_fermi_level
        # calculate_fermi_level internally uses g=2 for nspin=1, meaning:
        #   fne = 2 * sum(f(E) * wk)
        #
        # In cal_E_fermi (bisection method):
        #   q_cal = sum(f(E) * wk) compared with total_electrons / spindeg
        #
        # For consistency:
        # - spindeg=2: q_cal = total_electrons/2, so fne = total_electrons, ne = total_electrons
        # - spindeg=1: q_cal = total_electrons, so fne = 2*total_electrons, ne = 2*total_electrons
        if spindeg == 1:
            ne = total_electrons * 2  # SOC: compensate for internal g=2 factor
            log.info('Calculating Fermi energy for SOC case (spindeg=1).')
        else:
            ne = total_electrons  # Normal: matches g=2 in calculate_fermi_level
            log.info('Calculating Fermi energy in the case of spin-degeneracy.')

        # Call the new implementation
        Ef, occ, eband, eband_free = calculate_fermi_level(
            eigs=eigs_3d,
            wk=wk,
            ne=ne,
            sigma=sigma,
            method=method,
            with_eband=True  # eband is needed by downstream consumers (e.g., dftb_scc)
        )

        # Verify electron conservation
        ne_calc = np.sum(occ * wk.reshape(1, -1, 1)) # occ already includes internal spindeg factor
        diff_ne = abs(ne_calc - ne)
        # log.info(f'Fermi energy calculation completed using {method} smearing.')

        if diff_ne > q_tol:
            log.warning(
                f'Calculated charge deviates from target by more than q_tol={q_tol}. '
                f'Calculated: {ne_calc}, Target: {ne}. '
                f'Difference: {diff_ne:.6e}'
            )

            # Refinement: Newton-Raphson polish and rescale occ
            # For refinements, we work with 2D eigenvalues (nk, nbands)
            # Newton-Raphson polishing (only for Fermi-Dirac smearing)
            if method == 'fd':
                Ef = cls.newton_raphson_polish_fermi(
                    eigenvalues=eigenvalues,
                    Ef=Ef,
                    total_electrons=total_electrons,
                    wk=wk,
                    kT=sigma,
                    spindeg=spindeg,
                    elec_tol=q_tol
                )
                log.info(f'Polished E_fermi using Newton-Raphson: {Ef}')

            # Recompute occupation with (polished) Fermi energy
            # Apply spindeg factor so occ values are in [0, spindeg] range
            # This matches the convention from calculate_fermi_level and charge_pop.py
            if method == 'fd':
                occ = spindeg * cls.fermi_dirac_smearing(eigenvalues, kT=sigma, mu=Ef)
            else:
                occ = spindeg * cls.Gaussian_smearing(eigenvalues, sigma=sigma, mu=Ef)

        # Always rescale occ to ensure exact electron count
        ## untested code for spindeg=1 case
        occ = cls.rescale_occ(
            occ=occ,
            target_electrons=total_electrons,
            wk=wk
        )

        # Ensure occ is 2D (nk, nbands) for downstream consumers
        # calculate_fermi_level returns (nspin, nk, nbands) with nspin=1
        # but charge_pop.py expects (nk, nbands)
        if occ.ndim == 3:
            occ = occ.squeeze(axis=0)

        return Ef, occ, diff_ne, eband, eband_free

    @classmethod
    def fermi_dirac_smearing(cls, E, kT=0.025852, mu=0.0):
        """Fermi-Dirac distribution function.

        Wrapper around dptb.utils.occupy.ffd for backwards compatibility.

        Parameters
        ----------
        E : array_like
            Energy eigenvalues.
        kT : float
            Thermal energy in eV (Boltzmann constant * temperature).
            Default is 0.025852 eV (approximately 300 K).
        mu : float
            Chemical potential / Fermi energy.

        Returns
        -------
        np.ndarray
            Occupation numbers in range [0, 1].
        """
        x = (E - mu) / kT
        return ffd(x)

    @classmethod
    def Gaussian_smearing(cls, E, sigma=0.025852, mu=0.0):
        """Gaussian smearing distribution function.

        Wrapper around dptb.utils.occupy.fgau for backwards compatibility.

        Parameters
        ----------
        E : array_like
            Energy eigenvalues.
        sigma : float
            Smearing width in eV.
            Default is 0.025852 eV (approximately 300 K thermal energy).
        mu : float
            Chemical potential / Fermi energy.

        Returns
        -------
        np.ndarray
            Occupation numbers in range [0, 1].
        """
        x = (E - mu) / sigma
        return fgau(x)

    @classmethod
    def deriv_fermi_dirac_smearing(cls, E, kT=0.025852, mu=0.0):
        """Derivative of Fermi-Dirac occupation with respect to Fermi energy.

        Wrapper around dptb.utils.occupy.dffd for backwards compatibility.

        Computes df/dEf = (1/kT) * f * (1 - f) = (1/kT) * exp(x) / (1 + exp(x))^2
        where x = (E - mu) / kT.

        Parameters
        ----------
        E : np.ndarray
            Energy eigenvalues.
        kT : float
            Thermal energy in eV (Boltzmann constant * temperature).
            Default is 0.025852 eV (approximately 300 K).
        mu : float
            Chemical potential / Fermi energy.

        Returns
        -------
        np.ndarray
            Derivative df/dEf at each energy, same shape as E.

        Reference: DFTBplus src/dftbp/dftb/etemp.F90:derivElectronCount
        """
        x = (E - mu) / kT
        # dffd returns df/dx, multiply by 1/kT to get df/dEf
        return dffd(x) / kT

    @classmethod
    def newton_raphson_polish_fermi(
        cls,
        eigenvalues: np.ndarray,
        Ef: float,
        total_electrons: float,
        wk: np.ndarray,
        kT: float,
        spindeg: int = 2,
        elec_tol: float = 1e-15,
        max_iter: int = 100
    ) -> float:
        """Polish Fermi energy using Newton-Raphson method (Fermi-Dirac only).

        After bisection search, if the electron count error still exceeds the
        tolerance, this method refines the Fermi energy using Newton-Raphson
        iteration. This is particularly useful for achieving very tight
        convergence (e.g., 1e-15 tolerance as used in DFTBplus).

        Implements DFTBplus Stage 4 algorithm from etemp.F90 (lines 137-149).

        The algorithm:
        1. Compute current electron count N(Ef) and error |N - N_target|
        2. Compute derivative dN/dEf = (1/kT) * Σ wk * f * (1-f)
        3. Update: Ef_new = Ef - (N - N_target) / (dN/dEf)
        4. Accept update only if error decreases; stop otherwise

        Parameters
        ----------
        eigenvalues : np.ndarray
            Eigenvalues with shape (nk, nbands).
        Ef : float
            Initial Fermi energy (typically from bisection search).
        total_electrons : float
            Target total number of electrons (before spin degeneracy adjustment).
            The function internally divides by spindeg to get electrons per spin.
        wk : np.ndarray
            K-point weights with shape (nk,). Must sum to 1.
        kT : float
            Thermal energy in eV (Boltzmann * T).
        spindeg : int, optional
            Spin degeneracy factor. Use 2 for spin-degenerate systems (default),
            1 for SOC calculations where spin is not a good quantum number.
        elec_tol : float, optional
            Electron count tolerance. Default is 1e-15 (DFTBplus accuracy.F90).
        max_iter : int, optional
            Maximum number of Newton-Raphson iterations. Default is 100.

        Returns
        -------
        float
            Polished Fermi energy.

        Notes
        -----
        - Only applicable to Fermi-Dirac distribution
        - The iteration continues while the error improves
        - Returns the last Ef that gave improvement if iteration diverges

        Reference
        ---------
        DFTBplus src/dftbp/dftb/etemp.F90:Efilling, Stage 4 (lines 137-149)

        Examples
        --------
        >>> import numpy as np
        >>> from dptb.postprocess.elec_struc_cal import ElecStruCal
        >>>
        >>> # Eigenvalues for 4 k-points, 10 bands
        >>> eigs = np.random.uniform(-5, 5, (4, 10))
        >>> wk = np.ones(4) / 4
        >>> kT = 0.025852  # ~300K
        >>>
        >>> # Get initial Ef from bisection
        >>> Ef_bisection = ElecStruCal.cal_E_fermi(eigs, 12, spindeg=2, wk=wk)
        >>>
        >>> # Polish with Newton-Raphson
        >>> Ef_polished = ElecStruCal.newton_raphson_polish_fermi(
        ...     eigs, Ef_bisection, 12.0, wk, kT, spindeg=2
        ... )
        """
        wk = wk.reshape(-1, 1)  # (nk, 1) for broadcasting with (nk, nbands)

        # Adjust target electrons for spin degeneracy
        # Fermi-Dirac smearing gives occupations in 0-1 range (per spin channel)
        # So we compare with electrons per spin channel
        target_electrons_per_spin = total_electrons / spindeg

        # Compute initial occupation and electron count
        f = cls.fermi_dirac_smearing(eigenvalues, kT=kT, mu=Ef)
        N_calc = (wk * f).sum()
        error_old = abs(N_calc - target_electrons_per_spin)

        # Already converged
        if error_old <= elec_tol:
            return Ef

        Ef_best = Ef

        for _ in range(max_iter):
            # Compute derivative: dN/dEf = Σ wk * df/dEf
            # where df/dEf = (1/kT) * f * (1 - f)
            df_dEf = cls.deriv_fermi_dirac_smearing(eigenvalues, kT=kT, mu=Ef)
            dN_dEf = (wk * df_dEf).sum()

            # Avoid division by zero/near-zero
            if abs(dN_dEf) < 1e-30:
                log.debug('Newton-Raphson: derivative too small, stopping.')
                break

            # Newton-Raphson update
            Ef_new = Ef - (N_calc - target_electrons_per_spin) / dN_dEf

            # Compute new electron count and error
            f_new = cls.fermi_dirac_smearing(eigenvalues, kT=kT, mu=Ef_new)
            N_new = (wk * f_new).sum()
            error_new = abs(N_new - target_electrons_per_spin)

            # Check convergence
            if error_new <= elec_tol:
                log.debug(f'Newton-Raphson converged: error={error_new:.2e}')
                return Ef_new

            # Check if improvement was made
            if error_new >= error_old:
                # No improvement, keep last good value
                log.debug(f'Newton-Raphson: no improvement, stopping. '
                         f'Error: {error_old:.2e}')
                break

            # Accept the update
            Ef_best = Ef_new
            Ef = Ef_new
            N_calc = N_new
            error_old = error_new

        return Ef_best

    @staticmethod
    def rescale_occ(
        occ: np.ndarray,
        target_electrons: float,
        calculated_electrons: float = None,
        wk: np.ndarray = None
    ) -> np.ndarray:
        """Rescale occ to give exact electron count.

        After finding the Fermi energy, there may still be a small discrepancy
        between the calculated electron count and the target. This method
        scales all occupation numbers proportionally to achieve the exact
        target electron count.

        Implements DFTBplus Stage 5 algorithm from etemp.F90 (lines 154-158).

        Parameters
        ----------
        occ : np.ndarray
            Occupation numbers with shape (nk, nbands) or (nspin, nk, nbands).
        target_electrons : float
            Target number of electrons.
        calculated_electrons : float, optional
            Calculated electron count from occ. If None, it will be
            computed as sum(wk * occ) or sum(occ) if wk is None.
        wk : np.ndarray, optional
            K-point weights. Required if calculated_electrons is None and
            occ is k-point dependent. Shape (nk,).

        Returns
        -------
        np.ndarray
            Rescaled occ with the same shape as input.
            The sum (weighted by wk if provided) equals target_electrons.

        Notes
        -----
        - If calculated_electrons is very small (< 1e-30), returns occ
          unchanged to avoid division by zero.
        - This is a simple proportional scaling: occ_new = occ_old * N_target / N_calc

        Reference
        ---------
        DFTBplus src/dftbp/dftb/etemp.F90:Efilling, Stage 5 (lines 154-158)

        Examples
        --------
        >>> import numpy as np
        >>> from dptb.postprocess.elec_struc_cal import ElecStruCal
        >>>
        >>> # Example occ for 4 k-points, 10 bands
        >>> occ = np.random.uniform(0, 1, (4, 10))
        >>> wk = np.ones(4) / 4
        >>>
        >>> # Rescale to get exactly 12 electrons
        >>> occ_rescaled = ElecStruCal.rescale_occ(
        ...     occ, target_electrons=12.0, wk=wk
        ... )
        >>>
        >>> # Verify
        >>> n_elec = (wk.reshape(-1, 1) * occ_rescaled).sum()
        >>> print(f"Electron count: {n_elec}")  # Should be 12.0
        """
        assert occ.ndim in [2, 3], f'occ must be 2D or 3D, got {occ.ndim}D'
        # Compute calculated_electrons if not provided
        if calculated_electrons is None:
            if wk is not None:
                # Check shape
                assert wk.ndim == 1, \
                    f'wk must be 1D, got {wk.ndim}D'
                assert wk.shape[0] == occ.shape[0 if occ.ndim == 2 else 1], \
                    f'wk length {wk.shape[0]} does not match occ nk dimension {occ.shape[0 if occ.ndim == 2 else 1]}'
                # Handle different shapes
                if occ.ndim == 2:  # (nk, nbands)
                    wk_broadcast = wk.reshape(-1, 1)
                elif occ.ndim == 3:  # (nspin, nk, nbands)
                    wk_broadcast = wk.reshape(1, -1, 1)
                else:
                    raise ValueError(f'occ must be 2D or 3D, got {occ.ndim}D')
                calculated_electrons = (wk_broadcast * occ).sum()
            else:
                calculated_electrons = occ.sum()

        # Avoid division by zero
        if abs(calculated_electrons) < 1e-30:
            log.warning('rescale_occ: calculated_electrons ~ 0, '
                       'returning occ unchanged.')
            return occ.copy()

        # Rescale
        scale_factor = target_electrons / calculated_electrons
        if abs(scale_factor - 1.0) > 1e-6:
            log.warning(f'Rescaling occ by factor {scale_factor:.6e} '
                       f'to match target electrons {target_electrons} '
                       f'from calculated {calculated_electrons}.')
            return occ * scale_factor
        else:
            return occ.copy()


    @classmethod
    def cal_elec_bandE(cls,
                      E_fermi: float,
                      spindeg: int,
                      Temp: float,
                      eigvals: np.ndarray,
                      wk: np.ndarray,) -> float:
        """Calculate the electronic band-structure energy from the eigenvalues.
        Note: This function includes the spin degeneracy by multiplying the result by the `spindeg` factor.
        Parameters
        ----------
        E_fermi : float
            The estimated Fermi energy.
        spindeg : int
            The spin degeneracy factor.
        Temp : float
            The temperature for smearing.
        eigvals : np.ndarray
            The eigenvalues of the system.
        wk : np.ndarray
            The weights for each k-point.

        Returns
        -------
            The total electronic band-structure energy.
        """
        assert spindeg == 2, "This function is only for spin-degeneracy case."
        assert len(eigvals.shape) == 2, "Eigenvalues tensor must be 2-dimensional (nk, nstates)."
        nk = eigvals.shape[0]
        assert len(wk.shape) == 1 and wk.shape[0] == nk, "Weights tensor must be 1-dimensional with length equal to number of k-points."
        assert np.isclose(wk.sum(), 1.0), "Weights must sum to 1."

        # Calculate total energy
        fermi_prop = cls.fermi_dirac_smearing(  E = eigvals,
                                                kT= kB_eV_per_K * Temp,
                                                mu= E_fermi) # (nk, nstate)
        elec_totE = spindeg * (wk.reshape(-1,1) * (fermi_prop * eigvals)).sum() # with spin degeneracy
        return elec_totE
