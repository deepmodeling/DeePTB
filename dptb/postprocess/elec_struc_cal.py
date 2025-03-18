import numpy as np
from ase.io import read
import ase
import numpy as np
from typing import Union
import torch
from typing import Optional
import logging
log = logging.getLogger(__name__)
from dptb.data import AtomicData, AtomicDataDict
from dptb.nn.energy import Eigenvalues
from dptb.utils.argcheck import get_cutoffs_from_model_options
from copy import deepcopy
from dptb.utils.constants import Boltzmann,eV2J

# This class `ElecStruCal`  is designed to calculate electronic structure properties such as
# eigenvalues and Fermi energy based on provided input data and model. 
# It serve as a basic post-processing class to load data and provide Fermi energy.

class ElecStruCal(object):
    def __init__ (
            self, 
            model: torch.nn.Module,
            device: Union[str, torch.device]=None
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
        
        '''
        if  device is None:
            device = model.device
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.model = model
        self.model.eval()
        self.overlap = hasattr(model, 'overlap')

        if not self.model.transform:
            log.error('The model.transform is not True, please check the model.')
            raise RuntimeError('The model.transform is not True, please check the model.')
        
        if self.overlap:
            self.eigv = Eigenvalues(
                idp=model.idp,
                device=self.device,
                s_edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
                s_node_field=AtomicDataDict.NODE_OVERLAP_KEY,
                s_out_field=AtomicDataDict.OVERLAP_KEY,
                dtype=model.dtype,
            )
        else:
            self.eigv = Eigenvalues(
                idp=model.idp,
                device=self.device,
                dtype=model.dtype,
            )
        r_max, er_max, oer_max  = get_cutoffs_from_model_options(model.model_options)
        self.cutoffs = {'r_max': r_max, 'er_max': er_max, 'oer_max': oer_max}
    def get_data(self,data: Union[AtomicData, ase.Atoms, str],pbc:Union[bool,list]=None, device: Union[str, torch.device]=None, AtomicData_options:dict=None):
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
        
        Returns
        -------
            the loaded AtomicData object.
        
        '''
        atomic_options = deepcopy(self.cutoffs)
        if pbc is not None:
            # 这一句要结合后面AtomicData.from_ase(structase, **atomic_options) 看。在from_ase中
            # pbc = kwargs.pop("pbc", atoms.pbc), 所以当默认 调用get_dat 传入 pbc = None 时，
            # atomic_options 中并没有 pbc 这个key，所以在from_ase中，pbc = atoms.pbc 默认采用atoms的pbc
            # 当传入pbc 非None时，atomic_options中会有pbc这个key，所以from_ase中的pbc 将不会采用atoms的pbc。 
            # 逻辑线埋的比较深，需要注意。
            atomic_options.update({'pbc': pbc})

        if AtomicData_options is not None:
            if AtomicData_options.get('r_max', None) is not None:
                if atomic_options['r_max'] != AtomicData_options.get('r_max'):
                    atomic_options['r_max'] = AtomicData_options.get('r_max')
                    log.warning(f'Overwrite the r_max setting in the model with the r_max setting in the AtomicData_options: {AtomicData_options.get("r_max")}')
                    log.warning(f'This is very dangerous, please make sure you know what you are doing.')
            if AtomicData_options.get('er_max', None) is not None:
                if atomic_options['er_max'] != AtomicData_options.get('er_max'):
                    atomic_options['er_max'] = AtomicData_options.get('er_max')
                    log.warning(f'Overwrite the er_max setting in the model with the er_max setting in the AtomicData_options: {AtomicData_options.get("er_max")}')
                    log.warning(f'This is very dangerous, please make sure you know what you are doing.')
            if AtomicData_options.get('oer_max', None) is not None:
                if atomic_options['oer_max'] != AtomicData_options.get('oer_max'):
                    atomic_options['oer_max'] = AtomicData_options.get('oer_max')
                    log.warning(f'Overwrite the oer_max setting in the model with the oer_max setting in the AtomicData_options: {AtomicData_options.get("oer_max")}')
                    log.warning(f'This is very dangerous, please make sure you know what you are doing.')
        
        else:
            if atomic_options['r_max'] is None:
                log.error('The r_max is not provided in model_options, please provide it in AtomicData_options.')
                raise RuntimeError('The r_max is not provided in model_options, please provide it in AtomicData_options.')
            
        if isinstance(data, str):
            structase = read(data)
            data = AtomicData.from_ase(structase, **atomic_options)
        elif isinstance(data, ase.Atoms):
            structase = data
            data = AtomicData.from_ase(structase, **atomic_options)
        elif isinstance(data, AtomicData):
            # structase = data.to("cpu").to_ase()
            log.info('The data is already an instance of AtomicData. Then the data is used directly.')
            data = data
        else:
            raise ValueError('data should be either a string, ase.Atoms, or AtomicData')
        
        if device is None:
            device = self.device
        data = AtomicData.to_AtomicDataDict(data.to(device))
        data = self.model.idp(data)

        return data


    def get_eigs(self, data: Union[AtomicData, ase.Atoms, str], klist: np.ndarray, pbc:Union[bool,list]=None, AtomicData_options:dict=None):
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
        
        Returns
        -------
            The function `get_eigs` returns the loaded data and the energy eigenvalues as a numpy array.
        
        '''
            
        data  = self.get_data(data=data, pbc=pbc, device=self.device,AtomicData_options=AtomicData_options)
        # set the kpoint of the AtomicData
        data[AtomicDataDict.KPOINT_KEY] = \
            torch.nested.as_nested_tensor([torch.as_tensor(klist, dtype=self.model.dtype, device=self.device)])
        # get the eigenvalues
        data = self.model(data)
        if self.overlap == True:
            assert data.get(AtomicDataDict.EDGE_OVERLAP_KEY) is not None
        data = self.eigv(data)

        return data, data[AtomicDataDict.ENERGY_EIGENVALUE_KEY][0].detach().cpu().numpy()

    def get_fermi_level(self, data: Union[AtomicData, ase.Atoms, str], nel_atom: dict, \
                        meshgrid: list = None, klist: np.ndarray=None, pbc:Union[bool,list]=None,AtomicData_options:dict=None,
                        q_tol:float=1e-10,smearing_method:str='FD',temp:float=300):
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
            from dptb.utils.make_kpoints import kmesh_sampling_negf
            klist,wk = kmesh_sampling_negf(meshgrid=meshgrid, is_gamma_center=True, is_time_reversal=True)
            log.info(f'KPOINTS  kmesh sampling: {klist.shape[0]} kpoints')
        else:
            wk = np.ones(klist.shape[0])/klist.shape[0]
            log.info(f'KPOINTS  klist: {klist.shape[0]} kpoints')

        # eigenvalues would be used if provided, otherwise the eigenvalues would be calculated from the model on the specified k-points
        if not AtomicDataDict.ENERGY_EIGENVALUE_KEY in data:
            data, eigs = self.get_eigs(data=data, klist=klist, pbc=pbc, AtomicData_options=AtomicData_options) 
            log.info('Getting eigenvalues from the model.')
        else:
            log.info('The eigenvalues are already in data. will use them.')
            eigs = data[AtomicDataDict.ENERGY_EIGENVALUE_KEY][0].detach().cpu().numpy()
        
        if nel_atom is not None:
            atomtype_list = data[AtomicDataDict.ATOM_TYPE_KEY].flatten().tolist()
            atomtype_symbols = np.asarray(self.model.idp.type_names)[atomtype_list].tolist()
            total_nel = np.array([nel_atom[s] for s in atomtype_symbols]).sum()
            if hasattr(self.model,'soc_param'):
                spindeg = 1
            else:
                spindeg = 2
            E_fermi = self.cal_E_fermi(eigs, total_nel, spindeg, wk, 
                                       q_tol= q_tol, smearing_method = smearing_method,temp=temp)
            log.info(f'Estimated E_fermi: {E_fermi} based on the valence electrons setting nel_atom : {nel_atom} .')
        else:
            E_fermi = None
            raise RuntimeError('nel_atom should be provided to calculate Fermi energy.')
        
        return data, E_fermi


    
    @classmethod
    def cal_E_fermi(cls,eigenvalues: np.ndarray, total_electrons: int, spindeg: int=2,wk: np.ndarray=None,
                    q_tol:float=1e-10,smearing_method:str='FD',temp:float=300):
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
            
        
        # calculate boundaries
        min_Ef, max_Ef = eigenvalues.min(), eigenvalues.max()
        kT = Boltzmann/eV2J * temp
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
            if smearing_method == 'FD':
                q_cal = (wk * cls.fermi_dirac_smearing(eigenvalues,kT=kT, mu=Ef)).sum()
            elif smearing_method == 'Gaussian':
                q_cal = (wk * cls.Gaussian_smearing(eigenvalues,sigma = kT, mu=Ef)).sum()
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
    def fermi_dirac_smearing(cls, E, kT=0.025852, mu=0.0):
        x = (E - mu) / kT
        mask_min = x < -40.0 # 40 results e16 precision
        mask_max = x > 40.0
        mask_in_limit = ~(mask_min | mask_max)
        out = np.zeros_like(x)
        out[mask_min] = 1.0
        out[mask_max] = 0.0
        out[mask_in_limit] = 1.0 / (np.expm1(x[mask_in_limit]) + 2.0)
        return out
        
    @classmethod
    def Gaussian_smearing(cls, E, sigma=0.025852, mu=0.0):
        from scipy.special import erfc
        x = (mu - E) / sigma
        return 0.5 * erfc(-1*x)