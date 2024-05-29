import numpy as np
from dptb.utils.tools import j_must_have
from dptb.utils.make_kpoints  import ase_kpath, abacus_kpath, vasp_kpath
from ase.io import read
import ase
import numpy as np
from typing import Union
import matplotlib.pyplot as plt
import torch
from typing import Optional
import matplotlib
import logging
log = logging.getLogger(__name__)
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from dptb.data import AtomicData, AtomicDataDict
from dptb.nn.energy import Eigenvalues

class AbstractProcess(object):
    def __init__ (
            self, 
            model: torch.nn.Module,
            results_path: Optional[str]=None,
            use_gui=False,
            device: Union[str, torch.device]=None
            ):
        
        if  device is None:
            device = model.device
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.model = model
        self.model.eval()
        self.use_gui = use_gui
        self.results_path = results_path
        self.overlap = hasattr(model, 'overlap')

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

    def get_data(self,data: Union[AtomicData, ase.Atoms, str],AtomicData_options: dict={},device: Union[str, torch.device]=None):

        if isinstance(data, str):
            structase = read(data)
            data = AtomicData.from_ase(structase, **AtomicData_options)
        elif isinstance(data, ase.Atoms):
            structase = data
            data = AtomicData.from_ase(structase, **AtomicData_options)
        elif isinstance(data, AtomicData):
            # structase = data.to("cpu").to_ase()
            data = data
        else:
            raise ValueError('data should be either a string, ase.Atoms, or AtomicData')
        
        if device is None:
            device = self.device
        data = AtomicData.to_AtomicDataDict(data.to(device))
        data = self.model.idp(data)

        return data


    def get_eigs(self, data: Union[AtomicData, ase.Atoms, str], klist: np.ndarray, AtomicData_options: dict={}):
            
        data  = self.get_data(data=data, AtomicData_options=AtomicData_options, device=self.device)
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
                        kmesh: list = None,klist: np.ndarray=None, AtomicData_options: dict={},\
                        eigenvalues: np.ndarray=None):

        assert kmesh is not None or klist is not None, 'kmesh or klist should be provided.'
        assert isinstance(nel_atom, dict)
        
        # klist would be used if provided, otherwise kmesh would be used to generate klist
        if klist is None:
            from dptb.utils.make_kpoints import kmesh_sampling_negf
            klist,wk = kmesh_sampling_negf(meshgrid=kmesh, is_gamma_center=True, is_time_reversal=True)
            log.info(f'KPOINTS  kmesh sampling: {klist.shape[0]} kpoints')
        else:
            wk = np.ones(klist.shape[0])/klist.shape[0]
            log.info(f'KPOINTS  klist: {klist.shape[0]} kpoints')

        # eigenvalues would be used if provided, otherwise the eigenvalues would be calculated from the model
        if eigenvalues is None:
            data, eigs = self.get_eigs(data=data, klist=klist, AtomicData_options=AtomicData_options) 
            log.info('Getting eigenvalues from the model.')
        else:
            data = self.get_data(data=data, AtomicData_options=AtomicData_options, device=self.device)
            eigs = eigenvalues
            log.info('Using the provided eigenvalues.')
        
        if nel_atom is not None:
            atomtype_list = data[AtomicDataDict.ATOM_TYPE_KEY].flatten().tolist()
            atomtype_symbols = np.asarray(self.model.idp.type_names)[atomtype_list].tolist()
            total_nel = np.array([nel_atom[s] for s in atomtype_symbols]).sum()
            if hasattr(self.model,'soc_param'):
                spindeg = 1
            else:
                spindeg = 2
            E_fermi = self.cal_E_fermi(eigs, total_nel, spindeg, wk)
            log.info(f'Estimated E_fermi: {E_fermi} based on the valence electrons setting nel_atom : {nel_atom} .')
        else:
            E_fermi = None
            raise RuntimeError('nel_atom should be provided.')
        
        return data, E_fermi



    @classmethod
    def cal_E_fermi(cls, eigenvalues: np.ndarray, total_electrons: int, spindeg: int=2,wk: np.ndarray=None,q_tol=1e-10):
        nextafter = np.nextafter
        total_electrons = total_electrons / spindeg # This version is for the case of spin-degeneracy
        log.info('Calculating Fermi energy in the case of spin-degeneracy.')
        def fermi_dirac(E, kT=0.4, mu=0.0):
            return 1.0 / (np.expm1((E - mu) / kT) + 2.0)
        
        # calculate boundaries
        min_Ef, max_Ef = eigenvalues.min(), eigenvalues.max()
        Ef = (min_Ef + max_Ef) * 0.5

        if wk is None:
            wk = np.ones(eigenvalues.shape[0]) / eigenvalues.shape[0]
            log.info('wk is not provided, using equal weight for kpoints.')

        while nextafter(min_Ef, max_Ef) < max_Ef:
            # Calculate guessed charge
            wk = wk.reshape(-1,1)
            q_cal = (wk * fermi_dirac(eigenvalues, mu=Ef)).sum()

            if abs(q_cal - total_electrons) < q_tol:
                return Ef

            if q_cal >= total_electrons:
                max_Ef = Ef
            else:
                min_Ef = Ef
            Ef = (min_Ef + max_Ef) * 0.5

        return Ef