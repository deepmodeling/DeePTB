import torch
from typing import List
from dptb.negf.surface_green import selfEnergy
import logging
from dptb.negf.negf_utils import update_kmap, update_temp_file
import os
from dptb.utils.constants import Boltzmann, eV2J
import numpy as np
from dptb.negf.bloch import Bloch
import torch.profiler


log = logging.getLogger(__name__)

# """The data output of the intermidiate result should be like this:
# {each kpoint
#     "e_mesh":[],
#     "emap":[]
#     "se":[se(e0), se(e1),...], 
#     "sgf":[...e...]
# }
# There will be a kmap outside like: {(0,0,0):1, (0,1,2):2}, to locate which file it is to reads.
# """


class LeadProperty(object):
    '''
    The Lead class represents a lead in a structure and provides methods for calculating the self energy
    and gamma for the lead.

    Property
    ----------
    hamiltonian
        hamiltonian of the whole structure.
    structure
        structure of the lead.
    tab
        lead tab.
    voltage
        voltage of the lead.
    results_path
        output  path.
    kBT
        Boltzmann constant times temperature.
    efermi
        Fermi energy.
    mu
        chemical potential of the lead.
    gamma
        the broadening function of the isolated energy level of the device
    HL 
        hamiltonian within principal layer
    HLL 
        hamiiltonian between two adjacent principal layers
    HDL 
        hamiltonian between principal layer and device
    SL SLL and SDL 
        the overlap matrix, with the same meaning as HL HLL and HDL.
    

    Method
    ----------
    self_energy
        calculate  the self energy and surface green function at the given kpoint and energy.
    sigma2gamma
        calculate the Gamma function from the self energy.

    '''
    def __init__(self, tab, hamiltonian, structure, results_path, voltage,\
                 structure_leads_fold,bloch_sorted_indice, useBloch: bool=False, bloch_factor: List[int]=[1,1,1],bloch_R_list:List=None,\
                    e_T=300, efermi=0.0) -> None:
        self.hamiltonian = hamiltonian
        self.structure = structure
        self.tab = tab
        self.voltage = voltage
        self.results_path = results_path
        self.kBT = Boltzmann * e_T / eV2J
        self.e_T = e_T
        self.efermi = efermi
        self.mu = self.efermi - self.voltage
        self.kpoint = None
        self.voltage_old = None
        self.structure_leads_fold = structure_leads_fold
        
        self.useBloch = useBloch
        self.bloch_factor = bloch_factor
        self.bloch_sorted_indice = bloch_sorted_indice
        self.bloch_R_list = bloch_R_list

    def self_energy(self, kpoint, energy, eta_lead: float=1e-5, method: str="Lopez-Sancho", \
                    ):
        '''calculate and loads the self energy and surface green function at the given kpoint and energy.
        
        Parameters
        ----------
        kpoint
            the coordinates of a specific point in the Brillouin zone. 
        energy
            specific energy value.
        eta_lead : 
            the broadening parameter for calculating lead surface green function.
        method : 
            specify the method for calculating the self energy. At this stage it only supports "Lopez-Sancho".
        
        '''
        assert len(np.array(kpoint).reshape(-1)) == 3
        # according to given kpoint and e_mesh, calculating or loading the self energy and surface green function to self.
        if not isinstance(energy, torch.Tensor):
            energy = torch.tensor(energy) # Energy relative to Ef

        # if not hasattr(self, "HL"):
        #TODO: check here whether it is necessary to calculate the self energy every time


        if not self.useBloch:
            if not hasattr(self, "HL") or abs(self.voltage_old-self.voltage)>1e-6 or max(abs(self.kpoint-torch.tensor(kpoint)))>1e-6:
                self.HL, self.HLL, self.HDL, self.SL, self.SLL, self.SDL \
                    = self.hamiltonian.get_hs_lead(kpoint, tab=self.tab, v=self.voltage)
                self.voltage_old = self.voltage
                self.kpoint = torch.tensor(kpoint)
                

            self.se, _ = selfEnergy(
                ee=energy,
                hL=self.HL,
                hLL=self.HLL,
                sL=self.SL,
                sLL=self.SLL,
                hDL=self.HDL,
                sDL=self.SDL,             #TODO: check chemiPot settiing is correct or not
                chemiPot=self.efermi, # temmporarily change to self.efermi for the case in which applying lead bias to corresponding to Nanotcad
                etaLead=eta_lead, 
                method=method
            )
        
        else:
            if not hasattr(self, "HL") or abs(self.voltage_old-self.voltage)>1e-6 or max(abs(self.kpoint-torch.tensor(kpoint)))>1e-6:
                self.kpoint = torch.tensor(kpoint)
                self.voltage_old = self.voltage

            bloch_unfolder = Bloch(self.bloch_factor)
            kpoints_lead = bloch_unfolder.unfold_points(self.kpoint.tolist())
            se_k = []
            m_size = self.bloch_factor[2]*self.bloch_factor[1]*self.bloch_factor[0]
            for k in kpoints_lead:
                k = torch.tensor(k)
                self.HL, self.HLL, self.HDL, self.SL, self.SLL, self.SDL \
                    = self.hamiltonian.get_hs_lead(k, tab=self.tab, v=self.voltage)
                
                se, _ = selfEnergy(
                    ee=energy,
                    hL=self.HL,
                    hLL=self.HLL,
                    sL=self.SL,
                    sLL=self.SLL,
                    hDL=self.HDL,
                    sDL=self.SDL,             #TODO: check chemiPot settiing is correct or not
                    chemiPot=self.efermi, # temmporarily change to self.efermi for the case in which applying lead bias to corresponding to Nanotcad
                    etaLead=eta_lead, 
                    method=method
                )
                phase_factor_m = torch.zeros([m_size,m_size],dtype=torch.complex128)
                for i in range(m_size):
                    for j in range(m_size):
                        if i == j:
                            phase_factor_m[i,j] = 1
                        else:
                            phase_factor_m[i,j] = torch.exp(torch.tensor(1j)*2*torch.pi*torch.dot(self.bloch_R_list[j]-self.bloch_R_list[i],k))  

                se_k.append(torch.kron(phase_factor_m,se))                

            se_k = torch.sum(torch.stack(se_k),dim=0)/len(se_k)
            self.se = se_k[self.bloch_sorted_indice,:][:,self.bloch_sorted_indice]

            

    def sigmaLR2Gamma(self, se):
        '''calculate the Gamma function from the self energy.
        
        Gamma function is the broadening function of the isolated energy level of the device.

        Parameters
        ----------
        se
            The parameter "se" represents self energy, a complex matrix.
        
        Returns
        -------
        Gamma
            The Gamma function, $\Gamma = 1j(se-se^\dagger)$.
        
        '''
        return 1j * (se - se.conj().T)
    
    def fermi_dirac(self, x) -> torch.Tensor:
        return 1 / (1 + torch.exp((x - self.mu)/ self.kBT))
    
    @property
    def gamma(self):
        return self.sigmaLR2Gamma(self.se)