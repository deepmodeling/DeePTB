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
import ase

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
                 structure_leads_fold:ase.Atoms=None,bloch_sorted_indice:torch.Tensor=None, useBloch: bool=False, \
                    bloch_factor: List[int]=[1,1,1],bloch_R_list:List=None,\
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
        
        
        self.useBloch = useBloch
        self.bloch_factor = bloch_factor
        self.bloch_sorted_indice = bloch_sorted_indice
        self.bloch_R_list = bloch_R_list
        self.structure_leads_fold = structure_leads_fold
        if self.useBloch:
            assert self.bloch_sorted_indice is not None
            assert self.bloch_R_list is not None
            assert self.bloch_factor is not None
            assert self.structure_leads_fold is not None

    def self_energy(self, kpoint, energy, eta_lead: float=1e-5, method: str="Lopez-Sancho", \
                    save: bool=False, save_path: str=None, se_info_display: bool=False,
                    HS_inmem: bool=False):
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
        save :
            whether to save the self energy. 
        save_path :
            the path to save the self energy. If not specified, the self energy will be saved in the results_path.
        se_info_display :
            whether to display the information of the self energy calculation.   
        HS_inmem :
            whether to store the Hamiltonian and overlap matrix in memory. Default is False.     
        '''
        assert len(np.array(kpoint).reshape(-1)) == 3
        # according to given kpoint and e_mesh, calculating or loading the self energy and surface green function to self.
        if not isinstance(energy, torch.Tensor):
            energy = torch.tensor(energy) # Energy relative to Ef

        # if not hasattr(self, "HL"):
        #TODO: check here whether it is necessary to calculate the self energy every time

        
        if save_path is None:
            save_path = os.path.join(self.results_path, \
                                        "self_energy",\
                                        f"se_{self.tab}_k{kpoint[0]}_{kpoint[1]}_{kpoint[2]}_E{energy}.pth")
            parent_dir = os.path.dirname(save_path)
            if not os.path.exists(parent_dir): 
                os.makedirs(parent_dir)

        # If the file in save_path exists, then directly load the self energy from the file    
        if os.path.exists(save_path):

            if se_info_display: log.info(f"Loading self energy from {save_path}")     
            if not save_path.endswith(".pth"):
                # if the save_path is a directory, then the self energy file is stored in the directory
                save_path = os.path.join(save_path, \
                                        f"se_{self.tab}_k{kpoint[0]}_{kpoint[1]}_{kpoint[2]}_E{energy}.pth")
                assert os.path.exists(save_path), f"Cannot find the self energy file {save_path}"
            self.se = torch.load(save_path)
            return
        else:
            if se_info_display:
                log.info("-"*50)
                log.info(f"Not find stored {self.tab} self energy. Calculating it at kpoint {kpoint} and energy {energy}.")
                log.info("-"*50)

        if not self.useBloch:
            if not hasattr(self, "HL") or abs(self.voltage_old-self.voltage)>1e-6 or max(abs(self.kpoint-torch.tensor(kpoint)))>1e-6:
                self.HLk, self.HLLk, self.HDLk, self.SLk, self.SLLk, self.SDLk \
                    = self.hamiltonian.get_hs_lead(kpoint, tab=self.tab, v=self.voltage)
                self.voltage_old = self.voltage
                self.kpoint = torch.tensor(kpoint)

            HDL_reduced, SDL_reduced = self.HDL_reduced(self.HDLk, self.SDLk)
            
            self.se, _ = selfEnergy(
                ee=energy,
                hL=self.HLk,
                hLL=self.HLLk,
                sL=self.SLk,
                sLL=self.SLLk,
                hDL=HDL_reduced,
                sDL=SDL_reduced,             #TODO: check chemiPot settiing is correct or not
                chemiPot=self.efermi, # temmporarily change to self.efermi for the case in which applying lead bias to corresponding to Nanotcad
                etaLead=eta_lead, 
                method=method
            )

            # torch.save(self.se, os.path.join(self.results_path, f"se_nobloch_k{kpoint[0]}_{kpoint[1]}_{kpoint[2]}_{energy}.pth"))
        
        else:
            if not hasattr(self, "HL") or abs(self.voltage_old-self.voltage)>1e-6 or max(abs(self.kpoint-torch.tensor(kpoint)))>1e-6:
                self.kpoint = torch.tensor(kpoint)
                self.voltage_old = self.voltage

            bloch_unfolder = Bloch(self.bloch_factor)
            kpoints_bloch = bloch_unfolder.unfold_points(self.kpoint.tolist())
            sgf_k = []
            m_size = self.bloch_factor[1]*self.bloch_factor[0]
            for ik_lead,k_bloch in enumerate(kpoints_bloch):
                k_bloch = torch.tensor(k_bloch)
                self.HLk, self.HLLk, self.HDLk, self.SLk, self.SLLk, self.SDLk \
                    = self.hamiltonian.get_hs_lead(k_bloch, tab=self.tab, v=self.voltage)
                
                _, sgf = selfEnergy(
                    ee=energy,
                    hL=self.HLk,
                    hLL=self.HLLk,
                    sL=self.SLk,
                    sLL=self.SLLk,            #TODO: check chemiPot settiing is correct or not
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
                            phase_factor_m[i,j] = torch.exp(torch.tensor(1j)*2*torch.pi*torch.dot(self.bloch_R_list[j]-self.bloch_R_list[i],k_bloch))  
                phase_factor_m = phase_factor_m.contiguous()
                sgf = sgf.contiguous()
                sgf_k.append(torch.kron(phase_factor_m,sgf)) 
             

            sgf_k = torch.sum(torch.stack(sgf_k),dim=0)/len(sgf_k)
            sgf_k = sgf_k[self.bloch_sorted_indice,:][:,self.bloch_sorted_indice]
            b = self.HDLk.shape[1] # size of lead hamiltonian

            # reduce the Hamiltonian and overlap matrix based on the non-zero range of HDL
            HDL_reduced, SDL_reduced = self.HDL_reduced(self.HDLk, self.SDLk) 
            # HDL_reduced, SDL_reduced = self.HDL, self.SDL
            if not isinstance(energy, torch.Tensor):
                eeshifted = torch.scalar_tensor(energy, dtype=torch.complex128) + self.efermi
            else:
                eeshifted = energy + self.efermi
            # self.se = (eeshifted*self.SDL-self.HDL) @ sgf_k[:b,:b] @ (eeshifted*self.SDL.conj().T-self.HDL.conj().T)
            self.se = (eeshifted*SDL_reduced-HDL_reduced) @ sgf_k[:b,:b] @ (eeshifted*SDL_reduced.conj().T-HDL_reduced.conj().T)

        if not HS_inmem:
            del self.HLk, self.HLLk, self.HDLk, self.SLk, self.SLLk, self.SDLk

        if save:
            assert save_path is not None, "Please specify the path to save the self energy."
            if se_info_display: log.info(f"Saving self energy to {save_path}")
            torch.save(self.se, save_path)
            # if self.useBloch:
            #     torch.save(self.se, os.path.join(self.results_path, f"se_bloch_k{kpoint[0]}_{kpoint[1]}_{kpoint[2]}_{energy}.pth"))
            # else:
            #     torch.save(self.se, os.path.join(self.results_path, f"se_nobloch_k{kpoint[0]}_{kpoint[1]}_{kpoint[2]}_{energy}.pth"))

    @staticmethod
    def HDL_reduced(HDL: torch.Tensor, SDL: torch.Tensor) -> torch.Tensor:
        '''This function takes in Hamiltonian/Overlap matrix between lead and device and reduces 
        it based on the non-zero range of the Hamiltonian matrix.

            When the device part has only one orbital, the Hamiltonian matrix is not reduced.
        
        Parameters
        ----------
        HDL : torch.Tensor
            HDL is a torch.Tensor representing the Hamiltonian matrix between the first principal layer and the device.
        SDL : torch.Tensor
            SDL is a torch.Tensor representing the overlap matrix between the first principal layer and the device.
        
        Returns
        -------
        HDL_reduced, SDL_reduced
            The reduced Hamiltonian and overlap matrix.
        
        '''
        assert len(HDL.shape) == 2, "The shape of HDL should be 2."
        assert len(SDL.shape) == 2, "The shape of SDL should be 2."
        assert HDL.shape == SDL.shape, "The shape of HDL and SDL should be the same."

        HDL_nonzero_range = (HDL.nonzero().min(dim=0).values, HDL.nonzero().max(dim=0).values)
        # HDL_nonzero_range is a tuple((min_row,min_col),(max_row,max_col))
        if HDL.shape[0] == 1: # Only 1 orbital in the device
            HDL_reduced = HDL
            SDL_reduced = SDL
        elif HDL_nonzero_range[0][0] > 0: # Right lead
            HDL_reduced = HDL[HDL_nonzero_range[0][0]:, :]
            SDL_reduced = SDL[HDL_nonzero_range[0][0]:, :]
        else: # Left lead
            HDL_reduced = HDL[:HDL_nonzero_range[1][0]+1, :]
            SDL_reduced = SDL[:HDL_nonzero_range[1][0]+1, :]

        return HDL_reduced, SDL_reduced


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