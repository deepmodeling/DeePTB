import torch
from typing import List
from dptb.negf.surface_green import selfEnergy
import logging
from dptb.negf.utils import update_kmap, update_temp_file
import os
from dptb.utils.constants import *
import numpy as np

log = logging.getLogger(__name__)

"""The data output of the intermidiate result should be like this:
{each kpoint
    "e_mesh":[],
    "emap":[]
    "se":[se(e0), se(e1),...], 
    "sgf":[...e...]
}
There will be a kmap outside like: {(0,0,0):1, (0,1,2):2}, to locate which file it is to reads.
"""

            
            

            # get output



class Lead(object):
    def __init__(self, tab, hamiltonian, structure, results_path, voltage, e_T=300, efermi=0.0) -> None:
        self.hamiltonian = hamiltonian
        self.structure = structure
        self.tab = tab
        self.voltage = voltage
        self.results_path = results_path
        self.kBT = k * e_T / eV
        self.e_T = e_T
        self.efermi = efermi
        self.mu = self.efermi - self.voltage

    def self_energy(self, kpoint, e, eta_lead: float=1e-5, method: str="Lopez-Sancho"):
        assert len(np.array(kpoint).reshape(-1)) == 3
        # according to given kpoint and e_mesh, calculating or loading the self energy and surface green function to self.
        if not isinstance(e, torch.Tensor):
            e = torch.tensor(e)

        if not hasattr(self, "HL"):
            self.HL, self.HLL, self.HDL, self.SL, self.SLL, self.SDL = self.hamiltonian.get_hs_lead(kpoint, tab=self.tab, v=self.voltage)

        self.se, _ = selfEnergy(
            ee=e,
            hL=self.HL,
            hLL=self.HLL,
            sL=self.SL,
            sLL=self.SLL,
            hDL=self.HDL,
            sDL=self.SDL,
            chemiPot=self.mu,
            etaLead=eta_lead, 
            method=method
        )

    def sigmaLR2Gamma(self, se):
        return -1j * (se - se.conj())
    
    def fermi_dirac(self, x) -> torch.Tensor:
        return 1 / (1 + torch.exp((x - self.mu)/ self.kBT))
    
    @property
    def gamma(self):
        return self.sigmaLR2Gamma(self.se)