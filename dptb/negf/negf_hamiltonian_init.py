from typing import List
import torch
from dptb.negf.areshkin_pole_sum import pole_maker
from dptb.negf.recursive_green_cal import recursive_gf
from dptb.negf.surface_green import selfEnergy
from dptb.negf.negf_utils import quad, gauss_xw,update_kmap,leggauss
from dptb.negf.ozaki_res_cal import ozaki_residues
from dptb.negf.areshkin_pole_sum import pole_maker
from ase.io import read
from dptb.negf.scf_method import SCFMethod
import logging
import os
import torch.optim as optim
from dptb.utils.tools import j_must_have
import numpy as np
from dptb.utils.make_kpoints import kmesh_sampling

'''
a Hamiltonian object  that initializes and manipulates device and  lead Hamiltonians for NEGF
'''

log = logging.getLogger(__name__)

class NEGFHamiltonianInit(object):
    '''The Class for Hamiltonian object in negf module. 
    
        It is used to initialize and manipulate device and lead Hamiltonians for negf.
        It is different from the Hamiltonian object in the dptb module.
        
        Property
        ----------
        apiH: the API object for Hamiltonian
        unit: the unit of energy
        structase: the structure object for the device and leads
        stru_options: the options for structure from input file
        results_path: the path to store the results

        device_id: the start-atom id and end-atom id of the device in the structure file
        lead_ids: the start-atom id and end-atom id of the leads in the structure file


        Methods
        ----------
        initialize: initializes the device and lead Hamiltonians
        get_hs_device: get the device Hamiltonian and overlap matrix at a specific kpoint
        get_hs_lead: get the lead Hamiltonian and overlap matrix at a specific kpoint
        
    '''

    def __init__(self, apiH, structase, stru_options, results_path) -> None:
        self.apiH = apiH
        self.unit = apiH.unit
        self.structase = structase
        self.stru_options = stru_options
        self.results_path = results_path
        
        self.device_id = [int(x) for x in self.stru_options.get("device")["id"].split("-")]
        self.lead_ids = {}
        for kk in self.stru_options:
            if kk.startswith("lead"):
                self.lead_ids[kk] = [int(x) for x in self.stru_options.get(kk)["id"].split("-")]

        if self.unit == "Hartree":
            self.h_factor = 13.605662285137 * 2
        elif self.unit == "eV":
            self.h_factor = 1.
        elif self.unit == "Ry":
            self.h_factor = 13.605662285137
        else:
            log.error("The unit name is not correct !")
            raise ValueError

    def initialize(self, kpoints, block_tridiagnal=False):
        """initializes the device and lead Hamiltonians 
        
        construct device and lead Hamiltonians and return the structures respectively.The lead Hamiltonian 
        is k-resolved due to the transverse k point sampling.

        Args: 
                kpoints: k-points in the Brillouin zone with three coordinates (kx, ky, kz)
                block_tridiagnal: A boolean parameter that determines whether to block-tridiagonalize the
                    device Hamiltonian or not. 

        Returns: 
                structure_device and structure_leads corresponding to the structure of device and leads.

        Raises:
                RuntimeError: if the lead hamiltonian attained from device and lead calculation does not match.                
        
        """
        assert len(np.array(kpoints).shape) == 2

        HS_device = {}
        HS_leads = {}
        HS_device["kpoints"] = kpoints

        self.apiH.update_struct(self.structase, mode="device", stru_options=j_must_have(self.stru_options, "device"), pbc=self.stru_options["pbc"])
        # change parameters to match the structure projection
        n_proj_atom_pre = np.array([1]*len(self.structase))[:self.device_id[0]][self.apiH.structure.projatoms[:self.device_id[0]]].sum()
        n_proj_atom_device = np.array([1]*len(self.structase))[self.device_id[0]:self.device_id[1]][self.apiH.structure.projatoms[self.device_id[0]:self.device_id[1]]].sum()
        proj_device_id = [0,0]
        proj_device_id[0] = n_proj_atom_pre
        proj_device_id[1] = n_proj_atom_pre + n_proj_atom_device
        projatoms = self.apiH.structure.projatoms

        self.atom_norbs = [self.apiH.structure.proj_atomtype_norbs[i] for i in self.apiH.structure.proj_atom_symbols]
        self.apiH.get_HR()
        H, S = self.apiH.get_HK(kpoints=kpoints)
        d_start = int(np.sum(self.atom_norbs[:proj_device_id[0]]))
        d_end = int(np.sum(self.atom_norbs)-np.sum(self.atom_norbs[proj_device_id[1]:]))
        HD, SD = H[:,d_start:d_end, d_start:d_end], S[:, d_start:d_end, d_start:d_end]
        
        if not block_tridiagnal:
            HS_device.update({"HD":HD.cdouble()*self.h_factor, "SD":SD.cdouble()})
        else:
            hd, hu, hl, sd, su, sl = self.get_block_tridiagonal(HD*self.h_factor, SD)
            HS_device.update({"hd":hd, "hu":hu, "hl":hl, "sd":sd, "su":su, "sl":sl})

        torch.save(HS_device, os.path.join(self.results_path, "HS_device.pth"))
        structure_device = self.apiH.structure.projected_struct[self.device_id[0]:self.device_id[1]]
        
        structure_leads = {}
        for kk in self.stru_options:
            if kk.startswith("lead"):
                HS_leads = {}
                stru_lead = self.structase[self.lead_ids[kk][0]:self.lead_ids[kk][1]]
                self.apiH.update_struct(stru_lead, mode="lead", stru_options=self.stru_options.get(kk), pbc=self.stru_options["pbc"])
                # update lead id
                n_proj_atom_pre = np.array([1]*len(self.structase))[:self.lead_ids[kk][0]][projatoms[:self.lead_ids[kk][0]]].sum()
                n_proj_atom_lead = np.array([1]*len(self.structase))[self.lead_ids[kk][0]:self.lead_ids[kk][1]][projatoms[self.lead_ids[kk][0]:self.lead_ids[kk][1]]].sum()
                proj_lead_id = [0,0]
                proj_lead_id[0] = n_proj_atom_pre
                proj_lead_id[1] = n_proj_atom_pre + n_proj_atom_lead

                l_start = int(np.sum(self.atom_norbs[:proj_lead_id[0]]))
                l_end = int(l_start + np.sum(self.atom_norbs[proj_lead_id[0]:proj_lead_id[1]]) / 2)
                HL, SL = H[:,l_start:l_end, l_start:l_end], S[:, l_start:l_end, l_start:l_end] # lead hamiltonian
                HDL, SDL = H[:,d_start:d_end, l_start:l_end], S[:,d_start:d_end, l_start:l_end] # device and lead's hopping
                HS_leads.update({
                    "HL":HL.cdouble()*self.h_factor, 
                    "SL":SL.cdouble(), 
                    "HDL":HDL.cdouble()*self.h_factor, 
                    "SDL":SDL.cdouble()}
                    )

                
                structure_leads[kk] = self.apiH.structure.struct
                self.apiH.get_HR()
                h, s = self.apiH.get_HK(kpoints=kpoints)
                nL = int(h.shape[1] / 2)
                HLL, SLL = h[:, :nL, nL:], s[:, :nL, nL:] # H_{L_first2L_second}
                err_l = (h[:, :nL, :nL] - HL).abs().max()
                if  err_l >= 1e-4: # check the lead hamiltonian get from device and lead calculation matches each other
                    log.error(msg="ERROR, the lead's hamiltonian attained from diffferent methods does not match.")
                    raise RuntimeError
                elif 1e-7 <= err_l <= 1e-4:
                    log.warning(msg="WARNING, the lead's hamiltonian attained from diffferent methods have slight differences {:.7f}.".format(err_l))

                HS_leads.update({
                    "HLL":HLL.cdouble()*self.h_factor, 
                    "SLL":SLL.cdouble()}
                    )
                
                HS_leads["kpoints"] = kpoints
                
                torch.save(HS_leads, os.path.join(self.results_path, "HS_"+kk+".pth"))
        
        return structure_device, structure_leads
    
    def get_hs_device(self, kpoint, V, block_tridiagonal=False):
        """ get the device Hamiltonian and overlap matrix at a specific kpoint

        In diagonalization mode, the Hamiltonian and overlap matrix are block tridiagonalized,
        and hd,hu,hl refers to the diagnonal, upper and lower blocks of the Hamiltonian, respectively.
        The same rules apply to sd, su, sl.
        
        Args:
            kpoints: k-points in the Brillouin zone with three coordinates (kx, ky, kz)
            V: voltage bias
            block_tridiagonal:  a boolean flag that shows whether Hamiltonian has been diagonalized or not
        
        Returns:
            if not diagonalized, return the whole Hamiltonian and Overlap HD-V*SD, SD
            if diagonalized, return the block tridiagonalized Hamiltonian and Overlap component hd, hu, hl,
            sd, su, sl.
        """
        f = torch.load(os.path.join(self.results_path, "HS_device.pth"), weights_only=False)
        kpoints = f["kpoints"]

        ix = None
        for i, k in enumerate(kpoints):
            if np.abs(np.array(k) - np.array(kpoint)).sum() < 1e-8:
                ix = i
                break

        assert ix is not None

        if not block_tridiagonal:
            HD, SD = f["HD"][ix], f["SD"][ix]
        else:
            hd, sd, hl, su, sl, hu = f["hd"][ix], f["sd"][ix], f["hl"][ix], f["su"][ix], f["sl"][ix], f["hu"][ix]
        
        if block_tridiagonal:
            return hd, sd, hl, su, sl, hu
        else:
            return [HD - V*SD], [SD], [], [], [], []
    
    def get_hs_lead(self, kpoint, tab, v):
        """get the lead Hamiltonian and overlap matrix at a specific kpoint
        
        In diagonalization mode, the Hamiltonian and overlap matrix are block tridiagonalized,
        and hd,hu,hl refers to the diagnonal, upper and lower blocks of the Hamiltonian, respectively.
        The same rules apply to sd, su, sl.
        
        Args:
            kpoints: k-points in the Brillouin zone with three coordinates (kx, ky, kz)
            V: voltage bias
            block_tridiagonal:  a boolean flag that shows whether Hamiltonian has been diagonalized or not
        
        Returns:
            if not diagonalized, return the whole Hamiltonian and Overlap HD-V*SD, SD
            if diagonalized, return the block tridiagonalized Hamiltonian and Overlap component hd, hu, hl,
            sd, su, sl.
        """
        f = torch.load(os.path.join(self.results_path, "HS_{0}.pth".format(tab)), weights_only=False)
        kpoints = f["kpoints"]

        ix = None
        for i, k in enumerate(kpoints):
            if np.abs(np.array(k) - np.array(kpoint)).sum() < 1e-8:
                ix = i
                break

        assert ix is not None

        hL, hLL, hDL, sL, sLL, sDL = f["HL"][ix], f["HLL"][ix], f["HDL"][ix], \
                         f["SL"][ix], f["SLL"][ix], f["SDL"][ix]


        return hL-v*sL, hLL-v*sLL, hDL, sL, sLL, sDL

    def attach_potential():
        pass

    def write(self):
        pass

    @property
    def device_norbs(self):
        """ 
        return the number of atoms in the device Hamiltonian
        """
        return self.atom_norbs[self.device_id[0]:self.device_id[1]]

    # def get_hs_block_tridiagonal(self, HD, SD):

    #     return hd, hu, hl, sd, su, sl
