from typing import List
import torch
from dptb.negf.Areshkin import pole_maker
from dptb.negf.RGF import recursive_gf
from dptb.negf.surface_green import selfEnergy
from dptb.negf.utils import quad, gauss_xw
from dptb.negf.CFR import ozaki_residues
from dptb.negf.Areshkin import pole_maker
from ase.io import read
from dptb.negf.poisson import density2Potential, getImg
from dptb.negf.SCF import _SCF
from dptb.utils.constants import *
from dptb.negf.utils import update_kmap
from dptb.negf.utils import leggauss
import logging
import os
import torch.optim as optim
from dptb.utils.tools import j_must_have
from tqdm import tqdm
import numpy as np
from dptb.utils.make_kpoints import kmesh_sampling

'''
1. split the leads, the leads and contact, and contact. the atoms
'''

log = logging.getLogger(__name__)

class Hamiltonian(object):
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
        assert len(np.array(kpoints).shape) == 2

        HS_device = {}
        HS_leads = {}
        HS_device["kpoints"] = kpoints

        self.apiH.update_struct(self.structase, mode="device", stru_options=j_must_have(self.stru_options, "device"), pbc=self.stru_options["pbc"])
        # change parameters to match the structure projection
        n_proj_atom_pre = np.array([1]*len(self.structase))[:self.device_id[0]][self.apiH.structure.projatoms[:self.device_id[0]]].sum()
        n_proj_atom_device = np.array([1]*len(self.structase))[self.device_id[0]:self.device_id[1]][self.apiH.structure.projatoms[self.device_id[0]:self.device_id[1]]].sum()
        device_id = [0,0]
        device_id[0] = n_proj_atom_pre
        device_id[1] = n_proj_atom_pre + n_proj_atom_device
        projatoms = self.apiH.structure.projatoms

        self.atom_norbs = [self.apiH.structure.proj_atomtype_norbs[i] for i in self.apiH.structure.proj_atom_symbols]
        self.apiH.get_HR()
        H, S = self.apiH.get_HK(kpoints=kpoints)
        d_start = int(np.sum(self.atom_norbs[:device_id[0]]))
        d_end = int(np.sum(self.atom_norbs)-np.sum(self.atom_norbs[device_id[1]:]))
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
                lead_id = [0,0]
                lead_id[0] = n_proj_atom_pre
                lead_id[1] = n_proj_atom_pre + n_proj_atom_lead

                l_start = int(np.sum(self.atom_norbs[:lead_id[0]]))
                l_end = int(l_start + np.sum(self.atom_norbs[lead_id[0]:lead_id[1]]) / 2)
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
        f = torch.load(os.path.join(self.results_path, "HS_device.pth"))
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
        f = torch.load(os.path.join(self.results_path, "HS_{0}.pth".format(tab)))
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
        return self.atom_norbs[self.device_id[0]:self.device_id[1]]

    # def get_hs_block_tridiagonal(self, HD, SD):

    #     return hd, hu, hl, sd, su, sl
