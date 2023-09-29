from typing import List
import torch
from dptb.negf.Areshkin import pole_maker
from dptb.negf.RGF import recursive_gf
from dptb.negf.surface_green import selfEnergy
from dptb.negf.utils import quad, gauss_xw
from dptb.negf.CFR import ozaki_residues
from dptb.negf.hamiltonian import Hamiltonian
from dptb.negf.density import Ozaki
from dptb.negf.Areshkin import pole_maker
from dptb.negf.Device import Device
from dptb.negf.utils import update_kmap
from dptb.negf.Lead import Lead
from ase.io import read
from dptb.negf.poisson import density2Potential, getImg
from dptb.negf.SCF import _SCF
from dptb.utils.constants import *
from dptb.negf.utils import leggauss
import logging
import os
from dptb.utils.tools import j_must_have
from tqdm import tqdm
import numpy as np
from dptb.utils.make_kpoints import kmesh_sampling


log = logging.getLogger(__name__)

class NEGF(object):
    def __init__(self, apiHrk, run_opt, jdata):
        self.apiH = apiHrk
        if isinstance(run_opt['structure'],str):
            self.structase = read(run_opt['structure'])
        elif isinstance(run_opt['structure'],ase.Atoms):
            self.structase = run_opt['structure']
        else:
            raise ValueError('structure must be ase.Atoms or str')
        
        self.results_path = run_opt.get('results_path')
        self.jdata = jdata
        self.cdtype = torch.complex128
        self._device = "cpu"
        
        
        
        # get the parameters
        self.ele_T = jdata["ele_T"]
        self.kBT = k * self.ele_T / eV
        self.e_fermi = jdata["e_fermi"]
        self.stru_options = j_must_have(jdata, "stru_options")
        self.pbc = self.stru_options["pbc"]
        if not any(self.pbc):
            self.kpoints = np.array([[0,0,0]])
        else:
            self.kpoints = kmesh_sampling(self.jdata["stru_options"]["kmesh"])

        self.unit = jdata["unit"]
        self.scf = jdata["scf"]
        self.block_tridiagonal = jdata["block_tridiagonal"]
        self.properties = jdata["properties"]
        

        # computing the hamiltonian
        self.hamiltonian = Hamiltonian(apiH=self.apiH, structase=self.structase, stru_options=jdata["stru_options"], results_path=self.results_path)
        with torch.no_grad():
            struct_device, struct_leads = self.hamiltonian.initialize(kpoints=self.kpoints)
        self.generate_energy_grid()

        self.device = Device(self.hamiltonian, struct_device, results_path=self.results_path, efermi=self.e_fermi)
        self.device.set_leadLR(
                lead_L=Lead(
                hamiltonian=self.hamiltonian, 
                tab="lead_L", 
                structure=struct_leads["lead_L"], 
                results_path=self.results_path,
                e_T=self.ele_T,
                efermi=self.e_fermi, 
                voltage=self.jdata["stru_options"]["lead_L"]["voltage"]
            ),
                lead_R=Lead(
                    hamiltonian=self.hamiltonian, 
                    tab="lead_R", 
                    structure=struct_leads["lead_R"], 
                    results_path=self.results_path, 
                    e_T=self.ele_T,
                    efermi=self.e_fermi, 
                    voltage=self.jdata["stru_options"]["lead_R"]["voltage"]
            )
        )

        # initialize density class
        self.density_options = j_must_have(self.jdata, "density_options")
        if self.density_options["method"] == "Ozaki":
            self.density = Ozaki(R=self.density_options["R"], M_cut=self.density_options["M_cut"], n_gauss=self.density_options["n_gauss"])
        else:
            raise ValueError


    def generate_energy_grid(self):

        # computing parameters for NEGF
        
        cal_pole = False
        cal_int_grid = False

        if self.scf:
            v_list = [self.stru_options[i].get("voltage", None) for i in self.stru_options if i.startswith("lead")]
            v_list_b = [i == v_list[0] for i in v_list]
            if not all(v_list_b):
                cal_pole = True
                cal_int_grid = True
        elif "density" in self.properties or "potential" in self.properties:
            cal_pole = True
            v_list = [self.stru_options[i].get("voltage", None) for i in self.stru_options if i.startswith("lead")]
            v_list_b = [i == v_list[0] for i in v_list]
            if not all(v_list_b):
                cal_int_grid = True
        
        if "current" in self.properties:
            cal_int_grid = True

        if "DOS" in self.properties or "TC" in self.properties:
            self.uni_grid = torch.linspace(start=self.jdata["emin"], end=self.jdata["emax"], steps=int((self.jdata["emax"]-self.jdata["emin"])/self.jdata["espacing"]))

        if cal_pole:
            self.poles, self.residues = ozaki_residues(M_cut=self.jdata["M_cut"])
            self.poles = 1j* self.poles * self.kBT + self.device.lead_L.mu - self.device.mu

        if cal_int_grid:
            xl = min(v_list)-4*self.kBT
            xu = max(v_list)+4*self.kBT
            self.int_grid, self.int_weight = gauss_xw(xl=xl, xu=xu, n=int(self.density_options["n_gauss"]))

    def compute(self):

        # check if scf is required
        if self.scf:
            for k in self.kpoints:
                pass
        else:
            pass
        
        # computing output properties
        for k in self.kpoints:

            if hasattr(self, "uni_grid"):
                for e in self.uni_grid:
                    leads = self.stru_options.keys()
                    for ll in leads:
                        if ll.startswith("lead"):
                            getattr(self.device, ll).self_energy(
                                ee=e, 
                                kpoint=k, 
                                etaLead=self.jdata["eta_lead"],
                                method=self.jdata["sgf_solver"]
                                )
                            
                    self.device.green_function(
                        ee=e, 
                        kpoint=k, 
                        etaDevice=self.jdata["eta_device"], 
                        block_tridiagonal=self.block_tridiagonal
                        )
        
                    self.compute_properties(k, self.property)
    
    def fermi_dirac(self, x) -> torch.Tensor:
        return 1 / (1 + torch.exp(x / self.kBT))
    
    def compute_properties(self, kpoint, properties):
        
        out = {}
        # for k in self.kpoints:
        #     ik = update_kmap(self.results_path, kpoint=k)
        for p in properties:
            # log.info(msg="Computing {0} at k = {1}".format(p, k))
            prop = out.setdefault(p, [])
            prop.append(getattr(self, "compute_"+p)(kpoint))


    def compute_DOS(self, kpoint):
        return self.device.dos
    
    def compute_TC(self, kpoint):
        return self.device.tc
    
    def compute_density(self, kpoint):
        DM_eq, DM_neq = self.density.integrate(device=self.device, kpoint=kpoint)

        return DM_eq, DM_neq

    def compute_current(self, kpoint):
        self.device.green_function(ee=self.int_grid, kpoint=kpoint, block_tridiagonal=self.block_tridiagonal)
        return self.device.current
    
    def compute_current_nscf(self, kpoint):
        self.device.green_function(ee=self.uni_grid, kpoint=kpoint, block_tridiagonal=self.block_tridiagonal)
        return self.device.current_nscf


    def SCF(self):
        pass