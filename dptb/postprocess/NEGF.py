from typing import List
import torch
from dptb.negf.recursive_green_cal import recursive_gf
from dptb.negf.surface_green import selfEnergy
from dptb.negf.negf_utils import quad, gauss_xw,leggauss,update_kmap
from dptb.negf.ozaki_res_cal import ozaki_residues
from dptb.negf.negf_hamiltonian_init import NEGFHamiltonianInit
from dptb.negf.density import Ozaki
from dptb.negf.areshkin_pole_sum import pole_maker
from dptb.negf.device_property import DeviceProperty
from dptb.negf.lead_property import LeadProperty
from ase.io import read
import ase
from dptb.negf.poisson import Density2Potential, getImg
from dptb.negf.scf_method import SCFMethod
from dptb.utils.constants import Boltzmann, eV2J
import os
from dptb.utils.tools import j_must_have
import numpy as np
from dptb.utils.make_kpoints import kmesh_sampling
import logging

log = logging.getLogger(__name__)

# TODO : add common class to set all the dtype and precision.

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
        self.kBT = Boltzmann * self.ele_T / eV2J
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
        

        # computing the hamiltonian
        self.negf_hamiltonian = NEGFHamiltonianInit(apiH=self.apiH, structase=self.structase, stru_options=jdata["stru_options"], results_path=self.results_path)
        with torch.no_grad():
            struct_device, struct_leads = self.negf_hamiltonian.initialize(kpoints=self.kpoints)
        

        self.deviceprop = DeviceProperty(self.negf_hamiltonian, struct_device, results_path=self.results_path, efermi=self.e_fermi)
        self.deviceprop.set_leadLR(
                lead_L=LeadProperty(
                hamiltonian=self.negf_hamiltonian, 
                tab="lead_L", 
                structure=struct_leads["lead_L"], 
                results_path=self.results_path,
                e_T=self.ele_T,
                efermi=self.e_fermi, 
                voltage=self.jdata["stru_options"]["lead_L"]["voltage"]
            ),
                lead_R=LeadProperty(
                    hamiltonian=self.negf_hamiltonian, 
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
        
        # geting the output settings
        self.out_tc = jdata["out_tc"]
        self.out_dos = jdata["out_dos"]
        self.out_density = jdata["out_density"]
        self.out_potential = jdata["out_potential"]
        self.out_current = jdata["out_current"]
        self.out_current_nscf = jdata["out_current_nscf"]
        self.out_ldos = jdata["out_ldos"]
        self.out_lcurrent = jdata["out_lcurrent"]
        assert not (self.out_lcurrent and self.block_tridiagonal)
        self.generate_energy_grid()
        self.out = {}


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
        elif self.out_density or self.out_potential:
            cal_pole = True
            v_list = [self.stru_options[i].get("voltage", None) for i in self.stru_options if i.startswith("lead")]
            v_list_b = [i == v_list[0] for i in v_list]
            if not all(v_list_b):
                cal_int_grid = True

        if self.out_lcurrent:
            cal_int_grid = True
        
        if self.out_current:
            cal_int_grid = True

        if self.out_dos or self.out_tc or self.out_current_nscf or self.out_ldos:
            self.uni_grid = torch.linspace(start=self.jdata["emin"], end=self.jdata["emax"], steps=int((self.jdata["emax"]-self.jdata["emin"])/self.jdata["espacing"]))

        if cal_pole:
            self.poles, self.residues = ozaki_residues(M_cut=self.jdata["density_options"]["M_cut"])
            self.poles = 1j* self.poles * self.kBT + self.deviceprop.lead_L.mu - self.deviceprop.mu

        if cal_int_grid:
            xl = torch.tensor(min(v_list)-8*self.kBT)
            xu = torch.tensor(max(v_list)+8*self.kBT)
            self.int_grid, self.int_weight = gauss_xw(xl=xl, xu=xu, n=int(self.density_options["n_gauss"]))

    def compute(self):

        # check if scf is required
        if self.scf:
            # perform k-point sampling and scf calculation to get the converged density
            for k in self.kpoints:
                pass
        else:
            pass
        
        # computing output properties
        for ik, k in enumerate(self.kpoints):
            self.out = {}
            log.info(msg="Properties computation at k = [{:.4f},{:.4f},{:.4f}]".format(float(k[0]),float(k[1]),float(k[2])))

            # computing properties that is functions of E
            if hasattr(self, "uni_grid"):
                self.out["k"] = k
                for e in self.uni_grid:
                    log.info(msg="computing green's function at e = {:.3f}".format(float(e)))
                    leads = self.stru_options.keys()
                    for ll in leads:
                        if ll.startswith("lead"):
                            getattr(self.deviceprop, ll).self_energy(
                                energy=e, 
                                kpoint=k, 
                                eta_lead=self.jdata["eta_lead"],
                                method=self.jdata["sgf_solver"]
                                )
                            
                    self.deviceprop.cal_green_function(
                        energy=e, 
                        kpoint=k, 
                        eta_device=self.jdata["eta_device"], 
                        block_tridiagonal=self.block_tridiagonal
                        )

                    if self.out_dos:
                        prop = self.out.setdefault("DOS", [])
                        prop.append(self.compute_DOS(k))
                    if self.out_tc or self.out_current_nscf:
                        prop = self.out.setdefault("TC", [])
                        prop.append(self.compute_TC(k))
                    if self.out_ldos:
                        prop = self.out.setdefault("LDOS", [])
                        prop.append(self.compute_LDOS(k))

            if self.out_dos:
                self.out["DOS"] = torch.stack(self.out["DOS"])
            if self.out_tc or self.out_current_nscf:
                self.out["TC"] = torch.stack(self.out["TC"])
            
            if self.out_current_nscf:
                self.out["BIAS_POTENTIAL_NSCF"], self.out["CURRENT_NSCF"] = self.compute_current_nscf(k, self.uni_grid, self.out["TC"])
            
            # computing properties that are not functions of E (improvement can be made here in properties related to integration of energy window of fermi functions)
            if self.out_current:
                pass

            if self.out_density or self.out_potential:
                self.out["DM_eq"], self.out["DM_neq"] = self.compute_density(k)
            
            if self.out_potential:
                pass

            if self.out_lcurrent:
                lcurrent = 0
                for i, e in enumerate(self.int_grid):
                    log.info(msg="computing green's function at e = {:.3f}".format(float(e)))
                    leads = self.stru_options.keys()
                    for ll in leads:
                        if ll.startswith("lead"):
                            getattr(self.deviceprop, ll).self_energy(
                                energy=e, 
                                kpoint=k, 
                                eta_lead=self.jdata["eta_lead"],
                                method=self.jdata["sgf_solver"]
                                )
                            
                    self.deviceprop.cal_green_function(
                        energy=e,
                        kpoint=k, 
                        eta_device=self.jdata["eta_device"], 
                        block_tridiagonal=self.block_tridiagonal
                        )
                    
                    lcurrent += self.int_weight[i] * self.compute_lcurrent(k)
                self.out["LOCAL_CURRENT"] = lcurrent

            torch.save(self.out, self.results_path+"/negf.k{}.out.pth".format(ik))

            # plotting
            
    
    def fermi_dirac(self, x) -> torch.Tensor:
        return 1 / (1 + torch.exp(x / self.kBT))
    
    def compute_properties(self, kpoint, properties):
        
        # for k in self.kpoints:
        #     ik = update_kmap(self.results_path, kpoint=k)
        for p in properties:
            # log.info(msg="Computing {0} at k = {1}".format(p, k))
            prop = self.out.setdefault(p, [])
            prop.append(getattr(self, "compute_"+p)(kpoint))


    def compute_DOS(self, kpoint):
        return self.deviceprop.dos
    
    def compute_TC(self, kpoint):
        return self.deviceprop.tc
    
    def compute_LDOS(self, kpoint):
        return self.deviceprop.ldos
    
    def compute_current_nscf(self, kpoint, ee, tc):
        return self.deviceprop._cal_current_nscf_(ee, tc)

    def compute_density(self, kpoint):
        DM_eq, DM_neq = self.density.integrate(deviceprop=self.deviceprop, kpoint=kpoint)
        return DM_eq, DM_neq

    def compute_current(self, kpoint):
        self.deviceprop.cal_green_function(e=self.int_grid, kpoint=kpoint, block_tridiagonal=self.block_tridiagonal)
        return self.devidevicepropce.current
    
    def compute_lcurrent(self, kpoint):
        return self.deviceprop.lcurrent


    def SCF(self):
        pass