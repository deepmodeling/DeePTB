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
from dptb.negf.scf_method import SCFMethod
from dptb.utils.constants import Boltzmann, eV2J
import os
from dptb.utils.tools import j_must_have
import numpy as np
from dptb.utils.make_kpoints import kmesh_sampling_negf
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

        # check the consistency of the kmesh and pbc
        assert len(self.pbc) == 3, "pbc should be a list of length 3"
        for i in range(3):
            if self.pbc[i] == False and self.jdata["stru_options"]["kmesh"][i] > 1:
                raise ValueError("kmesh should be 1 for non-periodic direction")
            elif self.pbc[i] == False and self.jdata["stru_options"]["kmesh"][i] == 0:
                self.jdata["stru_options"]["kmesh"][i] = 1
                log.info(msg="Warning! kmesh should be set to 1 for non-periodic direction")
            elif self.pbc[i] == True and self.jdata["stru_options"]["kmesh"][i] == 0:
                raise ValueError("kmesh should be > 0 for periodic direction")

        if not any(self.pbc):
            self.kpoints,self.wk = np.array([[0,0,0]]),np.array([1.])
        else:
            self.kpoints,self.wk = kmesh_sampling_negf(self.jdata["stru_options"]["kmesh"], 
                                                       self.jdata["stru_options"]["gamma_center"],
                                                       self.jdata["stru_options"]["time_reversal_symmetry"],)

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
        
        self.out['k']=[]; self.out['wk']=[]
        if hasattr(self, "uni_grid"): self.out["E"] = self.uni_grid

        #  output kpoints information
        log.info(msg="------ k-point for NEGF -----\n")
        log.info(msg="Gamma Center: {0}".format(self.jdata["stru_options"]["gamma_center"])+"\n")
        log.info(msg="Time Reversal: {0}".format(self.jdata["stru_options"]["time_reversal_symmetry"])+"\n")
        log.info(msg="k-points Num: {0}".format(len(self.kpoints))+"\n")
        log.info(msg="k-points weights: {0}".format(self.wk)+"\n")
        log.info(msg="--------------------------------\n")

        for ik, k in enumerate(self.kpoints):
            self.out["k"].append(k)
            self.out['wk'].append(self.wk[ik])
            log.info(msg="Properties computation at k = [{:.4f},{:.4f},{:.4f}]".format(float(k[0]),float(k[1]),float(k[2])))
            # computing properties that is functions of E
            if hasattr(self, "uni_grid"):
                output_freq = int(len(self.uni_grid)/10)
                for ie,e in enumerate(self.uni_grid):
                    if ie % output_freq == 0:
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
                        # prop = self.out['DOS'].setdefault(str(k), [])
                        # prop.append(self.compute_DOS(k))
                        prop = self.out.setdefault('DOS', {})
                        propk = prop.setdefault(str(k), [])
                        propk.append(self.compute_DOS(k))
                    if self.out_tc or self.out_current_nscf:
                        # prop = self.out['TC'].setdefault(str(k), [])
                        # prop.append(self.compute_TC(k))
                        prop = self.out.setdefault('T_k', {})
                        propk = prop.setdefault(str(k), [])
                        propk.append(self.compute_TC(k))
                    if self.out_ldos:
                        # prop = self.out['LDOS'].setdefault(str(k), [])
                        # prop.append(self.compute_LDOS(k))
                        prop = self.out.setdefault('LDOS', {})
                        propk = prop.setdefault(str(k), [])
                        propk.append(self.compute_LDOS(k))

            if self.out_dos:
                self.out["DOS"][str(k)] = torch.stack(self.out["DOS"][str(k)])

            if self.out_tc or self.out_current_nscf:
                self.out["T_k"][str(k)] = torch.stack(self.out["T_k"][str(k)])
            
            # if self.out_current_nscf:
                # self.out["BIAS_POTENTIAL_NSCF"][str(k)], self.out["CURRENT_NSCF"][str(k)] \
                #     = self.compute_current_nscf(k, self.uni_grid, self.out["TC"][str(k)])
            
            # computing properties that are not functions of E (improvement can be made here in properties related to integration of energy window of fermi functions)
            if self.out_current:
                pass

            if self.out_density or self.out_potential:
                prop_DM_eq = self.out.setdefault('DM_eq', {})
                prop_DM_neq = self.out.setdefault('DM_neq', {})
                prop_DM_eq[str(k)], prop_DM_neq[str(k)] = self.compute_density(k)
            
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
                
                prop_local_current = self.out.setdefault('LOCAL_CURRENT', {})
                prop_local_current[str(k)] = lcurrent

        self.out["k"] = np.array(self.out["k"])
        self.out['T_avg'] = torch.tensor(self.out['wk']) @ torch.stack(list(self.out["T_k"].values()))

        if self.out_current_nscf:
            self.out["BIAS_POTENTIAL_NSCF"], self.out["CURRENT_NSCF"] \
                = self.compute_current_nscf(k, self.uni_grid, self.out["T_avg"])
        torch.save(self.out, self.results_path+"/negf.out.pth")

            
    
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