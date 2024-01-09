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
from negf.poisson_scf import poisson_negf_scf # TODO : move this to dptb.negf
from negf.poisson_init import Grid,Interface3D,Gate,Dielectric

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

        ## Poisson equation
        self.poisson_grid = jdata["poisson_grid"]
        self.gate_region = jdata["gate_region"]
        self.dielectric_region = jdata["dielectric_region"]


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

        if self.scf:
            if not self.out_density:
                raise RuntimeError("Error! scf calculation requires density matrix. Please set out_density to True")
            self.poisson_negf_scf()
        else:
            self.negf_compute(scf_require=False)


    def poisson_negf_scf(self,diff_acc=1e-6,max_iter=100,mix_rate=0.3):
       
        # create grid
        xg,yg,zg,xa,ya,za = self.read_grid(self.structase, self.poisson_grid) #TODO:write read_grid
        grid = Grid(xg,yg,zg,xa,ya,za)
        # create gate
        gate_list = []
        gates = self.gate_region.keys()
        for gg in gates:
            if gg.startswith("gate"):
                xmin,xmax = self.gate_region[gg].get("x_range",None).split('-')
                ymin,ymax = self.gate_region[gg].get("y_range",None).split('-')
                zmin,zmax = self.gate_region[gg].get("z_range",None).split('-')
                gate_init = Gate(xmin,xmax,ymin,ymax,zmin,zmax)
                gate_init.Ef = self.gate_region[gg].get("Ef",None)
                gate_list.append(gate_init)
                      
        # create dielectric
        dielectric_list = []
        dielectric = self.dielectric_region.keys()
        for dd in dielectric:
            if dd.startswith("dielectric"):
                xmin,xmax = self.dielectric_region[dd].get("x_range",None).split('-')
                ymin,ymax = self.dielectric_region[dd].get("y_range",None).split('-')
                zmin,zmax = self.dielectric_region[dd].get("z_range",None).split('-')
                dielectric_init = Dielectric(xmin,xmax,ymin,ymax,zmin,zmax)
                dielectric_init.eps = self.dielectric_region[dd].get("Ef",None)
                dielectric_list.append(dielectric_init)        

        # create interface
        interface_poisson = Interface3D(grid,gate_list,dielectric_list)

        max_diff = 1e30; iter_count=0
        while max_diff > diff_acc:

            # update Hamiltonian by modifying onsite energy with potential
            atom_gridpoint_index =  list(interface_poisson.grid.atom_index_dict.values())
            potential_atom = interface_poisson.phi[atom_gridpoint_index] # a vector with length of number of atoms
            # number of orbitals on atoms in device region
            device_atom_norbs = self.negf_hamiltonian.atom_norbs[self.negf_hamiltonian.proj_device_id[0]:self.negf_hamiltonian.proj_device_id[1]]
            
            potential_list = []
            for i in range(len(device_atom_norbs)):
                potential_list.append(potential_atom[i]*torch.ones(device_atom_norbs[i]))
            potential_tensor = torch.cat(potential_list)
            self.negf_compute(scf_require=True,Vbias=potential_tensor)
            

            # update electron density for solving Poisson equation
            DM_eq,DM_neq = self.out["DM_eq"], self.out["DM_neq"]
            DM = DM_eq + DM_neq
            elec_density = torch.diag(DM)
            density_list = []
            pre_atom_orbs = 0
            for i in range(len(device_atom_norbs)):
                density_list.append(torch.sum(elec_density[pre_atom_orbs : pre_atom_orbs+atom_gridpoint_index[i]]))
                pre_atom_orbs += device_atom_norbs[i]

            interface_poisson.free_charge[atom_gridpoint_index] = np.array(density_list)
            max_diff = interface_poisson.solve_poisson(method='pyamg')

            interface_poisson.phi = interface_poisson.phi + mix_rate*(interface_poisson.phi - interface_poisson.phi_old)
            interface_poisson.phi_old = interface_poisson.phi.copy()

            iter_count += 1
            print('Poisson iteration: ',iter_count,' max_diff: ',max_diff)
            if iter_count > max_iter:
                raise RuntimeError('Poisson iteration exceeds max_iter')


        # calculate transport properties with converged potential
        self.negf_compute(scf_require=False)


    def negf_compute(self,scf_require=False,Vbias=None):
        # check if scf is required
        # if self.scf:
        #     # perform k-point sampling and scf calculation to get the converged density
        #     for k in self.kpoints:
        #         pass
        # else:
        #     pass
        
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
                        block_tridiagonal=self.block_tridiagonal,
                        Vbias=Vbias
                        )
                    if scf_require==False:
                        if self.out_dos:
                            prop = self.out.setdefault("DOS", [])
                            prop.append(self.compute_DOS(k))
                        if self.out_tc or self.out_current_nscf:
                            prop = self.out.setdefault("TC", [])
                            prop.append(self.compute_TC(k))
                        if self.out_ldos:
                            prop = self.out.setdefault("LDOS", [])
                            prop.append(self.compute_LDOS(k))


            # whether scf_require is True or False, density are computed for Poisson-NEGF SCF
            if self.out_density or self.out_potential:
                self.out["DM_eq"], self.out["DM_neq"] = self.compute_density(k)
            
            if self.out_potential:
                pass

            if scf_require==False:
                if self.out_dos:
                    self.out["DOS"] = torch.stack(self.out["DOS"])
                if self.out_tc or self.out_current_nscf:
                    self.out["TC"] = torch.stack(self.out["TC"])
                if self.out_current_nscf:
                    self.out["BIAS_POTENTIAL_NSCF"], self.out["CURRENT_NSCF"] = self.compute_current_nscf(k, self.uni_grid, self.out["TC"]) 
                # computing properties that are not functions of E (improvement can be made here in properties related to integration of energy window of fermi functions)
                if self.out_current:
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


            if scf_require == False:
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