from typing import List
import torch
from dptb.negf.recursive_green_cal import recursive_gf
from dptb.negf.surface_green import selfEnergy
from dptb.negf.negf_utils import quad, gauss_xw,leggauss,update_kmap
from dptb.negf.ozaki_res_cal import ozaki_residues
from dptb.negf.negf_hamiltonian_init import NEGFHamiltonianInit
from dptb.negf.density import Ozaki,Fiori
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
from dptb.utils.make_kpoints import kmesh_sampling_negf
import logging
from dptb.negf.poisson_init import Grid,Interface3D,Gate,Dielectric
from typing import Optional, Union
# from pyinstrument import Profiler
from dptb.data import AtomicData, AtomicDataDict

log = logging.getLogger(__name__)

# TODO : add common class to set all the dtype and precision.

class NEGF(object):
    def __init__(self, 
                model: torch.nn.Module,
                AtomicData_options: dict, 
                structure: Union[AtomicData, ase.Atoms, str],
                ele_T: float,e_fermi: float,
                emin: float, emax: float, espacing: float,
                density_options: dict,
                unit: str,
                scf: bool, poisson_options: dict,
                stru_options: dict,eta_lead: float,eta_device: float,
                block_tridiagonal: bool,sgf_solver: str,
                out_tc: bool=False,out_dos: bool=False,out_density: bool=False,out_potential: bool=False,
                out_current: bool=False,out_current_nscf: bool=False,out_ldos: bool=False,out_lcurrent: bool=False,
                results_path: Optional[str]=None,
                overlap=False,
                torch_device: Union[str, torch.device]=torch.device('cpu'),
                **kwargs):
        
        
        # self.apiH = apiHrk
        
        self.model = model      
        self.results_path = results_path
        # self.jdata = jdata
        self.cdtype = torch.complex128
        self.torch_device = torch_device
        self.overlap = overlap
        
        # get the parameters
        self.ele_T = ele_T
        self.kBT = Boltzmann * self.ele_T / eV2J # change to eV
        self.e_fermi = e_fermi
        self.eta_lead = eta_lead; self.eta_device = eta_device
        self.emin = emin; self.emax = emax; self.espacing = espacing
        self.stru_options = stru_options
        self.sgf_solver = sgf_solver
        self.pbc = self.stru_options["pbc"]

        # check the consistency of the kmesh and pbc
        assert len(self.pbc) == 3, "pbc should be a list of length 3"
        for i in range(3):
            if self.pbc[i] == False and self.stru_options["kmesh"][i] > 1:
                raise ValueError("kmesh should be 1 for non-periodic direction")
            elif self.pbc[i] == False and self.stru_options["kmesh"][i] == 0:
                self.stru_options["kmesh"][i] = 1
                log.warning(msg="kmesh should be set to 1 for non-periodic direction! Automatically Setting kmesh to 1 in direction {}.".format(i))
            elif self.pbc[i] == True and self.stru_options["kmesh"][i] == 0:
                raise ValueError("kmesh should be > 0 for periodic direction")
            
        if not any(self.pbc):
            self.kpoints,self.wk = np.array([[0,0,0]]),np.array([1.])
        else:
            self.kpoints,self.wk = kmesh_sampling_negf(self.stru_options["kmesh"], 
                                                       self.stru_options["gamma_center"],
                                                     self.stru_options["time_reversal_symmetry"])
        log.info(msg="------ k-point for NEGF -----")
        log.info(msg="Gamma Center: {0}".format(self.stru_options["gamma_center"]))
        log.info(msg="Time Reversal: {0}".format(self.stru_options["time_reversal_symmetry"]))
        log.info(msg="k-points Num: {0}".format(len(self.kpoints)))
        if len(self.wk)<10:
            log.info(msg="k-points: {0}".format(self.kpoints))
            log.info(msg="k-points weights: {0}".format(self.wk))
        log.info(msg="--------------------------------")

        self.unit = unit
        self.scf = scf
        self.block_tridiagonal = block_tridiagonal
        # computing the hamiltonian  #需要改写NEGFHamiltonianInit   
        self.negf_hamiltonian = NEGFHamiltonianInit(model=model,
                                                    AtomicData_options=AtomicData_options, 
                                                    structure=structure,
                                                    block_tridiagonal=self.block_tridiagonal,
                                                    pbc_negf = self.pbc, 
                                                    stru_options=self.stru_options,
                                                    unit = self.unit, 
                                                    results_path=self.results_path,
                                                    overlap = self.overlap,
                                                    torch_device = self.torch_device)
        with torch.no_grad():
            struct_device, struct_leads = self.negf_hamiltonian.initialize(kpoints=self.kpoints,
                                                                           block_tridiagnal=self.block_tridiagonal)
        

        self.deviceprop = DeviceProperty(self.negf_hamiltonian, struct_device, results_path=self.results_path, efermi=self.e_fermi)
        self.deviceprop.set_leadLR(
                lead_L=LeadProperty(
                hamiltonian=self.negf_hamiltonian, 
                tab="lead_L", 
                structure=struct_leads["lead_L"], 
                results_path=self.results_path,
                e_T=self.ele_T,
                efermi=self.e_fermi, 
                voltage=self.stru_options["lead_L"]["voltage"]
            ),
                lead_R=LeadProperty(
                    hamiltonian=self.negf_hamiltonian, 
                    tab="lead_R", 
                    structure=struct_leads["lead_R"], 
                    results_path=self.results_path, 
                    e_T=self.ele_T,
                    efermi=self.e_fermi, 
                    voltage=self.stru_options["lead_R"]["voltage"]
            )
        )

        # initialize density class
        # self.density_options = j_must_have(self.jdata, "density_options")
        self.density_options = density_options
        if self.density_options["method"] == "Ozaki":
            self.density = Ozaki(R=self.density_options["R"], M_cut=self.density_options["M_cut"], n_gauss=self.density_options["n_gauss"])
        elif self.density_options["method"] == "Fiori":
            self.density = Fiori(n_gauss=self.density_options["n_gauss"])
        else:
            raise ValueError

        # number of orbitals on atoms in device region
        self.device_atom_norbs = self.negf_hamiltonian.h2k.atom_norbs[self.negf_hamiltonian.device_id[0]:self.negf_hamiltonian.device_id[1]]
        # np.save(self.results_path+"/device_atom_norbs.npy",self.device_atom_norbs)

        # geting the output settings
        self.out_tc = out_tc
        self.out_dos = out_dos
        self.out_density = out_density
        self.out_potential = out_potential
        self.out_current = out_current
        self.out_current_nscf = out_current_nscf
        self.out_ldos = out_ldos
        self.out_lcurrent = out_lcurrent
        assert not (self.out_lcurrent and self.block_tridiagonal)
        self.generate_energy_grid()
        self.out = {}

        ## Poisson equation settings
        self.poisson_options = poisson_options
        # self.LDOS_integral = {}  # for electron density integral
        self.free_charge = {} # net charge: hole - electron
        self.gate_region = [self.poisson_options[i] for i in self.poisson_options if i.startswith("gate")]
        self.dielectric_region = [self.poisson_options[i] for i in self.poisson_options if i.startswith("dielectric")]



    def generate_energy_grid(self):

        # computing parameters for NEGF
        
        cal_pole = False
        cal_int_grid = False

        if self.scf:
            v_list = [self.stru_options[i].get("voltage", None) for i in self.stru_options if i.startswith("lead")]
            v_list_b = [i == v_list[0] for i in v_list]
            if not all(v_list_b):
                if self.density_options["method"] == "Ozaki": 
                    cal_pole = True
                cal_int_grid = True
        elif self.out_density or self.out_potential:
            if self.density_options["method"] == "Ozaki":
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
            # Energy gird is set relative to Fermi level
            self.uni_grid = torch.linspace(start=self.emin, end=self.emax, steps=int((self.emax-self.emin)/self.espacing))

        if cal_pole and  self.density_options["method"] == "Ozaki":
            self.poles, self.residues = ozaki_residues(M_cut=self.density_options["M_cut"])
            self.poles = 1j* self.poles * self.kBT + self.deviceprop.lead_L.mu - self.deviceprop.mu

        if cal_int_grid:
            xl = torch.tensor(min(v_list)-8*self.kBT)
            xu = torch.tensor(max(v_list)+8*self.kBT)
            self.int_grid, self.int_weight = gauss_xw(xl=xl, xu=xu, n=int(self.density_options["n_gauss"]))

    def compute(self):

        if self.scf:
            # if not self.out_density:
            #     self.out_density = True
            #     raise UserWarning("SCF is required, but out_density is set to False. Automatically Setting out_density to True.")
            self.poisson_negf_scf(err=self.poisson_options['err'],tolerance=self.poisson_options['tolerance'],\
                                  max_iter=self.poisson_options['max_iter'],mix_rate=self.poisson_options['mix_rate'])
        else:
            potential_add = None
            self.negf_compute(scf_require=False,Vbias=potential_add)

    def poisson_negf_scf(self,err=1e-6,max_iter=1000,mix_rate=0.3,tolerance=1e-7):

        
        # profiler.start() 
        # create real-space grid
        grid = self.get_grid(self.poisson_options["grid"],self.deviceprop.structure)
        
        # create gate
        Gate_list = []
        for gg in range(len(self.gate_region)):
            gate_init = Gate(self.gate_region[gg].get("x_range",None).split(':'),\
                             self.gate_region[gg].get("y_range",None).split(':'),\
                             self.gate_region[gg].get("z_range",None).split(':'))
            gate_init.Ef = float(self.gate_region[gg].get("voltage",None)) # in unit of volt
            Gate_list.append(gate_init)
                      
        # create dielectric
        Dielectric_list = []
        for dd in range(len(self.dielectric_region)):
            dielectric_init = Dielectric(self.dielectric_region[dd].get("x_range",None).split(':'),\
                self.dielectric_region[dd].get("y_range",None).split(':'),\
                self.dielectric_region[dd].get("z_range",None).split(':'))
            dielectric_init.eps = float(self.dielectric_region[dd].get("relative permittivity",None))
            Dielectric_list.append(dielectric_init)        

        # create interface
        interface_poisson = Interface3D(grid,Gate_list,Dielectric_list)

        #initial guess for electrostatic potential
        log.info(msg="-----Initial guess for electrostatic potential----")
        interface_poisson.solve_poisson_NRcycle(method=self.poisson_options['solver'],tolerance=tolerance)
        atom_gridpoint_index =  list(interface_poisson.grid.atom_index_dict.values())
        log.info(msg="-------------------------------------------\n")

        max_diff_phi = 1e30; max_diff_list = [] 
        iter_count=0
        # Gummel type iteration
        while max_diff_phi > err:
            # update Hamiltonian by modifying onsite energy with potential
            atom_gridpoint_index =  list(interface_poisson.grid.atom_index_dict.values())
            # print("atom_gridpoint_index",atom_gridpoint_index)
            # np.save(self.results_path+"/atom_gridpoint_index.npy",atom_gridpoint_index)
            self.potential_at_atom = interface_poisson.phi[atom_gridpoint_index]
            # print([torch.full((norb,), p) for p, norb in zip(self.potential_at_atom, self.device_atom_norbs)])
            self.potential_at_orb = torch.cat([torch.full((norb,), p) for p, norb in zip(self.potential_at_atom, self.device_atom_norbs)])
            # torch.save(self.potential_at_orb, self.results_path+"/potential_at_orb.pth")

                      
            self.negf_compute(scf_require=True,Vbias=self.potential_at_orb)
            # Vbias makes sense for orthogonal basis as in NanoTCAD
            # TODO: check if Vbias makes sense for non-orthogonal basis 

            # update electron density for solving Poisson equation SCF
            # DM_eq,DM_neq = self.out["DM_eq"], self.out["DM_neq"]
            # elec_density = torch.diag(DM_eq+DM_neq)
            

            # elec_density_per_atom = []
            # pre_atom_orbs = 0
            # for i in range(len(device_atom_norbs)):
            #     elec_density_per_atom.append(torch.sum(elec_density[pre_atom_orbs : pre_atom_orbs+device_atom_norbs[i]]).numpy())
            #     pre_atom_orbs += device_atom_norbs[i]

            # TODO: check the sign of free_charge
            # TODO: check the spin degenracy
            # TODO: add k summation operation
            free_charge_allk = torch.zeros_like(torch.tensor(self.device_atom_norbs),dtype=torch.complex128)
            for ik,k in enumerate(self.kpoints):
                free_charge_allk += np.real(self.free_charge[str(k)].numpy()) * self.wk[ik]
            interface_poisson.free_charge[atom_gridpoint_index] = free_charge_allk
            
            interface_poisson.phi_old = interface_poisson.phi.copy()
            max_diff_phi = interface_poisson.solve_poisson_NRcycle(method=self.poisson_options['solver'],tolerance=tolerance)
            interface_poisson.phi = interface_poisson.phi + mix_rate*(interface_poisson.phi_old-interface_poisson.phi)
            

            iter_count += 1 # Gummel type iteration
            log.info(msg="Poisson-NEGF iteration: {}    Potential Diff Maximum: {}\n".format(iter_count,max_diff_phi))
            max_diff_list.append(max_diff_phi)

            if max_diff_phi <= err:
                log.info(msg="Poisson-NEGF SCF Converges Successfully!")
                

            if iter_count > max_iter:
                log.info(msg="Warning! Poisson-NEGF iteration exceeds the upper limit of iterations {}".format(int(max_iter)))
                # profiler.stop()
                # with open('profile_report.html', 'w') as report_file:
                #     report_file.write(profiler.output_html())
                # break

        self.poisson_out = {}
        self.poisson_out['potential'] = torch.tensor(interface_poisson.phi)
        self.poisson_out['potential_at_atom'] = self.potential_at_atom
        self.poisson_out['grid_point_number'] = interface_poisson.grid.Np
        self.poisson_out['grid'] = torch.tensor(interface_poisson.grid.grid_coord)
        self.poisson_out['free_charge_at_atom'] = torch.tensor(interface_poisson.free_charge[atom_gridpoint_index])
        self.poisson_out['max_diff_list'] = torch.tensor(max_diff_list)
        torch.save(self.poisson_out, self.results_path+"/poisson.out.pth")

        # calculate transport properties with converged potential
        self.negf_compute(scf_require=False,Vbias=self.potential_at_orb)

        # output the profile report in html format
        # if iter_count <= max_iter: 
        #     profiler.stop()
        #     with open('profile_report.html', 'w') as report_file:
        #         report_file.write(profiler.output_html())

    def negf_compute(self,scf_require=False,Vbias=None):
        
    
        assert scf_require is not None

        self.out['k']=[];self.out['wk']=[]
        if hasattr(self, "uni_grid"): self.out["uni_grid"] = self.uni_grid
    
        for ik, k in enumerate(self.kpoints):


            #  output kpoints information
            # if ik == 0:
            #     log.info(msg="------ k-point for NEGF -----")
            #     log.info(msg="Gamma Center: {0}".format(self.jdata["stru_options"]["gamma_center"]))
            #     log.info(msg="Time Reversal: {0}".format(self.jdata["stru_options"]["time_reversal_symmetry"]))
            #     log.info(msg="k-points Num: {0}".format(len(self.kpoints)))
            #     if len(self.wk)<10:
            #         log.info(msg="k-points: {0}".format(self.kpoints))
            #         log.info(msg="k-points weights: {0}".format(self.wk))
            #     log.info(msg="--------------------------------")

            self.out['k'].append(k)
            self.out['wk'].append(self.wk[ik])
            self.free_charge.update({str(k):torch.zeros_like(torch.tensor(self.device_atom_norbs),dtype=torch.complex128)})
            log.info(msg="Properties computation at k = [{:.4f},{:.4f},{:.4f}]".format(float(k[0]),float(k[1]),float(k[2])))

            if scf_require:
                if self.density_options["method"] == "Fiori":
                    leads = self.stru_options.keys()

                    for ll in leads:
                        if ll.startswith("lead"):
                            if Vbias is not None  and self.density_options["method"] == "Fiori":
                                # set voltage as -1*potential_at_orb[0] and -1*potential_at_orb[-1] for self-energy same as in NanoTCAD
                                if ll == 'lead_L' :
                                    getattr(self.deviceprop, ll).voltage = Vbias[0]
                                else:
                                    getattr(self.deviceprop, ll).voltage = Vbias[-1]

                    self.density.density_integrate_Fiori(
                        e_grid = self.uni_grid, 
                        kpoint=k,
                        Vbias=Vbias,
                        integrate_way = self.density_options["integrate_way"],  
                        deviceprop=self.deviceprop,
                        device_atom_norbs=self.device_atom_norbs,
                        potential_at_atom = self.potential_at_atom,
                        free_charge = self.free_charge,
                        eta_lead = self.eta_lead,
                        eta_device = self.eta_device
                        )
                else:
                    # TODO: add Ozaki support for NanoTCAD-style SCF
                    raise ValueError("Ozaki method does not support Poisson-NEGF SCF in this version.")
                

            # in non-scf case, computing properties in uni_gird
            else:
                if hasattr(self, "uni_grid"):
                    # dE = abs(self.uni_grid[1] - self.uni_grid[0])                       
                    output_freq = int(len(self.uni_grid)/10)

                    for ie, e in enumerate(self.uni_grid):
                        if ie % output_freq == 0:
                            log.info(msg="computing green's function at e = {:.3f}".format(float(e)))
                        leads = self.stru_options.keys()
                        for ll in leads:
                            if ll.startswith("lead"):
                                if Vbias is not None  and self.density_options["method"] == "Fiori":
                                    # set voltage as -1*potential_at_orb[0] and -1*potential_at_orb[-1] for self-energy same as in NanoTCAD
                                    if ll == 'lead_L' :
                                        getattr(self.deviceprop, ll).voltage = Vbias[0]
                                    else:
                                        getattr(self.deviceprop, ll).voltage = Vbias[-1]
                                
                                getattr(self.deviceprop, ll).self_energy(
                                    energy=e, 
                                    kpoint=k, 
                                    eta_lead=self.eta_lead,
                                    method=self.sgf_solver
                                    )
                                # self.out[str(ll)+"_se"][str(e.numpy())] = getattr(self.deviceprop, ll).se
                                
                        self.deviceprop.cal_green_function(
                            energy=e, kpoint=k, 
                            eta_device=self.eta_device,
                            block_tridiagonal=self.block_tridiagonal,
                            Vbias=Vbias
                            )
                        # self.out["gtrans"][str(e.numpy())] = gtrans

                        
                        if self.out_dos:
                            # prop = self.out.setdefault("DOS", [])
                            # prop.append(self.compute_DOS(k))
                            prop = self.out.setdefault('DOS', {})
                            propk = prop.setdefault(str(k), [])
                            propk.append(self.compute_DOS(k))
                        if self.out_tc or self.out_current_nscf:
                            # prop = self.out.setdefault("TC", [])
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
                        
                            
                    # over energy loop in uni_gird
                    # The following code is for output properties before NEGF ends
                    # TODO: check following code for multiple k points calculation
                
                    if self.out_density or self.out_potential:
                        if self.density_options["method"] == "Ozaki":
                            prop_DM_eq = self.out.setdefault('DM_eq', {})
                            prop_DM_neq = self.out.setdefault('DM_neq', {})
                            prop_DM_eq[str(k)], prop_DM_neq[str(k)] = self.compute_density_Ozaki(k,Vbias)
                        elif self.density_options["method"] == "Fiori":
                            log.warning("Fiori method does not support  output density in this version.")
                        else:
                            raise ValueError("Unknown method for density calculation.")
                    if self.out_potential:
                        pass
                    if self.out_dos:
                        self.out["DOS"][str(k)] = torch.stack(self.out["DOS"][str(k)])
                    if self.out_tc or self.out_current_nscf:
                        self.out["T_k"][str(k)] = torch.stack(self.out["T_k"][str(k)])
                    # if self.out_current_nscf:
                    #     self.out["BIAS_POTENTIAL_NSCF"], self.out["CURRENT_NSCF"] = self.compute_current_nscf(k, self.uni_grid, self.out["TC"]) 
                    # computing properties that are not functions of E (improvement can be made here in properties related to integration of energy window of fermi functions)
                    if self.out_current:
                        pass
            
                    # TODO: check the following code for multiple k points calculation
                    if self.out_lcurrent:
                        lcurrent = 0
                        log.info(msg="computing local current at k = [{:.4f},{:.4f},{:.4f}]".format(float(k[0]),float(k[1]),float(k[2])))
                        for i, e in enumerate(self.int_grid):
                            log.info(msg=" computing green's function at e = {:.3f}".format(float(e)))
                            leads = self.stru_options.keys()
                            for ll in leads:
                                if ll.startswith("lead"):
                                    getattr(self.deviceprop, ll).self_energy(
                                        energy=e, 
                                        kpoint=k, 
                                        eta_lead=self.eta_lead,
                                        method=self.sgf_solver
                                        )
                                    
                            self.deviceprop.cal_green_function(
                                energy=e,
                                kpoint=k, 
                                eta_device=self.eta_device, 
                                block_tridiagonal=self.block_tridiagonal
                                )
                            
                            lcurrent += self.int_weight[i] * self.compute_lcurrent(k)

                        prop_local_current = self.out.setdefault('LOCAL_CURRENT', {})
                        prop_local_current[str(k)] = lcurrent


        if scf_require==False:
            self.out["k"] = np.array(self.out["k"])
            self.out['T_avg'] = torch.tensor(self.out['wk']) @ torch.stack(list(self.out["T_k"].values()))
            # TODO:check the following code for multiple k points calculation
            if self.out_current_nscf:
                self.out["BIAS_POTENTIAL_NSCF"], self.out["CURRENT_NSCF"] = self.compute_current_nscf(self.uni_grid, self.out["T_avg"])
            torch.save(self.out, self.results_path+"/negf.out.pth")

                
            

    def get_grid(self,grid_info,structase):
        x_start,x_end,x_num = grid_info.get("x_range",None).split(':')
        xg = np.linspace(float(x_start),float(x_end),int(x_num))

        y_start,y_end,y_num = grid_info.get("y_range",None).split(':')
        yg = np.linspace(float(y_start),float(y_end),int(y_num))
        # yg = np.array([(float(y_start)+float(y_end))/2]) # TODO: temporary fix for 2D case

        z_start,z_end,z_num = grid_info.get("z_range",None).split(':')
        zg = np.linspace(float(z_start),float(z_end),int(z_num))

        device_atom_coords = structase.get_positions()
        xa,ya,za = device_atom_coords[:,0],device_atom_coords[:,1],device_atom_coords[:,2]

        # grid = Grid(xg,yg,zg,xa,ya,za)
        grid = Grid(xg,yg,za,xa,ya,za) #TODO: change back to zg
        return grid       
    
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
    
    def compute_current_nscf(self, ee, tc):
        return self.deviceprop._cal_current_nscf_(ee, tc)

    def compute_density_Ozaki(self, kpoint,Vbias):
        DM_eq, DM_neq = self.density.integrate(deviceprop=self.deviceprop, kpoint=kpoint, Vbias=Vbias)
        return DM_eq, DM_neq
     

    def compute_current(self, kpoint):
        self.deviceprop.cal_green_function(e=self.int_grid, kpoint=kpoint, block_tridiagonal=self.block_tridiagonal)
        return self.devidevicepropce.current
    
    def compute_lcurrent(self, kpoint):
        return self.deviceprop.lcurrent


    def SCF(self):
        pass

