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
from dptb.negf.poisson_init import Grid,Interface3D,Gate,Dielectric

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
        self.kBT = Boltzmann * self.ele_T / eV2J # change to eV
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

        # number of orbitals on atoms in device region
        self.device_atom_norbs = self.negf_hamiltonian.atom_norbs[self.negf_hamiltonian.proj_device_id[0]:self.negf_hamiltonian.proj_device_id[1]]
        np.save(self.results_path+"/device_atom_norbs.npy",self.device_atom_norbs)

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

        ## Poisson equation settings
        self.poisson_options = j_must_have(jdata, "poisson_options")
        # self.LDOS_integral = {}  # for electron density integral
        self.free_charge_nanotcad = {}
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
            # Energy gird is set relative to Fermi level
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
                self.out_density = True
                raise UserWarning("SCF is required, but out_density is set to False. Automatically Setting out_density to True.")
            self.poisson_negf_scf(err=self.poisson_options['err'],tolerance=self.poisson_options['tolerance'],\
                                  max_iter=self.poisson_options['max_iter'],mix_rate=self.poisson_options['mix_rate'])
        else:
            potential_add = None
            self.negf_compute(scf_require=False,Vbias=potential_add)

    def poisson_negf_scf(self,err=1e-6,max_iter=1000,mix_rate=0.3,tolerance=1e-7):
       
        # create real-space grid
        grid = self.get_grid(self.poisson_options["grid"],self.deviceprop.structure)
        
        # create gate
        Gate_list = []
        for gg in range(len(self.gate_region)):
            # xmin,xmax = self.gate_region[gg].get("x_range",None).split(':')
            # ymin,ymax = self.gate_region[gg].get("y_range",None).split(':')
            # zmin,zmax = self.gate_region[gg].get("z_range",None).split(':')
            # gate_init = Gate(float(xmin),float(xmax),float(ymin),float(ymax),float(zmin),float(zmax))
            gate_init = Gate(self.gate_region[gg].get("x_range",None).split(':'),\
                             self.gate_region[gg].get("y_range",None).split(':'),\
                             self.gate_region[gg].get("z_range",None).split(':'))
            gate_init.Ef = float(self.gate_region[gg].get("voltage",None)) # in unit of volt
            Gate_list.append(gate_init)
                      
        # create dielectric
        Dielectric_list = []
        for dd in range(len(self.dielectric_region)):
            # xmin,xmax = self.dielectric_region[dd].get("x_range",None).split(':')
            # ymin,ymax = self.dielectric_region[dd].get("y_range",None).split(':')
            # zmin,zmax = self.dielectric_region[dd].get("z_range",None).split(':')

            dielectric_init = Gate(self.dielectric_region[dd].get("x_range",None).split(':'),\
                self.dielectric_region[dd].get("y_range",None).split(':'),\
                self.dielectric_region[dd].get("z_range",None).split(':'))
            dielectric_init.eps = float(self.dielectric_region[dd].get("relative permittivity",None))
            Dielectric_list.append(dielectric_init)        

        # create interface
        interface_poisson = Interface3D(grid,Gate_list,Dielectric_list)

        #initial guess for electrostatic potential
        log.info(msg="-----Initial guess for electrostatic potential----")
        interface_poisson.solve_poisson(method=self.poisson_options['solver'],tolerance=tolerance)
        np.save(self.results_path+"/initial_guess_phi.npy",interface_poisson.phi)
        atom_gridpoint_index =  list(interface_poisson.grid.atom_index_dict.values())
        np.save(self.results_path+"/initial_guess_phi_at_atom.npy",interface_poisson.phi[atom_gridpoint_index])
        log.info(msg="-------------------------------------------\n")

        max_diff = 1e30; max_diff_list = [] 
        iter_count=0
        while max_diff > err:

            # update Hamiltonian by modifying onsite energy with potential
            atom_gridpoint_index =  list(interface_poisson.grid.atom_index_dict.values())
            np.save(self.results_path+"/atom_gridpoint_index.npy",atom_gridpoint_index)
            self.potential_at_atom = interface_poisson.phi[atom_gridpoint_index] # a vector with length of number of atoms
                       
            potential_list = []
            for i in range(len(self.device_atom_norbs)):
                potential_list.append(self.potential_at_atom[i]*torch.ones(self.device_atom_norbs[i]))
            self.potential_tensor = torch.cat(potential_list)
            torch.save(self.potential_tensor, self.results_path+"/potential_tensor.pth")

            #TODO: check the sign of potential_tensor: -1 is right or not.          
            self.negf_compute(scf_require=True,Vbias=self.potential_tensor)
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
            interface_poisson.free_charge[atom_gridpoint_index] =\
                np.real(self.free_charge_nanotcad[str(self.kpoints[0])].numpy())
            

            interface_poisson.phi_old = interface_poisson.phi.copy()

            max_diff = interface_poisson.solve_poisson(method=self.poisson_options['solver'],tolerance=tolerance)

            interface_poisson.phi = interface_poisson.phi + mix_rate*(interface_poisson.phi_old-interface_poisson.phi)
            

            iter_count += 1
            print('Poisson iteration: ',iter_count,' max_diff: ',max_diff)
            max_diff_list.append(max_diff)


            if iter_count > max_iter:
                log.info(msg="Warning! Poisson iteration exceeds max_iter {}".format(int(max_iter)))
                break

        self.poisson_out = {}
        self.poisson_out['potential'] = torch.tensor(interface_poisson.phi)
        self.poisson_out['grid_point_number'] = interface_poisson.grid.Np
        self.poisson_out['grid'] = torch.tensor(interface_poisson.grid.grid_coord)
        self.poisson_out['free_charge_at_atom'] = torch.tensor(interface_poisson.free_charge[atom_gridpoint_index])
        self.poisson_out['max_diff_list'] = torch.tensor(max_diff_list)
        
        torch.save(self.poisson_out, self.results_path+"/poisson.out.pth")

        # calculate transport properties with converged potential
        self.negf_compute(scf_require=False,Vbias=self.potential_tensor)


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
            self.out["lead_L_se"] = {}
            self.out["lead_R_se"] = {}
            self.out["gtrans"] = {}
            self.out['uni_grid'] = self.uni_grid
            log.info(msg="Properties computation at k = [{:.4f},{:.4f},{:.4f}]".format(float(k[0]),float(k[1]),float(k[2])))

            # computing properties that is functions of E
            if hasattr(self, "uni_grid"):
                self.out["k"] = k
                dE = abs(self.uni_grid[1] - self.uni_grid[0])
                self.free_charge_nanotcad.update({str(k):torch.zeros_like(torch.tensor(self.device_atom_norbs),dtype=torch.complex128)})
                
                output_freq = int(len(self.uni_grid)/10)
                for ie, e in enumerate(self.uni_grid):

                    if ie % output_freq == 0:
                        log.info(msg="computing green's function at e = {:.3f}".format(float(e)))
                    leads = self.stru_options.keys()
                    for ll in leads:
                        if ll.startswith("lead"):
                        # TODO: temporarily set the voltage to -1*potential_tensor[0] and -1*potential_tensor[-1]
                            if Vbias is not None:
                                if ll == 'lead_L' :
                                    getattr(self.deviceprop, ll).voltage = Vbias[0]
                                else:
                                    getattr(self.deviceprop, ll).voltage = Vbias[-1]
                            

                            getattr(self.deviceprop, ll).self_energy(
                                energy=e, 
                                kpoint=k, 
                                eta_lead=self.jdata["eta_lead"],
                                method=self.jdata["sgf_solver"]
                                )
                            self.out[str(ll)+"_se"][str(e.numpy())] = getattr(self.deviceprop, ll).se
                            
                    gtrans = self.deviceprop.cal_green_function(
                        energy=e, 
                        kpoint=k, 
                        eta_device=self.jdata["eta_device"], 
                        block_tridiagonal=self.block_tridiagonal,
                        Vbias=Vbias
                        )
                    
                    self.out["gtrans"][str(e.numpy())] = gtrans

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
                    else:

                        self.get_density_nanotcad(e, k, dE)


            # whether scf_require is True or False, density are computed for Poisson-NEGF SCF
            if self.out_density or self.out_potential:
                self.out["DM_eq"], self.out["DM_neq"] = self.compute_density(k,Vbias)
            
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
            

    def get_grid(self,grid_info,structase):
        x_start,x_end,x_num = grid_info.get("x_range",None).split(':')
        xg = np.linspace(float(x_start),float(x_end),int(x_num))

        y_start,y_end,y_num = grid_info.get("y_range",None).split(':')
        yg = np.linspace(float(y_start),float(y_end),int(y_num))

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
    
    def compute_current_nscf(self, kpoint, ee, tc):
        return self.deviceprop._cal_current_nscf_(ee, tc)

    def compute_density(self, kpoint,Vbias):
        DM_eq, DM_neq = self.density.integrate(deviceprop=self.deviceprop, kpoint=kpoint, Vbias=Vbias)
        return DM_eq, DM_neq

    def compute_current(self, kpoint):
        self.deviceprop.cal_green_function(e=self.int_grid, kpoint=kpoint, block_tridiagonal=self.block_tridiagonal)
        return self.devidevicepropce.current
    
    def compute_lcurrent(self, kpoint):
        return self.deviceprop.lcurrent


    def SCF(self):
        pass


    def get_density_nanotcad(self,e,kpoint,dE,eta_lead=1e-5, eta_device=0.,Vbias=None):

        tx, ty = self.deviceprop.g_trans.shape
        lx, ly = self.deviceprop.lead_L.se.shape
        rx, ry = self.deviceprop.lead_R.se.shape
        x0 = min(lx, tx)
        x1 = min(rx, ty)

        gammaL = torch.zeros(size=(tx, tx), dtype=torch.complex128)
        gammaL[:x0, :x0] += self.deviceprop.lead_L.gamma[:x0, :x0]
        gammaR = torch.zeros(size=(ty, ty), dtype=torch.complex128)
        gammaR[-x1:, -x1:] += self.deviceprop.lead_R.gamma[-x1:, -x1:]

        A_L = torch.mm(torch.mm(self.deviceprop.g_trans,gammaL),self.deviceprop.g_trans.conj().T)
        A_R = torch.mm(torch.mm(self.deviceprop.g_trans,gammaR),self.deviceprop.g_trans.conj().T)

        # Vbias = -1 * potential_tensor
        for Ei_index, Ei_at_atom in enumerate(-1*self.potential_at_atom):
            pre_orbs = sum(self.device_atom_norbs[:Ei_index])
        
            # electron density
            if e >= Ei_at_atom: 
                for j in range(self.device_atom_norbs[Ei_index]):
                    self.free_charge_nanotcad[str(kpoint)][Ei_index] +=\
                    2*(-1)/2/torch.pi*(A_L[pre_orbs+j,pre_orbs+j]*self.deviceprop.lead_L.fermi_dirac(e+self.deviceprop.lead_L.efermi) \
                                  +A_R[pre_orbs+j,pre_orbs+j]*self.deviceprop.lead_R.fermi_dirac(e+self.deviceprop.lead_R.efermi))*dE
                    # 2*(-1)/2/torch.pi*(A_L[pre_orbs+j,pre_orbs+j]*self.deviceprop.lead_L.fermi_dirac(e+self.deviceprop.lead_L.mu) \
                    #               +A_R[pre_orbs+j,pre_orbs+j]*self.deviceprop.lead_R.fermi_dirac(e+self.deviceprop.lead_R.mu))*dE
            # hole density
            else:
                for j in range(self.device_atom_norbs[Ei_index]):
                    self.free_charge_nanotcad[str(kpoint)][Ei_index] +=\
                    2*1/2/torch.pi*(A_L[pre_orbs+j,pre_orbs+j]*(1-self.deviceprop.lead_L.fermi_dirac(e+self.deviceprop.lead_L.efermi)) \
                                  +A_R[pre_orbs+j,pre_orbs+j]*(1-self.deviceprop.lead_R.fermi_dirac(e+self.deviceprop.lead_R.efermi)))*dE