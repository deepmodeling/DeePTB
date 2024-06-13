import torch
from dptb.negf.ozaki_res_cal import ozaki_residues
from dptb.negf.areshkin_pole_sum import pole_maker
import numpy as np
from dptb.negf.negf_utils import gauss_xw
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

'''
The density class is used to calculate the density of states of the device.
1. Density is the basic class, which contains the integrate method and slice method.
2. Ozaki  is used to calculate the density of states using the Ozaki method.
(At this stage, it only supports the calculation of the density with Ozaki method)
'''

class Density(object):
    '''Density Class

    Density is the basic density object, which contains the integrate method and slice method.

    Method
    ----------
    slice
        generate a 2D grid of real-space density with Gaussian broadening.

    '''
    def __init__(self) -> None:
        pass

    def integrate(self, deviceprop):
        pass

    def slice(self, deviceprop, density, fix_dim:str="z", vslice:float=0.3, h:float=0.01, sigma:float=0.05, plot:bool=False, optimize:bool=False):
        '''generate a 2D grid of real-space density along transmission direction with Gaussian broadening.
        
        The `slice` function takes in a device structure and density, and slices it along a specified
        dimension to generate a 2D grid of data points, which can be plotted if desired.
        
        Parameters
        ----------
        deviceprop
            DeviceProperty object that represents the device 
        density
            electron density of the device calcualted from the instance method, here the Ozaki method.
        fix_dim
            transmission direction, default is "z"
        vslice
            the value at which the slice is taken in the fixed dimension. 
        h
            the step size for the grid spacing. 
        sigma
            The parameter "sigma" represents the standard deviation of the Gaussian distribution used in the
        calculation of the density. 
        plot
            a boolean value that determines whether or not to generate a plot of the sliced density. 
        optimize
            a boolean flag that determines whether to optimize the slicing dimensions based on the density data. 
        
        Returns
        -------
            three variables: X, Y, and data.
        
        '''
        
        lx, ly, lz = deviceprop.structure.cell.array.max(axis=0)
        sx, sy, sz = deviceprop.structure.cell.array.min(axis=0)

        if optimize:
            lx, ly, lz = (density[:,:3] + 5 * sigma).max(dim=0)[0]
            sx, sy, sz = (density[:,:3] - 5 * sigma).min(dim=0)[0]
        
        index = {"x":[1,2],"y":[0,2],"z":[0,1]}
        if fix_dim == "x":
            X, Y = torch.arange(sy,ly+h, h), torch.arange(sz,lz+h, h)
            grid = torch.meshgrid(torch.scalar_tensor(vslice), torch.arange(sy,ly+h, h), torch.arange(sz,lz+h, h))
        elif fix_dim == "y":
            X, Y = torch.arange(sx,lx+h, h), torch.arange(sz,lz+h, h)
            grid = torch.meshgrid(torch.arange(sx,lx+h, h), torch.scalar_tensor(vslice), torch.arange(sz,lz+h, h))
        elif fix_dim == "z":
            X, Y = torch.arange(sx,lx+h, h), torch.arange(sy,ly+h, h)
            grid = torch.meshgrid(torch.arange(sx,lx+h, h), torch.arange(sy,ly+h, h), torch.scalar_tensor(vslice))
        else:
            log.error("The fix_dim parameters only allow x/y/z.")
            raise ValueError
        
        grid = torch.stack(grid).view(3,-1).T
        dist = torch.cdist(grid, density[:,:3].float(), p=2)**2

        data = (2*torch.pi*sigma)**-0.5 * density[:,-1].unsqueeze(0) * torch.exp(-dist/(2*sigma**2))

        data = data.sum(dim=1)

        if plot:
            norm = mpl.colors.Normalize(vmin=0., vmax=0.55)
            xmin, xmax = X.min(), X.max()
            ymin, ymax = Y.min(), Y.max()
            ax = plt.axes(xlim=(xmin,xmax), ylim=(ymin,ymax))
            pc = ax.pcolor(X,Y,data.reshape(len(X), len(Y)).T, cmap="Reds", norm=norm)
            plt.colorbar(pc, ax=ax)
            plt.plot()
        
        return X, Y, data

class Ozaki(Density):
    '''calculates the equilibrium and non-equilibrium density with Ozaki method.
        
    The `Ozaki` class is a subclass of `Density` that calculates the equilibrium and non-equilibrium
    density matrices and returns the onsite density. The Ozaki method details can be found in
    T. Ozaki, Continued Fraction Representation of the Fermi-Dirac Function for Large-Scale Electronic Structure 
    Calculations, Phys. Rev. B 75, 035123 (2007).


    Property
    ----------
    poles
        the poles of Ozaki method
    residues
        the residues of Ozaki method
    R
        radius parameter in Ozaki method
    n_gauss
        the number of points in the Gauss-Legendre quadrature method


    Method
    ----------
    integrate
        calculates the equilibrium and non-equilibrium density matrices for a given k-point.
    on_site_density
        calculate the onsite density for a given deviceprop and density matrix.
    
    '''
    def __init__(self, R, M_cut, n_gauss):
        super(Ozaki, self).__init__()
        self.poles, self.residues = ozaki_residues(M_cut)
        # here poles are in the unit of (e-mu) / kbT
        self.R = R
        self.n_gauss = n_gauss

    def integrate(self, deviceprop, kpoint, eta_lead=1e-5, eta_device=0.,Vbias=None):
        '''calculates the equilibrium and non-equilibrium density matrices for a given k-point.
        
        Parameters
        ----------
        deviceprop
            the Device Object of the DeviceProperty for which the integration is being performed. 
        kpoint
            point in the Brillouin zone of the lead. It is used to calculate the self-energy and 
            Green's function for the given kpoint.
        eta_lead
            the broadening parameter for the leads in the calculation of the self-energy.
        eta_device
            the broadening parameter for the deviceprop in the calculation of the Green function.
        
        Returns
        -------
            The function `integrate` returns two variables: `DM_eq` and `DM_neq`.
            DM_eq is the equilibrium density matrix, and DM_neq is the non-equilibrium density matrix.
        
        '''
        kBT = deviceprop.kBT
        # add 0th order moment
        poles = 1j* self.poles * kBT + deviceprop.lead_L.mu - deviceprop.mu # left lead expression for rho_eq
        deviceprop.lead_L.self_energy(kpoint=kpoint, energy=1j*self.R-deviceprop.mu)
        deviceprop.lead_R.self_energy(kpoint=kpoint, energy=1j*self.R-deviceprop.mu)
        deviceprop.cal_green_function(energy=1j*self.R-deviceprop.mu, kpoint=kpoint, block_tridiagonal=False,
                                      Vbias = Vbias)
        g0 = deviceprop.grd[0]
        DM_eq = 1.0j * self.R * g0
        for i, e in enumerate(poles):
            deviceprop.lead_L.self_energy(kpoint=kpoint, energy=e, eta_lead=eta_lead)
            deviceprop.lead_R.self_energy(kpoint=kpoint, energy=e, eta_lead=eta_lead)
            deviceprop.cal_green_function(energy=e, kpoint=kpoint, block_tridiagonal=False, eta_device=eta_device,\
                                          Vbias = Vbias)
            term = ((-4 * 1j * kBT) * deviceprop.grd[0] * self.residues[i]).imag
            DM_eq -= term
        
        DM_eq = DM_eq.real

        if abs(deviceprop.lead_L.voltage - deviceprop.lead_R.voltage) > 1e-14:
            # calculating Non-equilibrium density
            xl, xu = min(deviceprop.lead_L.voltage, deviceprop.lead_R.voltage), max(deviceprop.lead_L.voltage, deviceprop.lead_R.voltage)
            xl, xu = xl - 8*kBT, xu + 8*kBT
            xs, wlg = gauss_xw(xl=torch.scalar_tensor(xl), xu=torch.scalar_tensor(xu), n=self.n_gauss)
            DM_neq = 0.
            for i, e in enumerate(xs):
                deviceprop.lead_L.self_energy(kpoint=kpoint, energy=e, eta_lead=eta_lead)
                deviceprop.lead_R.self_energy(kpoint=kpoint, energy=e, eta_lead=eta_lead)
                deviceprop.cal_green_function(energy=e, kpoint=kpoint, block_tridiagonal=False, eta_device=eta_device)
                gr_gamma_ga = torch.mm(torch.mm(deviceprop.grd[0], deviceprop.lead_R.gamma), deviceprop.grd[0].conj().T).real
                gr_gamma_ga = gr_gamma_ga * (deviceprop.lead_R.fermi_dirac(e+deviceprop.mu) - deviceprop.lead_L.fermi_dirac(e+deviceprop.mu))
                DM_neq = DM_neq + wlg[i] * gr_gamma_ga
        else:
            DM_neq = 0.

        return DM_eq, DM_neq
    
    def get_density_onsite(self, deviceprop, DM):
        '''calculate the onsite density for a given deviceprop and density matrix.
        
        Parameters
        ----------
        deviceprop
            the Device Object of the DeviceProperty for which the integration is being performed. 
        DM
           a given density matrix.
        
        Returns
        -------
            the onsite density, which is a tensor containing the positions of atoms in the device structure and
        the corresponding density values for each atom.
        
        '''
        # assume DM is a cubic tensor
        if len(DM.shape) == 2:
            DM = DM.diag()
        elif not len(DM.shape) == 1:
            log.error("The DM must be the of shape [norbs] or [norbs, norbs]")


        norbs = [0]+deviceprop.norbs_per_atom
        accmap = np.cumsum(norbs)
        onsite_density = torch.stack([DM[accmap[i]:accmap[i+1]].sum() for i in range(len(accmap)-1)])
        
        onsite_density = torch.cat([torch.from_numpy(deviceprop.structure.positions), onsite_density.unsqueeze(-1)], dim=-1)

        return onsite_density
    


class Fiori(Density):

    def __init__(self, n_gauss=None):
        super(Fiori, self).__init__()
        self.n_gauss = n_gauss
        self.xs = None
        self.wlg = None
        self.e_grid_Fiori = None

    def density_integrate_Fiori(self,e_grid,kpoint,Vbias,block_tridiagonal,subblocks,integrate_way,deviceprop,
                                device_atom_norbs,potential_at_atom,free_charge, 
                                eta_lead=1e-5, eta_device=1e-5):
        if integrate_way == "gauss":
            assert self.n_gauss is not None, "n_gauss must be set in the Fiori class"
            if self.xs is None:
                self.xs, self.wlg = gauss_xw(xl=torch.scalar_tensor(e_grid[0]), xu=torch.scalar_tensor(e_grid[-1]), n=self.n_gauss)
                # self.xs = self.xs.numpy();self.wlg = self.wlg.numpy()
                self.e_grid_Fiori = e_grid
            elif self.e_grid_Fiori[0] != e_grid[0] or self.e_grid_Fiori[-1] != e_grid[-1]:
                self.xs, self.wlg = gauss_xw(xl=torch.scalar_tensor(e_grid[0]), xu=torch.scalar_tensor(e_grid[-1]), n=self.n_gauss)
                # self.xs = self.xs.numpy();self.wlg = self.wlg.numpy()
                self.e_grid_Fiori = e_grid
            integrate_range = self.xs
            pre_factor = self.wlg
        elif integrate_way == "direct":
            dE = e_grid[1] - e_grid[0]
            integrate_range = e_grid
            pre_factor = dE * torch.ones(len(e_grid))
        else:
            raise ValueError("integrate_way only supports gauss and direct in this version")



        for eidx, e in enumerate(integrate_range):
            deviceprop.lead_L.self_energy(kpoint=kpoint, energy=e, eta_lead=eta_lead)
            deviceprop.lead_R.self_energy(kpoint=kpoint, energy=e, eta_lead=eta_lead)
            deviceprop.cal_green_function(energy=e, kpoint=kpoint, block_tridiagonal=block_tridiagonal, eta_device=eta_device,Vbias = Vbias)

            tx, ty = deviceprop.g_trans.shape
            lx, ly = deviceprop.lead_L.se.shape
            rx, ry = deviceprop.lead_R.se.shape
            x0 = min(lx, tx)
            x1 = min(rx, ty)

            gammaL = torch.zeros(size=(tx, tx), dtype=torch.complex128)
            gammaL[:x0, :x0] += deviceprop.lead_L.gamma[:x0, :x0]
            gammaR = torch.zeros(size=(ty, ty), dtype=torch.complex128)
            gammaR[-x1:, -x1:] += deviceprop.lead_R.gamma[-x1:, -x1:]
            
            if not block_tridiagonal:
                A_Rd = [torch.mm(torch.mm(deviceprop.grd[i],gammaR),deviceprop.grd[i].conj().T) for i in range(len(deviceprop.grd))]
            else:
                A_Rd = [torch.mm(torch.mm(deviceprop.gr_lc[i],gammaR[-x1:, -x1:]),deviceprop.gr_lc[i].conj().T) for i in range(len(deviceprop.gr_lc))]
            
            A_Ld = [1j*(deviceprop.grd[i]-deviceprop.grd[i].conj().T)-A_Rd[i] for i in range(len(A_Rd))]
            gnd = [A_Ld[i]*deviceprop.lead_L.fermi_dirac(e+deviceprop.lead_L.efermi) \
                    +A_Rd[i]*deviceprop.lead_R.fermi_dirac(e+deviceprop.lead_R.efermi) for i in range(len(A_Ld))]
            gpd = [A_Ld[i] + A_Rd[i] - gnd[i] for i in range(len(A_Ld))]
                

            # Vbias = -1 * potential_at_orb
            for atom_index, Ei_at_atom in enumerate(-1*potential_at_atom):
                pre_orbs = sum(device_atom_norbs[:atom_index])
                last_orbs = pre_orbs + device_atom_norbs[atom_index]
                # electron density
                if e >= Ei_at_atom: 
                    if not block_tridiagonal:
                        free_charge[str(kpoint)][atom_index] +=\
                            pre_factor[eidx]*2*(-1)/2/torch.pi*torch.trace(gnd[0][pre_orbs:last_orbs,pre_orbs:last_orbs])
                        # free_charge[str(kpoint)][atom_index] +=\
                        #     pre_factor[eidx]*2*(-1)/2/torch.pi*torch.trace(deviceprop.gnd[0][pre_orbs:last_orbs,pre_orbs:last_orbs])                            
                    else:
                        block_indexs,orb_start,orb_end = self.get_subblock_index(subblocks,atom_index,device_atom_norbs)
                        if len(block_indexs) == 1:
                            free_charge[str(kpoint)][atom_index] += \
                            pre_factor[eidx]*2*(-1)/2/torch.pi*torch.trace(gnd[block_indexs[0]][orb_start:orb_end,orb_start:orb_end])
                        else:
                            for bindex in block_indexs:
                                if bindex == block_indexs[0]:
                                    free_charge[str(kpoint)][atom_index] += \
                                    pre_factor[eidx]*2*(-1)/2/torch.pi*torch.trace(gnd[bindex][orb_start:,orb_start:])
                                elif bindex == block_indexs[-1]:
                                    free_charge[str(kpoint)][atom_index] += \
                                    pre_factor[eidx]*2*(-1)/2/torch.pi*torch.trace(gnd[bindex][:orb_end,:orb_end])
                                else:
                                    free_charge[str(kpoint)][atom_index] += \
                                    pre_factor[eidx]*2*(-1)/2/torch.pi*torch.trace(gnd[bindex])
                # hole density
                else:
                    if not block_tridiagonal:                      
                        free_charge[str(kpoint)][atom_index] +=\
                        pre_factor[eidx]*2/2/torch.pi*torch.trace(gpd[0][pre_orbs:last_orbs,pre_orbs:last_orbs])
                        # free_charge[str(kpoint)][atom_index] += pre_factor[eidx]*2*1/2/torch.pi*torch.trace(gpd[0][pre_orbs:last_orbs,pre_orbs:last_orbs])
        
                    else:
                        block_indexs,orb_start,orb_end = self.get_subblock_index(subblocks,atom_index,device_atom_norbs)
                        if len(block_indexs) == 1:
                            free_charge[str(kpoint)][atom_index] += \
                            pre_factor[eidx]*2*1/2/torch.pi*torch.trace(gpd[block_indexs[0]][orb_start:orb_end,orb_start:orb_end])
                        else:
                            for bindex in block_indexs:
                                if bindex == block_indexs[0]:
                                    free_charge[str(kpoint)][atom_index] += \
                                    pre_factor[eidx]*2*1/2/torch.pi*torch.trace(gpd[bindex][orb_start:,orb_start:])
                                elif bindex == block_indexs[-1]:
                                    free_charge[str(kpoint)][atom_index] += \
                                    pre_factor[eidx]*2*1/2/torch.pi*torch.trace(gpd[bindex][:orb_end,:orb_end])
                                else:
                                    free_charge[str(kpoint)][atom_index] += \
                                    pre_factor[eidx]*2*1/2/torch.pi*torch.trace(gpd[bindex])
                            
    def get_subblock_index(self,subblocks,atom_index,device_atom_norbs):
        # print('atom_index:',atom_index)
        # print('subblocks:',subblocks)
        subblocks_cumsum = [0]+list(np.cumsum(subblocks))
        # print('subblocks_cumsum:',subblocks_cumsum)
        pre_orbs = sum(device_atom_norbs[:atom_index])
        last_orbs = pre_orbs + device_atom_norbs[atom_index]

        # print('pre_orbs:',pre_orbs)
        # print('last_orbs:',last_orbs)

        block_index = []
        for i in range(len(subblocks_cumsum)-1):
            if pre_orbs >= subblocks_cumsum[i] and last_orbs <= subblocks_cumsum[i+1]:
                block_index.append(i)
                orb_start = pre_orbs - subblocks_cumsum[i]
                orb_end = last_orbs - subblocks_cumsum[i]
                # print('1')
                break
            elif pre_orbs >= subblocks_cumsum[i] and pre_orbs < subblocks_cumsum[i+1] and last_orbs > subblocks_cumsum[i+1]:
                block_index.append(i)
                orb_start = pre_orbs - subblocks_cumsum[i]
                for j in range(i+1,len(subblocks_cumsum)-1):
                    block_index.append(j)
                    if last_orbs <= subblocks_cumsum[j+1]:
                        orb_end = last_orbs - subblocks_cumsum[j]
                        # print('2')
                        break
        # print('block_index',block_index)
        # print('orb_start',orb_start)
        # print('orb_end',orb_end)
        return block_index,orb_start,orb_end                 



    # def density_integrate_Fiori_gauss(self,e_grid,kpoint,Vbias,deviceprop,device_atom_norbs,potential_at_atom,free_charge, eta_lead=1e-5, eta_device=1e-5):

    #     if self.xs is None:
    #         self.xs, self.wlg = gauss_xw(xl=torch.scalar_tensor(e_grid[0]), xu=torch.scalar_tensor(e_grid[-1]), n=self.n_gauss)
    #         # self.xs = self.xs.numpy();self.wlg = self.wlg.numpy()
    #         self.e_grid_Fiori = e_grid
    #     elif self.e_grid_Fiori[0] != e_grid[0] or self.e_grid_Fiori[-1] != e_grid[-1]:
    #         self.xs, self.wlg = gauss_xw(xl=torch.scalar_tensor(e_grid[0]), xu=torch.scalar_tensor(e_grid[-1]), n=self.n_gauss)
    #         # self.xs = self.xs.numpy();self.wlg = self.wlg.numpy()
    #         self.e_grid_Fiori = e_grid

    #     for eidx, e in enumerate(self.xs):

    #         deviceprop.lead_L.self_energy(kpoint=kpoint, energy=e, eta_lead=eta_lead)
    #         deviceprop.lead_R.self_energy(kpoint=kpoint, energy=e, eta_lead=eta_lead)
    #         deviceprop.cal_green_function(energy=e, kpoint=kpoint, block_tridiagonal=False, eta_device=eta_device,Vbias = Vbias)

    #         tx, ty = deviceprop.g_trans.shape
    #         lx, ly = deviceprop.lead_L.se.shape
    #         rx, ry = deviceprop.lead_R.se.shape
    #         x0 = min(lx, tx)
    #         x1 = min(rx, ty)

    #         gammaL = torch.zeros(size=(tx, tx), dtype=torch.complex128)
    #         gammaL[:x0, :x0] += deviceprop.lead_L.gamma[:x0, :x0]
    #         gammaR = torch.zeros(size=(ty, ty), dtype=torch.complex128)
    #         gammaR[-x1:, -x1:] += deviceprop.lead_R.gamma[-x1:, -x1:]

    #         A_L = torch.mm(torch.mm(deviceprop.g_trans,gammaL),deviceprop.g_trans.conj().T)
    #         A_R = torch.mm(torch.mm(deviceprop.g_trans,gammaR),deviceprop.g_trans.conj().T)

    #         # Vbias = -1 * potential_at_orb
    #         for atom_index, Ei_at_atom in enumerate(-1*potential_at_atom):
    #             pre_orbs = sum(device_atom_norbs[:atom_index])
            
    #             # electron density
    #             if e >= Ei_at_atom: 
    #                 for j in range(device_atom_norbs[atom_index]):
    #                     free_charge[str(kpoint)][atom_index] +=\
    #                     self.wlg[eidx]*2*(-1)/2/torch.pi*(A_L[pre_orbs+j,pre_orbs+j]*deviceprop.lead_L.fermi_dirac(e+deviceprop.lead_L.efermi) \
    #                                 +A_R[pre_orbs+j,pre_orbs+j]*deviceprop.lead_R.fermi_dirac(e+deviceprop.lead_R.efermi))

    #             # hole density
    #             else:
    #                 for j in range(device_atom_norbs[atom_index]):
    #                     free_charge[str(kpoint)][atom_index] +=\
    #                     self.wlg[eidx]*2*1/2/torch.pi*(A_L[pre_orbs+j,pre_orbs+j]*(1-deviceprop.lead_L.fermi_dirac(e+deviceprop.lead_L.efermi)) \
    #                                 +A_R[pre_orbs+j,pre_orbs+j]*(1-deviceprop.lead_R.fermi_dirac(e+deviceprop.lead_R.efermi)))