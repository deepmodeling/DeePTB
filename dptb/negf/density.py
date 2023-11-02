import torch
from dptb.negf.ozaki_res_cal import Ozaki_residues
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

    def integrate(self, device):
        pass

    def slice(self, device, density, fix_dim="z", vslice=0.3, h=0.01, sigma=0.05, plot=False, optimize=False):
        '''generate a 2D grid of real-space density along transmission direction with Gaussian broadening.
        
        The `slice` function takes in a device structure and density, and slices it along a specified
        dimension to generate a 2D grid of data points, which can be plotted if desired.
        
        Parameters
        ----------
        device
            Device object that represents the device 
        density
            electron density of the device
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
        
        lx, ly, lz = device.structure.cell.array.max(axis=0)
        sx, sy, sz = device.structure.cell.array.min(axis=0)

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
        calculate the onsite density for a given device and density matrix.
    
    '''
    def __init__(self, R, M_cut, n_gauss):
        super(Ozaki, self).__init__()
        self.poles, self.residues = Ozaki_residues(M_cut)
        # here poles are in the unit of (e-mu) / kbT
        self.R = R
        self.n_gauss = n_gauss

    def integrate(self, device, kpoint, eta_lead=1e-5, eta_device=0.):
        '''calculates the equilibrium and non-equilibrium density matrices for a given k-point.
        
        Parameters
        ----------
        device
            the Device Object of the device for which the integration is being performed. 
        kpoint
            point in the Brillouin zone of the lead. It is used to calculate the self-energy and 
            Green's function for the given kpoint.
        eta_lead
            the broadening parameter for the leads in the calculation of the self-energy.
        eta_device
            the broadening parameter for the device in the calculation of the Green function.
        
        Returns
        -------
            The function `integrate` returns two variables: `DM_eq` and `DM_neq`.
            DM_eq is the equilibrium density matrix, and DM_neq is the non-equilibrium density matrix.
        
        '''
        kBT = device.kBT
        # add 0th order moment
        poles = 1j* self.poles * kBT + device.lead_L.mu - device.mu # left lead expression for rho_eq
        device.lead_L.self_energy(kpoint=kpoint, energy=1j*self.R-device.mu)
        device.lead_R.self_energy(kpoint=kpoint, energy=1j*self.R-device.mu)
        device.green_function(energy=1j*self.R-device.mu, kpoint=kpoint, block_tridiagonal=False)
        g0 = device.grd[0]
        DM_eq = 1.0j * self.R * g0
        for i, e in enumerate(poles):
            device.lead_L.self_energy(kpoint=kpoint, energy=e, eta_lead=eta_lead)
            device.lead_R.self_energy(kpoint=kpoint, energy=e, eta_lead=eta_lead)
            device.green_function(energy=e, kpoint=kpoint, block_tridiagonal=False, eta_device=eta_device)
            term = ((-4 * 1j * kBT) * device.grd[0] * self.residues[i]).imag
            DM_eq -= term
        
        DM_eq = DM_eq.real

        if abs(device.lead_L.voltage - device.lead_R.voltage) > 1e-14:
            # calculating Non-equilibrium density
            xl, xu = min(device.lead_L.voltage, device.lead_R.voltage), max(device.lead_L.voltage, device.lead_R.voltage)
            xl, xu = xl - 8*kBT, xu + 8*kBT
            xs, wlg = gauss_xw(xl=torch.scalar_tensor(xl), xu=torch.scalar_tensor(xu), n=self.n_gauss)
            DM_neq = 0.
            for i, e in enumerate(xs):
                device.lead_L.self_energy(kpoint=kpoint, energy=e, eta_lead=eta_lead)
                device.lead_R.self_energy(kpoint=kpoint, energy=e, eta_lead=eta_lead)
                device.green_function(e=e, kpoint=kpoint, block_tridiagonal=False, eta_device=eta_device)
                ggg = torch.mm(torch.mm(device.grd[0], device.lead_R.gamma), device.grd[0].conj().T).real
                ggg = ggg * (device.lead_R.fermi_dirac(e+device.mu) - device.lead_L.fermi_dirac(e+device.mu))
                DM_neq = DM_neq + wlg[i] * ggg
        else:
            DM_neq = 0.

        return DM_eq, DM_neq
    
    def get_density_onsite(self, device, DM):
        '''calculate the onsite density for a given device and density matrix.
        
        Parameters
        ----------
        device
            the Device Object of the device for which the integration is being performed. 
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


        norbs = [0]+device.norbs_per_atom
        accmap = np.cumsum(norbs)
        onsite_density = torch.stack([DM[accmap[i]:accmap[i+1]].sum() for i in range(len(accmap)-1)])
        
        onsite_density = torch.cat([torch.from_numpy(device.structure.positions), onsite_density.unsqueeze(-1)], dim=-1)

        return onsite_density