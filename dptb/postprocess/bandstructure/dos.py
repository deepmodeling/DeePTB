import numpy as np
from dptb.utils.tools import j_must_have
from dptb.utils.make_kpoints import monkhorst_pack,  gamma_center, kmesh_sampling
from ase.io import read
import ase
import matplotlib.pyplot as plt
import logging
log = logging.getLogger(__name__)


class doscalc (object):
    def __init__ (self, apiHrk, run_opt, jdata):
        self.apiH = apiHrk
        if isinstance(run_opt['structure'],str):
            self.structase = read(run_opt['structure'])
        elif isinstance(run_opt['structure'],ase.Atoms):
            self.structase = run_opt['structure']
        else:
            raise ValueError('structure must be ase.Atoms or str')
        
        self.jdata = jdata
        self.results_path = run_opt.get('results_path')
        self.apiH.update_struct(self.structase)


    def get_dos(self):
        self.dos_plot_options = j_must_have(self.jdata, 'dos')
        self.mesh_grid = self.dos_plot_options['mesh_grid']
        self.isgamma = self.dos_plot_options['gamma_center']
        self.kpoints = kmesh_sampling(meshgrid= self.mesh_grid,is_gamma_center=self.isgamma)
        sigma = self.dos_plot_options.get('sigma',0.1)
        npoints = self.dos_plot_options.get('npoints',100)
        width = self.dos_plot_options.get('width',None)
        self.eigenvalues, self.estimated_E_fermi = self.get_eigenvalues(kpoints=self.kpoints)

        self.omega, self.dos = self._calc_dos(sigma=sigma, npoints=npoints, width=width)
        
        eigenstatus =  {'kpoints': self.kpoints,
                        'omega': self.omega,
                        'dos':self.dos,
                        'sigma': sigma,
                        'width': [self.omega.min(), self.omega.max()],
                        'eigenvalues': self.eigenvalues,
                        'E_fermi': self.E_fermi}

        np.save(f'{self.results_path}/DOS',eigenstatus)
        return eigenstatus
        
    def dos_plot(self):
        plt.figure(figsize=(5,4),dpi=100)
        
        plt.plot(self.omega, self.dos, 'b-',lw=1)

        plt.xlim(self.omega.min(),self.omega.max())
        plt.xticks(np.linspace(self.omega.min(),self.omega.max(),5),fontsize=8)
        plt.yticks(fontsize=8)
        plt.ylabel('Density of states',fontsize=8)
        plt.xlabel('Energy (eV)',fontsize=8)

        plt.tick_params(direction='in')
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/dos.png',dpi=300)
        plt.show()


    def _calc_dos(self, sigma=0.1, npoints=100,  width=None, updata=False, kpoints=None):
        if updata:
            if kpoints is not None:
                self.eigenvalues, self.E_fermi = self.get_eigenvalues(kpoints=kpoints)
            else:
                self.eigenvalues, self.E_fermi = self.get_eigenvalues(kpoints=self.kpoints)
        else:
            if not hasattr(self, 'eigenvalues'):
                if kpoints is not None:
                    self.eigenvalues, self.E_fermi = self.get_eigenvalues(kpoints=kpoints)
                else:
                    self.eigenvalues, self.E_fermi = self.get_eigenvalues(kpoints=self.kpoints)
        
        if width is not None:
            emin,emax = width                
            if emin is None:
                emin = self.eigenvalues.min()
            if emax is None:
                emax = self.eigenvalues.max()
        else:
            emin, emax = self.eigenvalues.min()- 5*sigma, self.eigenvalues.max() + 5*sigma

        self.omega = np.linspace(emin, emax, npoints)
        
        xx = self.eigenvalues.reshape(-1,1) - self.E_fermi - self.omega.reshape(1,-1)
        self.dos = np.sum(np.exp(-(xx)**2 / (2 * sigma**2)), axis=0) / (sigma * np.sqrt(2 * np.pi))

        return self.omega, self.dos

    
    def get_eigenvalues(self, kpoints):
        all_bonds, hamil_blocks, overlap_blocks = self.apiH.get_HR()
        self.eigenvalues, self.estimated_E_fermi = self.apiH.get_eigenvalues(kpoints)
        if self.dos_plot_options.get('E_fermi',None) != None:
            self.E_fermi = self.dos_plot_options['E_fermi']
            log.info(f'set E_fermi from jdata: {self.E_fermi} , While the estimated value is {self.estimated_E_fermi} .')
        else:
            self.E_fermi = self.estimated_E_fermi
            log.info(f'set E_fermi by estimated value {self.estimated_E_fermi} .')

        return self.eigenvalues, self.E_fermi
    
