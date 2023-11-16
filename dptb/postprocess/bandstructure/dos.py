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
        
        self.dos_plot_options = jdata
        self.results_path = run_opt.get('results_path')
        self.apiH.update_struct(self.structase)
        self.use_gui = self.dos_plot_options.get("use_gui", False)

    def get_dos(self):
        self.mesh_grid = self.dos_plot_options['mesh_grid']
        self.isgamma = self.dos_plot_options['gamma_center']
        self.kpoints = kmesh_sampling(meshgrid= self.mesh_grid,is_gamma_center=self.isgamma)
        sigma = self.dos_plot_options.get('sigma',0.1)
        npoints = self.dos_plot_options.get('npoints',100)
        width = self.dos_plot_options.get('width',None)
        self.eigenvalues, self.E_fermi = self.get_eigenvalues(kpoints=self.kpoints)

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
        plt.xlabel('E - EF (eV)',fontsize=8)

        plt.tick_params(direction='in')
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/dos.png',dpi=300)
        if self.use_gui:
            plt.show()


    def _calc_dos(self, sigma=0.1, npoints=100,  width=None, updata=False, kpoints=None):
        if kpoints is not None:
            kpoint_use = kpoints
        else:
            kpoint_use = self.kpoints
        nkp = len(kpoint_use)

        if updata:
            self.eigenvalues, self.E_fermi = self.get_eigenvalues(kpoints=kpoint_use)
        else:
            if not hasattr(self, 'eigenvalues'):
                self.eigenvalues, self.E_fermi = self.get_eigenvalues(kpoints=kpoint_use)

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
        self.dos = np.sum(np.exp(-(xx)**2 / (2 * sigma**2)), axis=0) / (sigma * np.sqrt(2 * np.pi)) / nkp

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


class pdoscalc (object):
    def __init__ (self, apiHrk, run_opt, jdata):
        self.apiH = apiHrk
        if isinstance(run_opt['structure'],str):
            self.structase = read(run_opt['structure'])
        elif isinstance(run_opt['structure'],ase.Atoms):
            self.structase = run_opt['structure']
        else:
            raise ValueError('structure must be ase.Atoms or str')
        
        self.pdos_plot_options = jdata
        self.results_path = run_opt.get('results_path')
        self.apiH.update_struct(self.structase)

        self.num_orbs_per_atom = []
        for itype in apiHrk.structure.proj_atom_symbols:
            norbs = apiHrk.structure.proj_atomtype_norbs[itype]
            self.num_orbs_per_atom.append(norbs)
        
        self.use_gui = self.pdos_plot_options.get("use_gui", False)

    def get_pdos(self):
        self.mesh_grid = self.pdos_plot_options['mesh_grid']
        self.isgamma = self.pdos_plot_options['gamma_center']
        self.kpoints = kmesh_sampling(meshgrid= self.mesh_grid,is_gamma_center=self.isgamma)
        sigma = self.pdos_plot_options.get('sigma',0.1)
        npoints = self.pdos_plot_options.get('npoints',100)
        width = self.pdos_plot_options.get('width',None)
        self.eigenvalues, self.E_fermi, self.eigenvectors = self.get_eigenvalues(kpoints=self.kpoints)

        self.omega, self.pdos = self._calc_pdos(sigma=sigma, npoints=npoints, width=width)

        eigenstatus =  {'kpoints': self.kpoints,
                        'omega': self.omega,
                        'pdos':self.pdos,
                        'sigma': sigma,
                        'width': [self.omega.min(), self.omega.max()],
                        'eigenvalues': self.eigenvalues,
                        'E_fermi': self.E_fermi}

        np.save(f'{self.results_path}/proj_DOS',eigenstatus)
        return eigenstatus
        
    def pdos_plot(self, atom_index=None, orbital_index=None):
        
        if atom_index is None:
            atom_index = self.pdos_plot_options['atom_index']
        if orbital_index is None:
            orbital_index = self.pdos_plot_options['orbital_index']

        if isinstance(atom_index, int):
            atom_index = [atom_index]
        if isinstance(orbital_index, int):
            orbital_index = [orbital_index]

        numOrbs = np.array(self.num_orbs_per_atom)
    

        plt.figure(figsize=(5,4),dpi=100)
                
        pdosindex = []
        for ia in atom_index:
            for iorb in orbital_index:
                iind = np.sum(numOrbs[:ia]) + iorb
                pdosindex.append(iind)
                plt.plot(self.omega, self.pdos[iind], '-',lw=1, label=f'atom-{ia} orb-{iorb}')
        #if len(pdosindex) > 1:
        #    pdos_pick = np.sum(self.pdos[pdosindex],axis=0)
        #    plt.plot(self.omega, pdos_pick, '-',lw=1, label='Sum')
        
        plt.legend(fontsize=8)
        plt.xlim(self.omega.min(),self.omega.max())
        plt.xticks(np.linspace(self.omega.min(),self.omega.max(),5),fontsize=8)
        plt.yticks(fontsize=8)
        plt.ylabel('Projected density of states',fontsize=8)
        plt.xlabel('Energy (eV)',fontsize=8)

        plt.tick_params(direction='in')
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/proj_dos.png',dpi=300)
        if self.use_gui:
            plt.show()


    def _calc_pdos(self, sigma=0.1, npoints=100,  width=None, updata=False, kpoints=None):
        if kpoints is not None:
            kpoint_use = kpoints
        else:
            kpoint_use = self.kpoints
        nkp = len(kpoint_use)

        if updata:
            self.eigenvalues, self.E_fermi, self.eigenvectors = self.get_eigenvalues(kpoints=kpoint_use)
        else:
            if not hasattr(self, 'eigenvalues'):
                self.eigenvalues, self.E_fermi, self.eigenvectors = self.get_eigenvalues(kpoints=kpoint_use)
        
        if width is not None:
            emin,emax = width                
            if emin is None:
                emin = self.eigenvalues.min()
            if emax is None:
                emax = self.eigenvalues.max()
        else:
            emin, emax = self.eigenvalues.min()- 5*sigma, self.eigenvalues.max() + 5*sigma

        self.omega = np.linspace(emin, emax, npoints)
        
        # prob = self.eigenvectors.abs()**2
        prob = np.abs(self.eigenvectors)**2
        xx = (self.omega[np.newaxis,np.newaxis,:] - self.eigenvalues[:,:,np.newaxis] + self.E_fermi)
        dos_e_k = np.exp(-(xx)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
        pdos_e_k = prob[:,:,:,np.newaxis]  * dos_e_k[:,:,np.newaxis,:]
        self.pdos = np.sum(np.sum(pdos_e_k,axis=0),axis=0) / nkp

        return self.omega, self.pdos

    
    def get_eigenvalues(self, kpoints):
        all_bonds, hamil_blocks, overlap_blocks = self.apiH.get_HR()
        self.eigenvalues, self.estimated_E_fermi, self.eigenvectors = self.apiH.get_eigenvalues(kpoints, if_eigvec=True)
        if self.pdos_plot_options.get('E_fermi',None) != None:
            self.E_fermi = self.pdos_plot_options['E_fermi']
            log.info(f'set E_fermi from jdata: {self.E_fermi} , While the estimated value is {self.estimated_E_fermi} .')
        else:
            self.E_fermi = self.estimated_E_fermi
            log.info(f'set E_fermi by estimated value {self.estimated_E_fermi} .')

        return self.eigenvalues, self.E_fermi, self.eigenvectors
    
