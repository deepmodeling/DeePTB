import numpy as np
from dptb.utils.tools import j_must_have
from dptb.utils.make_kpoints import monkhorst_pack,  gamma_center, kmesh_sampling
from dptb.utils.make_kpoints import rot_revlatt_2D, kmesh_fs
from ase.io import read
import ase
from scipy.interpolate import  interp2d
from dptb.utils.tools import LorentzSmearing, GaussianSmearing
import matplotlib.pyplot as plt
import logging
log = logging.getLogger(__name__)


class fs2dcalc (object):
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
    
    
    def get_fs(self):
        self.fs_plot_options = j_must_have(self.jdata, 'FS2D')
        mesh_grid = self.fs_plot_options['mesh_grid']
        assert len(mesh_grid)==3, 'mesh_grid must have 3 elements'
        assert np.sum(np.array(mesh_grid) <=1)==1 and np.sum(np.array(mesh_grid) >=1)==3, 'mesh_grid must have only one element = 1'
        E0 = self.fs_plot_options.get('E0',0)
        sigma = self.fs_plot_options.get('sigma',0.1)
        intpfactor = self.fs_plot_options.get('intpfactor',1)

        index_2d = [0,1,2]
        xyzlabel = ['x','y','z']
        index_2d.remove(mesh_grid.index(1))
        self.plane_label = xyzlabel[index_2d[0]] + xyzlabel[index_2d[1]]
        rev_latt = np.mat(self.apiH.structure.struct.cell).I.T
        rev_latt_new, newcorr = rot_revlatt_2D(rev_latt,index=index_2d)
        self.kpoints = kmesh_fs(meshgrid=mesh_grid)
        N1, N2 = mesh_grid[index_2d[0]], mesh_grid[index_2d[1]]
        k1 = np.linspace(0,1,N1)
        k2 = np.linspace(0,1,N2)

        self.eigenvalues, self.E_fermi = self.get_eigenvalues(kpoints=self.kpoints)
        E0 = self.E_fermi + E0

        ist = np.sum(np.sum(self.eigenvalues - (E0 - 10*sigma) < 0, axis=0) / self.eigenvalues.shape[0]  >=1) -1
        ied = np.sum(np.sum(self.eigenvalues - (E0 + 10*sigma) > 0, axis=0) /  self.eigenvalues.shape[0] < 1) + 1
        if ist <0:
            ist = 0
        
        eig_pick = self.eigenvalues[:,ist:ied]
        eig_pick = np.reshape(eig_pick,(N1,N2,ied-ist))

        k1intp= np.linspace(0,1,N1 * intpfactor)
        k2intp = np.linspace(0,1,N2 * intpfactor)
        eig_pick_intp = np.zeros(shape=(N1*intpfactor, N2*intpfactor,ied-ist))
        for i in range(eig_pick.shape[2]):
            f =interp2d(k1,k2,eig_pick[:,:,i])
            eig_pick_intp[:,:,i] = f(k1intp,k2intp)

        specfunc_ek = LorentzSmearing(eig_pick_intp, E0, sigma=sigma)
        self.specfunc_k = np.sum(specfunc_ek,axis=2)

        mesh_intp = [1,1,1]
        mesh_intp[index_2d[0]] = N1 * intpfactor
        mesh_intp[index_2d[1]] = N2 * intpfactor

        kpointsintp = kmesh_fs(meshgrid=mesh_intp)
        kpoints_cart  = np.array(kpointsintp * rev_latt_new)
        self.XX=np.reshape(kpoints_cart[:,index_2d[0]],(N1*intpfactor,N2*intpfactor))
        self.YY=np.reshape(kpoints_cart[:,index_2d[1]],(N1*intpfactor,N2*intpfactor))
        
        eigenstatus =  {'XX': self.XX,
                        'YY': self.YY,
                        'fs2d':self.specfunc_k,
                        'sigma': sigma,
                        'smearing': "LorentzSmearing"
                        }
        
        np.save(f'{self.results_path}/DOS',eigenstatus)
        return eigenstatus

    def fs2d_plot(self):
        
        #plt.figure(figsize=(4,4),dpi=100)
        plt.figure()

        plt.contourf(self.XX,self.YY,self.specfunc_k,cmap='jet')

        plt.tick_params(direction='in')
        plt.tight_layout()
        plt.savefig(f'{self.results_path}/fs2d_' + self.plane_label +'.png',dpi=300)
        plt.show()


    def get_eigenvalues(self, kpoints):
        all_bonds, hamil_blocks, overlap_blocks = self.apiH.get_HR()
        self.eigenvalues, self.estimated_E_fermi = self.apiH.get_eigenvalues(kpoints)
        if self.fs_plot_options.get('E_fermi',None) != None:
            self.E_fermi = self.fs_plot_options['E_fermi']
            log.info(f'set E_fermi from jdata: {self.E_fermi} , While the estimated value is {self.estimated_E_fermi} .')
        else:
            self.E_fermi = self.estimated_E_fermi
            log.info(f'set E_fermi by estimated value {self.estimated_E_fermi} .')

        return self.eigenvalues, self.E_fermi