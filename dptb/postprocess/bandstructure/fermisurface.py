import numpy as np
from dptb.utils.tools import j_must_have
from dptb.utils.make_kpoints import monkhorst_pack,  gamma_center, kmesh_sampling
from dptb.utils.make_kpoints import rot_revlatt_2D, kmesh_fs
from ase.io import read
import ase
from scipy.interpolate import  interp2d, interpn
from dptb.utils.tools import LorentzSmearing, GaussianSmearing
import matplotlib.pyplot as plt
import logging
log = logging.getLogger(__name__)


class fs3dcalc(object):
    def __init__ (self, apiHrk, run_opt, jdata):
        self.apiH = apiHrk
        if isinstance(run_opt['structure'],str):
            self.structase = read(run_opt['structure'])
        elif isinstance(run_opt['structure'],ase.Atoms):
            self.structase = run_opt['structure']
        else:
            raise ValueError('structure must be ase.Atoms or str')
        
        self.fs_plot_options = jdata
        self.results_path = run_opt.get('results_path')
        self.apiH.update_struct(self.structase)

    def get_fs(self):

        mesh_grid = self.fs_plot_options['mesh_grid']
        assert len(mesh_grid)==3, 'mesh_grid must have 3 elements'
        E0 = self.fs_plot_options.get('E0',0)
        sigma = self.fs_plot_options.get('sigma',0.1)
        intpfactor = self.fs_plot_options.get('intpfactor',1)
        self.filename = self.fs_plot_options.get('fsname','FS.bxsf')
        
        N1, N2, N3 = mesh_grid
        Np1, Np2, Np3 = intpfactor*N1, intpfactor*N2, intpfactor*N3
        mesh_grid_intp = [Np1,Np2,Np3]

        line_xyz, self.kpoints = kmesh_fs(meshgrid= mesh_grid)
        self.eigenvalues, self.E_fermi = self.get_eigenvalues(kpoints=self.kpoints)

        # E0 = self.E_fermi + E0
        self.E0 = E0

        ist = np.sum(np.sum(self.eigenvalues - (self.E_fermi + E0 - 10*sigma) < 0, axis=0) / self.eigenvalues.shape[0]  >=1) -1
        ied = np.sum(np.sum(self.eigenvalues - (self.E_fermi + E0 + 10*sigma) > 0, axis=0) /  self.eigenvalues.shape[0] < 1) + 1
        if ist <0:
            ist = 0
        
        eig_pick = self.eigenvalues[:,ist:ied]
        eig_pick = np.reshape(eig_pick,(N1,N2,N3,ied-ist))

        eig_pick_intp = np.zeros((Np1*Np2*Np3, ied-ist))

        _, kpoints_intp = kmesh_fs(meshgrid=mesh_grid_intp)

        for i in range(ied-ist):
        # interpolate eigenvalues to a finer mesh, 3D interpolation scipy.interpolate.interpn is used.
            eig_pick_intp[:,i] = interpn(line_xyz, eig_pick[:,:,:,i],kpoints_intp)

        eig_pick_intp = eig_pick_intp - self.E_fermi
        self.eigenvalues_intp = eig_pick_intp
        self.mesh_grid_intp = mesh_grid_intp


    
    def fs_plot(self):
        log.info('We support plotting Fermi surface with Xcrysden software...')
        log.info('Please install Xcrysden software first...')
        log.info('Here, we generate the bxsf file for Xcrysden to plot Fermi surface.')
        log.info('Genetating ...')

        outfile_name = f'{self.results_path}/{self.filename}'

        self._out2bxsf(mesh_grid=self.mesh_grid_intp, eigenvalues=self.eigenvalues_intp, filename=outfile_name)

        log.info(f'Done! The bxsf file is saved as  {outfile_name}')
        log.info('For visualization, run the following command: xcrysden --bxsf FS.bxsf') 
        log.info('Enjoy it!') 


    def _out2bxsf(self, mesh_grid, eigenvalues, filename=None):
        """ Output eigenvalues to a bxsf file for xcrysden to plot. not support spin-polarized case.
        
        Parameters
        ----------
        mesh_grid: list
            The mesh grid of kpoints, [nkx, nky, nkz]
        eigenvalues: array
            The eigenvalues of the system, shape = (nkx*nky*nkz, nbands)

        Returns
        -------
        None

        """
        nkx,nky,nkz = mesh_grid
        num_total_bands= eigenvalues.shape[1] 
        rev_latt = np.array(np.mat(self.structase.cell).I.T)

        if filename is None:
            outfile_name = f'{self.results_path}/{self.filename}'

        else:
            outfile_name = filename

        # 1: Write headers blocks.
        
        out_file=open(outfile_name,'w')
        out_file.write('BEGIN_INFO\n')
        out_file.write('   #\n')
        out_file.write('   # Case:  {0}\n'.format(self.structase.symbols))
        out_file.write('   #\n')
        out_file.write('   # Launch as: xcrysden --bxsf FS.bxsf\n')
        out_file.write('   #\n')
        out_file.write('   Fermi Energy: {0}\n'.format(self.E0))  # Fermi  energy is set to 0,  surface of constant energy E0, when E0 =0 , it it fermi surface.  
        out_file.write(' END_INFO\n')
        out_file.write('\n')
        out_file.write(' BEGIN_BLOCK_BANDGRID_3D\n')
        out_file.write(' band_energies\n')
        out_file.write('   BEGIN_BANDGRID_3D\n')
        out_file.write("    {:d}\n".format(num_total_bands))
        out_file.write("    {:5d}{:5d}{:5d}\n".format(nkx,nky,nkz))
        out_file.write("    {:16.8f}{:16.8f}{:16.8f}\n".format(0.0, 0.0, 0.0))
        out_file.write('\n'.join(["    " + ''.join(["%16.8f" % xx for xx in row]) for row in rev_latt]))

        # 2: Write band grid for each band 
        # row-major ie C-syntax: 
        # C-syntax:
        #   for (i=0; i<nx; i++)
        #     for (j=0; j<ny; j++)
        #       for (k=0; k<nz; k++)
        #         printf("%f",eigenvalues[i][j][k])
  
        for j in range(eigenvalues.shape[1]): # j is index of band
            out_file.write('   BAND:  {0}\n'.format(j+1)) 
            # reshape band j in shape from (nkx*nky*nkz,) to  (nkx*nky, nkz).
            one_band=np.reshape(eigenvalues[:,j], (nkx*nky,nkz))
            # block_counter is index of y-z block.
            block_counter=0
            # there are 'nkx' number of blocks within one BAND
            for i in range(nkx):
                # there are 'nky' number of lines within one block
                for k in range(nky):
                    # there are 'nkz' number of columns within one line
                    # block_counter*nky+k row will be line to be printed
                    out_file.write('       '+'  '.join(map(str,one_band[block_counter*nky+k,:]))+'\n')
                # each blocks are separated by empty line
                out_file.write('\n')
                block_counter+=1

        # 3: Write ending blocks.  
        out_file.write('   END_BANDGRID_3D\n')
        out_file.write(' END_BLOCK_BANDGRID_3D')
        out_file.close()

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
        self.k_outplane = self.fs_plot_options.get('k_outplane',0.0)

        index_2d = [0,1,2]
        xyzlabel = ['x','y','z']
        index_2d.remove(mesh_grid.index(1))
        self.plane_label = xyzlabel[index_2d[0]] + xyzlabel[index_2d[1]]
        
        self.out_label = xyzlabel[mesh_grid.index(1)]

        rev_latt = np.mat(self.apiH.structure.struct.cell).I.T
        rev_latt_new, newcorr = rot_revlatt_2D(rev_latt,index=index_2d)
        _, self.kpoints = kmesh_fs(meshgrid=mesh_grid)
        self.kpoints = np.array(self.kpoints)
        self.kpoints[:, mesh_grid.index(1)] =self.k_outplane

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

        _, kpointsintp = kmesh_fs(meshgrid=mesh_intp)
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
        plt.savefig(f'{self.results_path}/fs2d_k{self.out_label}_{self.k_outplane}.png',dpi=300)
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