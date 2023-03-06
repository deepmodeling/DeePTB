import numpy as np
from dptb.utils.tools import j_must_have
from dptb.utils.make_kpoints  import ase_kpath, abscus_kpath
from ase.io import read
import ase
import matplotlib.pyplot as plt

class bandcalc (object):
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
    
    def get_bands(self):
        self.band_plot_options = j_must_have(self.jdata, 'bandstructure')
        kline_type = self.band_plot_options['kline_type']

        
        if kline_type == 'ase_kpath':
            kpath = self.band_plot_options['kpath']
            nkpoints = self.band_plot_options['nkpoints']
            self.klist, self.xlist, self.high_sym_kpoints, self.labels = ase_kpath(structase=self.structase,
                                                 pathstr=kpath, total_nkpoints=nkpoints)
        elif kline_type == 'abscus':
            kpath = self.band_plot_options['kpath']
            self.labels = self.band_plot_options['klabels']
            self.klist, self.xlist, self.high_sym_kpoints  = abscus_kpath(structase=self.structase, kpath=kpath)


        all_bonds, hamil_blocks, overlap_blocks = self.apiH.get_HR()
        self.eigenvalues, self.estimated_E_fermi = self.apiH.get_eigenvalues(self.klist)

        if self.jdata.get('E_fermi',None) != None:
            self.E_fermi = self.jdata['E_fermi']
        else:
            self.E_fermi = 0.0

        eigenstatus = {'klist': self.klist,
                        'xlist': self.xlist,
                        'high_sym_kpoints': self.high_sym_kpoints,
                        'labels': self.labels,
                        'eigenvalues': self.eigenvalues,
                        'E_fermi': self.E_fermi }

        np.save(f'{self.results_path}/eigenstatus',eigenstatus)

        return  eigenstatus

    
    def get_HR(self):
        all_bonds, hamil_blocks, overlap_blocks = self.apiH.get_HR()
        
        return all_bonds, hamil_blocks, overlap_blocks

    def band_plot(self):
        emin = self.band_plot_options.get('emin')
        emax = self.band_plot_options.get('emax')

        plt.figure(figsize=(5,5),dpi=100)
        plt.plot(self.xlist, self.eigenvalues - self.E_fermi, 'r-',lw=1)
        for ii in self.high_sym_kpoints:
            plt.axvline(ii,color='gray',lw=1,ls='--')
        plt.tick_params(direction='in')
        if not (emin is None or emax is None):
            plt.ylim(emin,emax)
        plt.xlim(self.xlist.min(),self.xlist.max())
        plt.ylabel('E - EF (eV)',fontsize=12)
        plt.yticks(fontsize=12)
        plt.xticks(self.high_sym_kpoints, self.labels, fontsize=12)
        plt.savefig(f'{self.results_path}/band.png',dpi=300)
        plt.show()



    

        
    
        

    

