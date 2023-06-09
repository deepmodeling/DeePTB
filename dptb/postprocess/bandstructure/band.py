import numpy as np
from dptb.utils.tools import j_must_have
from dptb.utils.make_kpoints  import ase_kpath, abacus_kpath, vasp_kpath
from ase.io import read
import ase
import matplotlib.pyplot as plt
import matplotlib
import logging
log = logging.getLogger(__name__)
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

class bandcalc(object):
    def __init__ (self, apiHrk, run_opt, jdata):
        self.apiH = apiHrk
        if isinstance(run_opt['structure'],str):
            self.structase = read(run_opt['structure'])
        elif isinstance(run_opt['structure'],ase.Atoms):
            self.structase = run_opt['structure']
        else:
            raise ValueError('structure must be ase.Atoms or str')
        
        self.band_plot_options = jdata
        self.results_path = run_opt.get('results_path')
        self.apiH.update_struct(self.structase)

        self.ref_band = self.band_plot_options.get("ref_band", None)
    
    def get_bands(self):
        kline_type = self.band_plot_options['kline_type']

        
        if kline_type == 'ase':
            kpath = self.band_plot_options['kpath']
            nkpoints = self.band_plot_options['nkpoints']
            self.klist, self.xlist, self.high_sym_kpoints, self.labels = ase_kpath(structase=self.structase,
                                                 pathstr=kpath, total_nkpoints=nkpoints)
        elif kline_type == 'abacus':
            kpath = self.band_plot_options['kpath']
            self.labels = self.band_plot_options['klabels']
            self.klist, self.xlist, self.high_sym_kpoints  = abacus_kpath(structase=self.structase, kpath=kpath)
        
        elif kline_type == 'vasp':
            kpath = self.band_plot_options['kpath']
            high_sym_kpoints_dict = self.band_plot_options['high_sym_kpoints']
            number_in_line = self.band_plot_options['number_in_line']
            self.klist, self.xlist, self.high_sym_kpoints, self.labels = vasp_kpath(structase=self.structase,
                                                 pathstr=kpath, high_sym_kpoints_dict=high_sym_kpoints_dict, number_in_line=number_in_line)
        else:
            log.error('Error, now, kline_type only support ase_kpath, abacus, or vasp.')
            raise ValueError
        
        all_bonds, hamil_blocks, overlap_blocks = self.apiH.get_HR()
        self.eigenvalues, self.estimated_E_fermi = self.apiH.get_eigenvalues(self.klist)

        if self.band_plot_options.get('E_fermi',None) != None:
            self.E_fermi = self.band_plot_options['E_fermi']
            log.info(f'set E_fermi from jdata: {self.E_fermi}, While the estimated value in line-mode is {self.estimated_E_fermi}')
        else:
            self.E_fermi = 0.0
            log.info(f'set E_fermi = 0.0, While the estimated value in line-mode is {self.estimated_E_fermi}')

        eigenstatus = {'klist': self.klist,
                        'xlist': self.xlist,
                        'high_sym_kpoints': self.high_sym_kpoints,
                        'labels': self.labels,
                        'eigenvalues': self.eigenvalues,
                        'E_fermi': self.E_fermi }

        np.save(f'{self.results_path}/bandstructure',eigenstatus)

        return  eigenstatus

    
    def get_HR(self):
        all_bonds, hamil_blocks, overlap_blocks = self.apiH.get_HR()
        
        return all_bonds, hamil_blocks, overlap_blocks

    def band_plot(self):
        matplotlib.rcParams['font.size'] = 7
        matplotlib.rcParams['pdf.fonttype'] = 42
        # plt.rcParams['font.sans-serif'] = ['Times New Roman']

        emin = self.band_plot_options.get('emin')
        emax = self.band_plot_options.get('emax')

        fig = plt.figure(figsize=(4.5,4),dpi=100)

        ax = fig.add_subplot(111)

        band_color = '#5d5d5d'
        # plot the line
        
        if self.ref_band:
            ref_eigenvalues = np.load(self.ref_band)
            if len(ref_eigenvalues.shape) == 3:
                ref_eigenvalues = ref_eigenvalues.reshape(ref_eigenvalues.shape[1:])
            elif len(ref_eigenvalues.shape) != 2:
                log.error("Reference Eigenvalues' shape mismatch.")
                raise ValueError

            if ref_eigenvalues.shape[0] != self.eigenvalues.shape[0]:
                log.error("Reference Eigenvalues' should have sampled from the sample kpath as model's prediction.")
                raise ValueError
            ref_eigenvalues = ref_eigenvalues - (np.min(ref_eigenvalues) - np.min(self.eigenvalues))
        
            nintp = len(self.xlist)//50 
            if nintp == 0:
                nintp = 1
            band_ref = ax.plot(self.xlist[::nintp], ref_eigenvalues[::nintp] - self.E_fermi, 'o', ms=4, color=band_color, alpha=0.8, label="Ref")
            band_pre = ax.plot(self.xlist, self.eigenvalues - self.E_fermi, color="tab:red", lw=1.5, alpha=0.8, label="DeePTB")


        else:
            ax.plot(self.xlist, self.eigenvalues - self.E_fermi, color="tab:red",lw=1.5, alpha=0.8)

        # add verticle line
        for ii in self.high_sym_kpoints[1:-1]:
            ax.axvline(ii, color='gray', lw=1,ls='--')

        # add shadow
        # for i in range(self.eigenvalues.shape[1]):
        #     ax.fill_between(self.xlist, self.eigenvalues[:,i] - self.E_fermi, -2, alpha=0.05, color=band_color)
        # add ticks
        
        if not (emin is None or emax is None):
            ax.set_ylim(emin,emax)

        ax.set_xlim(self.xlist.min()-0.03,self.xlist.max()+0.03)
        ax.set_ylabel('E - EF (eV)',fontsize=12)
        ax.yaxis.set_minor_locator(MultipleLocator(1.0))
        ax.tick_params(which='both', direction='in', labelsize=12, width=1.5)
        ax.tick_params(which='major', length=6)
        ax.tick_params(which='minor', length=4, color='gray')
        
        # ax.set_yticks(None, fontsize=12)
        ax.set_xticks(self.high_sym_kpoints, self.labels, fontsize=12)

        ax.grid(color='gray', alpha=0.2, linestyle='-', linewidth=1)
        ax.set_axisbelow(True)

        fig.patch.set_facecolor('#f2f2f2')
        fig.patch.set_alpha(1)
        for spine in ax.spines.values():
            spine.set_edgecolor('#5d5d5d')
            spine.set_linewidth(1.5)
        
        if self.ref_band:
            plt.legend(handles=[band_pre[0], band_ref[0]], loc="best")
        
        plt.tight_layout()
        # remove the box around the plot
        ax.set_frame_on(False)
        plt.savefig(f'{self.results_path}/band.png',dpi=300)
        plt.show()
