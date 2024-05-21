import numpy as np
from dptb.utils.tools import j_must_have
from dptb.utils.make_kpoints  import ase_kpath, abacus_kpath, vasp_kpath
from ase.io import read
import ase
from typing import Union
import matplotlib.pyplot as plt
import torch
from typing import Optional
import matplotlib
import logging
log = logging.getLogger(__name__)
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from dptb.data import AtomicData, AtomicDataDict
from dptb.nn.energy import Eigenvalues
from dptb.postprocess.abstract_process import AbstractProcess
# class bandcalc(object):
#     def __init__ (self, apiHrk, run_opt, jdata):
#         self.apiH = apiHrk
#         if isinstance(run_opt['structure'],str):
#             self.structase = read(run_opt['structure'])
#         elif isinstance(run_opt['structure'],ase.Atoms):
#             self.structase = run_opt['structure']
#         else:
#             raise ValueError('structure must be ase.Atoms or str')
        
#         self.band_plot_options = jdata
#         self.results_path = run_opt.get('results_path')
#         self.apiH.update_struct(self.structase)

#         self.ref_band = self.band_plot_options.get("ref_band", None)
#         self.use_gui = self.band_plot_options.get("use_gui", False)

#     def get_bands(self):
#         kline_type = self.band_plot_options['kline_type']

        
#         if kline_type == 'ase':
#             kpath = self.band_plot_options['kpath']
#             nkpoints = self.band_plot_options['nkpoints']
#             self.klist, self.xlist, self.high_sym_kpoints, self.labels = ase_kpath(structase=self.structase,
#                                                  pathstr=kpath, total_nkpoints=nkpoints)
#         elif kline_type == 'abacus':
#             kpath = self.band_plot_options['kpath']
#             self.labels = self.band_plot_options['klabels']
#             self.klist, self.xlist, self.high_sym_kpoints  = abacus_kpath(structase=self.structase, kpath=kpath)
        
#         elif kline_type == 'vasp':
#             kpath = self.band_plot_options['kpath']
#             high_sym_kpoints_dict = self.band_plot_options['high_sym_kpoints']
#             number_in_line = self.band_plot_options['number_in_line']
#             self.klist, self.xlist, self.high_sym_kpoints, self.labels = vasp_kpath(structase=self.structase,
#                                                  pathstr=kpath, high_sym_kpoints_dict=high_sym_kpoints_dict, number_in_line=number_in_line)
#         else:
#             log.error('Error, now, kline_type only support ase_kpath, abacus, or vasp.')
#             raise ValueError
        
#         all_bonds, hamil_blocks, overlap_blocks = self.apiH.get_HR()
#         self.eigenvalues, self.estimated_E_fermi = self.apiH.get_eigenvalues(self.klist)

#         if self.band_plot_options.get('E_fermi',None) != None:
#             self.E_fermi = self.band_plot_options['E_fermi']
#             log.info(f'set E_fermi from jdata: {self.E_fermi}, While the estimated value in line-mode is {self.estimated_E_fermi}')
#         else:
#             self.E_fermi = 0.0
#             log.info(f'set E_fermi = 0.0, While the estimated value in line-mode is {self.estimated_E_fermi}')

#         eigenstatus = {'klist': self.klist,
#                         'xlist': self.xlist,
#                         'high_sym_kpoints': self.high_sym_kpoints,
#                         'labels': self.labels,
#                         'eigenvalues': self.eigenvalues,
#                         'E_fermi': self.E_fermi }

#         np.save(f'{self.results_path}/bandstructure',eigenstatus)

#         return  eigenstatus

    
#     def get_HR(self):
#         all_bonds, hamil_blocks, overlap_blocks = self.apiH.get_HR()
        
#         return all_bonds, hamil_blocks, overlap_blocks

#     def band_plot(self):
#         matplotlib.rcParams['font.size'] = 7
#         matplotlib.rcParams['pdf.fonttype'] = 42
#         # plt.rcParams['font.sans-serif'] = ['Times New Roman']

#         emin = self.band_plot_options.get('emin')
#         emax = self.band_plot_options.get('emax')

#         fig = plt.figure(figsize=(4.5,4),dpi=100)

#         ax = fig.add_subplot(111)

#         band_color = '#5d5d5d'
#         # plot the line
        
#         if self.ref_band:
#             ref_eigenvalues = np.load(self.ref_band)
#             if len(ref_eigenvalues.shape) == 3:
#                 ref_eigenvalues = ref_eigenvalues.reshape(ref_eigenvalues.shape[1:])
#             elif len(ref_eigenvalues.shape) != 2:
#                 log.error("Reference Eigenvalues' shape mismatch.")
#                 raise ValueError

#             if ref_eigenvalues.shape[0] != self.eigenvalues.shape[0]:
#                 log.error("Reference Eigenvalues' should have sampled from the sample kpath as model's prediction.")
#                 raise ValueError
#             ref_eigenvalues = ref_eigenvalues - (np.min(ref_eigenvalues) - np.min(self.eigenvalues))

#             nkplot = (len(np.unique(self.high_sym_kpoints))-1) * 7
#             nintp = len(self.xlist) // nkplot 
#             if nintp == 0:
#                 nintp = 1
#             band_ref = ax.plot(self.xlist[::nintp], ref_eigenvalues[::nintp] - self.E_fermi, 'o', ms=4, color=band_color, alpha=0.8, label="Ref")
#             band_pre = ax.plot(self.xlist, self.eigenvalues - self.E_fermi, color="tab:red", lw=1.5, alpha=0.8, label="DeePTB")


#         else:
#             ax.plot(self.xlist, self.eigenvalues - self.E_fermi, color="tab:red",lw=1.5, alpha=0.8)

#         # add verticle line
#         for ii in self.high_sym_kpoints[1:-1]:
#             ax.axvline(ii, color='gray', lw=1,ls='--')

#         # add shadow
#         # for i in range(self.eigenvalues.shape[1]):
#         #     ax.fill_between(self.xlist, self.eigenvalues[:,i] - self.E_fermi, -2, alpha=0.05, color=band_color)
#         # add ticks
        
#         if not (emin is None or emax is None):
#             ax.set_ylim(emin,emax)

#         ax.set_xlim(self.xlist.min()-0.03,self.xlist.max()+0.03)
#         ax.set_ylabel('E - EF (eV)',fontsize=12)
#         ax.yaxis.set_minor_locator(MultipleLocator(1.0))
#         ax.tick_params(which='both', direction='in', labelsize=12, width=1.5)
#         ax.tick_params(which='major', length=6)
#         ax.tick_params(which='minor', length=4, color='gray')
        
#         # ax.set_yticks(None, fontsize=12)
#         ax.set_xticks(self.high_sym_kpoints, self.labels, fontsize=12)

#         ax.grid(color='gray', alpha=0.2, linestyle='-', linewidth=1)
#         ax.set_axisbelow(True)

#         fig.patch.set_facecolor('#f2f2f2')
#         fig.patch.set_alpha(1)
#         for spine in ax.spines.values():
#             spine.set_edgecolor('#5d5d5d')
#             spine.set_linewidth(1.5)
        
#         if self.ref_band:
#             plt.legend(handles=[band_pre[0], band_ref[0]], loc="best")
        
#         plt.tight_layout()
#         # remove the box around the plot
#         ax.set_frame_on(False)
#         plt.savefig(f'{self.results_path}/band.png',dpi=300)
#         if self.use_gui:
#             plt.show()


class Band(AbstractProcess):
            
    def get_bands(self, data: Union[AtomicData, ase.Atoms, str], kpath_kwargs: dict, AtomicData_options: dict={}):
        kline_type = kpath_kwargs['kline_type']

        # get the AtomicData structure and the ase structure
        if isinstance(data, str):
            structase = read(data)
        elif isinstance(data, ase.Atoms):
            structase = data
        elif isinstance(data, AtomicData):
            structase = data.to("cpu").to_ase()
        
        
        # data = AtomicData.to_AtomicDataDict(data.to(self.device))
        # data = self.model.idp(data)
            
        
        if kline_type == 'ase':
            kpath = kpath_kwargs['kpath']
            nkpoints = kpath_kwargs['nkpoints']
            klist, xlist, high_sym_kpoints, labels = ase_kpath(structase=structase, pathstr=kpath, total_nkpoints=nkpoints)

        elif kline_type == 'abacus':
            kpath = kpath_kwargs['kpath']
            labels = kpath_kwargs['klabels']
            klist, xlist, high_sym_kpoints  = abacus_kpath(structase=structase, kpath=kpath)
        
        elif kline_type == 'vasp':
            kpath = kpath_kwargs['kpath']
            high_sym_kpoints_dict = kpath_kwargs['high_sym_kpoints']
            number_in_line = kpath_kwargs['number_in_line']
            klist, xlist, high_sym_kpoints, labels = vasp_kpath(structase=structase,
                                                 pathstr=kpath, high_sym_kpoints_dict=high_sym_kpoints_dict, number_in_line=number_in_line)
        elif kline_type == 'array':
            klist = kpath_kwargs['kpath']
            high_sym_kpoints = kpath_kwargs.get('high_sym_kpoints', None)
            xlist = kpath_kwargs.get('xlist', None)
            labels = kpath_kwargs.get('labels', None)

        elif kline_type == "kmesh":
            from dptb.utils.make_kpoints import kmesh_sampling_negf
            kmesh = kpath_kwargs['kpath'] # TODO: add another field in input.json to specify the kmesh
            klist,wk = kmesh_sampling_negf(meshgrid=kmesh, is_gamma_center=True, is_time_reversal=True)
            log.info(f'kmesh sampling: {klist.shape[0]} kpoints')
        else:
            log.error('Error, now, kline_type only support ase_kpath, abacus, or vasp.')
            raise ValueError
        
        self.get_eigs(data, klist, AtomicData_options)
        
        # # set the kpoint of the AtomicData
        # data[AtomicDataDict.KPOINT_KEY] = torch.nested.as_nested_tensor([torch.as_tensor(klist, dtype=self.model.dtype, device=self.device)])

        # # get the eigenvalues
        # data = self.model(data)
        # if self.overlap == True:
        #     assert data.get(AtomicDataDict.EDGE_OVERLAP_KEY) is not None
        # data = self.eigv(data)

        # get the E_fermi from data
        nel_atom = kpath_kwargs.get('nel_atom', None)
        assert isinstance(nel_atom, dict) or nel_atom is None
        
        if nel_atom is not None:
            atomtype_list = self.data[AtomicDataDict.ATOM_TYPE_KEY].flatten().tolist()
            atomtype_symbols = np.asarray(self.model.idp.type_names)[atomtype_list].tolist()
            total_nel = np.array([nel_atom[s] for s in atomtype_symbols]).sum()
            if hasattr(self.model,'soc_param'):
                spindeg = 1
            else:
                spindeg = 2

            if kline_type != "kmesh":
                estimated_E_fermi = self.estimate_E_fermi(self.data[AtomicDataDict.ENERGY_EIGENVALUE_KEY][0].detach().cpu().numpy(), total_nel, spindeg)
            else:
                estimated_E_fermi = self.cal_E_fermi(self.data[AtomicDataDict.ENERGY_EIGENVALUE_KEY][0].detach().cpu().numpy(), total_nel, spindeg, wk)
            log.info(f'Estimated E_fermi: {estimated_E_fermi} based on the valence electrons setting nel_atom : {nel_atom} .')
        else:
            estimated_E_fermi = None

        if kline_type != "kmesh":
            self.eigenstatus = {'klist': klist,
                                'xlist': xlist,
                                'high_sym_kpoints': high_sym_kpoints,
                                'labels': labels,
                                'eigenvalues': self.data[AtomicDataDict.ENERGY_EIGENVALUE_KEY][0].detach().cpu().numpy(),
                                'E_fermi': estimated_E_fermi}

            if self.results_path is not None:
                np.save(f'{self.results_path}/bandstructure',self.eigenstatus)

            return self.eigenstatus
        else:
            return

    @classmethod
    def estimate_E_fermi(cls, eigenvalues: np.array, total_electrons: int, spindeg: int=2):
        assert len(eigenvalues.shape) == 2
        nk, nband  = eigenvalues.shape
        numek = total_electrons * nk // spindeg
        sorteigs =  np.sort(np.reshape(eigenvalues,[-1]))
        EF=(sorteigs[numek] + sorteigs[numek-1])/2

        return EF

    @classmethod
    def cal_E_fermi(cls, eigenvalues: np.array, total_electrons: int, spindeg: int=2,wk: np.array=None,q_tol=1e-10):
        nextafter = np.nextafter
        total_electrons = total_electrons / spindeg # This version is for the case of spin-degeneracy
        log.info('Calculating Fermi energy in the case of spin-degeneracy.')
        def fermi_dirac(E, kT=0.4, mu=0.0):
            return 1.0 / (np.expm1((E - mu) / kT) + 2.0)
        
        # calculate boundaries
        min_Ef, max_Ef = eigenvalues.min(), eigenvalues.max()
        Ef = (min_Ef + max_Ef) * 0.5
        while nextafter(min_Ef, max_Ef) < max_Ef:
            # Calculate guessed charge
            wk = wk.reshape(-1,1)
            q_cal = (wk * fermi_dirac(eigenvalues, mu=Ef)).sum()

            if abs(q_cal - total_electrons) < q_tol:
                return Ef

            if q_cal >= total_electrons:
                max_Ef = Ef
            else:
                min_Ef = Ef
            Ef = (min_Ef + max_Ef) * 0.5

        return Ef
    def band_plot(
            self, 
            ref_band: Union[str, np.array, torch.Tensor, bool]=None,
            E_fermi: Optional[float]=None,
            emin: Optional[float]=None,
            emax: Optional[float]=None,
            ):
        
        if isinstance(ref_band, str):
            ref_band = np.load(ref_band)

        if E_fermi != None and self.eigenstatus["E_fermi"] != E_fermi:
            log.info(f'use input fermi energy: {E_fermi}, While the estimated value in line-mode is {self.eigenstatus["E_fermi"]}')
        else:
            E_fermi = self.eigenstatus["E_fermi"]
            log.info(f'The fermi energy is not provided, use the estimated value in line-mode: {self.eigenstatus["E_fermi"]}')

        matplotlib.rcParams['font.size'] = 7
        matplotlib.rcParams['pdf.fonttype'] = 42
        # plt.rcParams['font.sans-serif'] = ['Times New Roman']

        fig = plt.figure(figsize=(4.5,4),dpi=100)

        ax = fig.add_subplot(111)

        band_color = '#5d5d5d'
        # plot the line
        
        if ref_band is not None:
            if len(ref_band.shape) == 3:
                assert ref_band.shape[0] == 1
                ref_band = ref_band.reshape(ref_band.shape[1:])
            elif len(ref_band.shape) != 2:
                log.error("Reference Eigenvalues' shape mismatch.")
                raise ValueError

            if ref_band.shape[0] != self.eigenstatus["eigenvalues"].shape[0]:
                log.error("Reference Eigenvalues' should have sampled from the sample kpath as model's prediction.")
                raise ValueError
            ref_band = ref_band - (np.min(ref_band) - np.min(self.eigenstatus["eigenvalues"]))

            nkplot = (len(np.unique(self.eigenstatus["high_sym_kpoints"]))-1) * 7
            nintp = len(self.eigenstatus["xlist"]) // nkplot 
            if nintp == 0:
                nintp = 1
            band_ref = ax.plot(self.eigenstatus["xlist"][::nintp], ref_band[::nintp] - E_fermi, 'o', ms=4, color=band_color, alpha=0.8, label="Ref")
            band_pre = ax.plot(self.eigenstatus["xlist"], self.eigenstatus["eigenvalues"] - E_fermi, color="tab:red", lw=1.5, alpha=0.8, label="DeePTB")

        else:
            ax.plot(self.eigenstatus["xlist"], self.eigenstatus["eigenvalues"] - E_fermi, color="tab:red",lw=1.5, alpha=0.8)

        # add verticle line
        for ii in self.eigenstatus["high_sym_kpoints"][1:-1]:
            ax.axvline(ii, color='gray', lw=1,ls='--')

        # add shadow
        # for i in range(self.eigenvalues.shape[1]):
        #     ax.fill_between(self.xlist, self.eigenvalues[:,i] - self.E_fermi, -2, alpha=0.05, color=band_color)
        # add ticks
        
        if not (emin is None or emax is None):
            ax.set_ylim(emin,emax)

        ax.set_xlim(self.eigenstatus["xlist"].min()-0.03,self.eigenstatus["xlist"].max()+0.03)
        ax.set_ylabel('E - EF (eV)',fontsize=12)
        ax.yaxis.set_minor_locator(MultipleLocator(1.0))
        ax.tick_params(which='both', direction='in', labelsize=12, width=1.5)
        ax.tick_params(which='major', length=6)
        ax.tick_params(which='minor', length=4, color='gray')
        
        # ax.set_yticks(None, fontsize=12)
        ax.set_xticks(self.eigenstatus["high_sym_kpoints"], self.eigenstatus["labels"], fontsize=12)

        ax.grid(color='gray', alpha=0.2, linestyle='-', linewidth=1)
        ax.set_axisbelow(True)

        fig.patch.set_facecolor('#f2f2f2')
        fig.patch.set_alpha(1)
        for spine in ax.spines.values():
            spine.set_edgecolor('#5d5d5d')
            spine.set_linewidth(1.5)
        
        if ref_band is not None:
            plt.legend(handles=[band_pre[0], band_ref[0]], loc="best")
        
        plt.tight_layout()
        # remove the box around the plot
        ax.set_frame_on(False)
        if self.results_path is not None:
            plt.savefig(f'{self.results_path}/band.png',dpi=300)
        if self.use_gui:
            plt.show()
