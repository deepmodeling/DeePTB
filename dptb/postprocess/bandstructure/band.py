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
from dptb.postprocess.elec_struc_cal import ElecStruCal

class Band(ElecStruCal):

    def __init__(self, model:torch.nn.Module, results_path: str=None, use_gui: bool=False, device: str='cpu'):
        super().__init__(model=model, device=device)
        self.results_path = results_path
        self.use_gui = use_gui
            
    def get_bands(self, data: Union[AtomicData, ase.Atoms, str], kpath_kwargs: dict, pbc:Union[bool,list]=None, AtomicData_options:dict=None):
        kline_type = kpath_kwargs['kline_type']

        # get  the ase structure
        if isinstance(data, str):
            structase = read(data)
        elif isinstance(data, ase.Atoms):
            structase = data
        elif isinstance(data, AtomicData):
            structase = data.to("cpu").to_ase()
        
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

        else:
            log.error('Error, now, kline_type only support ase_kpath, abacus, or vasp.')
            raise ValueError


        override_overlap = kpath_kwargs.get("override_overlap", None)       

        eig_solver = kpath_kwargs.get("eig_solver", None)
        
        data, eigenvalues = self.get_eigs(data=data, klist=klist, pbc=pbc, 
                                          AtomicData_options=AtomicData_options, 
                                          override_overlap=override_overlap, eig_solver=eig_solver)
        
        nel_atom = kpath_kwargs.get('nel_atom', None)

        if nel_atom is not None:
            data,estimated_E_fermi = self.get_fermi_level(  data=data, nel_atom=nel_atom,
                                                            klist = klist, pbc=pbc, 
                                                            AtomicData_options=AtomicData_options)
        else:
            estimated_E_fermi = None

        self.eigenstatus = {'klist': klist,
                            'xlist': xlist,
                            'high_sym_kpoints': high_sym_kpoints,
                            'labels': labels,
                            'eigenvalues': eigenvalues,
                            'E_fermi': estimated_E_fermi}

        if self.results_path is not None:
            np.save(f'{self.results_path}/bandstructure',self.eigenstatus)

        return self.eigenstatus


    @classmethod
    def estimate_E_fermi(cls, eigenvalues: np.array, total_electrons: int, spindeg: int=2):
        assert len(eigenvalues.shape) == 2
        nk, nband  = eigenvalues.shape
        numek = total_electrons * nk // spindeg
        sorteigs =  np.sort(np.reshape(eigenvalues,[-1]))
        EF=(sorteigs[numek] + sorteigs[numek-1])/2

        return EF


    def band_plot(
            self, 
            ref_band: Union[str, np.array, torch.Tensor, bool]=None,
            E_fermi: Optional[float]=None,
            emin: Optional[float]=None,
            emax: Optional[float]=None,
            ):
        
        if isinstance(ref_band, str):
            ref_band = np.load(ref_band)

        ylabel = 'E - EF (eV)'
        if E_fermi is not None:
            if abs(self.eigenstatus["E_fermi"] - E_fermi)>1e-3:
                log.info(f"Using input Fermi energy: {E_fermi:.4f} eV (estimated: {self.eigenstatus['E_fermi']:.4f} eV)")
            else:
                log.info(f"Using Fermi energy: {E_fermi:.4f} eV (matches estimated value)")
        elif self.eigenstatus["E_fermi"] is not None:
            E_fermi = self.eigenstatus["E_fermi"]
            log.info(f"No Fermi energy provided, using estimated value: {E_fermi:.4f} eV")
        else:
            E_fermi = 0.0
            ylabel = 'E (eV)'
            log.info("No Fermi energy available, setting to 0.0 eV")
        
        matplotlib.rcParams['font.size'] = 7
        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
        matplotlib.rcParams['axes.linewidth'] = 0.5
        matplotlib.rcParams['xtick.major.width'] =0.3
        matplotlib.rcParams['ytick.major.width'] =0.3
        # plt.rcParams['font.sans-serif'] = ['Times New Roman']

        fig = plt.figure(figsize=(3.2,2.8),dpi=200)

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
                log.info("kpoints in ref_band: ", ref_band.shape[0])
                log.info("kpoints in model's prediction: ", self.eigenstatus["eigenvalues"].shape[0])
                log.error("Reference Eigenvalues' should have sampled from the sample kpath as model's prediction.")
                raise ValueError
            ref_band = ref_band - (np.min(ref_band) - np.min(self.eigenstatus["eigenvalues"]))
            nintp = max(1 , self.eigenstatus["xlist"].shape[0] // 25)
            band_ref = ax.plot(self.eigenstatus["xlist"][::nintp], ref_band[::nintp] - E_fermi, 'o', ms=2, color=band_color, alpha=0.95, label="Ref")
            band_pre = ax.plot(self.eigenstatus["xlist"], self.eigenstatus["eigenvalues"] - E_fermi, color="tab:red", lw=0.5, alpha=0.95, label="DeePTB")

        else:
            ax.plot(self.eigenstatus["xlist"], self.eigenstatus["eigenvalues"] - E_fermi, color="tab:red",lw=0.5, alpha=0.95)

        # add verticle line
        for ii in self.eigenstatus["high_sym_kpoints"][1:-1]:
            ax.axvline(ii, color='gray', lw=0.3,ls='--')
        
        if not (emin is None or emax is None):
            ax.set_ylim(emin,emax)

        ax.set_xlim(self.eigenstatus["xlist"].min(),self.eigenstatus["xlist"].max())
        ax.set_ylabel(ylabel,fontsize=8)
        ax.yaxis.set_minor_locator(MultipleLocator(1.0))
        ax.tick_params(which='both', direction='in', labelsize=8)
        ax.tick_params(which='major', length=6)
        ax.tick_params(which='minor', length=4)
        
        ax.set_xticks(self.eigenstatus["high_sym_kpoints"], self.eigenstatus["labels"], fontsize=8)

        ax.set_axisbelow(True)
        
        if ref_band is not None:
            plt.legend(handles=[band_pre[0], band_ref[0]], loc="best")
        plt.tight_layout()
        if self.results_path is not None:
            plt.savefig(f'{self.results_path}/band.png',dpi=300)
        if plt.get_backend() != 'Agg':
            plt.show()
        else:
            plt.close()

