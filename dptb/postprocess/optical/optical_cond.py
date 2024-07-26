
import numpy as np
from dptb.data import AtomicDataDict
from dptb.data import AtomicData
import sys
import torch
from dptb.nn.hr2hk import HR2HK
from dptb.nn.hr2dhk import Hr2dHk
import math
from dptb.utils.make_kpoints import  kmesh_sampling_negf
import time
import logging
import os
from ase.io import read
import matplotlib.pyplot as plt
from dptb.utils.constants import  atomic_num_dict_r
try:
    from dptb.postprocess.fortran import ac_cond as acdf2py
except ImportError:
    acdf2py = None

log = logging.getLogger(__name__)

def fermi_dirac(e, mu, beta):
    return 1/(1+torch.exp(beta*(e-mu)))

def gauss(x,mu,sigma):
    res = torch.exp(-0.5*((x-mu)/sigma)**2)/(sigma*torch.sqrt(2*torch.tensor(math.pi)))
    return res

class AcCond:
    def __init__(self, model:torch.nn.Module, results_path: str=None, use_gui: bool=False, device: str='cpu'):
        self.model = model
        self.results_path = results_path
        self.use_gui = use_gui
        self.device = device
        os.makedirs(results_path, exist_ok=True)
        if acdf2py is None:
            log.warning('ac_cond_f is not available, please install the fortran code to calculate the AC conductivity')
            sys.exit(1)

    def get_accond(self,
                        struct,
                        AtomicData_options,
                        emax,
                        num_omega= 1000,
                        mesh_grid=[1,1,1],
                        nk_per_loop=None,
                        delta=0.03,
                        e_fermi=0,
                        valence_e=None,
                        gap_corr=0,
                        T=300, 
                        direction='xx',
                        g_s=2):
        
        self.direction = direction

        log.info('<><><><>'*5)
        # 调用from_ase方法，生成一个硅的AtomicData类型数据
        dataset = AtomicData.from_ase(atoms=read(struct),**AtomicData_options)
        data = AtomicData.to_AtomicDataDict(dataset)
        if valence_e is not None and abs(gap_corr) > 1e-3:
            uniq_type, counts = np.unique(data['atomic_numbers'].numpy(), return_counts=True)
            tot_num_e = 0
            for i in range(len(uniq_type)):
                symbol = atomic_num_dict_r[uniq_type[i]]
                assert symbol in valence_e
                tot_num_e += counts[i] * valence_e[symbol]
            
            num_val = tot_num_e // g_s
        else:
            num_val = None

        self.omegas, self.ac_cond_gauss, self.ac_cond_linhard = AcCond.cal_cond(model=self.model, data=data, e_fermi=e_fermi, mesh_grid=mesh_grid, 
                                                                    emax=emax, num_val=num_val, gap_corr=gap_corr, num_omega=num_omega, nk_per_loop=nk_per_loop, 
                                                                    delta=delta, T=T, direction=direction, g_s=g_s)

        np.save(f"{self.results_path}/AC_{self.direction}_cond_sig_{delta}.npy", {'energy':self.omegas, 'ac_cond_g': self.ac_cond_gauss, 'ac_cond_l': self.ac_cond_linhard})
        log.info('<><><><>'*5)

    def accond_plot(self):
        fig = plt.figure(figsize=(6,6),dpi=200)
        plt.plot(self.omegas, self.ac_cond_gauss.real, label='Gaussian')
        plt.plot(self.omegas, self.ac_cond_linhard.real, label='Linhard:real')
        plt.plot(self.omegas, self.ac_cond_linhard.imag, label='Linhard:real')
        plt.legend()
        plt.xlabel("Energy (eV)")
        plt.ylabel(f"sigma_{self.direction}")
        plt.savefig(f"{self.results_path}/AC_{self.direction}.png")
        if self.use_gui:
            plt.show()
        plt.close()

    @staticmethod
    def cal_cond(model, data, e_fermi, mesh_grid, emax, num_val=None, gap_corr=0, num_omega= 1000, nk_per_loop=None, delta=0.005, T=300, direction='xx', g_s=2):
        h2k = HR2HK(
            idp=model.idp,
            device=model.device,
            dtype=model.dtype)

        h2dk = Hr2dHk(
            idp=model.idp,
            device=model.device,
            dtype=model.dtype)
        data = model.idp(data)
        data = model(data)

        log.info('application of the model is done')

        KB = 8.617333262e-5
        beta = 1/(KB*T)
        kpoints, kweight = kmesh_sampling_negf(mesh_grid)
        assert len(direction) == 2
        kpoints = torch.as_tensor(kpoints, dtype=torch.float32)
        kweight = torch.as_tensor(kweight, dtype=torch.float64)
        assert kpoints.shape[0] == kweight.shape[0]

        tot_numk = kpoints.shape[0]
        if nk_per_loop is None:
            log.warning('nk_per_loop is not set, will use all kpoints in one loop, which may cause memory error.')
            nk_per_loop = tot_numk
        num_loop = math.ceil(tot_numk / nk_per_loop)
        omegas = torch.linspace(0,emax,num_omega, dtype=torch.float64)

        log.info(f'tot_numk: {tot_numk}, nk_per_loop: {nk_per_loop}, num_loop: {num_loop}')

        ac_cond = np.zeros((len(omegas)),dtype=np.complex128)
        ac_cond_ik = np.zeros((len(omegas)),dtype=np.complex128)

        ac_cond_linhard = np.zeros((len(omegas)),dtype=np.complex128)
        ac_cond_linhard_ik = np.zeros((len(omegas)),dtype=np.complex128)

        for ik in range(num_loop):
            t_start = time.time()
            log.info('<><><><><'*5)
            log.info(f'loop {ik+1} in {num_loop} circles')
            istart = ik * nk_per_loop
            iend = min((ik + 1) * nk_per_loop, tot_numk)
            kpoints_ = kpoints[istart:iend]
            kweight_ = kweight[istart:iend]

            data[AtomicDataDict.KPOINT_KEY] = torch.nested.as_nested_tensor([kpoints_])
            data = h2k(data)
            dhdk = h2dk(data,direction=direction)

            # Hamiltonian = data['hamiltonian'].detach().to(torch.complex128)
            # dhdk = {k: v.detach().to(torch.complex128) for k, v in dhdk.items()}

            log.info(f'    - get H and dHdk ...')

            eigs, eigv = torch.linalg.eigh(data['hamiltonian'])

            if num_val is not None and abs(gap_corr) > 1e-3:
                log.info(f'    - gap correction is applied with {gap_corr}')
                assert num_val > 0
                assert eigs[:,num_val].min() - eigs[:,num_val-1].max() > 1e-3 , f'the gap between the VBM {num_val-1} and the CBM {num_val} is too small'

                eigs[:,:num_val] = eigs[:,:num_val] - gap_corr/2
                eigs[:,num_val:] = eigs[:,num_val:] + gap_corr/2

            log.info(f'    - diagonalization of H ...')

            dh1 = eigv.conj().transpose(1,2) @ dhdk[direction[0]] @ eigv
            if direction[0] == direction[1]:
                dh2 = dh1
            else:
                dh2 = eigv.conj().transpose(1,2) @ dhdk[direction[1]] @ eigv

            p1p2 = dh1 * dh2.transpose(1,2)


            log.info(f'    - get p matrix from dHdk ...')

            p1p2.to(torch.complex128)
            eigs.to(torch.float64)

            eig_diff = eigs[:,:,None] - eigs[:,None,:]

            fdv = fermi_dirac(eigs, e_fermi, beta)
            fd_diff = fdv[:,:,None] - fdv[:,None,:]
            #fd_ed = torch.zeros_like(eig_diff)
            ind = torch.abs(eig_diff) > 1e-6
            ind2 = torch.abs(eig_diff) <= 1e-6
            fd_diff[ind] = fd_diff[ind] / eig_diff[ind]
            fd_diff[ind2] = 0.0 

            p1p2 = p1p2 * fd_diff
            p1p2 = p1p2 * kweight_[:,None,None]

            kpoints_.shape[1]
            ac_cond_ik = ac_cond_ik * 0
            acdf2py.ac_cond_gauss(eig_diff.permute(1,2,0).detach().numpy(), p1p2.permute(1,2,0).detach().numpy(), omegas, delta, 1, kpoints_.shape[0], ac_cond_ik)
            acdf2py.ac_cond_f(eig_diff.permute(1,2,0).detach().numpy(), p1p2.permute(1,2,0).detach().numpy(), omegas, delta, 1, kpoints_.shape[0], ac_cond_linhard_ik)
            ac_cond = ac_cond + ac_cond_ik
            ac_cond_linhard = ac_cond_linhard + ac_cond_linhard_ik

            log.info(f'    - get ac_cond ...')
            t_end = time.time()
            log.info(f'time cost: {t_end-t_start:.4f} s in loop {ik+1}')

        ac_cond = 1.0j * ac_cond * np.pi    
        volume = data['cell'][0] @ (data['cell'][1].cross(data['cell'][2],dim=0))
        prefactor = 2 * g_s * 1j / (volume.numpy())
        ac_cond = ac_cond * prefactor
        ac_cond_linhard = ac_cond_linhard * prefactor

        return omegas, ac_cond, ac_cond_linhard
