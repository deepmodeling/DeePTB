
import numpy as np
from dptb.data import AtomicDataDict
import torch
from dptb.nn.hr2hk import HR2HK
from dptb.nn.hr2dhk import Hr2dHk
from dptb.postprocess.fortran import ac_cond as acdf2py
import math
from dptb.utils.make_kpoints import  kmesh_sampling_negf
import time


def fermi_dirac(e, mu, beta):
    return 1/(1+torch.exp(beta*(e-mu)))

def gauss(x,mu,sigma):
    res = torch.exp(-0.5*((x-mu)/sigma)**2)/(sigma*torch.sqrt(2*torch.tensor(math.pi)))
    return res


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
    
    print('application of the model is done')

    KB = 8.617333262e-5
    beta = 1/(KB*T)
    kpoints, kweight = kmesh_sampling_negf(mesh_grid)
    assert len(direction) == 2
    kpoints = torch.as_tensor(kpoints, dtype=torch.float32)
    kweight = torch.as_tensor(kweight, dtype=torch.float64)
    assert kpoints.shape[0] == kweight.shape[0]

    tot_numk = kpoints.shape[0]
    if nk_per_loop is None:
        nk_per_loop = tot_numk
    num_loop = math.ceil(tot_numk / nk_per_loop)
    omegas = torch.linspace(0,emax,num_omega, dtype=torch.float64)

    print('tot_numk:',tot_numk, 'nk_per_loop:',nk_per_loop, 'num_loop:',num_loop)

    ac_cond = np.zeros((len(omegas)),dtype=np.complex128)
    ac_cond_ik = np.zeros((len(omegas)),dtype=np.complex128)
    
    ac_cond_linhard = np.zeros((len(omegas)),dtype=np.complex128)
    ac_cond_linhard_ik = np.zeros((len(omegas)),dtype=np.complex128)
    
    for ik in range(num_loop):
        t_start = time.time()
        print('<><><><><'*5)
        print(f'loop {ik+1} in {num_loop} circles')
        istart = ik * nk_per_loop
        iend = min((ik + 1) * nk_per_loop, tot_numk)
        kpoints_ = kpoints[istart:iend]
        kweight_ = kweight[istart:iend]

        data[AtomicDataDict.KPOINT_KEY] = torch.nested.as_nested_tensor([kpoints_])
        data = h2k(data)
        dhdk = h2dk(data,direction=direction)
        
        # Hamiltonian = data['hamiltonian'].detach().to(torch.complex128)
        # dhdk = {k: v.detach().to(torch.complex128) for k, v in dhdk.items()}

        print(f'    - get H and dHdk ...')

        eigs, eigv = torch.linalg.eigh(data['hamiltonian'])
        
        if num_val is not None:
            assert num_val > 0
            assert eigs[:,num_val].min() - eigs[:,num_val-1].max() > 1e-3 , f'the gap between the VBM {num_val-1} and the CBM {num_val} is too small'
            if abs(gap_corr)> 1e-3:
                print(f'    - gap correction is applied {gap_corr}')
                eigs[:,:num_val] = eigs[:,:num_val] - gap_corr/2
                eigs[:,num_val:] = eigs[:,num_val:] + gap_corr/2
        print(f'    - diagonalization of H ...')

        dh1 = eigv.conj().transpose(1,2) @ dhdk[direction[0]] @ eigv
        if direction[0] == direction[1]:
            dh2 = dh1
        else:
            dh2 = eigv.conj().transpose(1,2) @ dhdk[direction[1]] @ eigv

        p1p2 = dh1 * dh2.transpose(1,2)


        print(f'    - get p matrix from dHdk ...')

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

        print(f'    - get ac_cond ...')
        t_end = time.time()
        print(f'time cost: {t_end-t_start:.4f} s in loop {ik+1}')

    volume = data['cell'][0] @(data['cell'][1].cross(data['cell'][2]))
    prefactor = g_s * 1j / (volume.numpy())
    ac_cond = ac_cond * prefactor
    ac_cond.imag = -ac_cond.imag*np.pi
    ac_cond_linhard = ac_cond_linhard * prefactor
    
    return omegas, ac_cond, ac_cond_linhard
