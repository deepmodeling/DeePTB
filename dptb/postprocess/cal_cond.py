
import numpy as np
from dptb.data import AtomicDataDict
import torch
from dptb.nn.hr2hk import HR2HK
from dptb.nn.hr2dhk import HR2dHK
import math
import numpy as np
from tbplas.fortran import f2py
from dptb.utils.make_kpoints import  kmesh_sampling_negf
import time
import logging


log = logging.getLogger(__name__)

def fermi_dirac(e, mu, beta):
    return 1/(1+torch.exp(beta*(e-mu)))

def gauss(x,mu,sigma):
    res = torch.exp(-0.5*((x-mu)/sigma)**2)/(sigma*torch.sqrt(2*torch.tensor(math.pi)))
    return res


def cal_cond(model, data, e_fermi, mesh_grid, emax, num_omega= 1000, nk_per_loop=None, delta=0.005, T=300, direction='xx', g_s=2):
    
    h2k = HR2HK(
        idp=model.idp,
        device=model.device,
        dtype=model.dtype)
    
    h2dk = HR2dHK(
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
        nk_per_loop = tot_numk
    num_loop = math.ceil(tot_numk / nk_per_loop)
    omegas = torch.linspace(0,emax,num_omega, dtype=torch.float64)

    log.info('tot_numk:',tot_numk, 'nk_per_loop:',nk_per_loop, 'num_loop:',num_loop)

    ac_cond = np.zeros((len(omegas)),dtype=np.complex128)
    ac_cond_ik = np.zeros((len(omegas)),dtype=np.complex128)
    
    
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
        f2py.ac_cond_gauss(eig_diff.permute(1,2,0).detach().numpy(), p1p2.permute(1,2,0).detach().numpy(), omegas, delta, 1, kpoints_.shape[0], ac_cond_ik)
        ac_cond = ac_cond + ac_cond_ik

        log.info(f'    - get ac_cond ...')
        t_end = time.time()
        log.info(f'time cost: {t_end-t_start:.4f} s in loop {ik+1}')

    volume = data['cell'][0] @(data['cell'][1].cross(data['cell'][2]))
    if volume == 0:
        log.warning('Volume is 0, please check the cell parameters. \nFor 3D bulk materials, the volume should be positive. but for 1D,2D or non-periodic systems, the volume could be 0.')
        volume = 1.0
    if volume < 0:
        log.warning(f'Volume is negative {volume}, please check the cell parameters. We will take the absolute value of the volume.')
        volume = - volume
    prefactor = g_s * 1j / (volume.numpy())
    ac_cond = ac_cond * prefactor
    
    return omegas, ac_cond
