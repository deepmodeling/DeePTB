import ase.transport
import torch
from dptb.utils.constants import Boltzmann, eV2J,pi
from dptb.negf.recursive_green_cal import recursive_gf
from fmm3dpy import lfmm3d
from dptb.negf.areshkin_pole_sum import pole_maker
from dptb.negf.surface_green import selfEnergy
from dptb.negf.negf_utils import finite_difference
import numpy as np
from tqdm import tqdm
import time
from dptb.negf.utils import quad


'''
The used transformation of different quantities will be implemented here
'''
kBT = Boltzmann * T / eV2J

def fermi_dirac(x):
    return 1 / (1 + torch.exp(x/kBT))

# class _fermi_dirac(torch.autograd.Function):
#     @staticmethod
#     def forward(x, k, T):
#         ctx.save_for_backward(x)
#         if T == 0 and x == 0:
#             return



def sigmaLR2Gamma(se):
    return -1j*(se-se.conj())

def gamma2SigmaLRIn(gamma, ee, u):
    return gamma * fermi_dirac(ee-u)

def gamma2SigmaLROut(gamma, ee, u):
    return gamma * (1-fermi_dirac(ee-u))

# def acousticSigma(Da, rho, vs, a, sd, gnd, gpd, grd):
#     temp = (Da ** 2) * (k * T) / (vs**2 * rho * a**3)
#     N = len(gnd)
#     sigmaInA = [None for _ in range(N)]
#     sigmaOutA = [None for _ in range(N)]
#     sigmaRA = [None for _ in range(N)]
#
#     for i in range(N):
#         sigmaInA[i] = (temp*sd[i]).matmul(gnd[i])
#         sigmaOutA[i] = (temp*sd[i]).matmul(gpd[i])
#         sigmaRA[i] = (temp*sd[i]).matmul(grd[i])
#
#     return sigmaInA, sigmaOutA, sigmaRA

def opticalSigma():

    pass


def calTT(gammaL, gammaR, gtrans):
    return (gammaL @ gtrans @ gammaR @ gtrans.conj().T).real.trace()

def calCurrent(ul, ur, n_int=100, **hmt_ovp):
    xl = min(ul, ur) - 1
    xu = max(ul, ur) + 1

    dic = {}
    params = [ul, ur]
    for k, p in hmt_ovp.items():
        if isinstance(p, torch.Tensor):
            dic[k] = len(params)
            params.append(p)

        elif isinstance(p, (list, tuple)):
            dic[k] = len(params)
            params += list(p)
            dic[k] = (dic[k], len(params))

    def fn(ee, *params):
        xl = min(params[0], params[1])
        xu = max(params[0], params[1])
        seL, _ = selfEnergy(hd=params[dic['lhd']], hu=params[dic['lhu']], sd=params[dic['lsd']],
                            su=params[dic['lsu']], ee=ee, left=True, voltage=params[0])
        seR, _ = selfEnergy(hd=params[dic['rhd']], hu=params[dic['rhu']], sd=params[dic['rsd']],
                            su=params[dic['rsu']], ee=ee, left=False, voltage=params[1])
        g_trans, _, _, _, _ = recursive_gf(ee, hl=params[dic['hl'][0]:dic['hl'][1]], hd=params[dic['hd'][0]:dic['hd'][1]],
                                       hu=params[dic['hu'][0]:dic['hu'][1]], sd=params[dic['sd'][0]:dic['sd'][1]],
                                       su=params[dic['su'][0]:dic['su'][1]],
                                       sl=params[dic['sl'][0]:dic['sl'][1]], left_se=seL, right_se=seR, seP=None,
                                       s_in=None,
                                       s_out=None)
        s01, s02 = params[dic['hd'][0]].shape
        seL = seL[:s01, :s02]
        s11, s12 = params[dic['hd'][1]-1].shape
        seR = seR[-s11:, -s12:]
        gammaL, gammaR = sigmaLR2Gamma(seL), sigmaLR2Gamma(seR)
        TT = calTT(gammaL, gammaR, g_trans)

        return (fermi_dirac(ee - xu) - fermi_dirac(ee - xl)) * TT

    return quad(fcn=fn, xl=xl, xu=xu, params=params, n=n_int) / pi

def calEqDensity(pole, residue, basis_size, ul, ur, **hmt_ovp):
    N_pole = len(pole)

    def fn(ee):
        seL, _ = selfEnergy(hd=hmt_ovp['lhd'], hu=hmt_ovp['lhu'], sd=hmt_ovp['lsd'],
                            su=hmt_ovp['lsu'], ee=ee, left=True, voltage=ul, etaLead=0.)
        seR, _ = selfEnergy(hd=hmt_ovp['rhd'], hu=hmt_ovp['rhu'], sd=hmt_ovp['rsd'], su=hmt_ovp['rsu'],
                            ee=ee, left=False, voltage=ur, etaLead=0)

        _, grd, _, _, _ = recursive_gf(ee, hl=hmt_ovp['hl'], hd=hmt_ovp['hd'], hu=hmt_ovp['hu'], sd=hmt_ovp['sd'], su=hmt_ovp['su'],
                                           sl=hmt_ovp['sl'], left_se=seL, right_se=seR, seP=None, s_in=None,
                                           s_out=None, eta=0.)
        return torch.cat([i.diag() for i in grd], dim=0)

    # calculating density
    p = torch.zeros((basis_size,), dtype=torch.float64)
    # for i in tqdm(range(N_pole), desc="Calculating EqDensity"):
    for i in range(N_pole):
        grd = fn(pole[i])
        # print(residue)
        p = p - residue[i] * grd

    return 2*p.imag



def calNeqDensity(ul, ur, n_int=100, bSE=None, **hmt_ovp):
    xl = min(ul, ur) - 4*kBT
    xu = max(ul, ur) + 4*kBT

    dic = {}
    if bSE is None:
        params = []
        for p, v in hmt_ovp.items():
            if isinstance(v, torch.Tensor):
                dic[p] = len(params)
                params.append(v)

            elif isinstance(v, (list, tuple)):
                dic[p] = len(params)
                params += list(v)
                dic[p] = (dic[p], len(params))
        # params = [i for i in params if isinstance(i, torch.Tensor)]
        def fn(ee, *params):
            seL, _ = selfEnergy(hd=params[dic['lhd']], hu=params[dic['lhu']], sd=params[dic['lsd']],
                                su=params[dic['lsu']], ee=ee, left=True, voltage=ul)
            seR, _ = selfEnergy(hd=params[dic['rhd']], hu=params[dic['rhu']], sd=params[dic['rsd']],
                                su=params[dic['rsu']], ee=ee, left=False, voltage=ur)
            _, grd, _, _, _ = recursive_gf(ee, hl=params[dic['hl'][0]:dic['hl'][1]], hd=params[dic['hd'][0]:dic['hd'][1]], hu=params[dic['hu'][0]:dic['hu'][1]], sd=params[dic['sd'][0]:dic['sd'][1]],
                                           su=params[dic['su'][0]:dic['su'][1]],
                                           sl=params[dic['sl'][0]:dic['sl'][1]], left_se=seL, right_se=seR, seP=None, s_in=None,
                                           s_out=None)
            dp_neq = torch.cat([-2*i.diag() for i in grd], dim=0)
            return dp_neq.imag
        # print("Calculating NeqDensity")
        return quad(fcn=fn, xl=xl, xu=xu, params=params, n=n_int) / (2*pi)
    else:
        b_seL, b_seR = bSE
        n = len(b_seL)

        xlg, wlg = np.polynomial.legendre.leggauss(n)
        ndim = len(xu.shape)
        xlg = torch.tensor(xlg, dtype=xu.dtype, device=xu.device)[(...,) + (None,) * ndim]  # (n, *nx)
        wlg = torch.tensor(wlg, dtype=xu.dtype, device=xu.device)[(...,) + (None,) * ndim]  # (n, *nx)
        wlg *= 0.5 * (xu - xl)
        xs = xlg * (0.5 * (xu - xl)) + (0.5 * (xu + xl))  # (n, *nx)
        _, grd, _, _, _ = recursive_gf(xs[0], hl=hmt_ovp['hl'], hd=hmt_ovp['hd'], hu=hmt_ovp['hu'], sd=hmt_ovp['sd'],
                                       su=hmt_ovp['su'],
                                       sl=hmt_ovp['sl'], left_se=b_seL[0], right_se=b_seR[0], seP=None, s_in=None,
                                       s_out=None)
        dp_neq = torch.cat([-2 * i.diag() for i in grd], dim=0)
        res = wlg[0] * dp_neq.imag
        for i in range(1, n):
            _, grd, _, _, _ = recursive_gf(xs[i], hl=hmt_ovp['hl'], hd=hmt_ovp['hd'], hu=hmt_ovp['hu'],
                                           sd=hmt_ovp['sd'],
                                           su=hmt_ovp['su'],
                                           sl=hmt_ovp['sl'], left_se=b_seL[i], right_se=b_seR[i], seP=None, s_in=None,
                                           s_out=None)
            dp_neq = torch.cat([-2 * i.diag() for i in grd], dim=0)
            res += wlg[i] * dp_neq.imag

        return res / (2*pi)

def getxyzdensity(offset, siteDensity):
    # potential might be a 1d list, where each site correspond a spatial coordinate in corrd
    density = torch.zeros((len(offset),), dtype=torch.float64)
    for i in range(len(offset)-1):
        density[i] += siteDensity[offset[i]:offset[i+1]].sum()
    density[-1] += siteDensity[offset[-1]:].sum()

    return density

def citeCoord2Coord(offset, siteCoord):
    return siteCoord[offset]

def attachPotential(offset, hd, V, basis_size):
    offset_ = list(offset) + [basis_size]
    site_V = torch.cat([V[i].repeat(offset_[i+1]-offset_[i]) for i in range(len(offset_)-1)], dim=0)
    start = 0
    hd_V = []
    for i in range(len(hd)):
        hd_V.append(hd[i] - torch.diag(site_V[start:start+len(hd[i])]))
        start = start + len(hd[i])

    return hd_V

# def getImg(n, coord, d, dim):
#     zj = coord[:, 2]
#     img1 = torch.stack([zj - (2 * i + 2) * d for i in range(1, n + 1)], dim=1).unsqueeze(1)
#     img2 = torch.stack([-zj - 2 * i * d for i in range(1, n + 1)], dim=1).unsqueeze(1)
#     img3 = torch.stack([zj + (2 * i + 2) * d for i in range(1, n + 1)], dim=1).unsqueeze(1)
#     img4 = torch.stack([-zj + (2 * i + 2) * d for i in range(1, n + 1)], dim=1).unsqueeze(1)
#     img_z = torch.cat([img1, img2, img3, img4], dim=1).view(-1, 4*n).unsqueeze(2)
#     xy = coord[:, :2].view(-1, 1, 2).expand(-1, 4 * n, 2)
#     xyz = torch.cat((xy, img_z), dim=2).view(-1, 3)
#
#     return xyz

def getImg(n, coord, d, dim=2):
    zj = coord[:, dim]
    img1 = torch.stack([zj - (2 * i + 2) * d for i in range(1, n + 1)], dim=1).unsqueeze(1)
    img2 = torch.stack([-zj - 2 * i * d for i in range(1, n + 1)], dim=1).unsqueeze(1)
    img3 = torch.stack([zj + (2 * i + 2) * d for i in range(1, n + 1)], dim=1).unsqueeze(1)
    img4 = torch.stack([-zj + (2 * i + 2) * d for i in range(1, n + 1)], dim=1).unsqueeze(1)
    img = torch.cat([img1, img2, img3, img4], dim=1).view(-1, 4 * n).unsqueeze(2)
    if dim==2:
        xy = coord[:, :2].view(-1, 1, 2).expand(-1, 4 * n, 2)
        xyz = torch.cat((xy,img), dim=2).view(-1, 3)
    elif dim==1:
        x = coord[:, 0].view(-1, 1, 1).expand(-1, 4 * n, 1)
        z = coord[:, 2].view(-1, 1, 1).expand(-1, 4 * n, 1)
        xyz = torch.cat((x,img,z), dim=2).view(-1, 3)
    elif dim==0:
        yz = coord[:, 1:].view(-1, 1, 2).expand(-1, 4 * n, 2)
        xyz = torch.cat((img,yz), dim=2).view(-1, 3)
    else:
        raise ValueError

    return xyz

class density2Potential(torch.autograd.Function):
    '''
    This solves a poisson equation with dirichlet boundary condition
    '''
    @staticmethod
    def forward(ctx, coord, density, n, d, d_trans):
        imgCoord = getImg(n=n, coord=coord, d=d, dim=d_trans)

        img_density = density.view(-1,1,1).expand(-1,1,n)
        img_density = torch.cat([img_density, -img_density, img_density, -img_density], dim=1)
        img_density = img_density.reshape(-1)
        V = []
        if coord.requires_grad:
            pgt = 2
        else:
            pgt = 1

        grad_coord = []

        # for i in tqdm(range(density.shape[0]), desc="Calculating Image Charge Summation"):
        #     density_ = torch.cat([density[0:i],density[i+1:]], dim=0)
        #     density_ = torch.cat([density_, img_density], dim=0)
        #     coord_ = torch.cat([coord[0:i], coord[i+1:]], dim=0)
        #     coord_ = torch.cat([coord_, imgCoord], dim=0)
        #
        #     out = lfmm3d(eps=1e-10, sources=coord_.transpose(1, 0).numpy(), charges=density_.numpy(), dipvec=None,
        #                         targets=coord[i].unsqueeze(1).numpy(), pgt=pgt)
        #
        #     V.append(out.pottarg[0])
        #
        #     if coord.requires_grad:
        #         grad_coord.append(out.gradtarg)
        # ctx.save_for_backward(coord, torch.tensor(grad_coord), imgCoord, torch.tensor(n))
        # return torch.tensor(V) / (4*pi)

        # for i in tqdm(range(density.shape[0]), desc="Calculating Image Charge Summation"):
        for i in range(density.shape[0]):
            density_ = torch.cat([density[0:i],density[i+1:]], dim=0)
            coord_ = torch.cat([coord[0:i], coord[i+1:]], dim=0)

            out = lfmm3d(eps=1e-10, sources=coord_.transpose(1, 0).numpy(), charges=density_.numpy(), dipvec=None,
                                targets=coord[i].unsqueeze(1).numpy(), pgt=pgt)
            V.append(out.pottarg[0])

            if coord.requires_grad:
                grad_coord.append(out.gradtarg)

        out = lfmm3d(eps=1e-10, sources=imgCoord.transpose(1, 0).numpy(), charges=img_density.numpy(), dipvec=None,
                                targets=coord.transpose(1,0).numpy(), pgt=pgt)
        V = torch.tensor(V) + torch.tensor(out.pottarg)
        if coord.requires_grad:
            grad_coord = torch.tensor(grad_coord) + torch.tensor(out.gradtarg)
        ctx.save_for_backward(coord, torch.tensor(grad_coord), imgCoord, torch.tensor(n))
        return V / (4*pi)


    @staticmethod
    def backward(ctx, *grad_outputs):
        # to avoid the overflow and overcomplexity, the backward can also be viewed as a fmm.
        coord, grad_coord, imgCoord, n = ctx.saved_tensors
        grad_density = []
        grad_outputs = grad_outputs[0].reshape(-1)
        img_grad_outputs = grad_outputs.view(-1, 1, 1).expand(-1, 1, n)
        img_grad_outputs = torch.cat([img_grad_outputs, -img_grad_outputs, img_grad_outputs, -img_grad_outputs])
        img_grad_outputs = img_grad_outputs.reshape(-1)
        for i in range(grad_outputs.shape[0]):
            grad_outputs_ = torch.cat([grad_outputs[0:i],grad_outputs[i+1:]], dim=0)
            coord_ = torch.cat([coord[0:i], coord[i+1:]], dim=0)

            grad_out = lfmm3d(eps=1e-15, sources=coord_.transpose(1, 0).detach().numpy(), charges=grad_outputs_.detach().numpy(), dipvec=None,
                                targets=coord[i].unsqueeze(1).detach().numpy(), pgt=1)
            grad_density.append(grad_out.pottarg[0])
        grad_out = lfmm3d(eps=1e-15, sources=imgCoord.transpose(1, 0).detach().numpy(), charges=img_grad_outputs.detach().numpy(), dipvec=None,
                                targets=coord.transpose(1,0).detach().numpy(), pgt=1)
        grad_density = torch.tensor(grad_density) + torch.tensor(grad_out.pottarg)

        if len(grad_coord) == 0:
            return None, grad_density / (4*pi), None, None, None
        else:
            return grad_coord.squeeze(-1), grad_density / (4*pi), None, None, None

def calVdrop(ul, tCoord, zs, zd, ur):
    # assume ul is source side, correspond zs
    return ul + (ur - ul)*(tCoord-zs)/(zd-zs)

def TT_with_hTB(hamiltonian, V_ext, el, er, ul, ur, n, fd_step=1e-6, ifseebeck=False, ifFD=False, ifASE=False, dtype=torch.float64):
    hL, hD, hR, sL, sD, sR = hamiltonian.get_hamiltonians()
    hl_list, hd_list, hr_list, sl_list, sd_list, sr_list, subblocks = \
        hamiltonian.get_hamiltonians_block_tridiagonal(optimized=True)
    hd_ = attachPotential(hamiltonian._offsets, hd_list, V_ext, hamiltonian.basis_size)

    ee_list = torch.linspace(start=el, end=er, steps=n)
    seebeck = []
    seebeckFD = []
    transmission = []

    def fn(ee):
        seL, _ = selfEnergy(ee=ee, hd=hD, hu=hL.conj().T, sd=sD, su=sL.conj().T, left=True, voltage=ul)
        seR, _ = selfEnergy(ee=ee, hd=hD, hu=hR, sd=sD, su=sR, left=False, voltage=ur)
        g_trans, _, _, _, _ = recursive_gf(ee, hl=hl_list, hd=hd_, hu=hr_list, sd=sd_list, su=sr_list,
                                           sl=sl_list, left_se=seL, right_se=seR, seP=None, s_in=None, s_out=None)
        s01, s02 = hd_[0].shape
        seL = seL[:s01, :s02]
        s11, s12 = hd_[-1].shape
        seR = seR[-s11:, -s12:]
        gammaL, gammaR = sigmaLR2Gamma(seL), sigmaLR2Gamma(seR)
        TT = calTT(gammaL, gammaR, g_trans)
        return TT
    start = time.time()
    if ifseebeck:
        ee_list.requires_grad_()
    for ee in tqdm(ee_list, desc="Transmission"):
        TT = fn(ee)
        transmission.append(TT)
        if ifseebeck:
            seebeck.append(torch.autograd.grad(TT, ee)[0])
    endAD = time.time()
    if ifFD:
        for ee in tqdm(ee_list, desc="FD Seebeck"):
            seebeckFD.append(finite_difference(fn, ee, h=fd_step, dtype=dtype))
        endFD = time.time()



    if ifASE:
        n_L = hd_list[0].shape[0]
        n_R = hd_list[-1].shape[0]
        with torch.no_grad():
            TT = ase.transport.TransportCalculator(energies=ee_list.numpy(), h=hD.numpy(), s=sD.numpy(),
            h1=hD.numpy(), s1=sD.numpy(), h2=hD.numpy(), s2=sD.numpy()).get_transmission()
    if ifseebeck and ifFD:
        print("Time for AD: {0}, for FD {1}.".format(endAD - start, endFD - endAD))
        if ifASE:
            return torch.stack(transmission), torch.stack(seebeck), torch.stack(seebeckFD), TT
        else: return torch.stack(transmission), torch.stack(seebeck), torch.stack(seebeckFD)
    elif ifseebeck:
        return torch.stack(transmission), torch.stack(seebeck)
    elif ifFD:
        return torch.stack(transmission), torch.stack(seebeckFD)
    else:
        return torch.stack(transmission)


def IV_with_hTB(hamiltonian, V_list, ifderivative=False, n_int=100):
    hL, hD, hR, sL, sD, sR = hamiltonian.get_hamiltonians()
    hl_list, hd_list, hr_list, sl_list, sd_list, sr_list, subblocks = \
        hamiltonian.get_hamiltonians_block_tridiagonal(optimized=True)

    current = []
    dIdV = []
    for (u, V_ext) in V_list.item():
        if ifderivative:
            u.requires_grad_()
        hd_ = attachPotential(hamiltonian._offsets, hd_list, V_ext, hamiltonian.basis_size)
        I = calCurrent(ul=u[0], ur=u[1], n_int=n_int, hd=hd_, hu=hr_list, hl=hl_list, sd=sd_list, su=sr_list, sl=sl_list,
                       lhd=hD, lhu=hL.conj().T, lhl=hL, lsd=hD, lsu=sL.conj().T, lsl=sL,
                       rhd=hD, rhu=hR, rhl=hR.conj().T, rsd=sD, rsu=sR, rsl=sR.conj().T)
        if ifderivative:
            current.append(I)
            dIdV = torch.autograd.grad(I, u[1])

    if ifderivative:
        return torch.stack(current), torch.stack(dIdV)
    else:
        return torch.stack(current)


if __name__ == '__main__':
    # coord = torch.tensor([[0,0,0],[1,0,0], [0,0,2.5],[1,0,2.75],[0,0,5]], dtype=torch.double)
    # density = torch.tensor([1.,1.25,1.25,1.75,1.], dtype=torch.double).requires_grad_()
    # n=100
    # d=7
    # # grad = torch.autograd.grad(out.sum(), density)
    # # print(grad)
    # torch.autograd.gradcheck(lambda test_density: density2Potential.apply(coord, test_density, n, d).sum(), inputs=density, eps=1e-6)

    calNeqDensity(ul=torch.tensor(0.), ur=torch.tensor(1.), )
