import numpy as np
import torch

def creat_basis_lm(orb):
    """Creat |lm,s> stated for orbital."""
    if orb=='p':
        basis = []
        for m in [-1,0,1]:
            for spin in [1,-1]:
                basis.append([1,m,spin])
    if orb=='d':
        basis = []
        for m in [-2,-1,0,1,2]:
            for spin in [1,-1]:
                basis.append([2,m,spin])
    if orb=='s':
        print('we do not consider the soc in s orbital')
    if orb=='f':
        print('for now, soc for f orbital is not added.')
        exit()
    return basis

def get_matrix_lmbasis(basis, device='cpu', dtype=torch.float32):
    """Creat Hsoc matrix  in |lm,s> basis."""
    ndim = len(basis)
    MatLpSm = torch.zeros([ndim,ndim], device=device, dtype=dtype)
    MatLmSp = torch.zeros([ndim,ndim], device=device, dtype=dtype)
    MatLzSz = torch.zeros([ndim,ndim], device=device, dtype=dtype)
    for i in range(len(basis)):
        raw = i
        cof,bas = MapLpSm(basis[i])
        if bas in basis:
            col = basis.index(bas)
            MatLpSm[raw,col] = cof
            
        cof,bas = MapLmSp(basis[i])
        if bas in basis:
            col = basis.index(bas)
            MatLmSp[raw,col] = cof
        
        cof,bas = MapLzSz(basis[i])
        if bas in basis:
            col = basis.index(bas)
            MatLzSz[raw,col] = cof
    LdotS = 0.5*(MatLpSm + MatLmSp + MatLzSz)
    return LdotS

def MapLpSm(lms):
    """L+ S_ |lm,s>."""
    l,m,s = lms[0],lms[1],lms[2]
    if s ==  1:
        cof = np.sqrt((l-m)*(l+m+1))
    if s == -1:
        cof = 0
    return cof,[l,m+1,s-2]


def MapLmSp(lms):
    """L- S+ |lm,s>."""
    l,m,s = lms[0],lms[1],lms[2]
    if s ==  1:
        cof = 0
    if s == -1:
        cof = np.sqrt((l+m)*(l-m+1))
    return cof,[l,m-1,s+2]


def MapLzSz(lms):
    """Lz Sz |lm,s>."""
    l,m,s = lms[0],lms[1],lms[2]
    if s ==  1:
        cof = m
    if s == -1:
        cof = -m
    return cof,[l,m,s]