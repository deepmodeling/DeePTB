import numpy as np
import torch
import logging
from dptb.utils.constants import CUBIC_MAG_NUM_DICT, LM_MAG_NUM_DICT

log = logging.getLogger(__name__)

# l=0,  m=0, s
#
#       m=-1, py
# l=1,  m= 0, pz
#       m= 1, px
#
#       m= -2, dxy 
#       m= -1, dyz
# l=2,  m=  0, dz2
#       m=  1, dxz
#       m=  2 dx2-y2

# sktb 变换中轨道的顺序选择 m从小到大排列。see dptb.hamiltonian.transform_sk for more details.
# 因此定义soc矩阵的 CUBIC_MAG_NUM_DICT 中每个角动量的轨道，也是按照m从小到大排序。
# 值得注意的是这个排序只是一种约定，保持代码中一致就可以。


# magnetic quantum number
def lm2cubic_mat(cubic_mag_num, lm_mag_num, device='cpu', dtype=torch.float32):
    '''> The function `lm2cubic_mat` takes in two lists of magnetic quantum numbers, one for the cubic
    basis and one for the lm basis, and returns a matrix that transforms the lm basis into the cubic basis
    
    Parameters
    ----------
    cubic_mag_num
        the magnetic quantum numbers of the cubic basis
    lm_mag_num
        the magnetic quantum numbers of the lm basis
    
    Returns
    -------
        The matrix that transforms from  the lm harmonics basis to the cubic harmonics basis.
    
    '''
    assert len(cubic_mag_num) in [1, 3, 5], "The number of magnetic_quantum_numbers must be 1, 3, 5"
    assert len(lm_mag_num) == len(cubic_mag_num), "The number of lm_mag_nummust the same  as cubic_mag_num"
    
    if dtype is torch.float32:
        cdtype = torch.complex64
    elif dtype is torch.float64:
        cdtype = torch.complex128
    else:
        log.error(msg="the dtype is not supported! now only float64, float32 is supported!")

    t2 = torch.tensor(2.0)
    s2_1=1.0/torch.sqrt(t2)
    M = torch.zeros([len(cubic_mag_num),len(cubic_mag_num)],device=device, dtype=cdtype)


    for i in range(len(cubic_mag_num)):
        mq = cubic_mag_num[i]
        if mq == 0:
            j = lm_mag_num.index(mq)
            M[i,j] = 1
        elif mq < 0:
            j = lm_mag_num.index(mq)
            M[i,j] = 1.0j*s2_1
            j = lm_mag_num.index(-1*mq)
            M[i,j] = 1.0j*s2_1*(-1)**(mq+1)
        elif mq > 0:
            j = lm_mag_num.index(-1*mq)
            M[i,j] = 1.0*s2_1
            j = lm_mag_num.index(mq)
            M[i,j] = 1.0*s2_1*(-1)**(mq)
        else:
            raise Exception
    return M

def creat_basis_lm(orb: str):
    '''It creates a list of  |lm,s> states for orbital, where each list contains the quantum numbers of 
            a basis state in the form of [l,m,s]
    
    Parameters
    ----------
    orb
        the orbital type, 's', 'p', 'd'.  'f' is not suppoted yet.     
    Returns
    -------
        A list of lists. Each list contains the quantum numbers for a single state.
    
    '''
    assert orb in ['s', 'p', 'd'], 'The orb parameter must be one of the s, p ,d values  in the format of str'    
    l_value = ['s', 'p', 'd'].index(orb)
    mlist = LM_MAG_NUM_DICT[orb]
    basis = []
    for m in mlist:
        for spin in [1,-1]:
            basis.append([l_value, m, spin])
    return basis

def get_matrix_lmbasis(basis, device='cpu', dtype=torch.float32):
    '''> The function `get_matrix_lmbasis` takes a list of basis states and returns the matrix
    representation of the operator L cdot S in the basis of |l,m,s >
    
    Parameters
    ----------
    basis
        a list of basis states in the form of [l,m,s]
    device, optional
        the device to use for the tensors.
    dtype
        the data type of the matrix elements.
    
    Returns
    -------
        The matrix representation of the operator L.S in the basis |l,m,s>
    
    '''

    ndim = len(basis)
    MatLpSm = torch.zeros([ndim,ndim], device=device, dtype=dtype)
    MatLmSp = torch.zeros([ndim,ndim], device=device, dtype=dtype)
    MatLzSz = torch.zeros([ndim,ndim], device=device, dtype=dtype)
    if basis == None:
        LdotS = torch.zeros([2,2], device=device, dtype=dtype)
    else:
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


def get_soc_matrix_cubic_basis(orbital: str,cubic_mag_num=None, lm_mag_num=None, device='cpu', dtype=torch.float32):  
    '''The function `get_soc_matrix_cubic_basis` takes in an orbital type (s, p, or d) and returns the
    spin-orbit coupling matrix in the cubic basis.
    
    Parameters
    ----------
    orbital : str
        's', 'p', 'd'
    
    Returns
    -------
        The spin-orbit coupling matrix in the cubic basis. And the orbital follows the order: up up up ..., down, down, ....
        ie:
        [[upup],[updn]
         [dnup],[dndn]]
    '''
    if cubic_mag_num is None:
        cubic_mag_num = CUBIC_MAG_NUM_DICT[orbital]
    if lm_mag_num is None:
        lm_mag_num = LM_MAG_NUM_DICT[orbital]

    if dtype is torch.float32:
        cdtype = torch.complex64
    elif dtype is torch.float64:
        cdtype = torch.complex128
    else:
        log.error(msg="the dtype is not supported! now only float64, float32 is supported!")

    num_orb = {'s':1,'p':3,'d':5}
    assert len(cubic_mag_num)  == num_orb[orbital], "The number of magnetic_quantum_numbers is not correct!"
    assert len(lm_mag_num) == num_orb[orbital],  "The number of magnetic_quantum_numbers is not correct!"   
    assert orbital in ['s','p','d']   

    lm_basis = creat_basis_lm(orbital)
    Mtrans = lm2cubic_mat(cubic_mag_num, lm_mag_num)
    Msoc_lm = get_matrix_lmbasis(lm_basis)
    Msoc_lm_clx = torch.complex(Msoc_lm, torch.zeros_like(Msoc_lm))

    trans = torch.kron(Mtrans, torch.eye(2)).T
    transHT = torch.conj(trans.T)
    Msoc_cubic = transHT @ Msoc_lm_clx @ trans
    
    # 'transfor the spin basis form up down up down ... to up up ... down down...'
    norbs = len(cubic_mag_num)
    assert len(Msoc_cubic) == 2*norbs
    Msoc_updn_block = torch.zeros([2*norbs,2*norbs],device=device,dtype=cdtype)
    Msoc_updn_block[0    :  norbs,    0:  norbs] = Msoc_cubic[0:2*norbs:2,0:2*norbs:2]
    Msoc_updn_block[norbs:2*norbs,norbs:2*norbs] = Msoc_cubic[1:2*norbs:2,1:2*norbs:2]
    Msoc_updn_block[0    :  norbs,norbs:2*norbs] = Msoc_cubic[0:2*norbs:2,1:2*norbs:2]
    Msoc_updn_block[norbs:2*norbs,    0:  norbs] = Msoc_cubic[1:2*norbs:2,0:2*norbs:2]

    return Msoc_updn_block

