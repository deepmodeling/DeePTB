import numpy as np

def interp_kpath(structase, kpath):
    kpath = np.asarray(kpath)
    assert kpath.shape[-1] == 4
    assert  len(kpath.shape) == 2
    kpoints = kpath[:,0:3]
    num_kp = kpath[:,3].astype(int)
    assert num_kp[-1] == 1

    kpath_list = []
    for i in range(len(kpoints)-1):
        tmp = np.linspace(kpoints[i],kpoints[i+1],num_kp[i]+1)[0:num_kp[i]]
        kpath_list.append(tmp)
    
    kpath_list.append(kpoints[-1:])
    kpath_list = np.concatenate(kpath_list,axis=0)

    #rev_latt = 2*np.pi*np.mat(ase_struct.cell).I
    rev_latt = np.mat(structase.cell).I
    kdiff = kpoints[1:] - kpoints[:-1]
    kdiff_cart = np.asarray(kdiff * rev_latt)
    kdist  = np.linalg.norm(kdiff_cart,axis=1)
    
    kdist_list = []
    high_sym_kpoints = []
    for i in range(len(kdist)):
        if num_kp[i]==1:
            kdist[i]=0
    for i in range(len(kdist)):
        tmp = np.linspace(np.sum(kdist[:i]), np.sum(kdist[:i+1]),num_kp[i]+1)[0:num_kp[i]]
        high_sym_kpoints.append(tmp[0])
        if i==0:
            kdist_list = tmp.copy()
        else:
            kdist_list = np.concatenate([kdist_list,tmp],axis=0)
    kdist_list = np.append(kdist_list,[np.sum(kdist)])
    high_sym_kpoints.append(np.sum(kdist))
    high_sym_kpoints = np.asarray(high_sym_kpoints)

    return kpath_list, kdist_list, high_sym_kpoints


def ase_kpath(structase, pathstr:str, total_nkpoints:int):
    '''> The function `ase_kpath` takes in a structure, a string of high symmetry points, and the total
    number of k-points to be used in the band structure calculation. It returns a list of k-points, a
    list of x-values, a list of high symmetry k-points, and a list of labels
    
    Parameters
    ----------
    structase
        the ase structure object
    pathstr : str
        a string that defines the path in reciprocal space.
    total_nkpoints : int
        the total number of k-points along the path
    '''
    
    kpath = structase.cell.bandpath(pathstr, npoints=total_nkpoints)
    xlist, high_sym_kpoints, labels = kpath.get_linear_kpoint_axis()
    klist = kpath.kpts
    return klist, xlist, high_sym_kpoints, labels

