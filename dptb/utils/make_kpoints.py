import numpy as np
import ase
import logging
log = logging.getLogger(__name__)

def monkhorst_pack(meshgrid=[1,1,1]):
    """ Generate k-points using Monkhorst-Pack method based on given meshgrid.
    
    Parameters
    ----------
    meshgrid : list. [N1, N2, N3]
        A list of 3 integers, the number of k-points in each direction.
    
    Returns
    -------
    kpoints : numpy.ndarray
        A numpy array of k-points.

    """
    if len(meshgrid) != 3  or not (np.array(meshgrid,dtype=int) > 0).all():
        log.error("Error! meshgrid must be a list of 3 positive integers!")
        raise ValueError

    kpoints = np.indices(meshgrid).transpose((1, 2, 3, 0)).reshape((-1, 3))
    kpoints = (kpoints + 0.5) / meshgrid - 0.5
    return kpoints

def gamma_center(meshgrid=[1,1,1]):
    """ Generates a gamma centered k-point mesh based on the given meshgrid.

    Parameters
    ----------
    meshgrid : list. [N1, N2, N3]
        A list of 3 integers, the number of k-points in each direction.

    Returns
    -------
    kpoints : numpy.ndarray
        A numpy array of k-points.
    """
    if len(meshgrid) != 3  or not (np.array(meshgrid,dtype=int) > 0).all():
        log.error("Error! meshgrid must be a list of 3 positive integers!")
        raise ValueError

    kpoints = np.indices(meshgrid).transpose((1, 2, 3, 0)).reshape((-1, 3))
    kpoints = (kpoints) / meshgrid
    return kpoints

def kgrid_spacing(structase,kspacing:float,sampling='MP'):
    """Generate k-points based on the given k-spacing and sampling method.
    
    Parameters
    ----------
    structase : ase.Atoms
        The structure in ASE format.
    kspacing : float
        The k-spacing.
    sampling : str
        The sampling method. 'MP' for Monkhorst-Pack method, 'Gamma' for gamma centered method.

    Returns
    -------
    kpoints : numpy.ndarray
        A numpy array of k-points.
    """
    
    assert isinstance(structase,ase.Atoms)
    rev_latt = 2*np.pi*np.mat(structase.cell).I
    meshgrid = np.ceil(np.dot(rev_latt.T, np.array([1/kspacing, 1/kspacing, 1/kspacing]))).astype(int)

    if sampling == 'MP':
        kpoints = monkhorst_pack(meshgrid)
    elif sampling == 'Gamma':
        kpoints = gamma_center(meshgrid)
    else:
        log.error("Error! sampling must be either 'MP' or 'Gamma'!, by default it is MP using the monkhorst_pack method.")
        raise ValueError

    return kpoints


def abacus_kpath(structase, kpath):
    '''> The function `abacus_kpath` takes in a structure and a list of high symmetry points. It returns a list of k-points, a list of x-values, and a list of high symmetry k-points.
    
    Parameters
    ----------
    structase : ase.Atoms
        The structure in ASE format.
    kpath : list
        A list of high symmetry points. Each high symmetry point is a list of 4 elements: [kx, ky, kz, nk], where nk is the number of k-points to be used in the path between this high symmetry point and the next one.

    Returns
    -------
    kpath_list : list
        A list of k-points.
    kdist_list : list
        A list of x-values.
    high_sym_kpoints : list
        A list of high symmetry k-points.
    '''
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
    rev_latt = np.mat(structase.cell).I.T
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

def vasp_kpath(structase, pathstr:str, high_sym_kpoints_dict:dict, number_in_line:int):
    """The function `vasp_kpath` takes in a structure, a string of high symmetry points, a dictionary of high symmetry points, and the number of k-points in each line. 
    It returns a list of k-points, a list of x-values, a list of high symmetry k-points, and a list of labels.

    Parameters:
    -----------
    structase: ase structure object
    pathstr: str
        a string that defines the path in reciprocal space.
    high_sym_kpoints: dict
        a dictionary of high symmetry points
    number_in_line: int
        the number of k-points in each line

    Returns:
    --------
    klist: np.array, float, shape [N,3]
        a list of k-points
    xlist: np.array, float, shape [N]
        a list of x-values
    xlist_label: list[float]
        a list of high symmetry k-points
    klabels: list[str]
    """

    kpath = []
    klist = []

    for i  in range(len(pathstr)):
        kline = (pathstr[i].split('-'))
        kpath.append([high_sym_kpoints_dict[kline[0]],high_sym_kpoints_dict[kline[1]]])
        kline_list = np.linspace(high_sym_kpoints_dict[kline[0]], high_sym_kpoints_dict[kline[1]], number_in_line)
        klist.append(kline_list)
        if i == 0:  
            klabels = [(pathstr[i].split('-')[0])]
        else:
            if pathstr[i].split('-')[0] == pathstr[i-1].split('-')[1]:
                klabels.append(pathstr[i].split('-')[0])
            else:
                klabels.append(pathstr[i-1].split('-')[1] + '|' + pathstr[i].split('-')[0])
            if i == len(pathstr)-1:
                klabels.append(pathstr[i].split('-')[1])

    kpath = np.asarray(kpath)
    klist = np.concatenate(klist)


    rev_latt = np.mat(structase.cell).I.T
    #rev_latt = 2*np.pi*np.mat(ase_struct.cell).I
    kdiff = kpath[:,1] - kpath[:,0]
    kdiff_cart = np.asarray(kdiff * rev_latt)
    kdist  = np.linalg.norm(kdiff_cart,axis=1)

    xlist_label = [0] 
    for i in range(len(kdist)):
        if i == 0:
            xlist = np.linspace(0,kdist[i],number_in_line)
        else:
            xlist = np.concatenate([xlist, xlist[-1] + np.linspace(0,kdist[i],number_in_line)])
        xlist_label.append(xlist[-1])
    
    return klist, xlist, xlist_label, klabels
