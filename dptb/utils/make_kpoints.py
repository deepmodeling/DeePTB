import numpy as np
import ase
import logging
log = logging.getLogger(__name__)


def rot_revlatt_2D(rev_latt,index=[0,1]): # 0, x; 1,y, 2,z
    """ Transform the coordinate system of reciprocal lattice vectors. 
    The new coordinate system is defined by the two reciprocal lattice vectors with index [0,1] in the original coordinate system. 
    The new x-axis is along the reciprocal lattice vector with index 0, and the new z-axis is perpendicular to the new x-axis and the reciprocal lattice vector with index 1.
    The new y-axis is perpendicular to the new x-axis and the new z-axis. 
    The new coordinate system is right-handed. 
    The new reciprocal lattice vectors are returned as a 3x3 matrix. 
    The transformation matrix is also returned. The new reciprocal lattice vectors are obtained by new_rev_latt = rev_latt @ newcorr.I


    Parameters
    ----------
    rev_latt : numpy.matrix
        The reciprocal lattice vectors in the original coordinate system. A 3x3 matrix.
    index : list. [i1, i2]
        A list of 2 integers, the index of the two reciprocal lattice vectors to be used to define the new coordinate system. 
        The index of the reciprocal lattice vector is 0, 1, or 2, corresponding to the x, y, and z direction, respectively.

    Returns
    -------
    rev_latt_new : numpy.matrix
        The reciprocal lattice vectors in the new coordinate system. A 3x3 matrix.
    newcorr : numpy.matrix
        The transformation matrix. The new reciprocal lattice vectors are obtained by new_rev_latt = rev_latt @ newcorr.I
    """

    if isinstance(rev_latt, np.matrix):
        if rev_latt.shape != (3,3):
            log.error("Error! rev_latt must be a 3x3 matrix!")
            raise ValueError
    else:
        log.error("Error! rev_latt must be a 3x3 matrix!")
        raise ValueError
    
    index_left  = [0,1,2]
    for i in index:
        index_left.remove(i)

    vec1 = np.array(rev_latt[index[0]]).reshape(-1)
    vec2 = np.array(rev_latt[index[1]]).reshape(-1)
    vec3 = np.array(rev_latt[index_left[0]]).reshape(-1)

    avec1 = vec1/np.linalg.norm(vec1)
    avec3 = np.cross(avec1,vec2)/np.linalg.norm(np.cross(avec1,vec2))
    avec2 = np.cross(avec3,avec1)
    if np.dot(np.cross(avec1,avec2),avec3) < 0:
        avec3 = -avec3
    newcorr = np.zeros((3,3))    
    newcorr[index[0]] = avec1
    newcorr[index[1]] = avec2
    newcorr[index_left[0]] = avec3
    newcorr = np.mat(newcorr)

    rev_latt_new = rev_latt @ newcorr.I

    return rev_latt_new, newcorr


def kmesh_fs(meshgrid=[1,1,1]):
    """ Generate k-points on mesh for fermi surface calculation. The k-points are centered at Gamma point. and with endpoints [0,1]. 

    Parameters
    ----------
    meshgrid : list. [N1, N2, N3]
        A list of 3 integers, the number of k-points in each direction.
    
    """

    Nx, Ny, Nz = meshgrid
    lx, ly, lz = np.linspace(0, 1, Nx), np.linspace(0, 1, Ny), np.linspace(0, 1, Nz)
    xx, yy, zz = np.meshgrid(lx, ly, lz, indexing='ij')
    kgrids  = np.array([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]).T

    return (lx,ly,lz), kgrids


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
    
    This function is modified from ASE.
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




def kmesh_sampling(meshgrid=[1,1,1], is_gamma_center=True):
    """ Generate k-points using Monkhorst-Pack method based on given meshgrid. The k-points are centered at Gamma point by default.
     
    """

    kpoints = np.indices(meshgrid).transpose((1, 2, 3, 0)).reshape((-1, 3))

    if is_gamma_center:
        kpoints = gamma_center(meshgrid)
    else:
        kpoints = monkhorst_pack(meshgrid)
    return kpoints


def kmesh_sampling_negf(meshgrid=[1,1,1], is_gamma_center=True, is_time_reversal=True):
    """ Generate k-points for NEGF based on given meshgrid. Through time symmetry reduction, the number of k-points is reduced.
     
    """

    if is_time_reversal:
        kpoints,wk = time_symmetry_reduce(meshgrid, is_gamma_center=is_gamma_center)
            
    else:
        kpoints = kmesh_sampling(meshgrid, is_gamma_center=is_gamma_center)
        wk = np.ones(len(kpoints))/len(kpoints)
    
    return kpoints,wk


def time_symmetry_reduce(meshgrid=[1,1,1], is_gamma_center=True):
    '''Reduce the number of k-points in a meshgrid by applying symmetry operations.

    For gamma centered meshgrid, k-points range from 0 to 1 in each dimension initially. 
    For non-gamma centered meshgrid, k-points range from -0.5 to 0.5 in each dimension initially.

    With time symmetry reduction, the number of k-points is reduced and limited to [0,0.5] in x-direction.
    
    Parameters
    ----------
    meshgrid
        The `meshgrid` parameter specifies the number of k-points in each direction. 
    is_gamma_center
        The parameter "is_gamma_center" is a boolean value that determines whether the k-point mesh must be
    centered around the gamma point (0, 0, 0) or not. 
    
    Returns
    -------
        the reduced k-points and their corresponding weights.
    
    '''

    k_points = kmesh_sampling(meshgrid, is_gamma_center=is_gamma_center)
    k_points_with_tr = []
    kweight = []

    
    if is_gamma_center:
        k_points[k_points>0.5] = k_points[k_points>0.5] - 1

    k_points = np.round(k_points, decimals=5)

    for kp in k_points:
        if (-kp).tolist() not in k_points_with_tr:
            k_points_with_tr.append(kp.tolist())
            kweight.append(1)
        else:
            kweight[k_points_with_tr.index((-kp).tolist())] += 1

    k_points_with_tr = np.array(k_points_with_tr)

    # make the reduced kpoints in [0,0.5] in x-direction
    if is_gamma_center:
        k_points_with_tr[k_points_with_tr < 0] += 1 
    else: # MP sampling
        k_points_with_tr =  -1 * k_points_with_tr # due to time revesal symmetry

    # sort the k-points
    k_sort_indx = np.lexsort((k_points_with_tr[:, 2], k_points_with_tr[:, 1], k_points_with_tr[:, 0]))
    k_points_with_tr = k_points_with_tr[k_sort_indx]
    kweight = np.array(kweight)/len(k_points) # normalize the weight to one
    kweight = kweight[k_sort_indx]
    assert abs(kweight.sum() - 1.0) < 1e-5, "The sum of weight is not 1.0"

    return k_points_with_tr, kweight




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
    # rev_latt =(np.matrix(structase.cell).I.T)
    rev_latt =  np.linalg.inv(np.array(structase.cell).T)
    kdiff = kpoints[1:] - kpoints[:-1]
    # kdiff_cart = np.asarray(kdiff * rev_latt)
    kdiff_cart = np.dot(kdiff, rev_latt)
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

def vasp_kpath(structase, pathstr:list[str], high_sym_kpoints_dict:dict, number_in_line:int):
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
