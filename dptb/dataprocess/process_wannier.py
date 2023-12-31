import re
import numpy as np
from ase import Atoms
from dptb.utils.constants import anglrMId, Orbital_Order_Wan_Default, Orbital_Order_SK
from dptb.structure.structure import BaseStruct
from dptb.nnsktb.onsiteDB import onsite_energy_database

def get_wannier_blocks(file:str, struct:BaseStruct, wannier_proj_orbital:dict, orb_wan:dict=None):
    """ get the hopping matrices in the order of dptb.
    """
    
    Rlatt, hopps, indR0 = read_hr(file)
    wannier_orbital_order, sk_orbital_order, iatom_nors = wan_orbital_orders(struct, wannier_proj_orbital, orb_wan)
    hopping_bonds = transfrom_Hwan(hopps, Rlatt, indR0, struct, wannier_orbital_order, sk_orbital_order, iatom_nors)

    return hopping_bonds


def read_hr(file='wannier90_hr.dat'):
    """ Read wannier90_hr.dat. The wannier TB parameters files are generated by wannier90.x.
    
    Parameters:
    ----------
        file: filename of wannier90_hr.dat
    
    Returns:
    --------
        Rlatt: lattice vectors of Wigner-Seitz grid points, shape (nrpts,3)
        hopps: hopping matrices, shape (nrpts,num_wann,num_wann)
        indR0: index of R=0, where the hopps[indR0] corresponding to the onsite Hamiltonian block.
    
    """

    f=open(file,'r')
    data=f.readlines()
    #read hopping matrix
    num_wann = int(data[1])
    nrpts = int(data[2])
    r_hop= np.zeros([num_wann,num_wann], dtype=complex)
    #hop=[]
    #skip n lines of degeneracy of each Wigner-Seitz grid point
    skiplines = int(np.ceil(nrpts / 15.0))
    istart = 3 + skiplines
    deg=[]
    for i in range(3,istart):
        deg.append(np.array([int(j) for j in data[i].split()]))
    deg=np.concatenate(deg,0)
    
    icount=0
    ii=-1
    Rlatt = []
    hopps = []
    for i in range(istart,len(data)):
        line=data[i].split()
        m = int(line[3]) - 1
        n = int(line[4]) - 1
        r_hop[m,n] = complex(round(float(line[5]),6),round(float(line[6]),6))
        icount+=1
        if(icount % (num_wann*num_wann) == 0):
            ii+=1
            R = np.array([float(x) for x in line[0:3]])
            Rlatt.append(R)
            hopps.append(r_hop)
            r_hop= np.zeros([num_wann,num_wann], dtype=complex)
    Rlatt=np.asarray(Rlatt,dtype=int)
    hopps=np.asarray(hopps)
    deg = np.reshape(deg,[nrpts,1,1])
    hopps=hopps/deg

    for i in range(nrpts):
        if (Rlatt[i]==0).all():
            indR0 = i
    
    return Rlatt, hopps, indR0


def wan_orbital_orders(struct:BaseStruct, wannier_proj_orbital:dict, orb_wan:dict=None):
    """ get the wannier orbital orders for the wannier orbitals in wannier90_hr.dat.
     by default is shoule be in the order of :
      atom-0-s, atom-0-pz,atom-0-px, atom-0-py, atom-1-s, atom-1-pz, ..., etc.
    
    Parameters:
    -----------
        structase: ase.Atoms
        wannier_proj_orbital: dict,the orbital defined in projection in wannierizaion process. 
            e.g.: {'N': ['s','p'], 'B': ['s']} or {'N': 'p', 'B': 's'}
        orb_wan: dict, the orbital order in wannier90_hr.dat.
            e.g.: {'s': ['s'], 'p': ['pz','px','py'], 'd': ['dz2','dxz','dyz','dx2-y2','dxy']}

    Returns:
    --------
        wannier_orbital_order: list, the orbital order in wannier90_hr.dat.
        sk_orbital_order: list, the orbital order in dptb.
        iatom_nors: list, the number of orbitals on every atom.

    """

    proj_atom_anglr_m = struct.proj_atom_anglr_m
    if orb_wan is None:
        orb_wan = Orbital_Order_Wan_Default
    orb_sk = Orbital_Order_SK
    
    # take the projected_struct as from the structure class, which should be Atoms object.
    projected_struct = struct.projected_struct
    assert isinstance(projected_struct,Atoms), 'projected_struct should be ase.Atoms'
    # check the consistency of wannier_proj_orbital and proj_atom_anglr_m.
    assert set(wannier_proj_orbital.keys()) == set(proj_atom_anglr_m.keys())
    for ii in proj_atom_anglr_m:
        assert len(wannier_proj_orbital[ii]) == len(proj_atom_anglr_m[ii]), 'proj_atom_anglr_m and wannier_proj_orbital are not consistent'
        for iorb in proj_atom_anglr_m[ii]:
            ishell_symbol = ''.join(re.findall(r'[A-Za-z]',iorb))
            assert ishell_symbol in wannier_proj_orbital[ii], 'proj_atom_anglr_m and wannier_proj_orbital are not consistent'
    
    # ------------------------------
    # get the wannier orbitals in order.
    # ------------------------------
    # wannier_orbital_order: the wannier orbitals in the order of wannier90_hr.dat
    #                   e.g.: ['0-s','0-pz','0-px', ... ]
    # sk_orbital_order: the orbitals in the order of dptb.
    # iatom_nors: the total number of orbitals on every atom.
    # ------------------------------

    iatom_nors = []   # number of atoms on every atoms:
    wannier_orbital_order  = []
    sk_orbital_order = []

    projected_struct_symbols = projected_struct.get_chemical_symbols() # list of atom symbols in the projected_struct
    for ia in range(len(projected_struct_symbols)):
        iatom_symbols  = projected_struct_symbols[ia]   # atom symbol of the ia-th atom in the projected_struct
        ii_num_orbs = 0
        if isinstance (wannier_proj_orbital[iatom_symbols],list): 
            iorblist = wannier_proj_orbital[iatom_symbols]
        elif isinstance (wannier_proj_orbital[iatom_symbols],str):
            iorblist = [wannier_proj_orbital[iatom_symbols]]
        else:
            raise ValueError('wannier_proj_orbital should be a list or a string')

        for iorb in iorblist:
            ii_num_orbs += 2 * anglrMId[iorb] + 1
            for ii_orb in orb_wan[iorb]:
                wannier_orbital_order.append(f'{ia}-{ii_orb}')
 
        iatom_nors.append(ii_num_orbs)

        if isinstance (proj_atom_anglr_m[iatom_symbols],list):
            iorblist = proj_atom_anglr_m[iatom_symbols]
        elif isinstance (proj_atom_anglr_m[iatom_symbols],str):
            iorblist = [proj_atom_anglr_m[iatom_symbols]]
        else:
            raise ValueError('proj_atom_anglr_m should be a list or a string')
        
        for iorb in iorblist:
            ishell_symbol = ''.join(re.findall(r'[A-Za-z]',iorb))
            for ii_orb in orb_sk[ishell_symbol]:
                sk_orbital_order.append(f'{ia}-{ii_orb}')

    assert len(wannier_orbital_order) == len(wannier_orbital_order), 'wannier_orb_in and sk_orb_in are not consistent'
    assert set(wannier_orbital_order) == set(wannier_orbital_order), 'wannier_orb_in and sk_orb_in are not consistent'

    iatom_nors=np.array(iatom_nors,dtype=int)

    return wannier_orbital_order, sk_orbital_order, iatom_nors

def get_onsite_shift(hopps_r00, struct, wannier_orbital_order, unit='eV'):
    '''The function `get_onsite_shift` calculates the onsite shift of a given orbital in a crystal
    structure based on the hopping matrix elements in wannier and a database of onsite energies.
    
    Parameters
    ----------
    hopps_r00
        The variable `hopps_r00` represents the onsite Hamiltonian matrix elements. It is a square matrix
    where each element represents the interaction energy between two orbitals on the same atom.
    
    struct
        The `struct` parameter is an object that represents the structure of the system. It likely contains
    information about the positions of atoms in the system and other relevant properties.
    
    wannier_orbital_order
        The `wannier_orbital_order` parameter is a list that specifies the order of the Wannier orbitals.
    Each element in the list represents a Wannier orbital and is in the format
    "atom_index-orbital_symbol". For example, if there are 3 atoms and
    
    unit, optional
        The `unit` parameter specifies the unit in which the onsite shift will be calculated. It can take
    one of three values: 'eV', 'Ry', or 'Hartree'.
    
    Returns
    -------
        the value of the onsite shift, which is calculated based on the input parameters.
    
    '''
    
    projected_struct = struct.projected_struct
    projected_struct_symbols = projected_struct.get_chemical_symbols() # list of atom symbols in the projected_struct

    onsite_diag_elements = dict(zip(wannier_orbital_order, np.diag(hopps_r00).real))
    min_key = min(onsite_diag_elements, key=onsite_diag_elements.get)
    atom_ind = int(min_key.split('-')[0])
    orb_symbol = min_key.split('-')[1][0]
    atom_symbol = projected_struct_symbols[atom_ind]

    proj_atom_anglr_m = struct.proj_atom_anglr_m

    if unit == 'eV':
        factor = 13.605662285137 * 2 # Hartree to eV
    elif unit == 'Ry':
        factor = 2.0  # Hartree to Ry
    elif unit == 'Hartree':
        factor = 1.0 
    else:
        raise ValueError('unit must be eV, Ry or Hartree')

    onsite_e_db={}
    for i in proj_atom_anglr_m:
        onsite_e_db[i]={}
        for iorb in proj_atom_anglr_m[i]:
            ishell_symbol = ''.join(re.findall(r'[A-Za-z]',iorb))
            onsite_e_db[i][ishell_symbol] = onsite_energy_database[i][iorb] * factor
    
    database_onsite_e_min = onsite_e_db[atom_symbol][orb_symbol]

    onsite_shift = onsite_diag_elements[min_key] - database_onsite_e_min

    return onsite_shift

def transfrom_Hwan(hopps, Rlatt, indR0, struct, wannier_orbital_order, sk_orbital_order, iatom_nors):
    """ transform the hopping matrices from the order of wannier90_hr.dat to the order of dptb.
    
    Parameters:
    -----------
        hopps: hopping matrices, shape (nrpts,num_wann,num_wann)
        Rlatt: lattice vectors of Wigner-Seitz grid points, shape (nrpts,3)
        wannier_orbital_order: list, the orbital order in wannier90_hr.dat.
        sk_orbital_order: list, the orbital order in dptb.
        iatom_nors: list, the number of orbitals on every atom.

    Returns:
    --------
        hopping_bonds: dict, the hopping matrices in the order of dptb.
            e.g.: hopping_bonds = {'0_0_0_0_0': H_0,0 block at R=000, '0_1_0_0_1': H_0,1 block at R=001, ... }
    """
    
    norb = len(sk_orbital_order)
    Mateye = np.eye(norb,dtype=int)
    mtrans = np.zeros([norb,norb],dtype=int)
    for i in range(norb):
        iorb = sk_orbital_order[i]
        assert iorb in wannier_orbital_order
        ind = wannier_orbital_order.index(iorb)
        mtrans[i] +=  Mateye[ind] 

    onsite_shift = get_onsite_shift(hopps[indR0], struct, wannier_orbital_order, unit='eV')

    hopps_skorder = mtrans @ hopps @ mtrans.T
    hopps_skorder[indR0] = hopps_skorder[indR0] - onsite_shift * np.eye(norb)

    hopping_bonds = {}
    for ir in range(len(Rlatt)):
        iR = Rlatt[ir]
        for ia in range(len(iatom_nors)):
            ist, ied = (np.sum(iatom_nors[:ia]),np.sum(iatom_nors[:ia+1]))
            for ja in range(len(iatom_nors)):
                jst, jed = (np.sum(iatom_nors[:ja]),np.sum(iatom_nors[:ja+1]))
                hopping_bonds[f'{ia}_{ja}_{iR[0]}_{iR[1]}_{iR[2]}'] = hopps_skorder[ir,ist:ied,jst:jed].real
    
    return hopping_bonds