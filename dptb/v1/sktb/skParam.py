import os
import sys
import numpy as  np
from scipy.interpolate import interp1d
from dptb.utils.tools import get_uniq_symbol, format_readline
from dptb.utils.constants import NumHvals,MaxShells


def sk_init(proj_atom_anglr_m, sk_file_path, **kwargs):
    '''It takes in a list of atom types, a dictionary of angular momentum for each atom type, and a path to
    the Slater-Koster files, and returns a list of Slater-Koster files, a list of Slater-Koster file
    types, and a dictionary of atom orbital information
    
    Parameters
    ----------
    proj_atom_anglr_m
        a dictionary of the angular momentum of each atom type. e.g. {'H': ['s'], 'O': ['s','p']}
    sk_file_path
        the path to the directory where the Slater-Koster files are stored.
    
    Returns
    -------
        skfiles: a list of lists of strings. Each string is a path to a file.
        skfile_types: a list of lists of atom type for the sk files. 
            eg: [[...,['C','C'],['C','H'],...],
                 [...,['H','C'],['H','H'],...]]
        atom_orb_infor: a dictionary of atom and orbital information.
            atom_orb_infor ={
            "atom_types":proj_atom_type,
            "anglr_momentum_values":anglr_momentum_values,
            "num_orbs":num_orbs
            }
        proj_atom_type: a list of atom types. no duplicates.
        anglr_momentum_values: a list of angular momentum values, for each atom type.
        num_orbs: a list of the number of orbitals for each atom type.
    '''
    skparas={
        "separator":"-",
        "suffix":".skf"
    }
    skparas.update(kwargs)
    skfiles = {}

    proj_atom_type = get_uniq_symbol(list(proj_atom_anglr_m.keys()))
    
    for itype in proj_atom_type:
        for jtype in proj_atom_type:
            filename = sk_file_path + '/' + itype + skparas["separator"] + jtype + skparas["suffix"]
            if not os.path.exists(filename):
                print('Didn\'t find the skfile: ' + filename)
                sys.exit()
            skfiles[itype+ '-' +jtype] = filename

    # proj_type_norbs = {}
    # proj_type_momentum_ids={}
    # for itype in proj_atom_type:
    #     anglrm_i = []
    #     norbs_i = []
    #     for ishell  in proj_atom_anglr_m[itype]:  #[s','p', 'd', ...]
    #         ishell_value = anglrMId[ishell]
    #         norbs_i.append(2 * ishell_value + 1)
    #         anglrm_i.append(ishell_value)
    #     proj_type_norbs[itype] = (norbs_i)
    #     proj_type_momentum_ids[itype] = anglrm_i

    return skfiles


def read_skfiles(skfiles):
    '''It reads the Slater-Koster files names and returns the grid distance, number of grids, 
    and the Slater-Koster integrals
    
    Parameters
    ----------
    skfiles
        a list of skfiles.

    Note: the Slater-Koster files name now consider that all the atoms are interacted with each other.
    Therefore, the number of Slater-Koster files is num_atom_types *2.
    
    Returns
    -------
    grid_distance
        list, intervals between the grid points from different sk files.
    num_grids, 
        list, total number of grids from different sk files.
    HSintgrl
        list, the Slater-Koster integrals from different sk files.

    '''
    assert isinstance(skfiles, dict)
    skfiletypes = list(skfiles.keys())
    num_sk_files = len(skfiletypes)
    grid_distance = {}
    num_grids = {}
    HSintgrl = {}

    SiteE = {}
    HubdU = {}
    Occu = {}

    for isktype in skfiletypes:
        filename = skfiles[isktype]
        print('# Reading SlaterKoster File......')
        print('# ' + filename)
        fr = open(filename)
        data = fr.readlines()
        fr.close()
        # Line 1
        datline = format_readline(data[0])
        gridDist, ngrid = float(datline[0]), int(datline[1])
        ngrid = ngrid - 1
        grid_distance[isktype] = (gridDist)
        num_grids[isktype] = (ngrid)

        HSvals = np.zeros([ngrid, NumHvals * 2])
        atomtypes = isktype.split(sep='-')
        if atomtypes[0]==atomtypes[1]:
            print('# This file is a Homo-nuclear case!')
            # Line 2 for Homo-nuclear case
            datline = format_readline(data[1])
            # Ed Ep Es, spe, Ud Up Us, Od Op Os.
            # order from d p s -> s p d.
            SiteE[atomtypes[0]] = np.array([float(datline[2 - ish]) for ish in range(MaxShells)])
            HubdU[atomtypes[0]] = np.array([float(datline[6 - ish]) for ish in range(MaxShells)])
            Occu[atomtypes[0]]  = np.array([float(datline[9 - ish]) for ish in range(MaxShells)])

            for il in range(3, 3 + ngrid):
                datline = format_readline(data[il])
                HSvals[il - 3] = np.array([float(val) for val in datline[0:2 * NumHvals]])
        else:
            print('# This is for Hetero-nuclear case!')
            for il in range(2, 2 + ngrid):
                datline = format_readline(data[il])
                HSvals[il - 2] = np.array([float(val) for val in datline[0:2 * NumHvals]])
        HSintgrl[isktype] = HSvals

    return grid_distance, num_grids, HSintgrl, SiteE, HubdU, Occu


def interp_sk_gridvalues(skfile_types, grid_distance, num_grids, HSintgrl):
    '''It reads the Slater-Koster files names and skfile_types 
    to generate the  interpolates of the Slater-Koster integrals. 
    
    Parameters
    ----------
    skfile_types
        a list of skfiles types like: 'C-C', 'C-H', 'H-C', ...
    grid_distance
       dict, the grid intervals for each skfile type. e.g. {'C-C':0.1, 'C-H':0.1, ...}
    num_grids
        dict, the number of grids for each skfile type. e.g. {'C-C':100, 'C-H':100, ...}
    HSintgrl
        dict, the H intergrals and ovelaps for each skfile type. e.g. {'C-C': H S values., 'C-H': ,H S values ...}
    
    Returns
    -------
    max_min_bond_length
        dict, the max and min bond length for each skfile type. e.g. {'C-C': [0.1, 10], 'C-H': [0.1, 10], ...}
    interp_skfunc
        dict, the interpolation functions for each skfile type. e.g. {'C-C': interpolation function, 'C-H': interpolation function ...}
    
    '''
    MaxDistail = 1.0
    eps = 1.0E-6
    interp_skfunc = {}
    max_min_bond_length = {}

    for isktype in skfile_types:
        xlist = np.arange(1,num_grids[isktype]+1)*grid_distance[isktype]
        xlist = np.append(xlist,[xlist[-1]+MaxDistail],axis=0)
        max_tmp = num_grids[isktype] * grid_distance[isktype] + MaxDistail - eps 
        min_tmp = grid_distance[isktype]

        target = HSintgrl[isktype]
        target = np.append(target,np.zeros([1,2* NumHvals]),axis=0)
        intpfunc = interp1d(xlist, target , axis=0)
        interp_skfunc[isktype] = intpfunc
        max_min_bond_length[isktype] = [max_tmp,min_tmp]

    return max_min_bond_length, interp_skfunc




