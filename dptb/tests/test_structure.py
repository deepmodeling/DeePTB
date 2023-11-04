import pytest
import ase
from ase.build import graphene_nanoribbon
import ase.neighborlist
import torch
import os
import sys
import logging
import numpy as np
from dptb.structure.structure import BaseStruct
from dptb.utils.tools import get_uniq_symbol
from dptb.utils.constants import anglrMId, atomic_num_dict, atomic_num_dict_r

@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
        return str(request.config.rootdir)

class param:
    def __init__(self, AtomType, ProjAtomType, ProjAnglrM, CutOff, ValElec):
        self.AtomType = AtomType
        self.ProjAtomType = ProjAtomType
        self.ProjAnglrM = ProjAnglrM
        self.CutOff = CutOff
        self.ValElec = ValElec

def generate_system():
    atoms = graphene_nanoribbon(1.5, 1, type='armchair', saturated=True)
    basestruct = BaseStruct(atom=atoms, format='ase', cutoff=1.5, proj_atom_anglr_m={'C':['s','p'], 'H':['s']},proj_atom_neles={'C':4, 'H':1})
    return basestruct


def test_BaseStruct(root_directory):
    filename = root_directory + '/dptb/tests/data/hBN/hBN.vasp'
    proj_atom_anglr_m = {"N":["s","p"],"B":["s","p"]}
    proj_atom_neles = {"N":5,"B":3}
    CutOff = 4
    onsitelist = torch.tensor([[7, 0, 7, 0, 0, 0, 0],
       [5, 1, 5, 1, 0, 0, 0]])
    bondlist = torch.tensor([[ 7,  0,  5,  1, -2,  0,  0],
       [ 7,  0,  7,  0, -1,  0,  0],
       [ 7,  0,  5,  1, -1,  0,  0],
       [ 7,  0,  5,  1,  1,  0,  0],
       [ 7,  0,  5,  1, -1,  1,  0],
       [ 7,  0,  7,  0,  0,  1,  0],
       [ 7,  0,  5,  1,  0,  1,  0],
       [ 7,  0,  7,  0,  1,  1,  0],
       [ 7,  0,  5,  1,  1,  1,  0],
       [ 7,  0,  5,  1,  0,  2,  0],
       [ 7,  0,  5,  1,  1,  2,  0],
       [ 7,  0,  5,  1,  0,  0,  0],
       [ 7,  0,  5,  1, -1, -1,  0],
       [ 7,  0,  5,  1, -2, -1,  0],
       [ 7,  0,  5,  1,  0, -1,  0],
       [ 5,  1,  5,  1,  1,  1,  0],
       [ 5,  1,  5,  1,  0, -1,  0],
       [ 5,  1,  5,  1, -1,  0,  0]])
    
    bond_dist_vec = torch.tensor([[ 3.8249233e+00, -9.8198050e-01, -1.8898225e-01,  0.0000000e+00],
       [ 2.5039999e+00, -1.0000000e+00,  0.0000000e+00,  0.0000000e+00],
       [ 1.4456851e+00, -8.6602539e-01, -5.0000000e-01,  0.0000000e+00],
       [ 3.8249230e+00,  9.8198050e-01, -1.8898226e-01,  0.0000000e+00],
       [ 2.8913701e+00, -8.6602545e-01,  4.9999997e-01,  0.0000000e+00],
       [ 2.5039999e+00, -5.0000000e-01,  8.6602539e-01,  0.0000000e+00],
       [ 1.4456849e+00, -5.0252535e-08,  1.0000000e+00,  0.0000000e+00],
       [ 2.5039999e+00,  5.0000000e-01,  8.6602539e-01,  0.0000000e+00],
       [ 2.8913701e+00,  8.6602539e-01,  5.0000000e-01,  0.0000000e+00],
       [ 3.8249230e+00, -3.2732686e-01,  9.4491118e-01,  0.0000000e+00],
       [ 3.8249230e+00,  3.2732683e-01,  9.4491118e-01,  0.0000000e+00],
       [ 1.4456850e+00,  8.6602539e-01, -5.0000006e-01,  0.0000000e+00],
       [ 2.8913701e+00, -2.5091678e-08, -1.0000000e+00,  0.0000000e+00],
       [ 3.8249233e+00, -6.5465367e-01, -7.5592893e-01,  0.0000000e+00],
       [ 3.8249230e+00,  6.5465367e-01, -7.5592899e-01,  0.0000000e+00],
       [ 2.5039999e+00,  5.0000000e-01,  8.6602539e-01,  0.0000000e+00],
       [ 2.5039999e+00,  5.0000000e-01, -8.6602539e-01,  0.0000000e+00],
       [ 2.5039999e+00, -1.0000000e+00,  0.0000000e+00,  0.0000000e+00]],
      dtype=torch.float32)
    struct = BaseStruct(atom=filename,format='vasp',
        cutoff=CutOff,proj_atom_anglr_m=proj_atom_anglr_m,proj_atom_neles=proj_atom_neles, onsitemode='uniform')
    bonds, bonds_onsite = struct.get_bond()
    assert struct.proj_atom_anglr_m == proj_atom_anglr_m
    assert struct.atomtype == ['N','B']
    assert struct.proj_atomtype == ['N','B']
    assert struct.proj_atomtype_norbs == {'N':4,'B':4}
    assert (struct.proj_atom_symbols == ['N','B'])
    assert (struct.atom_symbols == ['N','B']).all()
    assert (struct.__bonds__[:,0:7].int() == bondlist).all()
    assert (np.abs(struct.__bonds__[:,7:11].float()-bond_dist_vec) < 1e-6).all()
    assert (struct.__bonds_onsite__ == onsitelist).all()

    bond_index_map = {'N-N': {'s-s': [0], 's-p': [1], 'p-s': [1], 'p-p': [2, 3]},
                      'N-B': {'s-s': [0], 's-p': [1], 'p-s': [2], 'p-p': [3, 4]},
                      'B-N': {'s-s': [0], 's-p': [2], 'p-s': [1], 'p-p': [3, 4]},
                      'B-B': {'s-s': [0], 's-p': [1], 'p-s': [1], 'p-p': [2, 3]}}
    bond_num_hops = {'N-N': 4, 'N-B': 5, 'B-N': 5, 'B-B': 4}
    onsite_index_map = {'N': {'s': [0], 'p': [1]}, 'B': {'s': [0], 'p': [1]}}
    onsite_num = {'N': 2, 'B': 2}

    assert struct.bond_index_map == bond_index_map
    assert struct.bond_num_hops == bond_num_hops
    assert struct.onsite_index_map == onsite_index_map
    assert struct.onsite_num == onsite_num

    atom_symb1_inbond  = [atomic_num_dict_r[int(i)] for i in bonds[:,0]]
    proj_atom_sym1_inbond =[struct.proj_atom_symbols[int(i)] for i in bonds[:,1]]

    assert (atom_symb1_inbond == proj_atom_sym1_inbond)

    atom_symb2_inbond  = [atomic_num_dict_r[int(i)] for i in bonds[:,2]]
    proj_atom_sym2_inbond =[struct.proj_atom_symbols[int(i)] for i in bonds[:,3]]
    assert (atom_symb2_inbond == proj_atom_sym2_inbond)

    struct.update_struct(atom=filename,format='vasp',onsitemode='split')
    assert struct.onsite_index_map == {'N': {'s': [0], 'p': [1, 2, 3]}, 'B': {'s': [0], 'p': [1, 2, 3]}}
    assert struct.onsite_num == {'N': 4, 'B': 4}

def test_Struct_IndMap_case1(root_directory):
    filename = root_directory + '/dptb/tests/data/hBN/hBN.vasp'
    proj_atom_anglr_m = {"N":["s","p"],"B":["s","p"]}
    proj_atom_neles = {"N":5,"B":3}
    CutOff = 4
    struct = BaseStruct(atom=filename,format='vasp',
        cutoff=CutOff,proj_atom_anglr_m=proj_atom_anglr_m,proj_atom_neles=proj_atom_neles, onsitemode='uniform')
    assert struct.proj_atom_anglr_m == proj_atom_anglr_m
    assert struct.atomtype == ['N','B']
    assert struct.proj_atomtype == ['N','B']
    assert struct.proj_atomtype_norbs == {'N':4,'B':4}
    assert (struct.proj_atom_symbols == ['N','B'])
    assert (struct.atom_symbols == ['N','B']).all()

    bond_index_map = {'N-N': {'s-s': [0], 's-p': [1], 'p-s': [1], 'p-p': [2, 3]},
                      'N-B': {'s-s': [0], 's-p': [1], 'p-s': [2], 'p-p': [3, 4]},
                      'B-N': {'s-s': [0], 's-p': [2], 'p-s': [1], 'p-p': [3, 4]},
                      'B-B': {'s-s': [0], 's-p': [1], 'p-s': [1], 'p-p': [2, 3]}}
    bond_num_hops = {'N-N': 4, 'N-B': 5, 'B-N': 5, 'B-B': 4}
    onsite_index_map = {'N': {'s': [0], 'p': [1]}, 'B': {'s': [0], 'p': [1]}}
    onsite_num = {'N': 2, 'B': 2}

    assert struct.bond_index_map == bond_index_map
    assert struct.bond_num_hops == bond_num_hops
    assert struct.onsite_index_map == onsite_index_map
    assert struct.onsite_num == onsite_num

    struct.update_struct(atom=filename,format='vasp',onsitemode='split')
    assert struct.onsite_index_map == {'N': {'s': [0], 'p': [1, 2, 3]}, 'B': {'s': [0], 'p': [1, 2, 3]}}
    assert struct.onsite_num == {'N': 4, 'B': 4}


def test_Struct_IndMap_case2(root_directory):
    filename = root_directory + '/dptb/tests/data/hBN/hBN.vasp'
    proj_atom_anglr_m = {"N":["2s","2p"],"B":["2s","2p"]}
    proj_atom_neles = {"N":5,"B":3}
    CutOff = 4
    struct = BaseStruct(atom=filename,format='vasp',
        cutoff=CutOff,proj_atom_anglr_m=proj_atom_anglr_m,proj_atom_neles=proj_atom_neles, onsitemode='uniform')
    assert struct.proj_atom_anglr_m == proj_atom_anglr_m
    assert struct.atomtype == ['N','B']
    assert struct.proj_atomtype == ['N','B']
    assert struct.proj_atomtype_norbs == {'N':4,'B':4}
    assert (struct.proj_atom_symbols == ['N','B'])
    assert (struct.atom_symbols == ['N','B']).all()


    bond_index_map = {'N-N': {'2s-2s': [0], '2s-2p': [1], '2p-2s': [1], '2p-2p': [2, 3]},
                      'N-B': {'2s-2s': [0], '2s-2p': [1], '2p-2s': [2], '2p-2p': [3, 4]},
                      'B-N': {'2s-2s': [0], '2s-2p': [2], '2p-2s': [1], '2p-2p': [3, 4]},
                      'B-B': {'2s-2s': [0], '2s-2p': [1], '2p-2s': [1], '2p-2p': [2, 3]}}

    bond_num_hops = {'N-N': 4, 'N-B': 5, 'B-N': 5, 'B-B': 4}
    onsite_index_map = {'N': {'2s': [0], '2p': [1]}, 'B': {'2s': [0], '2p': [1]}}
    onsite_num = {'N': 2, 'B': 2}

    assert struct.bond_index_map == bond_index_map
    assert struct.bond_num_hops == bond_num_hops
    assert struct.onsite_index_map == onsite_index_map
    assert struct.onsite_num == onsite_num

    struct.update_struct(atom=filename,format='vasp',onsitemode='split')
    assert struct.onsite_index_map == {'N': {'2s': [0], '2p': [1, 2, 3]}, 'B': {'2s': [0], '2p': [1, 2, 3]}}
    assert struct.onsite_num == {'N': 4, 'B': 4}

def test_Struct_IndMap_case3(root_directory):
    filename = root_directory + '/dptb/tests/data/hBN/hBN.vasp'
    proj_atom_anglr_m = {"N":["2s","2p",'s*'],"B":["2s","2p"]}
    proj_atom_neles = {"N":5,"B":3}
    CutOff = 4
    struct = BaseStruct(atom=filename,format='vasp',
        cutoff=CutOff,proj_atom_anglr_m=proj_atom_anglr_m,proj_atom_neles=proj_atom_neles, onsitemode='uniform')
    assert struct.proj_atom_anglr_m == proj_atom_anglr_m
    assert struct.atomtype == ['N','B']
    assert struct.proj_atomtype == ['N','B']
    assert struct.proj_atomtype_norbs == {'N':5,'B':4}
    assert (struct.proj_atom_symbols == ['N','B'])
    assert (struct.atom_symbols == ['N','B']).all()


    bond_index_map = {'N-N': {'2s-2s': [0], '2s-2p': [1], '2s-s*': [2], '2p-2s': [1],    '2p-2p': [3, 4], '2p-s*': [5], 's*-2s': [2], 's*-2p': [5], 's*-s*': [6]},
                      'N-B': {'2s-2s': [0], '2s-2p': [1], '2p-2s': [2], '2p-2p': [3, 4], 's*-2s': [5],    's*-2p': [6]},
                      'B-N': {'2s-2s': [0], '2s-2p': [2], '2s-s*': [5], '2p-2s': [1],    '2p-2p': [3, 4], '2p-s*': [6]},
                      'B-B': {'2s-2s': [0], '2s-2p': [1], '2p-2s': [1], '2p-2p': [2, 3]}}

    bond_num_hops =  {'N-N': 7, 'N-B': 7, 'B-N': 7, 'B-B': 4}
    onsite_index_map = {'N': {'2s': [0], '2p': [1], 's*': [2]}, 'B': {'2s': [0], '2p': [1]}}
    onsite_num = {'N': 3, 'B': 2}

    assert struct.bond_index_map == bond_index_map
    assert struct.bond_num_hops == bond_num_hops
    assert struct.onsite_index_map == onsite_index_map
    assert struct.onsite_num == onsite_num

    struct.update_struct(atom=filename,format='vasp',onsitemode='split')
    assert struct.onsite_index_map == {'N': {'2s': [0], '2p': [1, 2, 3],'s*':[4]}, 'B': {'2s': [0], '2p': [1, 2, 3]}}
    assert struct.onsite_num == {'N': 5, 'B': 4}