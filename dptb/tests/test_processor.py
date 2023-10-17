import numpy as np
import torch
from dptb.dataprocess.processor import Processor
from dptb.structure.structure import BaseStruct
from dptb.utils.tools import get_uniq_symbol
import pytest
from ase.build import graphene_nanoribbon
import logging

batch_env_true = torch.tensor([[ 0.0000000000e+00,  7.0000000000e+00,  0.0000000000e+00,
          5.0000000000e+00,  1.0000000000e+00, -1.0000000000e+00,
          0.0000000000e+00,  0.0000000000e+00,  6.9171357155e-01,
         -8.6602538824e-01, -5.0000000000e-01,  0.0000000000e+00],
        [ 0.0000000000e+00,  7.0000000000e+00,  0.0000000000e+00,
          5.0000000000e+00,  1.0000000000e+00,  0.0000000000e+00,
          1.0000000000e+00,  0.0000000000e+00,  6.9171363115e-01,
         -5.0252534578e-08,  1.0000000000e+00,  0.0000000000e+00],
        [ 0.0000000000e+00,  7.0000000000e+00,  0.0000000000e+00,
          5.0000000000e+00,  1.0000000000e+00,  0.0000000000e+00,
          0.0000000000e+00,  0.0000000000e+00,  6.9171363115e-01,
          8.6602538824e-01, -5.0000005960e-01,  0.0000000000e+00],
        [ 0.0000000000e+00,  5.0000000000e+00,  1.0000000000e+00,
          7.0000000000e+00,  0.0000000000e+00,  0.0000000000e+00,
         -1.0000000000e+00,  0.0000000000e+00,  6.9171363115e-01,
          5.0252534578e-08, -1.0000000000e+00,  0.0000000000e+00],
        [ 0.0000000000e+00,  5.0000000000e+00,  1.0000000000e+00,
          7.0000000000e+00,  0.0000000000e+00,  0.0000000000e+00,
          0.0000000000e+00,  0.0000000000e+00,  6.9171363115e-01,
         -8.6602538824e-01,  5.0000005960e-01,  0.0000000000e+00],
        [ 0.0000000000e+00,  5.0000000000e+00,  1.0000000000e+00,
          7.0000000000e+00,  0.0000000000e+00,  1.0000000000e+00,
          0.0000000000e+00,  0.0000000000e+00,  6.9171357155e-01,
          8.6602538824e-01,  5.0000000000e-01,  0.0000000000e+00]])

batch_env_itype_jtype_true = {'N-B': torch.tensor([[ 0.0000000000e+00,  7.0000000000e+00,  0.0000000000e+00,
           5.0000000000e+00,  1.0000000000e+00, -1.0000000000e+00,
           0.0000000000e+00,  0.0000000000e+00,  6.9171357155e-01,
          -8.6602538824e-01, -5.0000000000e-01,  0.0000000000e+00],
         [ 0.0000000000e+00,  7.0000000000e+00,  0.0000000000e+00,
           5.0000000000e+00,  1.0000000000e+00,  0.0000000000e+00,
           1.0000000000e+00,  0.0000000000e+00,  6.9171363115e-01,
          -5.0252534578e-08,  1.0000000000e+00,  0.0000000000e+00],
         [ 0.0000000000e+00,  7.0000000000e+00,  0.0000000000e+00,
           5.0000000000e+00,  1.0000000000e+00,  0.0000000000e+00,
           0.0000000000e+00,  0.0000000000e+00,  6.9171363115e-01,
           8.6602538824e-01, -5.0000005960e-01,  0.0000000000e+00]]),
 'B-N': torch.tensor([[ 0.0000000000e+00,  5.0000000000e+00,  1.0000000000e+00,
           7.0000000000e+00,  0.0000000000e+00,  0.0000000000e+00,
          -1.0000000000e+00,  0.0000000000e+00,  6.9171363115e-01,
           5.0252534578e-08, -1.0000000000e+00,  0.0000000000e+00],
         [ 0.0000000000e+00,  5.0000000000e+00,  1.0000000000e+00,
           7.0000000000e+00,  0.0000000000e+00,  0.0000000000e+00,
           0.0000000000e+00,  0.0000000000e+00,  6.9171363115e-01,
          -8.6602538824e-01,  5.0000005960e-01,  0.0000000000e+00],
         [ 0.0000000000e+00,  5.0000000000e+00,  1.0000000000e+00,
           7.0000000000e+00,  0.0000000000e+00,  1.0000000000e+00,
           0.0000000000e+00,  0.0000000000e+00,  6.9171357155e-01,
           8.6602538824e-01,  5.0000000000e-01,  0.0000000000e+00]])}


batch_bond_onsite_true = torch.tensor([[0., 7., 0., 7., 0., 0., 0., 0.],
                                       [0., 5., 1., 5., 1., 0., 0., 0.]])

batch_bond_true = torch.tensor([[ 0.0000000000e+00,  7.0000000000e+00,  0.0000000000e+00,
                                      5.0000000000e+00,  1.0000000000e+00, -1.0000000000e+00,
                                      0.0000000000e+00,  0.0000000000e+00,  1.4456851482e+00,
                                     -8.6602538824e-01, -5.0000000000e-01,  0.0000000000e+00],
                                    [ 0.0000000000e+00,  7.0000000000e+00,  0.0000000000e+00,
                                      5.0000000000e+00,  1.0000000000e+00,  0.0000000000e+00,
                                      1.0000000000e+00,  0.0000000000e+00,  1.4456849098e+00,
                                     -5.0252534578e-08,  1.0000000000e+00,  0.0000000000e+00],
                                    [ 0.0000000000e+00,  7.0000000000e+00,  0.0000000000e+00,
                                      5.0000000000e+00,  1.0000000000e+00,  0.0000000000e+00,
                                      0.0000000000e+00,  0.0000000000e+00,  1.4456850290e+00,
                                      8.6602538824e-01, -5.0000005960e-01,  0.0000000000e+00]])


batch_onsitenv_true = torch.tensor([[ 0.0000000000e+00,  7.0000000000e+00,  0.0000000000e+00,
          5.0000000000e+00,  1.0000000000e+00, -1.0000000000e+00,
          0.0000000000e+00,  0.0000000000e+00,  1.4456851482e+00,
         -8.6602538824e-01, -5.0000000000e-01,  0.0000000000e+00],
        [ 0.0000000000e+00,  7.0000000000e+00,  0.0000000000e+00,
          5.0000000000e+00,  1.0000000000e+00,  0.0000000000e+00,
          1.0000000000e+00,  0.0000000000e+00,  1.4456849098e+00,
         -5.0252534578e-08,  1.0000000000e+00,  0.0000000000e+00],
        [ 0.0000000000e+00,  7.0000000000e+00,  0.0000000000e+00,
          5.0000000000e+00,  1.0000000000e+00,  0.0000000000e+00,
          0.0000000000e+00,  0.0000000000e+00,  1.4456850290e+00,
          8.6602538824e-01, -5.0000005960e-01,  0.0000000000e+00],
        [ 0.0000000000e+00,  5.0000000000e+00,  1.0000000000e+00,
          7.0000000000e+00,  0.0000000000e+00,  0.0000000000e+00,
         -1.0000000000e+00,  0.0000000000e+00,  1.4456849098e+00,
          5.0252534578e-08, -1.0000000000e+00,  0.0000000000e+00],
        [ 0.0000000000e+00,  5.0000000000e+00,  1.0000000000e+00,
          7.0000000000e+00,  0.0000000000e+00,  0.0000000000e+00,
          0.0000000000e+00,  0.0000000000e+00,  1.4456850290e+00,
         -8.6602538824e-01,  5.0000005960e-01,  0.0000000000e+00],
        [ 0.0000000000e+00,  5.0000000000e+00,  1.0000000000e+00,
          7.0000000000e+00,  0.0000000000e+00,  1.0000000000e+00,
          0.0000000000e+00,  0.0000000000e+00,  1.4456851482e+00,
          8.6602538824e-01,  5.0000000000e-01,  0.0000000000e+00]])


batch_onsitenv_itype_jtype_true = {'N-B': torch.tensor([[ 0.0000000000e+00,  7.0000000000e+00,  0.0000000000e+00,
           5.0000000000e+00,  1.0000000000e+00, -1.0000000000e+00,
           0.0000000000e+00,  0.0000000000e+00,  1.4456851482e+00,
          -8.6602538824e-01, -5.0000000000e-01,  0.0000000000e+00],
         [ 0.0000000000e+00,  7.0000000000e+00,  0.0000000000e+00,
           5.0000000000e+00,  1.0000000000e+00,  0.0000000000e+00,
           1.0000000000e+00,  0.0000000000e+00,  1.4456849098e+00,
          -5.0252534578e-08,  1.0000000000e+00,  0.0000000000e+00],
         [ 0.0000000000e+00,  7.0000000000e+00,  0.0000000000e+00,
           5.0000000000e+00,  1.0000000000e+00,  0.0000000000e+00,
           0.0000000000e+00,  0.0000000000e+00,  1.4456850290e+00,
           8.6602538824e-01, -5.0000005960e-01,  0.0000000000e+00]]),
    'B-N': torch.tensor([[ 0.0000000000e+00,  5.0000000000e+00,  1.0000000000e+00,
           7.0000000000e+00,  0.0000000000e+00,  0.0000000000e+00,
          -1.0000000000e+00,  0.0000000000e+00,  1.4456849098e+00,
           5.0252534578e-08, -1.0000000000e+00,  0.0000000000e+00],
         [ 0.0000000000e+00,  5.0000000000e+00,  1.0000000000e+00,
           7.0000000000e+00,  0.0000000000e+00,  0.0000000000e+00,
           0.0000000000e+00,  0.0000000000e+00,  1.4456850290e+00,
          -8.6602538824e-01,  5.0000005960e-01,  0.0000000000e+00],
         [ 0.0000000000e+00,  5.0000000000e+00,  1.0000000000e+00,
           7.0000000000e+00,  0.0000000000e+00,  1.0000000000e+00,
           0.0000000000e+00,  0.0000000000e+00,  1.4456851482e+00,
           8.6602538824e-01,  5.0000000000e-01,  0.0000000000e+00]])}

@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    return str(request.config.rootdir)

def test_getenv(root_directory):

    filename = root_directory
    filename += '/dptb/tests/data/hBN/hBN.vasp'

    proj_atom_anglr_m = {"N": ["s", "p"], "B": ["s", "p"]}
    proj_atom_neles = {"N": 5, "B": 3}
    CutOff = 2
    struct = BaseStruct(atom=filename, format='vasp',
                        cutoff=CutOff, proj_atom_anglr_m=proj_atom_anglr_m, proj_atom_neles=proj_atom_neles)

    struct_list = [struct]
    kpoints_list = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    eig_list = [np.zeros([2,10]), np.zeros([2,10])]
    processor = Processor(structure_list=struct_list, kpoint=kpoints_list, eigen_list=eig_list, batchsize=1, env_cutoff=2.0)
    batch_env = processor.get_env(cutoff=2.0,sorted=None)

    assert (batch_env - batch_env_true < 1e-8).all()

    batch_env = processor.get_env(cutoff=2.0,sorted='st')
    assert isinstance(batch_env, dict)
    assert (batch_env[0] - batch_env_true < 1e-8).all()
   
    processor.__struct_workspace__[0].if_env_ready = False
    batch_env_itype_jtype = processor.get_env(cutoff=2.0, sorted="itype-jtype")
    assert isinstance(batch_env_itype_jtype, dict)
    for ikey in batch_env_itype_jtype:
        assert ikey in batch_env_itype_jtype_true
        assert (batch_env_itype_jtype[ikey] - batch_env_itype_jtype_true[ikey] < 1e-8).all()

def test_atomtype(root_directory):
    filename = root_directory
    filename += '/dptb/tests/data/hBN/hBN.vasp'
    proj_atom_anglr_m = {"N": ["s", "p"], "B": ["s", "p"]}
    proj_atom_neles = {"N": 5, "B": 3}
    CutOff = 4
    atoms = graphene_nanoribbon(1.5, 1, type='armchair', saturated=True)
    basestruct = BaseStruct(atom=atoms, format='ase', cutoff=1.5, proj_atom_anglr_m={'C': ['s', 'p'], 'H': ['s']},proj_atom_neles={'C':4, 'H':1})

    struct = BaseStruct(atom=filename, format='vasp',
                        cutoff=CutOff, proj_atom_anglr_m=proj_atom_anglr_m, proj_atom_neles=proj_atom_neles)

    struct_list = [struct, basestruct]
    kpoints_list = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    eig_list = [np.zeros([2,10]), np.zeros([2,10])]
    processor = Processor(structure_list=struct_list, kpoint=kpoints_list, eigen_list=eig_list, batchsize=2, env_cutoff=5)
    assert get_uniq_symbol(processor.atomtype) == get_uniq_symbol(['N','B','C','H'])

def test_proj_atomtype(root_directory):
    filename = root_directory
    filename +=  '/dptb/tests/data/hBN/hBN.vasp'
    proj_atom_anglr_m = {"N": ["s", "p"], "B": ["s", "p"]}
    proj_atom_neles = {"N": 5, "B": 3}
    CutOff = 4
    atoms = graphene_nanoribbon(1.5, 1, type='armchair', saturated=True)
    basestruct = BaseStruct(atom=atoms, format='ase', cutoff=1.5, proj_atom_anglr_m={'C': ['s', 'p']},proj_atom_neles={'C':4})

    struct = BaseStruct(atom=filename, format='vasp',
                        cutoff=CutOff, proj_atom_anglr_m=proj_atom_anglr_m, proj_atom_neles=proj_atom_neles)

    struct_list = [struct, basestruct]
    kpoints_list = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    eig_list = [np.zeros([2,10]), np.zeros([2,10])]
    processor = Processor(structure_list=struct_list, kpoint=kpoints_list, eigen_list=eig_list, batchsize=2, env_cutoff=5)
    print(processor.proj_atomtype)
    assert get_uniq_symbol(processor.proj_atomtype) == get_uniq_symbol(['N', 'B', 'C'])

def test_getbond(root_directory):

    filename = root_directory
    filename += '/dptb/tests/data/hBN/hBN.vasp'

    proj_atom_anglr_m = {"N": ["s", "p"], "B": ["s", "p"]}
    proj_atom_neles = {"N": 5, "B": 3}
    CutOff = 2
    struct = BaseStruct(atom=filename, format='vasp',
                        cutoff=CutOff, proj_atom_anglr_m=proj_atom_anglr_m, proj_atom_neles=proj_atom_neles)

    struct_list = [struct]
    kpoints_list = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    eig_list = [np.zeros([2,10]), np.zeros([2,10])]
    processor = Processor(structure_list=struct_list, kpoint=kpoints_list, eigen_list=eig_list, batchsize=1, env_cutoff=5)

    batch_bond, batch_bond_onsite = processor.get_bond(sorted=None)

    assert torch.equal(batch_bond_onsite, batch_bond_onsite_true)
    assert  (batch_bond - batch_bond_true < 1e-8).all()


    batch_bond, batch_bond_onsite  = processor.get_bond(sorted='st')
    assert isinstance(batch_bond, dict)
    assert isinstance(batch_bond_onsite, dict)
    assert torch.equal(batch_bond_onsite[0], batch_bond_onsite_true)
    assert  (batch_bond[0] - batch_bond_true < 1e-8).all()

def test_iter(root_directory):
    filename = root_directory
    filename += '/dptb/tests/data/hBN/hBN.vasp'
    proj_atom_anglr_m = {"N": ["s", "p"], "B": ["s", "p"]}
    proj_atom_neles = {"N": 5, "B": 3}

    CutOff = 4
    atoms = graphene_nanoribbon(1.5, 1, type='armchair', saturated=True)
    basestruct = BaseStruct(atom=atoms, format='ase', cutoff=1.5, proj_atom_anglr_m={'C': ['s', 'p']},proj_atom_neles={'C':4})

    struct = BaseStruct(atom=filename, format='vasp',
                        cutoff=CutOff, proj_atom_anglr_m=proj_atom_anglr_m,proj_atom_neles=proj_atom_neles)

    struct_list = [struct, basestruct, struct, basestruct]
    kpoints_list = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    eig_list = np.array([np.zeros([2,10]), np.zeros([2,10]),np.zeros([2,10]),np.zeros([2,10])])
    processor = Processor(structure_list=struct_list, kpoint=kpoints_list, eigen_list=eig_list, 
                          wannier_list=[None for _ in range(len(struct_list))], batchsize=1, env_cutoff=5)
    processor.get_env()

    i = 0
    for data in processor:
        print(data[0].size())
        i += 1
        if i > 4:
            raise ValueError
    if i != 4:
        raise ValueError
    
def test_get_onsitenv(root_directory):

    filename = root_directory
    filename += '/dptb/tests/data/hBN/hBN.vasp'

    proj_atom_anglr_m = {"N": ["s", "p"], "B": ["s", "p"]}
    proj_atom_neles = {"N": 5, "B": 3}
    CutOff = 2
    struct = BaseStruct(atom=filename, format='vasp',
                        cutoff=CutOff, proj_atom_anglr_m=proj_atom_anglr_m, proj_atom_neles=proj_atom_neles)

    struct_list = [struct]
    kpoints_list = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    eig_list = [np.zeros([2,10]), np.zeros([2,10])]
    processor = Processor(structure_list=struct_list, kpoint=kpoints_list, eigen_list=eig_list, batchsize=1, env_cutoff=2.0, onsite_cutoff=2.0)
    batch_onsitenv = processor.get_onsitenv(sorted=None)

    assert (batch_onsitenv - batch_onsitenv_true < 1e-8).all()

    batch_onsitenv = processor.get_onsitenv(sorted='st')
    assert isinstance(batch_onsitenv, dict)
    assert (batch_onsitenv[0] - batch_onsitenv_true < 1e-8).all()
   
    processor.__struct_workspace__[0].if_onsitenv_ready = False
    batch_onsitenv_itype_jtype = processor.get_onsitenv(sorted="itype-jtype")
    assert isinstance(batch_onsitenv_itype_jtype, dict)
    for ikey in batch_onsitenv_itype_jtype:
        assert ikey in batch_onsitenv_itype_jtype_true
        assert (batch_onsitenv_itype_jtype[ikey] - batch_onsitenv_itype_jtype_true[ikey] < 1e-8).all()
  

def test_next(root_directory):

    filename = root_directory
    filename += '/dptb/tests/data/hBN/hBN.vasp'

    proj_atom_anglr_m = {"N": ["s", "p"], "B": ["s", "p"]}
    proj_atom_neles = {"N": 5, "B": 3}
    CutOff = 2
    struct = BaseStruct(atom=filename, format='vasp',
                        cutoff=CutOff, proj_atom_anglr_m=proj_atom_anglr_m, proj_atom_neles=proj_atom_neles)

    struct_list = [struct]
    kpoints_list = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    eig_list = np.zeros([1, 2,10])

    processor = Processor(structure_list=struct_list, batchsize=1, kpoint=kpoints_list, eigen_list=eig_list, wannier_list=[None for _ in range(len(struct_list))],
            env_cutoff=2.0,onsite_cutoff=2.0, sorted_onsite="st", sorted_bond="st", sorted_env="itype-jtype")
    
    for data in processor:
        assert len(data) == 8
        assert isinstance(data[0], dict)
        assert isinstance(data[1], dict)
        assert torch.equal(data[1][0], batch_bond_onsite_true)
        assert  (data[0][0] - batch_bond_true < 1e-8).all()

        assert isinstance(data[2], dict)
        for ikey in data[2]:
            assert ikey in batch_env_itype_jtype_true
            assert (data[2][ikey] - batch_env_itype_jtype_true[ikey] < 1e-8).all()

        assert processor.onsitemode is None
        assert data[3] is None

        assert len(data[4]) == 1
        assert isinstance(data[4][0],BaseStruct)

        assert (data[5] - kpoints_list  < 1e-8).all()
        assert (data[6] - eig_list  < 1e-8).all()

    processor = Processor(structure_list=struct_list, batchsize=1, kpoint=kpoints_list, eigen_list=eig_list, wannier_list=[None for _ in range(len(struct_list))],
            env_cutoff=2.0,onsite_cutoff=2.0, sorted_onsite="st", sorted_bond="st", sorted_env="itype-jtype")
    processor.onsitemode = 'strain'
    
    for data in processor:
        assert len(data) == 8
        assert isinstance(data[0], dict)
        assert isinstance(data[1], dict)
        assert torch.equal(data[1][0], batch_bond_onsite_true)
        assert  (data[0][0] - batch_bond_true < 1e-8).all()

        assert isinstance(data[2], dict)
        for ikey in data[2]:
            assert ikey in batch_env_itype_jtype_true
            assert (data[2][ikey] - batch_env_itype_jtype_true[ikey] < 1e-8).all()

        assert processor.onsitemode is 'strain'
        assert isinstance(data[3], dict)
        assert  (data[3][0] - batch_onsitenv_true < 1e-8).all()

        assert len(data[4]) == 1
        assert isinstance(data[4][0],BaseStruct)

        assert (data[5] - kpoints_list  < 1e-8).all()
        assert (data[6] - eig_list  < 1e-8).all()

    
    processor = Processor(structure_list=struct_list, batchsize=1, kpoint=kpoints_list, eigen_list=eig_list, wannier_list=[None for _ in range(len(struct_list))],
            env_cutoff=2.0,onsite_cutoff=2.0, sorted_onsite="st", sorted_bond="st", sorted_env="itype-jtype")
    processor.onsitemode = 'NRL'
    
    for data in processor:
        assert len(data) == 8
        assert isinstance(data[0], dict)
        assert isinstance(data[1], dict)
        assert torch.equal(data[1][0], batch_bond_onsite_true)
        assert  (data[0][0] - batch_bond_true < 1e-8).all()

        assert isinstance(data[2], dict)
        for ikey in data[2]:
            assert ikey in batch_env_itype_jtype_true
            assert (data[2][ikey] - batch_env_itype_jtype_true[ikey] < 1e-8).all()

        assert processor.onsitemode is 'NRL'
        assert isinstance(data[3], dict)
        assert  (data[3][0] - batch_onsitenv_true < 1e-8).all()

        assert len(data[4]) == 1
        assert isinstance(data[4][0],BaseStruct)

        assert (data[5] - kpoints_list  < 1e-8).all()
        assert (data[6] - eig_list  < 1e-8).all()