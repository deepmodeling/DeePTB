import torch 
import numpy as np
import pytest
import os
from dptb.nnops.nnapi import NNSK, DeePTB
from ase.io import read,write
from dptb.structure.structure import BaseStruct

@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
        return str(request.config.rootdir)


def test_nnskapi(root_directory):
    filepath = f'{root_directory}/dptb/tests/data'
    checkpoint = f'{filepath}/hbn_nnsk.pb'
    assert os.path.exists(checkpoint), f'{checkpoint} not found!'
    proj_atom_anglr_m = {'N': ['2s', '2p'], 'B': ['2s', '2p']}
    proj_atom_neles = {'N': 5, 'B': 3}
    cutoff= 4
    nnsk = NNSK(checkpoint, proj_atom_anglr_m)
    strase = read(f'{root_directory}/examples/TBmodel/hBN/check/hBN.vasp')
    struct = BaseStruct(atom=strase, format='ase', cutoff=cutoff, proj_atom_anglr_m=proj_atom_anglr_m, proj_atom_neles=proj_atom_neles, time_symm=True)
    
    snapase = struct.struct
    kpath = snapase.cell.bandpath('GMKG', npoints=120)
    xlist, high_sym_kpoints, labels = kpath.get_linear_kpoint_axis()
    klist = kpath.kpts
    all_bonds, hamil_blocks, overlap_blocks = nnsk.get_HR(struct)

    refbond= np.array([[ 7,  0,  7,  0,  0,  0,  0],
       [ 5,  1,  5,  1,  0,  0,  0],
       [ 7,  0,  5,  1, -2,  0,  0],
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

    assert (all_bonds == refbond).all()
    assert overlap_blocks is None
    assert len(hamil_blocks) == len(all_bonds)



    eigks, EF = nnsk.get_eigenvalues(klist)

    assert eigks.shape == (120,8)
    assert np.abs(EF - -6.587416648864746) < 1e-5 
    
    refeig0 = np.array([-22.371159 , -10.782168 ,  -7.0572667,  -7.0572667,  -3.6658115,
        -2.6096034,   4.7781563,   4.7781568], dtype=np.float32)
    assert (np.abs(eigks[0]-refeig0) < 1e-5).all()
