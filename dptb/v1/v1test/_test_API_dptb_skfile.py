import torch 
import numpy as np
import pytest
import os
from dptb.nnops.v1.nnapi import NNSK, DeePTB
from ase.io import read,write
from dptb.structure.structure import BaseStruct

@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
        return str(request.config.rootdir)

def test_dptbapi_skfile(root_directory):
    filepath = f'{root_directory}/dptb/tests/data'
    dptb_checkpoint=f'{filepath}/hbn_dptb_skfile.pb'
    sk_file_path = f'{root_directory}/examples/slakos'
    assert os.path.exists(dptb_checkpoint), f'{dptb_checkpoint} not found!'
    assert os.path.exists(sk_file_path), f'{sk_file_path} not found!'

    proj_atom_anglr_m = {'N': ['2s', '2p'], 'B': ['2s', '2p']}
    proj_atom_neles = {'N': 5, 'B': 3}
    cutoff= 4
    dptb = DeePTB(dptb_checkpoint = dptb_checkpoint, proj_atom_anglr_m = proj_atom_anglr_m,sktbmode='skfile', sk_file_path=sk_file_path)
    strase = read(f'{root_directory}/examples/TBmodel/hBN/check/hBN.vasp')
    struct = BaseStruct(atom=strase, format='ase', cutoff=cutoff, proj_atom_anglr_m=proj_atom_anglr_m, proj_atom_neles=proj_atom_neles, time_symm=True)
    snapase = struct.struct
    kpath = snapase.cell.bandpath('GMKG', npoints=120)
    xlist, high_sym_kpoints, labels = kpath.get_linear_kpoint_axis()
    klist = kpath.kpts
    all_bonds, hamil_blocks, overlap_blocks = dptb.get_HR(structure=struct,env_cutoff=3.5)
    
    refbond = np.array([[ 7,  0,  7,  0,  0,  0,  0],
       [ 5,  1,  5,  1,  0,  0,  0],
       [ 7,  0,  5,  1, -2,  0,  0],
       [ 7,  0,  5,  1, -1,  0,  0],
       [ 7,  0,  5,  1,  1,  0,  0],
       [ 7,  0,  5,  1, -1,  1,  0],
       [ 7,  0,  5,  1,  0,  1,  0],
       [ 7,  0,  5,  1,  1,  1,  0],
       [ 7,  0,  5,  1,  0,  2,  0],
       [ 7,  0,  5,  1,  1,  2,  0],
       [ 7,  0,  5,  1,  0,  0,  0],
       [ 7,  0,  5,  1, -1, -1,  0],
       [ 7,  0,  5,  1, -2, -1,  0],
       [ 7,  0,  5,  1,  0, -1,  0],
       [ 7,  0,  7,  0, -1,  0,  0],
       [ 7,  0,  7,  0,  0,  1,  0],
       [ 7,  0,  7,  0,  1,  1,  0],
       [ 5,  1,  5,  1,  1,  1,  0],
       [ 5,  1,  5,  1,  0, -1,  0],
       [ 5,  1,  5,  1, -1,  0,  0]])
    
    assert (all_bonds == refbond).all()
    assert overlap_blocks is None
    assert len(hamil_blocks) == len(all_bonds)


    hkmat, skmat = dptb.get_HK(kpoints=klist)
    assert hkmat.shape == torch.Size([120, 8, 8])
    assert skmat.shape == torch.Size([120, 8, 8])

    hk00 = np.array([-6.628017e-01+0.j,  0.000000e+00+0.j,  0.000000e+00+0.j,
        0.000000e+00+0.j, -8.332963e-01+0.j,  9.057112e-08+0.j,
        0.000000e+00+0.j,  5.966285e-08+0.j], dtype=np.complex64)
    assert (np.abs(hkmat[0][0].detach().numpy() -  hk00) < 1e-5).all()
    eigks, EF = dptb.get_eigenvalues(kpoints=klist)
    assert eigks.shape == (120,8)

    refeig0 = np.array([-22.558338 , -10.220116 ,  -6.2584987,  -6.2584934,   4.1156316,
         9.046395 ,  14.905643 ,  14.905655 ], dtype=np.float32)
    assert np.abs(EF - -2.3601527214050293) < 1e-5 
    assert (np.abs(eigks[0]-refeig0) < 1e-5).all()

