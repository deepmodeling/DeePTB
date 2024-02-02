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

def test_dptbapi_nnsk(root_directory):
   # TODO: fix the train for dptb + nnsk and test.
   pass
   """
    filepath = f'{root_directory}/dptb/tests/data'
    dptb_checkpoint=f'{filepath}/hbn_dptb_skfile.pb'
    nnsk_checkpoint = f'{filepath}/hbn_nnsk.pb'
    assert os.path.exists(dptb_checkpoint), f'{dptb_checkpoint} not found!'
    assert os.path.exists(nnsk_checkpoint), f'{nnsk_checkpoint} not found!'

    proj_atom_anglr_m = {'N': ['2s', '2p'], 'B': ['2s', '2p']}
    proj_atom_neles = {'N': 5, 'B': 3}
    cutoff= 4
    dptb = DeePTB(dptb_checkpoint = dptb_checkpoint, proj_atom_anglr_m = proj_atom_anglr_m,sktbmode='nnsk', nnsk_checkpoint=nnsk_checkpoint)
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

    hk00 = np.array([-5.4950452e-01+0.j,  0.0000000e+00+0.j,  0.0000000e+00+0.j,
        0.0000000e+00+0.j, -2.6814380e-01+0.j, -1.6661943e-08+0.j,
        0.0000000e+00+0.j, -3.7543941e-09+0.j], dtype=np.complex64)
    assert (np.abs(hkmat[0][0].detach().numpy() -  hk00) < 1e-5).all()
    eigks, EF = dptb.get_eigenvalues(kpoints=klist)
    assert eigks.shape == (120,8)

    refeig0 = np.array([-2.1452765e+01, -9.3084345e+00, -6.7620611e+00, -5.1572165e+00,
       -5.1572142e+00,  8.6744241e-03,  7.9295797e+00,  7.9295816e+00 ], dtype=np.float32)
    assert np.abs(EF - -4.828580856323242) < 1e-5 
    assert (np.abs(eigks[0]-refeig0) < 1e-5).all()
   """