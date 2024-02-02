import numpy as np
import torch
import pytest
from dptb.dataprocess.process_wannier import read_hr, wan_orbital_orders, transfrom_Hwan, get_onsite_shift
from dptb.structure.structure import BaseStruct

@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    """
    :return:
    """
    return str(request.config.rootdir)


def test_read_hr(root_directory):
    # Test 1.1: Read a single wannier90_hr.dat file
    wannierfile = root_directory + '/dptb/tests/data/wan/wannier90_hr.dat'
    Rlatt, hopps, indR0 = read_hr(wannierfile)

    Rlatt_true = np.load(root_directory + '/dptb/tests/data/wan/Rlatt.npy')
    hopps_true = np.load(root_directory + '/dptb/tests/data/wan/hopps.npy')

    assert indR0 == 428
    assert Rlatt.shape == (857, 3)
    assert hopps.shape == (857, 8, 8)
    assert np.allclose(Rlatt, Rlatt_true)
    assert np.allclose(hopps, hopps_true)

def test_wan_orbital_orders(root_directory):

    proj_atom_anglr_m = {'Si':['3s','3p']}
    proj_atom_neles={'Si':4}
    CutOff = 2.5
    struct =BaseStruct(atom=root_directory + '/dptb/tests/data/wan/silicon.vasp',format='vasp',
        cutoff=CutOff,proj_atom_anglr_m=proj_atom_anglr_m,proj_atom_neles=proj_atom_neles, onsitemode='uniform')
    
    wannier_orbital_order, sk_orbital_order, iatom_nors = wan_orbital_orders(struct=struct, wannier_proj_orbital={'Si':['s','p']})
    
    wan_orb_order_true = ['0-s', '0-pz', '0-px', '0-py', '1-s', '1-pz', '1-px', '1-py']
    sk_orb_order_true = ['0-s', '0-py', '0-pz', '0-px', '1-s', '1-py', '1-pz', '1-px']


    assert wannier_orbital_order == wan_orb_order_true
    assert sk_orbital_order == sk_orb_order_true
    assert np.allclose(iatom_nors, np.array([4,4]))

def test_onsite_shift(root_directory):
    hopps = np.load(root_directory + '/dptb/tests/data/wan/hopps.npy')
    indR0 = 428 
    hopps_r00 = hopps[indR0]

    proj_atom_anglr_m = {'Si':['3s','3p']}
    proj_atom_neles={'Si':4}
    CutOff = 2.5
    struct =BaseStruct(atom=root_directory + '/dptb/tests/data/wan/silicon.vasp',format='vasp',
        cutoff=CutOff,proj_atom_anglr_m=proj_atom_anglr_m,proj_atom_neles=proj_atom_neles, onsitemode='uniform')
    
    wannier_orbital_order, sk_orbital_order, iatom_nors = wan_orbital_orders(struct=struct, wannier_proj_orbital={'Si':['s','p']})


    onsite_shift =  get_onsite_shift(hopps_r00, struct, wannier_orbital_order, unit='eV')

    assert abs(onsite_shift - 13.448753996967032) < 1e-8

def test_transfrom_Hwan(root_directory):
    proj_atom_anglr_m = {'Si':['3s','3p']}
    proj_atom_neles={'Si':4}
    CutOff = 2.5
    struct =BaseStruct(atom=root_directory + '/dptb/tests/data/wan/silicon.vasp',format='vasp',
        cutoff=CutOff,proj_atom_anglr_m=proj_atom_anglr_m,proj_atom_neles=proj_atom_neles, onsitemode='uniform')
    
    wannier_orbital_order, sk_orbital_order, iatom_nors = wan_orbital_orders(struct=struct, wannier_proj_orbital={'Si':['s','p']})
    
    Rlatt = np.load(root_directory + '/dptb/tests/data/wan/Rlatt.npy')
    hopps = np.load(root_directory + '/dptb/tests/data/wan/hopps.npy')
    indR0 = 428
    hopping_bonds = transfrom_Hwan(hopps, Rlatt, indR0, struct, wannier_orbital_order, sk_orbital_order, iatom_nors)
    
    hopping_bonds_true = np.load(root_directory + '/dptb/tests/data/wan/hop_bondwise.npy', allow_pickle=True).tolist()
    
    assert isinstance(hopping_bonds, dict)

    assert hopping_bonds_true.keys() == hopping_bonds.keys()
    for ikey in hopping_bonds_true.keys():
        assert np.allclose(hopping_bonds_true[ikey], hopping_bonds[ikey])

