import pytest 
from dptb.nn.sktb.bondlengthDB import bond_length_list, bond_length, bond_length_full_dict 
from dptb.utils.constants import atomic_num_dict_r


def test_bond_length_list():
    for ii in range(1,len(atomic_num_dict_r)):
        atomic_number = ii 
        atom_symbol = atomic_num_dict_r[atomic_number]
        if atom_symbol in bond_length:
            assert bond_length_full_dict[atom_symbol] == bond_length[atom_symbol]
            assert bond_length_list[ii-1] == bond_length[atom_symbol] / 1.8897259886
        else:
            assert bond_length_full_dict[atom_symbol] is None
            assert bond_length_list[ii-1] == -100
        
from dptb.nn.sktb.cov_radiiDB import Covalent_radii, R_cov_list
from dptb.utils.constants import atomic_num_dict

def test_Covalent_radii():
    for key, val in atomic_num_dict.items():
        if key in Covalent_radii:
            assert Covalent_radii[key] == R_cov_list[val-1] 
        else:
            assert R_cov_list[val-1] == -100
