import pytest 
from dptb.nn.sktb.bondlengthDB import bond_length_list, bond_length, bond_length_full_dict 
from dptb.utils.constants import atomic_num_dict_r


def test_bond_length_list():
    for ii in range(1,len(atomic_num_dict_r)):
        atomic_number = ii 
        atom_symbol = atomic_num_dict_r[atomic_number]
        if atom_symbol in bond_length:
            assert bond_length_full_dict[atom_symbol] == bond_length[atom_symbol]
            assert bond_length_list[ii-1] == bond_length[atom_symbol]
        else:
            assert bond_length_full_dict[atom_symbol] is None
            assert bond_length_list[ii-1] == -100
        