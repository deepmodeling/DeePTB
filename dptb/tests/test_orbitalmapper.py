import pytest
from typing import Dict, Union, List, Optional
import torch
from collections import OrderedDict
from dptb.data.transforms import OrbitalMapper
from e3nn.o3._irreps import Irreps

def test_orbital_mapper_init_str_spdf():
    # 创建一个OrbitalMapper实例
    basis = {"A": "2s2p3d1f", "B": "1s2f3d1f"}
    with pytest.raises(KeyError) as excinfo:
        OrbitalMapper(basis=basis,  device=torch.device("cpu"))
    assert 'A' in str(excinfo.value)

    basis = {"C": "2s2p3d1f", "O": "1s2f3d1f"}
    with pytest.raises(ValueError) as excinfo:
        OrbitalMapper(basis=basis,  device=torch.device("cpu"))
    assert "Duplicate orbitals found in the basis" in str(excinfo.value)

    basis = {"C": "2s2p3d1f", "O": "1s2p3d1f"}
    orbmap = OrbitalMapper(basis=basis, method="e3tb", device=torch.device("cpu"))
    assert orbmap.method == "e3tb"
    assert orbmap.basis =={'C': ['1s', '2s', '1p', '2p', '1d', '2d', '3d', '1f'],
                           'O': ['1s', '1p', '2p', '1d', '2d', '3d', '1f']}
    assert orbmap.orbtype_count == {'s': 2, 'p': 2, 'd': 3, 'f': 1, 'g':0, 'h':0}

    orbtype_count = orbmap.orbtype_count
    assert orbmap.full_basis_norb == 1 * orbtype_count["s"] + 3 * orbtype_count["p"] \
                                        + 5 * orbtype_count["d"] + 7 * orbtype_count["f"] == 30
    
    orbmap.reduced_matrix_element == int(((orbtype_count["s"] + 9 * orbtype_count["p"] + 25 * orbtype_count["d"] + 49 * orbtype_count["f"]) + \
                                                    orbmap.full_basis_norb ** 2)/2) == 522
    assert orbmap.full_basis == ['1s', '2s', '1p', '2p', '1d', '2d', '3d', '1f']

    assert orbmap.basis_to_full_basis == {'C': {'1s': '1s',
                                                '2s': '2s',
                                                '1p': '1p',
                                                '2p': '2p',
                                                '1d': '1d',
                                                '2d': '2d',
                                                '3d': '3d',
                                                '1f': '1f'},
                                               'O': {'1s': '1s',
                                                '1p': '1p',
                                                '2p': '2p',
                                                '1d': '1d',
                                                '2d': '2d',
                                                '3d': '3d',
                                                '1f': '1f'}}
    assert orbmap.full_basis_to_basis == orbmap.basis_to_full_basis
    assert torch.all(orbmap.atom_norb == torch.tensor([30, 29]))

    assert torch.all(orbmap.mask_to_basis == torch.tensor([[ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True],
        [ True, False,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True,  True,  True,  True,  True,  True,  True]]))

def test_orbital_mapper_init_str_spdfgh():
    basis = {"C": "2s2p3d1f1g1h", "O": "1s2p3d1f2g]"}
    orbmap = OrbitalMapper(basis=basis, method="e3tb", device=torch.device("cpu"))
    
    assert orbmap.method == "e3tb"
    assert orbmap.basis =={'C': ['1s', '2s', '1p', '2p', '1d', '2d', '3d', '1f', '1g', '1h'],
                           'O': ['1s', '1p', '2p', '1d', '2d', '3d', '1f', '1g', '2g']}
    assert orbmap.orbtype_count == {'s': 2, 'p': 2, 'd': 3, 'f': 1, 'g': 2, 'h': 1}

    orbtype_count = orbmap.orbtype_count
    assert orbmap.full_basis_norb == 1 * orbtype_count["s"] + 3 * orbtype_count["p"] \
                                        + 5 * orbtype_count["d"] + 7 * orbtype_count["f"] \
                                        + 9 * orbtype_count["g"] + 11 * orbtype_count["h"] == 59
    assert orbmap.reduced_matrix_element == int(((orbtype_count["s"] + 9 * orbtype_count["p"] + 25 * orbtype_count["d"] + 49 * orbtype_count["f"]) + \
                                                    + 81*orbtype_count["g"] + 121*orbtype_count["h"] +  orbmap.full_basis_norb ** 2)/2)  == 1954
    assert orbmap.full_basis == ['1s', '2s', '1p', '2p', '1d', '2d', '3d', '1f', '1g', '2g', '1h']

    assert orbmap.basis_to_full_basis == {'C': {'1s': '1s',
                                                '2s': '2s',
                                                '1p': '1p',
                                                '2p': '2p',
                                                '1d': '1d',
                                                '2d': '2d',
                                                '3d': '3d',
                                                '1f': '1f',
                                                '1g': '1g',
                                                '1h': '1h'},
                                           'O': {'1s': '1s',
                                                '1p': '1p',
                                                '2p': '2p',
                                                '1d': '1d',
                                                '2d': '2d',
                                                '3d': '3d',
                                                '1f': '1f',
                                                '1g': '1g',
                                                '2g': '2g'}}
    assert orbmap.full_basis_to_basis == orbmap.basis_to_full_basis
    assert torch.all(orbmap.atom_norb == torch.tensor([50, 47]))
    assert torch.all(orbmap.mask_to_basis == torch.tensor([[ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                                                             True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                                                             True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                                                             True,  True,  True,  True,  True,  True,  True,  True,  True, False,
                                                            False, False, False, False, False, False, False, False,  True,  True,
                                                             True,  True,  True,  True,  True,  True,  True,  True,  True],
                                                           [ True, False,  True,  True,  True,  True,  True,  True,  True,  True,
                                                             True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                                                             True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                                                             True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                                                             True,  True,  True,  True,  True,  True,  True,  True, False, False,
                                                            False, False, False, False, False, False, False, False, False]]))


def test_orbital_mapper_init_list_spdf_e3tb():
    basis = {"C": ['2s','2p','d*'], "O": ['2s','2p','f*']}
    orbmap = OrbitalMapper(basis=basis, method="e3tb", device=torch.device("cpu"))
    assert orbmap.method == "e3tb"
    assert orbmap.basis =={'C': ['2s', '2p', 'd*'], 'O': ['2s', '2p', 'f*']}
    assert orbmap.orbtype_count == {'s': 1, 'p': 1, 'd': 1, 'f': 1, 'g': 0, 'h': 0}
    orbtype_count = orbmap.orbtype_count
    assert orbmap.full_basis_norb == 1 * orbtype_count["s"] + 3 * orbtype_count["p"] \
                                        + 5 * orbtype_count["d"] + 7 * orbtype_count["f"] == 16
    assert orbmap.reduced_matrix_element == int(((orbtype_count["s"] + 9 * orbtype_count["p"] + 25 * orbtype_count["d"] + 49 * orbtype_count["f"]) + \
                                                    orbmap.full_basis_norb ** 2)/2) == 170
    assert orbmap.full_basis == ['1s', '1p', '1d', '1f']
    assert orbmap.basis_to_full_basis == {'C': {'2s': '1s', '2p': '1p', 'd*': '1d'},
                                          'O': {'2s': '1s', '2p': '1p', 'f*': '1f'}}
    assert orbmap.full_basis_to_basis == {'C': {'1s': '2s', '1p': '2p', '1d': 'd*'},
                                          'O': {'1s': '2s', '1p': '2p', '1f': 'f*'}}
    assert torch.all(orbmap.atom_norb == torch.tensor([9, 11]))
    assert torch.all(orbmap.mask_to_basis == torch.tensor([[ True,  True,  True,  True,  True,  True,  True,  True,  True, False,
                                                            False, False, False, False, False, False],
                                                           [ True,  True,  True,  True, False, False, False, False, False,  True,
                                                             True,  True,  True,  True,  True,  True]]))

def test_orbital_mapper_init_str_sktb():
    basis = {"C": "2s2p3d1f1g1h", "O": "1s2p3d1f2g]"}
    with pytest.raises(AssertionError) as excinfo:
        OrbitalMapper(basis=basis, method="sktb", device=torch.device("cpu"))
    assert "The method should be e3tb when the basis is given as string." in str(excinfo.value)

def test_orbital_mapper_init_list_spdf_sktb():
    basis = {"C": ['2s','2p','s*'], "O": ['2s','2p','f*']}
    orbmap = OrbitalMapper(basis=basis, method="sktb", device=torch.device("cpu"))
    assert orbmap.method == "sktb"
    assert orbmap.basis =={'C': ['s*', '2s', '2p'], 'O': ['2s', '2p', 'f*']}
    assert orbmap.orbtype_count == {'s': 2, 'p': 1, 'd': 0, 'f': 1, 'g': 0, 'h': 0}
    orbtype_count = orbmap.orbtype_count
    assert orbmap.full_basis_norb == 1 * orbtype_count["s"] + 3 * orbtype_count["p"] \
                                        + 5 * orbtype_count["d"] + 7 * orbtype_count["f"] == 12
    assert orbmap.reduced_matrix_element == 15
    assert orbmap.full_basis == ['1s', '2s', '1p', '1f']
    assert orbmap.n_onsite_Es == len(orbmap.full_basis)+1
    assert orbmap.basis_to_full_basis == {'C': {'s*': '1s', '2s': '2s', '2p': '1p'},
                                          'O': {'2s': '1s', '2p': '1p', 'f*': '1f'}}
    assert orbmap.full_basis_to_basis == {'C': {'1s': 's*', '2s': '2s', '1p': '2p'},
                                          'O': {'1s': '2s', '1p': '2p', '1f': 'f*'}}
    assert torch.all(orbmap.atom_norb == torch.tensor([5, 11]))
    assert torch.all(orbmap.mask_to_basis == torch.tensor([[ True,  True,  True,  True,  True, False, False, False, False, False, False, False],
                                                           [ True, False,  True,  True,  True,  True,  True,  True,  True,  True, True,  True]]))

def test_get_orbpairtype_maps():
    basis = {"C": "2s2p3d1f1g1h", "O": "1s2p3d1f2g]"}
    orbmap = OrbitalMapper(basis=basis, method="e3tb", device=torch.device("cpu"))
    expected_orbpairtype_maps = {'s-s': slice(0, 3, None),
                                 's-p': slice(3, 15, None),
                                 's-d': slice(15, 45, None),
                                 's-f': slice(45, 59, None),
                                 's-g': slice(59, 95, None),
                                 's-h': slice(95, 117, None),
                                 'p-p': slice(117, 144, None),
                                 'p-d': slice(144, 234, None),
                                 'p-f': slice(234, 276, None),
                                 'p-g': slice(276, 384, None),
                                 'p-h': slice(384, 450, None),
                                 'd-d': slice(450, 600, None),
                                 'd-f': slice(600, 705, None),
                                 'd-g': slice(705, 975, None),
                                 'd-h': slice(975, 1140, None),
                                 'f-f': slice(1140, 1189, None),
                                 'f-g': slice(1189, 1315, None),
                                 'f-h': slice(1315, 1392, None),
                                 'g-g': slice(1392, 1635, None),
                                 'g-h': slice(1635, 1833, None),
                                 'h-h': slice(1833, 1954, None)}
    assert orbmap.get_orbpairtype_maps() == expected_orbpairtype_maps

    basis = {"C": ['2s','2p','s*'], "O": ['2s','2p','f*']}
    orbmap = OrbitalMapper(basis=basis, method="sktb", device=torch.device("cpu"))
    expected_orbpairtype_maps = {'s-s': slice(0, 3, None),
                                 's-p': slice(3, 5, None),
                                 's-f': slice(5, 7, None),
                                 'p-p': slice(7, 9, None),
                                 'p-f': slice(9, 11, None),
                                 'f-f': slice(11, 15, None)}
    assert orbmap.get_orbpairtype_maps() == expected_orbpairtype_maps

def test_get_orbpair_maps():
    basis = {"C": "2s2p3d1f1g1h", "O": "1s2p3d1f2g]"}
    orbmap = OrbitalMapper(basis=basis, method="e3tb", device=torch.device("cpu"))
    expected_orbpair_maps = {'1s-1s': slice(0, 1, None),
                             '1s-2s': slice(1, 2, None),
                             '1s-1p': slice(3, 6, None),
                             '1s-2p': slice(6, 9, None),
                             '1s-1d': slice(15, 20, None),
                             '1s-2d': slice(20, 25, None),
                             '1s-3d': slice(25, 30, None),
                             '1s-1f': slice(45, 52, None),
                             '1s-1g': slice(59, 68, None),
                             '1s-2g': slice(68, 77, None),
                             '1s-1h': slice(95, 106, None),
                             '2s-2s': slice(2, 3, None),
                             '2s-1p': slice(9, 12, None),
                             '2s-2p': slice(12, 15, None),
                             '2s-1d': slice(30, 35, None),
                             '2s-2d': slice(35, 40, None),
                             '2s-3d': slice(40, 45, None),
                             '2s-1f': slice(52, 59, None),
                             '2s-1g': slice(77, 86, None),
                             '2s-2g': slice(86, 95, None),
                             '2s-1h': slice(106, 117, None),
                             '1p-1p': slice(117, 126, None),
                             '1p-2p': slice(126, 135, None),
                             '1p-1d': slice(144, 159, None),
                             '1p-2d': slice(159, 174, None),
                             '1p-3d': slice(174, 189, None),
                             '1p-1f': slice(234, 255, None),
                             '1p-1g': slice(276, 303, None),
                             '1p-2g': slice(303, 330, None),
                             '1p-1h': slice(384, 417, None),
                             '2p-2p': slice(135, 144, None),
                             '2p-1d': slice(189, 204, None),
                             '2p-2d': slice(204, 219, None),
                             '2p-3d': slice(219, 234, None),
                             '2p-1f': slice(255, 276, None),
                             '2p-1g': slice(330, 357, None),
                             '2p-2g': slice(357, 384, None),
                             '2p-1h': slice(417, 450, None),
                             '1d-1d': slice(450, 475, None),
                             '1d-2d': slice(475, 500, None),
                             '1d-3d': slice(500, 525, None),
                             '1d-1f': slice(600, 635, None),
                             '1d-1g': slice(705, 750, None),
                             '1d-2g': slice(750, 795, None),
                             '1d-1h': slice(975, 1030, None),
                             '2d-2d': slice(525, 550, None),
                             '2d-3d': slice(550, 575, None),
                             '2d-1f': slice(635, 670, None),
                             '2d-1g': slice(795, 840, None),
                             '2d-2g': slice(840, 885, None),
                             '2d-1h': slice(1030, 1085, None),
                             '3d-3d': slice(575, 600, None),
                             '3d-1f': slice(670, 705, None),
                             '3d-1g': slice(885, 930, None),
                             '3d-2g': slice(930, 975, None),
                             '3d-1h': slice(1085, 1140, None),
                             '1f-1f': slice(1140, 1189, None),
                             '1f-1g': slice(1189, 1252, None),
                             '1f-2g': slice(1252, 1315, None),
                             '1f-1h': slice(1315, 1392, None),
                             '1g-1g': slice(1392, 1473, None),
                             '1g-2g': slice(1473, 1554, None),
                             '1g-1h': slice(1635, 1734, None),
                             '2g-2g': slice(1554, 1635, None),
                             '2g-1h': slice(1734, 1833, None),
                             '1h-1h': slice(1833, 1954, None)}
    assert orbmap.get_orbpair_maps() == expected_orbpair_maps
                            
    basis = {"C": ['2s','2p','s*'], "O": ['2s','2p','f*']}
    orbmap = OrbitalMapper(basis=basis, method="sktb", device=torch.device("cpu"))
    expected_orbpair_maps = {'1s-1s': slice(0, 1, None),
                                 '1s-2s': slice(1, 2, None),
                                 '1s-1p': slice(3, 4, None),
                                 '1s-1f': slice(5, 6, None),
                                 '2s-2s': slice(2, 3, None),
                                 '2s-1p': slice(4, 5, None),
                                 '2s-1f': slice(6, 7, None),
                                 '1p-1p': slice(7, 9, None),
                                 '1p-1f': slice(9, 11, None),
                                 '1f-1f': slice(11, 15, None)}
    assert orbmap.get_orbpair_maps() == expected_orbpair_maps

def test_get_skonsite_maps():
    basis = {"C": "2s2p3d1f1g1h", "O": "1s2p3d1f2g]"}
    orbmap = OrbitalMapper(basis=basis, method="e3tb", device=torch.device("cpu"))
    with pytest.raises(AssertionError) as excinfo:  
        orbmap.get_skonsite_maps()
    assert "Only sktb orbitalmapper have skonsite maps" in str(excinfo.value)

    basis = {"C": ['2s','2p','s*'], "O": ['2s','2p','f*']}
    orbmap = OrbitalMapper(basis=basis, method="sktb", device=torch.device("cpu"))

    assert orbmap.get_skonsite_maps() == {'1s-1s': slice(0, 1, None),
                                          '1s-2s': slice(1, 2, None),
                                          '2s-2s': slice(2, 3, None),
                                          '1p-1p': slice(3, 4, None),
                                          '1f-1f': slice(4, 5, None)}

def test_get_skonsitetype_maps():
    basis = {"C": "2s2p3d1f1g1h", "O": "1s2p3d1f2g]"}
    orbmap = OrbitalMapper(basis=basis, method="e3tb", device=torch.device("cpu"))
    with pytest.raises(AssertionError) as excinfo:  
        orbmap.get_skonsitetype_maps()
    assert "Only sktb orbitalmapper have skonsite maps" in str(excinfo.value)

    basis = {"C": ['2s','2p','s*'], "O": ['2s','2p','f*']}
    orbmap = OrbitalMapper(basis=basis, method="sktb", device=torch.device("cpu"))

    assert orbmap.get_skonsitetype_maps() == {'s': slice(0, 3, None), 'p': slice(3, 4, None), 'f': slice(4, 5, None)}

def test_get_sksoctype_maps():
    basis = {"C": "2s2p3d1f1g1h", "O": "1s2p3d1f2g]"}
    orbmap = OrbitalMapper(basis=basis, method="e3tb", device=torch.device("cpu"))
    with pytest.raises(AssertionError) as excinfo:  
        orbmap.get_sksoctype_maps()
    assert "Only sktb orbitalmapper have sksoctype maps" in str(excinfo.value)

    basis = {"C": ['2s','2p','s*'], "O": ['2s','2p','f*']}
    orbmap = OrbitalMapper(basis=basis, method="sktb", device=torch.device("cpu"))

    assert orbmap.get_sksoctype_maps() == {'s': slice(0, 2, None), 'p': slice(2, 3, None), 'f': slice(3, 4, None)}

def test_get_sksoc_maps():
    basis = {"C": "2s2p3d1f1g1h", "O": "1s2p3d1f2g]"}
    orbmap = OrbitalMapper(basis=basis, method="e3tb", device=torch.device("cpu"))
    with pytest.raises(AssertionError) as excinfo:  
        orbmap.get_sksoc_maps()
    assert "Only sktb orbitalmapper have sksoc maps" in str(excinfo.value)

    basis = {"C": ['2s','2p','s*'], "O": ['2s','2p','f*']}
    orbmap = OrbitalMapper(basis=basis, method="sktb", device=torch.device("cpu"))

    assert orbmap.get_sksoc_maps() == {'1s': slice(0, 1, None),
                                          '2s': slice(1, 2, None),
                                          '1p': slice(2, 3, None),
                                          '1f': slice(3, 4, None)}    

def test_get_orbital_maps():
    basis = {"C": "2s2p3d1f1g1h", "O": "1s2p3d1f2g]"}
    orbmap = OrbitalMapper(basis=basis, method="e3tb", device=torch.device("cpu"))
    expected_orbital_maps = {'C': {'1s': slice(0, 1, None),
                            '2s': slice(1, 2, None),
                            '1p': slice(2, 5, None),
                            '2p': slice(5, 8, None),
                            '1d': slice(8, 13, None),
                            '2d': slice(13, 18, None),
                            '3d': slice(18, 23, None),
                            '1f': slice(23, 30, None),
                            '1g': slice(30, 39, None),
                            '1h': slice(39, 50, None)},
                           'O': {'1s': slice(0, 1, None),
                            '1p': slice(1, 4, None),
                            '2p': slice(4, 7, None),
                            '1d': slice(7, 12, None),
                            '2d': slice(12, 17, None),
                            '3d': slice(17, 22, None),
                            '1f': slice(22, 29, None),
                            '1g': slice(29, 38, None),
                            '2g': slice(38, 47, None)}}
    assert orbmap.get_orbital_maps() == expected_orbital_maps
    basis = {"C": ['2s','2p','s*'], "O": ['2s','2p','f*']}
    orbmap = OrbitalMapper(basis=basis, method="sktb", device=torch.device("cpu"))
    assert orbmap.get_orbital_maps() == {'C': {'s*': slice(0, 1, None),
                                               '2s': slice(1, 2, None),
                                               '2p': slice(2, 5, None)},
                                         'O': {'2s': slice(0, 1, None),
                                               '2p': slice(1, 4, None),
                                               'f*': slice(4, 11, None)}}

def test_get_irreps():
    basis = {"C": ['2s','2p','s*'], "O": ['2s','2p','f*']}
    orbmap = OrbitalMapper(basis=basis, method="sktb", device=torch.device("cpu"))
    with pytest.raises(AssertionError) as excinfo:  
        orbmap.get_irreps()
    assert "Only support e3tb method for now." in str(excinfo.value)

    basis = {"C": "2s2p3d1f1g1h", "O": "1s2p3d1f2g]"}
    orbmap = OrbitalMapper(basis=basis, method="e3tb", device=torch.device("cpu"))
    orbpair_irreps = orbmap.get_irreps()
    assert orbpair_irreps == Irreps("1x0e+1x0e+1x0e+1x1o+1x1o+1x1o+1x1o+1x2e+1x2e+1x2e+"+
                                  "1x2e+1x2e+1x2e+1x3o+1x3o+1x4e+1x4e+1x4e+1x4e+1x5o+1x5o+"+
                                  "1x0e+1x1e+1x2e+1x0e+1x1e+1x2e+1x0e+1x1e+1x2e+1x1o+1x2o+1x3o+"+
                                  "1x1o+1x2o+1x3o+1x1o+1x2o+1x3o+1x1o+1x2o+1x3o+1x1o+1x2o+1x3o+1x1o+"+
                                  "1x2o+1x3o+1x2e+1x3e+1x4e+1x2e+1x3e+1x4e+1x3o+1x4o+1x5o+1x3o+1x4o+1x5o+"+
                                  "1x3o+1x4o+1x5o+1x3o+1x4o+1x5o+1x4e+1x5e+1x6e+1x4e+1x5e+1x6e+1x0e+1x1e+1x2e+"+
                                  "1x3e+1x4e+1x0e+1x1e+1x2e+1x3e+1x4e+1x0e+1x1e+1x2e+1x3e+1x4e+1x0e+1x1e+1x2e+"+
                                  "1x3e+1x4e+1x0e+1x1e+1x2e+1x3e+1x4e+1x0e+1x1e+1x2e+1x3e+1x4e+1x1o+1x2o+1x3o+"+
                                  "1x4o+1x5o+1x1o+1x2o+1x3o+1x4o+1x5o+1x1o+1x2o+1x3o+1x4o+1x5o+1x2e+1x3e+1x4e+"+
                                  "1x5e+1x6e+1x2e+1x3e+1x4e+1x5e+1x6e+1x2e+1x3e+1x4e+1x5e+1x6e+1x2e+1x3e+1x4e+"+
                                  "1x5e+1x6e+1x2e+1x3e+1x4e+1x5e+1x6e+1x2e+1x3e+1x4e+1x5e+1x6e+1x3o+1x4o+1x5o+"+
                                  "1x6o+1x7o+1x3o+1x4o+1x5o+1x6o+1x7o+1x3o+1x4o+1x5o+1x6o+1x7o+1x0e+1x1e+1x2e+"+
                                  "1x3e+1x4e+1x5e+1x6e+1x1o+1x2o+1x3o+1x4o+1x5o+1x6o+1x7o+1x1o+1x2o+1x3o+1x4o+"+
                                  "1x5o+1x6o+1x7o+1x2e+1x3e+1x4e+1x5e+1x6e+1x7e+1x8e+1x0e+1x1e+1x2e+1x3e+1x4e+"+
                                  "1x5e+1x6e+1x7e+1x8e+1x0e+1x1e+1x2e+1x3e+1x4e+1x5e+1x6e+1x7e+1x8e+1x0e+1x1e+"+
                                  "1x2e+1x3e+1x4e+1x5e+1x6e+1x7e+1x8e+1x1o+1x2o+1x3o+1x4o+1x5o+1x6o+1x7o+1x8o+"+
                                  "1x9o+1x1o+1x2o+1x3o+1x4o+1x5o+1x6o+1x7o+1x8o+1x9o+1x0e+1x1e+1x2e+1x3e+1x4e+"+
                                  "1x5e+1x6e+1x7e+1x8e+1x9e+1x10e")
    assert orbpair_irreps == orbmap.get_irreps()
    assert orbmap.no_parity is False

    orbpair_irreps = orbmap.get_irreps(no_parity=True)
    assert orbmap.no_parity is True
    assert orbpair_irreps == Irreps("1x0e+1x0e+1x0e+1x1e+1x1e+1x1e+1x1e+1x2e+1x2e+1x2e+1x2e+1x2e+1x2e+1x3e+"+
                                    "1x3e+1x4e+1x4e+1x4e+1x4e+1x5e+1x5e+1x0e+1x1e+1x2e+1x0e+1x1e+1x2e+1x0e+"+
                                    "1x1e+1x2e+1x1e+1x2e+1x3e+1x1e+1x2e+1x3e+1x1e+1x2e+1x3e+1x1e+1x2e+1x3e+"+
                                    "1x1e+1x2e+1x3e+1x1e+1x2e+1x3e+1x2e+1x3e+1x4e+1x2e+1x3e+1x4e+1x3e+1x4e+"+
                                    "1x5e+1x3e+1x4e+1x5e+1x3e+1x4e+1x5e+1x3e+1x4e+1x5e+1x4e+1x5e+1x6e+1x4e+"+
                                    "1x5e+1x6e+1x0e+1x1e+1x2e+1x3e+1x4e+1x0e+1x1e+1x2e+1x3e+1x4e+1x0e+1x1e+"+
                                    "1x2e+1x3e+1x4e+1x0e+1x1e+1x2e+1x3e+1x4e+1x0e+1x1e+1x2e+1x3e+1x4e+1x0e+"+
                                    "1x1e+1x2e+1x3e+1x4e+1x1e+1x2e+1x3e+1x4e+1x5e+1x1e+1x2e+1x3e+1x4e+1x5e+"+
                                    "1x1e+1x2e+1x3e+1x4e+1x5e+1x2e+1x3e+1x4e+1x5e+1x6e+1x2e+1x3e+1x4e+1x5e+"+
                                    "1x6e+1x2e+1x3e+1x4e+1x5e+1x6e+1x2e+1x3e+1x4e+1x5e+1x6e+1x2e+1x3e+1x4e+"+
                                    "1x5e+1x6e+1x2e+1x3e+1x4e+1x5e+1x6e+1x3e+1x4e+1x5e+1x6e+1x7e+1x3e+1x4e+"+
                                    "1x5e+1x6e+1x7e+1x3e+1x4e+1x5e+1x6e+1x7e+1x0e+1x1e+1x2e+1x3e+1x4e+1x5e+"+
                                    "1x6e+1x1e+1x2e+1x3e+1x4e+1x5e+1x6e+1x7e+1x1e+1x2e+1x3e+1x4e+1x5e+1x6e+"+
                                    "1x7e+1x2e+1x3e+1x4e+1x5e+1x6e+1x7e+1x8e+1x0e+1x1e+1x2e+1x3e+1x4e+1x5e+"+
                                    "1x6e+1x7e+1x8e+1x0e+1x1e+1x2e+1x3e+1x4e+1x5e+1x6e+1x7e+1x8e+1x0e+1x1e+"+
                                    "1x2e+1x3e+1x4e+1x5e+1x6e+1x7e+1x8e+1x1e+1x2e+1x3e+1x4e+1x5e+1x6e+1x7e+"+
                                    "1x8e+1x9e+1x1e+1x2e+1x3e+1x4e+1x5e+1x6e+1x7e+1x8e+1x9e+1x0e+1x1e+1x2e+"+
                                    "1x3e+1x4e+1x5e+1x6e+1x7e+1x8e+1x9e+1x10e")

def test_masks_e3tb():
    basis = {"C": "2s2p1d", "O": "1s1p]"}
    orbmap = OrbitalMapper(basis=basis, method="e3tb", device=torch.device("cpu"))
    assert orbmap.mask_to_erme.shape == torch.Size([4, 107])
    assert orbmap.mask_to_nrme.shape == torch.Size([2, 107])

    basis = {"C": "2s2p1d", "O": "2s2p1d]"}
    orbmap = OrbitalMapper(basis=basis, method="e3tb", device=torch.device("cpu"))
    assert orbmap.mask_to_erme.shape == torch.Size([4, 107])
    assert orbmap.mask_to_nrme.shape == torch.Size([2, 107])
    basis = {"C": "2s1p1d", "O": "1s1p]"}
    orbmap = OrbitalMapper(basis=basis, method="e3tb", device=torch.device("cpu"))
    assert orbmap.mask_to_erme.shape == torch.Size([4, 68])
    assert orbmap.mask_to_nrme.shape == torch.Size([2, 68])

    expected_mask_to_erme = torch.tensor([[ True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True,  True,  True,  True,  True],
                                          [ True,  True, False,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                                          [ True,  True, False,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
                                          [ True, False, False,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]])

    assert torch.equal(orbmap.mask_to_erme, expected_mask_to_erme)

    expected_mask_to_nrme = torch.tensor([[ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,True,  True,  True,  True,  True,  True,  True,  True,  True,  True,True,  True,  True,  True,  True,  True,  True,  True,  True,  True,True,  True,  True,  True,  True,  True,  True,  True,  True,  True,True,  True,  True,  True,  True,  True,  True,  True,  True,  True,True,  True,  True,  True,  True,  True,  True,  True,  True,  True,True,  True,  True,  True,  True,  True,  True,  True],
                                          [ True, False, False,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False,  True,  True,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]])
    assert torch.equal(orbmap.mask_to_nrme, expected_mask_to_nrme)


    assert orbmap.mask_to_ndiag.shape == torch.Size([2, 68])
    expected_mask_to_ndiag = torch.tensor([[ True, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False,  True, False, False, False,  True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,  True, False, False, False, False, False,  True, False, False, False, False, False,  True, False, False, False, False, False,  True, False, False, False, False, False,  True],
                                           [ True, False, False, False, False, False, False, False, False, False,  False, False, False, False, False, False, False, False, False,  True,  False, False, False,  True, False, False, False,  True, False, False,  False, False, False, False, False, False, False, False, False, False,  False, False, False, False, False, False, False, False, False, False,  False, False, False, False, False, False, False, False, False, False,  False, False, False, False, False, False, False, False]])
    assert torch.equal(orbmap.mask_to_ndiag, expected_mask_to_ndiag)

def test_masks_sktb():
    basis = {"C": ['2s','2p','s*'], "O": ['2s','2p','f*']}
    orbmap = OrbitalMapper(basis=basis, method="sktb", device=torch.device("cpu"))
    assert orbmap.mask_to_erme.shape == torch.Size([4, 15])
    assert orbmap.mask_to_nrme.shape == torch.Size([2, 15])

    expected_mask_to_erme = torch.tensor([[ True,  True,  True,  True,  True, False, False,  True,  True, False, False, False, False, False, False],
                                          [ True,  True, False,  True,  True,  True,  True,  True,  True,  True, True, False, False, False, False],
                                          [ True,  True, False,  True,  True,  True,  True,  True,  True,  True, True, False, False, False, False],
                                          [ True, False, False,  True, False,  True, False,  True,  True,  True,True,  True,  True,  True,  True]])

    assert torch.equal(orbmap.mask_to_erme, expected_mask_to_erme)

    expected_mask_to_nrme = torch.tensor([[ True,  True,  True,  True,  True, False, False,  True,  True, False,False, False, False, False, False],
                                          [ True, False, False,  True, False,  True, False,  True,  True,  True,True,  True,  True,  True,  True]])
    assert torch.equal(orbmap.mask_to_nrme, expected_mask_to_nrme)

    assert not hasattr(orbmap, "mask_to_ndiag")

def test_equality_operator():
    # 测试相等性操作符
    basis = {"C": "2s1p1d", "O": "1s1p]"}
    orbmap = OrbitalMapper(basis=basis, method="e3tb", device=torch.device("cpu"))
    orbmap2 = OrbitalMapper(basis=basis, method="e3tb", device=torch.device("cpu"))
    assert orbmap == orbmap2

    # 不同的basis或method应该不相等
    basis = {"C": "2s1p1d", "O": "1s1p1d]"}
    orbmap2 = OrbitalMapper(basis=basis, method="e3tb", device=torch.device("cpu"))
    assert orbmap != orbmap2


    basis = {"C": ['2s','2p','s*'], "O": ['2s','2p','f*']}
    orbmap = OrbitalMapper(basis=basis, method="sktb", device=torch.device("cpu"))
    orbmap2 = OrbitalMapper(basis=basis, method="sktb", device=torch.device("cpu"))
    assert orbmap == orbmap2

    basis = {"C": ['2s','2p','s*'], "O": ['2s','2p','f*']}
    orbmap2 = OrbitalMapper(basis=basis, method="e3tb", device=torch.device("cpu"))
    assert orbmap != orbmap2

    basis = {"C": ['2s','2p','d*'], "O": ['2s','2p','f*']}
    orbmap2 = OrbitalMapper(basis=basis, method="sktb", device=torch.device("cpu"))
    assert orbmap != orbmap2

