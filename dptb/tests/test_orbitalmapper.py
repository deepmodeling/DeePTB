import pytest
from typing import Dict, Union, List, Optional
import torch
from collections import OrderedDict
from dptb.data.transforms import OrbitalMapper


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

#     def test_init(self, orbital_mapper):
#         # 测试初始化是否正确设置了basis和orbtype_count
#         expected_basis = {"A": ["s*", "1s", "2s", "2p", "1d", "2d", "3d", "1f"], "B": ["s*", "1s", "2f", "3d", "1f"]}
#         assert orbital_mapper.basis == expected_basis
#         assert orbital_mapper.orbtype_count == {"s": 4, "p": 3, "d": 2, "f": 2}
# 
#     def test_get_orbpairtype_maps(self, orbital_mapper):
#         # 测试get_orbpairtype_maps方法是否正确创建了轨道对类型到切片的映射
#         expected_orbpairtype_maps = {
#             "s-s": slice(0, 9),
#             "s-p": slice(9, 45),
#             "p-p": slice(45, 81),
#             "d-d": slice(81, 105),
#             "f-f": slice(105, 119)
#         }
#         assert orbital_mapper.get_orbpairtype_maps() == expected_orbpairtype_maps
# 
#     def test_get_orbpair_maps(self, orbital_mapper):
#         # 测试get_orbpair_maps方法是否正确创建了轨道对到索引的映射
#         expected_orbpair_maps = {
#             "1s-1s": slice(0, 1),
#             "2s-1s": slice(1, 2),
#             "2p-1s": slice(2, 6),
#             "1d-1s": slice(6, 10),
#             "2d-1s": slice(10, 14),
#             "3d-1s": slice(14, 18),
#             "1f-1s": slice(18, 19),
#             # ... 其他映射 ...
#         }
#         assert orbital_mapper.get_orbpair_maps() == expected_orbpair_maps
# 
#     def test_get_skonsite_maps(self, orbital_mapper):
#         # 测试get_skonsite_maps方法是否正确创建了sk onsite映射
#         # 由于这个方法的输出依赖于内部的skonsitetype_maps，我们需要先确保skonsitetype_maps是正确的
#         expected_skonsite_maps = {
#             "A": {"s": slice(0, 4), "p": slice(4, 9), "d": slice(9, 14), "f": slice(14, 20)}
#         assert orbital_mapper.get_skonsite_maps() == expected_skonsite_maps
# 
#     def test_get_skonsitetype_maps(self, orbital_mapper):
#         # 测试get_skonsitetype_maps方法是否正确创建了sk onsite类型映射
#         expected_skonsitetype_maps = {
#             "s": slice(0, 4),
#             "p": slice(4, 9),
#             "d": slice(9, 14),
#             "f": slice(14, 20)
#         }
#         assert orbital_mapper.get_skonsitetype_maps() == expected_skonsitetype_maps
# 
#     def test_get_orbital_maps(self, orbital_mapper):
#         # 测试get_orbital_maps方法是否正确创建了轨道映射
#         expected_orbital_maps = {
#             "A": {"1s": slice(0, 1), "2s": slice(1, 2), "2p": slice(2, 6), "1d": slice(6, 10), "2d": slice(10, 14), "3d": slice(14, 18), "1f": slice(18, 19)},
#             "B": {"1s": slice(19, 20), "2f": slice(20, 22), "3d": slice(22, 26), "1f": slice(26, 27)}
#         assert orbital_mapper.get_orbital_maps() == expected_orbital_maps
# 
#     def test_get_irreps(self, orbital_mapper):
#         # 测试get_irreps方法是否正确创建了轨道对的不可约表示
#         # 这里我们需要模拟o3.Irreps的行为，因为在测试环境中我们可能没有o3模块
#         expected_irreps = ["some_irreps_list"]
#         with pytest.raises(NotImplementedError):
#             orbital_mapper.get_irreps()
# 
#     def test_equality_operator(self, orbital_mapper):
#         # 测试相等性操作符
#         other_mapper = OrbitalMapper(basis=orbital_mapper.basis, device=torch.device("cpu"))
#         assert orbital_mapper == other_mapper
# 
#         # 不同的basis或method应该不相等
#         other_mapper_different_basis = OrbitalMapper(basis={"A": "2s1p", "B": "1s2f"}, device=torch.device("cpu"))
#         assert orbital_mapper != other_mapper_different_basis
# 
#         other_mapper_different_method = OrbitalMapper(basis=orbital_mapper.basis, method="sktb", device=torch.device("cpu"))
#         assert orbital_mapper != other_mapper_different_method
# 
# # 运行测试
# if __name__ == "__main__":
#     pytest.main([__file__])