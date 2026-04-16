from typing import Dict, Optional, Union, List
from dptb.data.AtomicDataDict import Type
from dptb.utils.tools import get_uniq_symbol
from dptb.utils.constants import anglrMId
import re
import warnings

import torch

import ase.data
import e3nn.o3 as o3

from dptb.data import AtomicData, AtomicDataDict


class TypeMapper:
    """Based on a configuration, map atomic numbers to types."""

    num_types: int
    chemical_symbol_to_type: Optional[Dict[str, int]]
    type_to_chemical_symbol: Optional[Dict[int, str]]
    type_names: List[str]
    _min_Z: int

    def __init__(
        self,
        type_names: Optional[List[str]] = None,
        chemical_symbol_to_type: Optional[Dict[str, int]] = None,
        type_to_chemical_symbol: Optional[Dict[int, str]] = None,
        chemical_symbols: Optional[List[str]] = None,
        device=torch.device("cpu"),
    ):
        self.device = device
        if chemical_symbols is not None:
            if chemical_symbol_to_type is not None:
                raise ValueError(
                    "Cannot provide both `chemical_symbols` and `chemical_symbol_to_type`"
                )
            # repro old, sane NequIP behaviour
            # checks also for validity of keys
            atomic_nums = [ase.data.atomic_numbers[sym] for sym in chemical_symbols]
            # https://stackoverflow.com/questions/29876580/how-to-sort-a-list-according-to-another-list-python
            chemical_symbols = [
                e[1] for e in sorted(zip(atomic_nums, chemical_symbols)) # low to high
            ]
            chemical_symbol_to_type = {k: i for i, k in enumerate(chemical_symbols)}
            del chemical_symbols

        if type_to_chemical_symbol is not None:
            type_to_chemical_symbol = {
                int(k): v for k, v in type_to_chemical_symbol.items()
            }
            assert all(
                v in ase.data.chemical_symbols for v in type_to_chemical_symbol.values()
            )

        # Build from chem->type mapping, if provided
        self.chemical_symbol_to_type = chemical_symbol_to_type
        if self.chemical_symbol_to_type is not None:
            # Validate
            for sym, type in self.chemical_symbol_to_type.items():
                assert sym in ase.data.atomic_numbers, f"Invalid chemical symbol {sym}"
                assert 0 <= type, f"Invalid type number {type}"
            assert set(self.chemical_symbol_to_type.values()) == set(
                range(len(self.chemical_symbol_to_type))
            )
            if type_names is None:
                # Make type_names
                type_names = [None] * len(self.chemical_symbol_to_type)
                for sym, type in self.chemical_symbol_to_type.items():
                    type_names[type] = sym
            else:
                # Make sure they agree on types
                # We already checked that chem->type is contiguous,
                # so enough to check length since type_names is a list
                assert len(type_names) == len(self.chemical_symbol_to_type)
            # Make mapper array
            valid_atomic_numbers = [
                ase.data.atomic_numbers[sym] for sym in self.chemical_symbol_to_type
            ]
            self._min_Z = min(valid_atomic_numbers)
            self._max_Z = max(valid_atomic_numbers)
            Z_to_index = torch.full(
                size=(1 + self._max_Z - self._min_Z,), fill_value=-1, dtype=torch.long, device=device
            )
            for sym, type in self.chemical_symbol_to_type.items():
                Z_to_index[ase.data.atomic_numbers[sym] - self._min_Z] = type
            self._Z_to_index = Z_to_index
            self._index_to_Z = torch.zeros(
                size=(len(self.chemical_symbol_to_type),), dtype=torch.long, device=device
            )
            for sym, type_idx in self.chemical_symbol_to_type.items():
                self._index_to_Z[type_idx] = ase.data.atomic_numbers[sym]
            self._valid_set = set(valid_atomic_numbers)
            true_type_to_chemical_symbol = {
                type_id: sym for sym, type_id in self.chemical_symbol_to_type.items()
            }
            if type_to_chemical_symbol is not None:
                assert type_to_chemical_symbol == true_type_to_chemical_symbol
            else:
                type_to_chemical_symbol = true_type_to_chemical_symbol

        # check
        if type_names is None:
            raise ValueError(
                "None of chemical_symbols, chemical_symbol_to_type, nor type_names was provided; exactly one is required"
            )
        # validate type names
        assert all(
            n.isalnum() for n in type_names
        ), "Type names must contain only alphanumeric characters"
        # Set to however many maps specified -- we already checked contiguous
        self.num_types = len(type_names)
        # Check type_names
        self.type_names = type_names
        self.type_to_chemical_symbol = type_to_chemical_symbol
        if self.type_to_chemical_symbol is not None:
            assert set(type_to_chemical_symbol.keys()) == set(range(self.num_types))

    def __call__(
        self, data: Union[AtomicDataDict.Type, AtomicData], types_required: bool = True
    ) -> Union[AtomicDataDict.Type, AtomicData]:
        if AtomicDataDict.ATOM_TYPE_KEY in data:
            if AtomicDataDict.ATOMIC_NUMBERS_KEY in data:
                warnings.warn(
                    "Data contained both ATOM_TYPE_KEY and ATOMIC_NUMBERS_KEY; ignoring ATOMIC_NUMBERS_KEY"
                )
        elif AtomicDataDict.ATOMIC_NUMBERS_KEY in data:
            assert (
                self.chemical_symbol_to_type is not None
            ), "Atomic numbers provided but there is no chemical_symbols/chemical_symbol_to_type mapping!"
            atomic_numbers = data[AtomicDataDict.ATOMIC_NUMBERS_KEY]
            del data[AtomicDataDict.ATOMIC_NUMBERS_KEY]

            data[AtomicDataDict.ATOM_TYPE_KEY] = self.transform(atomic_numbers)
        else:
            if types_required:
                raise KeyError(
                    "Data doesn't contain any atom type information (ATOM_TYPE_KEY or ATOMIC_NUMBERS_KEY)"
                )
        return data

    def transform(self, atomic_numbers):
        """core function to transform an array to specie index list"""

        if atomic_numbers.min() < self._min_Z or atomic_numbers.max() > self._max_Z:
            bad_set = set(torch.unique(atomic_numbers).cpu().tolist()) - self._valid_set
            raise ValueError(
                f"Data included atomic numbers {bad_set} that are not part of the atomic number -> type mapping!"
            )

        types = self._Z_to_index.to(device=atomic_numbers.device)[atomic_numbers - self._min_Z]
        
        if -1 in types:
            bad_set = set(torch.unique(atomic_numbers).cpu().tolist()) - self._valid_set
            raise ValueError(
                f"Data included atomic numbers {bad_set} that are not part of the atomic number -> type mapping!"
            )

        return types

    def untransform(self, atom_types):
        """Transform atom types back into atomic numbers"""
        index_to_Z = self._index_to_Z.to(device=atom_types.device)
        return index_to_Z[atom_types]

    @property
    def has_chemical_symbols(self) -> bool:
        return self.chemical_symbol_to_type is not None

    @staticmethod
    def format(
        data: list, type_names: List[str], element_formatter: str = ".6f"
    ) -> str:
        """
            Formats a list of data elements along with their type names.

        Parameters:
            data (list): The data elements to be formatted. This should be a list of numbers.
            type_names (List[str]): The type names corresponding to the data elements. This should be a list of strings.
            element_formatter (str, optional): The format in which the data elements should be displayed. Defaults to ".6f".

        Returns:
            str: A string representation of the data elements along with their type names.

        Raises:
            ValueError: If `data` is not None, not 0-dimensional, or not 1-dimensional with length equal to the length of `type_names`.
        
        Example:
            >>> data = [1.123456789, 2.987654321]
            >>> type_names = ['Type1', 'Type2']
            >>> print(TypeMapper.format(data, type_names))
                [Type1: 1.123457, Type2: 2.987654]    
        """ 
        data = torch.as_tensor(data) if data is not None else None
        if data is None:
            return f"[{', '.join(type_names)}: None]"
        elif data.ndim == 0:
            return (f"[{', '.join(type_names)}: {{:{element_formatter}}}]").format(data)
        elif data.ndim == 1 and len(data) == len(type_names):
            return (
                "["
                + ", ".join(
                    f"{{{i}[0]}}: {{{i}[1]:{element_formatter}}}"
                    for i in range(len(data))
                )
                + "]"
            ).format(*zip(type_names, data))
        else:
            raise ValueError(
                f"Don't know how to format data=`{data}` for types {type_names} with element_formatter=`{element_formatter}`"
            )


class BondMapper(TypeMapper):
    def __init__(
            self, 
            chemical_symbols: Optional[List[str]] = None, 
            chemical_symbols_to_type:Union[Dict[str, int], None]=None,
            device=torch.device("cpu"),
            ):
        super(BondMapper, self).__init__(chemical_symbol_to_type=chemical_symbols_to_type, chemical_symbols=chemical_symbols, device=device)

        self.bond_types = [None] * self.num_types ** 2
        self.reduced_bond_types = [None] * ((self.num_types * (self.num_types + 1)) // 2)
        self.bond_to_type = {}
        self.type_to_bond = {}
        self.reduced_bond_to_type = {}
        self.type_to_reduced_bond = {}
        for asym, ai in self.chemical_symbol_to_type.items():
            for bsym, bi in self.chemical_symbol_to_type.items():
                self.bond_types[ai * self.num_types + bi] = asym + "-" + bsym
                if ai <= bi:
                    self.reduced_bond_types[(2*self.num_types-ai+1) * ai // 2 + bi-ai] = asym + "-" + bsym
        for i, bt in enumerate(self.bond_types):
            self.bond_to_type[bt] = i
            self.type_to_bond[i] = bt
        for i, bt in enumerate(self.reduced_bond_types):
            self.reduced_bond_to_type[bt] = i
            self.type_to_reduced_bond[i] = bt
        
        ZZ_to_index = torch.full(
                size=(len(self._Z_to_index), len(self._Z_to_index)), fill_value=-1, device=device, dtype=torch.long
            )
        ZZ_to_reduced_index = torch.full(
                size=(len(self._Z_to_index), len(self._Z_to_index)), fill_value=-1, device=device, dtype=torch.long
            )
        

        for abond, aidx in self.bond_to_type.items(): # type_names has a ascending order according to atomic number
            asym, bsym = abond.split("-")
            ZZ_to_index[ase.data.atomic_numbers[asym]-self._min_Z, ase.data.atomic_numbers[bsym]-self._min_Z] = aidx

        for abond, aidx in self.reduced_bond_to_type.items(): # type_names has a ascending order according to atomic number        
            asym, bsym = abond.split("-")
            ZZ_to_reduced_index[ase.data.atomic_numbers[asym]-self._min_Z, ase.data.atomic_numbers[bsym]-self._min_Z] = aidx
        
        
        self._ZZ_to_index = ZZ_to_index
        self._ZZ_to_reduced_index = ZZ_to_reduced_index

        self._index_to_ZZ = torch.zeros(
                size=(len(self.bond_to_type),2), dtype=torch.long, device=device
            )
        self._reduced_index_to_ZZ = torch.zeros(
                size=(len(self.reduced_bond_to_type),2), dtype=torch.long, device=device
            )
        
        for abond, aidx in self.bond_to_type.items():
            asym, bsym = abond.split("-")
            self._index_to_ZZ[aidx] = torch.tensor([ase.data.atomic_numbers[asym], ase.data.atomic_numbers[bsym]], dtype=torch.long, device=device)

        for abond, aidx in self.reduced_bond_to_type.items():
            asym, bsym = abond.split("-")
            self._reduced_index_to_ZZ[aidx] = torch.tensor([ase.data.atomic_numbers[asym], ase.data.atomic_numbers[bsym]], dtype=torch.long, device=device)


    def transform_atom(self, atomic_numbers):
        return self.transform(atomic_numbers)
    
    def transform_bond(self, iatomic_numbers, jatomic_numbers):
        
        if iatomic_numbers.device != jatomic_numbers.device:
            raise ValueError("iatomic_numbers and jatomic_numbers should be on the same device!")
        
        if iatomic_numbers.min() < self._min_Z or iatomic_numbers.max() > self._max_Z:
            bad_set = set(torch.unique(iatomic_numbers).cpu().tolist()) - self._valid_set
            raise ValueError(
                f"Data included atomic numbers {bad_set} that are not part of the atomic number -> type mapping!"
            )
        
        if jatomic_numbers.min() < self._min_Z or jatomic_numbers.max() > self._max_Z:
            bad_set = set(torch.unique(jatomic_numbers).cpu().tolist()) - self._valid_set
            raise ValueError(
                f"Data included atomic numbers {bad_set} that are not part of the atomic number -> type mapping!"
            )


        bondtypes = self._ZZ_to_index.to(device=iatomic_numbers.device)[iatomic_numbers - self._min_Z, 
                                                                    jatomic_numbers - self._min_Z]
        
        if -1 in bondtypes:
            bad_set1 = set(torch.unique(iatomic_numbers).cpu().tolist()) - self._valid_set
            bad_set2 = set(torch.unique(jatomic_numbers).cpu().tolist()) - self._valid_set
            bad_set = bad_set1.union(bad_set2)
            raise ValueError(
                f"Data included atomic numbers {bad_set} that are not part of the atomic number -> type mapping!"
            )
        
        return bondtypes
    
    def transform_reduced_bond(self, iatomic_numbers, jatomic_numbers):
        
        if iatomic_numbers.device != jatomic_numbers.device:
            raise ValueError("iatomic_numbers and jatomic_numbers should be on the same device!")
        
        if not torch.all((iatomic_numbers -jatomic_numbers)<=0):
            raise ValueError("iatomic_numbers[i] should <= jatomic_numbers[i]")

        if iatomic_numbers.min() < self._min_Z or iatomic_numbers.max() > self._max_Z:
            bad_set = set(torch.unique(iatomic_numbers).cpu().tolist()) - self._valid_set
            raise ValueError(
                f"Data included atomic numbers {bad_set} that are not part of the atomic number -> type mapping!"
            )
        
        if jatomic_numbers.min() < self._min_Z or jatomic_numbers.max() > self._max_Z:
            bad_set = set(torch.unique(jatomic_numbers).cpu().tolist()) - self._valid_set
            raise ValueError(
                f"Data included atomic numbers {bad_set} that are not part of the atomic number -> type mapping!"
            )


        red_bondtypes = self._ZZ_to_reduced_index.to(device=iatomic_numbers.device)[
                                    iatomic_numbers - self._min_Z, jatomic_numbers - self._min_Z]
        
        if -1 in red_bondtypes:
            bad_set1 = set(torch.unique(iatomic_numbers).cpu().tolist()) - self._valid_set
            bad_set2 = set(torch.unique(jatomic_numbers).cpu().tolist()) - self._valid_set
            bad_set = bad_set1.union(bad_set2)
            raise ValueError(
                f"Data included atomic numbers {bad_set} that are not part of the atomic number -> type mapping!"
            )
        
        return red_bondtypes
    
    def untransform_atom(self, atom_types):
        """Transform atom types back into atomic numbers"""
        return self.untransform(atom_types)
    
    def untransform_bond(self, bond_types):
        """Transform bond types back into atomic numbers"""
        index_to_ZZ = self._index_to_ZZ.to(device=bond_types.device)
        return index_to_ZZ[bond_types]

    def untransform_reduced_bond(self, bond_types):
        """Transform reduced bond types back into atomic numbers"""
        reduced_index_to_ZZ = self._reduced_index_to_ZZ.to(device=bond_types.device)
        return reduced_index_to_ZZ[bond_types]
    
    @property
    def has_bond(self) -> bool:
        return self.bond_to_type is not None
    
    def __call__(
            self, data: Union[AtomicDataDict.Type, AtomicData], types_required: bool = True
            ) -> Union[AtomicDataDict.Type, AtomicData]:
        if AtomicDataDict.EDGE_TYPE_KEY in data:
            if AtomicDataDict.ATOMIC_NUMBERS_KEY in data:
                warnings.warn(
                    "Data contained both EDGE_TYPE_KEY and ATOMIC_NUMBERS_KEY; ignoring ATOMIC_NUMBERS_KEY"
                )
        elif AtomicDataDict.ATOMIC_NUMBERS_KEY in data:
            assert (
                self.bond_to_type is not None
            ), "Atomic numbers provided but there is no chemical_symbols/chemical_symbol_to_type mapping!"
            atomic_numbers = data[AtomicDataDict.ATOMIC_NUMBERS_KEY]

            assert (
                AtomicDataDict.EDGE_INDEX_KEY in data
            ), "The bond type mapper need a EDGE index as input."

            data[AtomicDataDict.EDGE_TYPE_KEY] = \
                self.transform_bond(
                    atomic_numbers[data[AtomicDataDict.EDGE_INDEX_KEY][0]],
                    atomic_numbers[data[AtomicDataDict.EDGE_INDEX_KEY][1]]
                    )
        else:
            if types_required:
                raise KeyError(
                    "Data doesn't contain any atom type information (EDGE_TYPE_KEY or ATOMIC_NUMBERS_KEY)"
                )
        return super().__call__(data=data, types_required=types_required)


# orbital_mapper.py
import torch
import re
import logging
import sys
from typing import Dict, Union, List, Optional
from e3nn import o3

# ================== 依赖：BondMapper（若外部没提供就用 Mock） ==================
# 真实项目里一般来自:
# from dptb.data.transforms import BondMapper
if 'BondMapper' not in locals():
    class BondMapper:
        def __init__(self, chemical_symbols=None, chemical_symbol_to_type=None, device='cpu'):
            if chemical_symbol_to_type is not None:
                self.chemical_symbol_to_type = dict(chemical_symbol_to_type)
                inv = {t: s for s, t in self.chemical_symbol_to_type.items()}
                self.type_names = [inv[i] for i in range(len(inv))]
                self.chemical_symbols = list(self.type_names)
            else:
                self.chemical_symbols = chemical_symbols or []
                self.type_names = list(self.chemical_symbols)
                self.chemical_symbol_to_type = {s: i for i, s in enumerate(self.type_names)}

            self.device = device

            # 全排列 bond types
            self.bond_types = []
            for a in self.chemical_symbols:
                for b in self.chemical_symbols:
                    self.bond_types.append(f"{a}-{b}")
            self.bond_to_type = {b: i for i, b in enumerate(self.bond_types)}


anglrMId = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5}

# ================== 配置日志 ==================
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
log = logging.getLogger("OrbitalMapper")


class OrbitalMapper(BondMapper):
    def __init__(
            self,
            basis: Dict[str, Union[List[str], str]],
            chemical_symbol_to_type: Optional[Dict[str, int]] = None,
            method: str = "e3tb",
            device: Union[str, torch.device] = torch.device("cpu"),
            # ====== 新增：SOC（最小改动，不影响 non-SOC 默认行为）======
            has_soc: bool = False,
            soc_upper_triangle_bonds: bool = True,
            soc_complex_doubling: bool = True,
            ):

        """
        原始 OrbitalMapper（你贴的版本）+ SOC 最小改动：

        - non-SOC：默认 has_soc=False，逻辑与原来一致（包括 bond_types 全排列）
        - SOC：has_soc=True 时：
            * 默认只取 bond_types 上三角（species type i<=j），通过 soc_upper_triangle_bonds 控制
            * e3tb 的每个 (l1,l2) block 特征维度从 (2l1+1)(2l2+1) 扩展为：
                - spinful decomposition: 4 * (2l1+1)(2l2+1)  （Spatial + (Spatial x 1) 的维度恒等）
                - 若 soc_complex_doubling=True：再 *2 （Real + Imag）
            * get_irreps() 在原逻辑基础上最小扩展：SOC 时用 (Spatial + Spatialx1) 并可选 complex doubling

        注意：
        - 这里的 SOC 指“spinful / 非共线”那套输出（对应你之前 DeepH-E3 的 irreps_from_l1l2 思路）
        - non-SOC 输出与 reduced_matrix_element 的一致性保持不变
        """

        # TODO: use OrderedDict to fix the order of the dict used as index map
        if chemical_symbol_to_type is not None:
            assert set(basis.keys()) == set(chemical_symbol_to_type.keys())
            super(OrbitalMapper, self).__init__(chemical_symbol_to_type=chemical_symbol_to_type, device=device)
        else:
            super(OrbitalMapper, self).__init__(chemical_symbols=list(basis.keys()), device=device)

        self.basis = basis
        self.method = method
        self.device = device

        # ===== 新增：SOC flags =====
        self.has_soc = bool(has_soc)
        self.soc_upper_triangle_bonds = bool(soc_upper_triangle_bonds)
        self.soc_complex_doubling = bool(soc_complex_doubling)

        if self.method not in ["e3tb", "sktb"]:
            raise ValueError

        # ===== SOC 默认 bond_types 上三角（最小改动：只在 has_soc=True 时改变 bond_types）=====
        self.bond_types_full = list(self.bond_types)
        self.bond_to_type_full = dict(self.bond_to_type)
        if self.has_soc and self.soc_upper_triangle_bonds:
            self._apply_upper_triangle_bond_types()

        # ===== SOC feature 维度因子：spinful(4) * (complex doubling ? 2 : 1) =====
        # non-SOC: factor = 1
        self._soc_feature_factor = 1
        if self.has_soc:
            if self.method != "e3tb":
                # 最小改动：不在 sktb 上强行引入 SOC（避免 silently wrong）
                raise NotImplementedError("SOC branch currently only implemented for method='e3tb'")
            self._soc_feature_factor = 4 * (2 if self.soc_complex_doubling else 1)

        # ===== 原始逻辑：basis 解析 =====
        if isinstance(self.basis[self.type_names[0]], str):
            assert method == "e3tb", "The method should be e3tb when the basis is given as string."
            all_orb_types = []
            for iatom, ibasis in self.basis.items():
                letters = [letter for letter in ibasis if letter.isalpha()]
                all_orb_types = all_orb_types + letters
                if len(letters) != len(set(letters)):
                    raise ValueError(f"Duplicate orbitals found in the basis {ibasis} of atom {iatom}")
            all_orb_types = set(all_orb_types)
            orbtype_count = {"s": 0, "p": 0, "d": 0, "f": 0, "g": 0, "h": 0}

            if not all_orb_types.issubset(set(orbtype_count.keys())):
                raise ValueError(f"Invalid orbital types {all_orb_types} found in the basis. now only support {set(orbtype_count.keys())}.")

            orbs = map(lambda bs: re.findall(r'[1-9]+[A-Za-z]', bs), self.basis.values())
            for ib in orbs:
                for io in ib:
                    assert len(io) == 2
                    if int(io[0]) > orbtype_count[io[1]]:
                        orbtype_count[io[1]] = int(io[0])
            # split into list basis
            basis = {k: [] for k in self.type_names}
            for ib in self.basis.keys():
                for io in ["s", "p", "d", "f", "g", "h"]:
                    if io in self.basis[ib]:
                        basis[ib].extend([str(i) + io for i in range(1, int(re.findall(r'[1-9]+' + io, self.basis[ib])[0][0]) + 1)])
            self.basis = basis

        elif isinstance(self.basis[self.type_names[0]], list):
            nb = len(self.type_names)
            orbtype_count = {"s": [0] * nb, "p": [0] * nb, "d": [0] * nb, "f": [0] * nb, "g": [0] * nb, "h": [0] * nb}
            for ib, bt in enumerate(self.type_names):
                for io in self.basis[bt]:
                    orb = re.findall(r'[A-Za-z]', io)[0]
                    orbtype_count[orb][ib] += 1

            for ko in orbtype_count.keys():
                orbtype_count[ko] = max(orbtype_count[ko])
        else:
            raise ValueError(f"Invalid basis {self.basis} found. now only support string or list basis.")

        self.orbtype_count = orbtype_count

        # ===== 原始逻辑：full_basis_norb =====
        full_basis_norb = 0
        for ko in orbtype_count.keys():
            assert ko in anglrMId
            full_basis_norb = full_basis_norb + (2 * anglrMId[ko] + 1) * orbtype_count[ko]
        self.full_basis_norb = full_basis_norb

        # ===== 原始逻辑：reduced_matrix_element（SOC 时按 factor 扩展，其他不动）=====
        if self.method == "e3tb":
            total_onsite_block_elements = 0
            for ko in orbtype_count.keys():
                total_onsite_block_elements += orbtype_count[ko] * (2 * anglrMId[ko] + 1) ** 2

            base_rme = int((self.full_basis_norb ** 2 + total_onsite_block_elements) / 2)

            # SOC 最小改动：只在 has_soc=True 时按因子扩大
            self.reduced_matrix_element = base_rme * self._soc_feature_factor
        else:
            # sktb 原始逻辑不改（且 SOC 不允许到这里）
            self.reduced_matrix_element = (
                1 * orbtype_count["s"] * orbtype_count["s"] +
                2 * orbtype_count["s"] * orbtype_count["p"] +
                2 * orbtype_count["s"] * orbtype_count["d"] +
                2 * orbtype_count["s"] * orbtype_count["f"] +
                2 * orbtype_count["s"] * orbtype_count["g"] +
                2 * orbtype_count["s"] * orbtype_count["h"]
            ) + \
            2 * (
                1 * orbtype_count["p"] * orbtype_count["p"] +
                2 * orbtype_count["p"] * orbtype_count["d"] +
                2 * orbtype_count["p"] * orbtype_count["f"] +
                2 * orbtype_count["p"] * orbtype_count["g"] +
                2 * orbtype_count["p"] * orbtype_count["h"]
            ) + \
            3 * (
                1 * orbtype_count["d"] * orbtype_count["d"] +
                2 * orbtype_count["d"] * orbtype_count["f"] +
                2 * orbtype_count["d"] * orbtype_count["g"] +
                2 * orbtype_count["d"] * orbtype_count["h"]
            ) + \
            4 * (
                1 * orbtype_count["f"] * orbtype_count["f"] +
                2 * orbtype_count["f"] * orbtype_count["g"] +
                2 * orbtype_count["f"] * orbtype_count["h"]
            ) + \
            5 * (
                1 * orbtype_count["g"] * orbtype_count["g"] +
                2 * orbtype_count["g"] * orbtype_count["h"]
            ) + \
            6 * (
                1 * orbtype_count["h"] * orbtype_count["h"]
            )

            self.reduced_matrix_element = self.reduced_matrix_element + orbtype_count["s"] + 2 * orbtype_count["p"] + 3 * orbtype_count["d"] + 4 * orbtype_count["f"] + 5 * orbtype_count["g"] + 6 * orbtype_count["h"]
            self.reduced_matrix_element = int(self.reduced_matrix_element / 2)
            self.n_onsite_Es = 0.5 * (orbtype_count["s"] ** 2 + orbtype_count["s"]) \
                               + 0.5 * (orbtype_count["p"] ** 2 + orbtype_count["p"]) \
                               + 0.5 * (orbtype_count["d"] ** 2 + orbtype_count["d"]) \
                               + 0.5 * (orbtype_count["f"] ** 2 + orbtype_count["f"]) \
                               + 0.5 * (orbtype_count["g"] ** 2 + orbtype_count["g"]) \
                               + 0.5 * (orbtype_count["h"] ** 2 + orbtype_count["h"])
            self.n_onsite_Es = int(self.n_onsite_Es)
            self.n_onsite_socLs = orbtype_count["s"] + orbtype_count["p"] + orbtype_count["d"] + orbtype_count["f"] + orbtype_count["g"] + orbtype_count["h"]

        # ===== 原始逻辑：sort basis =====
        for ib in self.basis.keys():
            self.basis[ib] = sorted(
                self.basis[ib],
                key=lambda s: (
                    anglrMId[re.findall(r"[a-z]", s)[0]],
                    re.findall(r"[1-9*]", s)[0] if re.findall(r"[1-9*]", s) else '0'
                )
            )

        # ===== 原始逻辑：full basis =====
        full_basis = []
        for io in ["s", "p", "d", "f", "g", "h"]:
            full_basis = full_basis + [str(i) + io for i in range(1, orbtype_count[io] + 1)]
        self.full_basis = full_basis

        # ===== 原始逻辑：basis_to_full_basis =====
        self.basis_to_full_basis = {}
        self.atom_norb = torch.zeros(len(self.type_names), dtype=torch.long, device=self.device)
        for ib in self.basis.keys():
            count_dict = {"s": 0, "p": 0, "d": 0, "f": 0, "g": 0, "h": 0}
            self.basis_to_full_basis.setdefault(ib, {})
            for o in self.basis[ib]:
                io = re.findall(r"[a-z]", o)[0]
                l = anglrMId[io]
                count_dict[io] += 1
                self.atom_norb[self.chemical_symbol_to_type[ib]] += 2 * l + 1
                self.basis_to_full_basis[ib][o] = str(count_dict[io]) + io

        # ===== 原始逻辑：full_basis_to_basis =====
        self.full_basis_to_basis = {}
        for at, maps in self.basis_to_full_basis.items():
            self.full_basis_to_basis[at] = {}
            for k, v in maps.items():
                self.full_basis_to_basis[at].update({v: k})

        # ===== 原始逻辑：mask_to_basis =====
        self.mask_to_basis = torch.zeros(len(self.type_names), self.full_basis_norb, device=self.device, dtype=torch.bool)

        for ib in self.basis.keys():
            ibasis = list(self.basis_to_full_basis[ib].values())
            ist = 0
            for io in self.full_basis:
                l = anglrMId[io[1]]
                if io in ibasis:
                    self.mask_to_basis[self.chemical_symbol_to_type[ib]][ist:ist + 2 * l + 1] = True
                ist += 2 * l + 1

        assert (self.mask_to_basis.sum(dim=1).int() - self.atom_norb).abs().sum() <= 1e-6

        self.get_orbpair_maps()

        # ===== 原始逻辑：mask_to_erme / mask_to_nrme =====
        self.mask_to_erme = torch.zeros(len(self.bond_types), self.reduced_matrix_element, dtype=torch.bool, device=self.device)
        self.mask_to_nrme = torch.zeros(len(self.type_names), self.reduced_matrix_element, dtype=torch.bool, device=self.device)
        for ib, bb in self.basis.items():
            for io in bb:
                iof = self.basis_to_full_basis[ib][io]
                for jo in bb:
                    jof = self.basis_to_full_basis[ib][jo]
                    if self.orbpair_maps.get(iof + "-" + jof) is not None:
                        self.mask_to_nrme[self.chemical_symbol_to_type[ib]][self.orbpair_maps[iof + "-" + jof]] = True

        for ib in self.bond_to_type.keys():
            a, b = ib.split("-")
            for io in self.basis[a]:
                iof = self.basis_to_full_basis[a][io]
                for jo in self.basis[b]:
                    jof = self.basis_to_full_basis[b][jo]
                    if self.orbpair_maps.get(iof + "-" + jof) is not None:
                        self.mask_to_erme[self.bond_to_type[ib]][self.orbpair_maps[iof + "-" + jof]] = True
                    elif self.orbpair_maps.get(jof + "-" + iof) is not None:
                        self.mask_to_erme[self.bond_to_type[ib]][self.orbpair_maps[jof + "-" + iof]] = True

        # ===== 原始逻辑：mask_to_ndiag / full_mask_to_diag（e3tb）=====
        if self.method == "e3tb":
            self.mask_to_ndiag = torch.zeros(len(self.type_names), self.reduced_matrix_element, dtype=torch.bool, device=self.device)
            for ib, bb in self.basis.items():
                for io in bb:
                    iof = self.basis_to_full_basis[ib][io]
                    if self.orbpair_maps.get(iof + "-" + iof) is not None:
                        sli = self.orbpair_maps[iof + "-" + iof]
                        l = anglrMId[re.findall("[a-z]", iof)[0]]
                        indices = torch.arange(2 * l + 1)
                        indices = indices + indices * (2 * l + 1)
                        indices += sli.start
                        assert indices.max() < sli.stop
                        self.mask_to_ndiag[self.chemical_symbol_to_type[ib]][indices] = True

            self.full_mask_to_diag = torch.zeros(self.reduced_matrix_element, dtype=torch.bool, device=self.device)
            for orbs, islice in self.orbpair_maps.items():
                fio, fjo = orbs.split('-')
                if fio == fjo:
                    self.full_mask_to_diag[islice] = True

        log.info(
            f"Initialized OrbitalMapper(method={self.method}, has_soc={self.has_soc}, "
            f"soc_upper_triangle_bonds={self.soc_upper_triangle_bonds}, soc_complex_doubling={self.soc_complex_doubling})"
        )

    # ================== 新增：SOC bond types 上三角筛选 ==================
    def _apply_upper_triangle_bond_types(self) -> None:
        sym_to_idx = {s: i for i, s in enumerate(self.type_names)}

        kept = []
        for b in self.bond_types:
            a, c = b.split("-")
            if sym_to_idx[a] <= sym_to_idx[c]:
                kept.append(b)

        self.bond_types = kept
        self.bond_to_type = {b: i for i, b in enumerate(self.bond_types)}

    def get_orbpairtype_maps(self):
        """
        原始逻辑 + SOC 最小改动：
        - e3tb: n_rme = (2l+1)(2l'+1) * soc_factor
        - sktb: 原样
        """
        self.orbpairtype_maps = {}
        ist = 0
        for i, io in enumerate(["s", "p", "d", "f", "g", "h"]):
            if self.orbtype_count[io] != 0:
                for jo in ["s", "p", "d", "f", "g", "h"][i:]:
                    if self.orbtype_count[jo] != 0:
                        orb_pair = io + "-" + jo
                        il, jl = anglrMId[io], anglrMId[jo]
                        if self.method == "e3tb":
                            n_rme = (2 * il + 1) * (2 * jl + 1) * self._soc_feature_factor
                        else:
                            n_rme = min(il, jl) + 1
                        numhops = self.orbtype_count[io] * self.orbtype_count[jo] * n_rme
                        if io == jo:
                            numhops += self.orbtype_count[jo] * n_rme
                            numhops = int(numhops / 2)
                        self.orbpairtype_maps[orb_pair] = slice(ist, ist + numhops)
                        ist += numhops

        return self.orbpairtype_maps

    def get_orbpair_maps(self):

        if hasattr(self, "orbpair_maps"):
            return self.orbpair_maps

        if not hasattr(self, "orbpairtype_maps"):
            self.get_orbpairtype_maps()

        self.orbpair_maps = {}
        for i, io in enumerate(self.full_basis):
            for jo in self.full_basis[i:]:
                full_basis_pair = io + "-" + jo
                ir, jr = int(full_basis_pair[0]), int(full_basis_pair[3])
                iio, jjo = full_basis_pair[1], full_basis_pair[4]
                il, jl = anglrMId[iio], anglrMId[jjo]

                if self.method == "e3tb":
                    n_feature = (2 * il + 1) * (2 * jl + 1) * self._soc_feature_factor
                else:
                    n_feature = min(il, jl) + 1

                if iio == jjo:
                    start = self.orbpairtype_maps[iio + "-" + jjo].start + \
                            n_feature * ((2 * self.orbtype_count[jjo] + 2 - ir) * (ir - 1) / 2 + (jr - ir))
                else:
                    start = self.orbpairtype_maps[iio + "-" + jjo].start + \
                            n_feature * ((ir - 1) * self.orbtype_count[jjo] + (jr - 1))
                start = int(start)
                self.orbpair_maps[io + "-" + jo] = slice(start, start + n_feature)

        return self.orbpair_maps

    def get_orbpair_soc_maps(self):
        # 原函数中 hasattr 检查写错了（orbpairt_soc_maps），这里做最小修正避免重复计算
        if hasattr(self, "orbpair_soc_maps"):
            return self.orbpair_soc_maps

        self.orbpair_soc_maps = {}
        ist = 0
        for i, io in enumerate(self.full_basis):
            full_basis_pair = io + "-" + io   # io-io only
            ir, jr = int(full_basis_pair[0]), int(full_basis_pair[3])
            iio, jjo = full_basis_pair[1], full_basis_pair[4]
            il, jl = anglrMId[iio], anglrMId[jjo]

            if self.method == 'e3tb':
                n_feature = int((2 * il + 1) * (2 * jl + 1) * 4 / 2)
            else:
                raise NotImplementedError
            ist = int(ist)
            self.orbpair_soc_maps[full_basis_pair] = slice(ist, ist + n_feature)
            ist += n_feature

        reduced_soc_matrix_elemet = 0
        for ko in self.orbtype_count.keys():
            reduced_soc_matrix_elemet += self.orbtype_count[ko] * (2 * anglrMId[ko] + 1) ** 2 * 4 / 2

        self.reduced_soc_matrix_elemet = int(reduced_soc_matrix_elemet)
        return self.orbpair_soc_maps

    def get_skonsite_maps(self):
        assert self.method == "sktb", "Only sktb orbitalmapper have skonsite maps."

        if hasattr(self, "skonsite_maps"):
            return self.skonsite_maps

        if not hasattr(self, "skonsitetype_maps"):
            self.get_skonsitetype_maps()

        self.mask_diag = torch.zeros(self.n_onsite_Es, dtype=torch.bool, device=self.device)
        self.skonsite_maps = {}
        for i, io in enumerate(self.full_basis):
            for j, jo in enumerate(self.full_basis[i:]):
                ir, jr = int(io[0]), int(jo[0])
                iio, jjo = io[1], jo[1]
                if iio == jjo:
                    full_basis_pair = io + "-" + jo
                    start = int(self.skonsitetype_maps[iio].start + ((2 * self.orbtype_count[jjo] - ir + 2) * (ir - 1) / 2 + jr - ir))
                    self.skonsite_maps[full_basis_pair] = slice(start, start + 1)
                    if io == jo:
                        self.mask_diag[start] = True

        return self.skonsite_maps

    def get_skonsitetype_maps(self):
        self.skonsitetype_maps = {}
        ist = 0

        assert self.method == "sktb", "Only sktb orbitalmapper have skonsite maps."
        for i, io in enumerate(["s", "p", "d", "f", "g", "h"]):
            if self.orbtype_count[io] != 0:
                numonsites = int(0.5 * (self.orbtype_count[io] ** 2 + self.orbtype_count[io]))
                self.skonsitetype_maps[io] = slice(ist, ist + numonsites)
                ist += numonsites

        return self.skonsitetype_maps

    def get_sksoctype_maps(self):
        self.sksoctype_maps = {}
        ist = 0

        assert self.method == "sktb", "Only sktb orbitalmapper have sksoctype maps."
        for i, io in enumerate(["s", "p", "d", "f", "g", "h"]):
            if self.orbtype_count[io] != 0:
                numonsites = self.orbtype_count[io]
                self.sksoctype_maps[io] = slice(ist, ist + numonsites)
                ist += numonsites

        return self.sksoctype_maps

    def get_sksoc_maps(self):
        assert self.method == "sktb", "Only sktb orbitalmapper have sksoc maps."

        if hasattr(self, "sksoc_maps"):
            return self.sksoc_maps

        if not hasattr(self, "sksoctype_maps"):
            self.get_sksoctype_maps()

        self.sksoc_maps = {}
        for i, io in enumerate(self.full_basis):
            ir = int(io[0])
            iio = io[1]
            start = int(self.sksoctype_maps[iio].start + (ir - 1))
            self.sksoc_maps[io] = slice(start, start + 1)

        return self.sksoc_maps

    def get_orbital_maps(self):
        self.orbital_maps = {}
        self.norbs = {}

        for ib in self.basis.keys():
            orbital_list = self.basis[ib]
            slices = {}
            start_index = 0

            self.norbs.setdefault(ib, 0)
            for orb in orbital_list:
                orb_l = re.findall(r'[A-Za-z]', orb)[0]
                increment = (2 * anglrMId[orb_l] + 1)
                self.norbs[ib] += increment
                end_index = start_index + increment

                slices[orb] = slice(start_index, end_index)
                start_index = end_index

            self.orbital_maps[ib] = slices

        return self.orbital_maps

    # ================== 关键：get_irreps（non-SOC 保持原样，SOC 最小扩展） ==================
    def get_irreps(self, no_parity: bool = False):
        assert self.method == "e3tb", "Only support e3tb method for now."

        # cache：把 SOC 参数也纳入 key（不影响 non-SOC）
        cache_key = (no_parity, self.has_soc, self.soc_complex_doubling, self._soc_feature_factor)
        if hasattr(self, "_orbpair_irreps_cache") and self._orbpair_irreps_cache.get("key") == cache_key:
            return self._orbpair_irreps_cache["irreps"]

        if not hasattr(self, "orbpairtype_maps"):
            self.get_orbpairtype_maps()

        # parity factor 与原实现一致
        factor = 1 if no_parity else -1

        irs = []

        # 遍历 pairtype slice（原逻辑）
        for pair, sli in self.orbpairtype_maps.items():
            l1, l2 = anglrMId[pair[0]], anglrMId[pair[2]]
            p = factor ** (l1 + l2)

            if not self.has_soc:
                # ===== 原始 non-SOC 逻辑（保持一致）=====
                required_ls = range(abs(l1 - l2), l1 + l2 + 1)
                required_irreps = [(1, (l, p)) for l in required_ls]
                nblock = int((sli.stop - sli.start) / (2 * l1 + 1) / (2 * l2 + 1))
                irs += required_irreps * nblock
            else:
                # ===== SOC 最小扩展：Spatial + (Spatial x 1)，并可选 complex doubling =====
                base_ls = list(range(abs(l1 - l2), l1 + l2 + 1))
                block_list = [(1, (L, p)) for L in base_ls]  # spatial

                # add spatial x 1
                for L in base_ls:
                    for Ls in range(abs(L - 1), L + 2):
                        block_list.append((1, (Ls, p)))

                # complex doubling：Real + Imag（与 DeepH-E3 思路一致）
                if self.soc_complex_doubling:
                    block_list = block_list + block_list

                block_irreps = o3.Irreps(block_list)
                block_dim = block_irreps.dim
                nblock = int((sli.stop - sli.start) / block_dim)

                # 仍按“重复 block 的 irreps 序列”方式构造（与原逻辑一致的风格）
                irs += block_list * nblock

        self.orbpair_irreps = o3.Irreps(irs)

        self._orbpair_irreps_cache = {"key": cache_key, "irreps": self.orbpair_irreps}
        return self.orbpair_irreps

    def get_irreps_sim(self, no_parity: bool = False):
        return self.get_irreps(no_parity=no_parity).sort()[0].simplify()

    def get_irreps_ess(self, no_parity: bool = False):
        irp_e = []
        for mul, (l, p) in self.get_irreps_sim(no_parity=no_parity):
            if (-1) ** l == p:
                irp_e.append((mul, (l, p)))
        return o3.Irreps(irp_e)

    def __eq__(self, other):
        return self.basis == other.basis and self.method == other.method
