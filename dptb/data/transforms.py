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
                e[1] for e in sorted(zip(atomic_nums, chemical_symbols))  # low to high
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
            chemical_symbols_to_type: Union[Dict[str, int], None] = None,
            device=torch.device("cpu"),
    ):
        super(BondMapper, self).__init__(chemical_symbol_to_type=chemical_symbols_to_type,
                                         chemical_symbols=chemical_symbols, device=device)

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
                    self.reduced_bond_types[(2 * self.num_types - ai + 1) * ai // 2 + bi - ai] = asym + "-" + bsym
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

        for abond, aidx in self.bond_to_type.items():  # type_names has a ascending order according to atomic number
            asym, bsym = abond.split("-")
            ZZ_to_index[ase.data.atomic_numbers[asym] - self._min_Z, ase.data.atomic_numbers[bsym] - self._min_Z] = aidx

        for abond, aidx in self.reduced_bond_to_type.items():  # type_names has a ascending order according to atomic number
            asym, bsym = abond.split("-")
            ZZ_to_reduced_index[
                ase.data.atomic_numbers[asym] - self._min_Z, ase.data.atomic_numbers[bsym] - self._min_Z] = aidx

        self._ZZ_to_index = ZZ_to_index
        self._ZZ_to_reduced_index = ZZ_to_reduced_index

        self._index_to_ZZ = torch.zeros(
            size=(len(self.bond_to_type), 2), dtype=torch.long, device=device
        )
        self._reduced_index_to_ZZ = torch.zeros(
            size=(len(self.reduced_bond_to_type), 2), dtype=torch.long, device=device
        )

        for abond, aidx in self.bond_to_type.items():
            asym, bsym = abond.split("-")
            self._index_to_ZZ[aidx] = torch.tensor([ase.data.atomic_numbers[asym], ase.data.atomic_numbers[bsym]],
                                                   dtype=torch.long, device=device)

        for abond, aidx in self.reduced_bond_to_type.items():
            asym, bsym = abond.split("-")
            self._reduced_index_to_ZZ[aidx] = torch.tensor(
                [ase.data.atomic_numbers[asym], ase.data.atomic_numbers[bsym]], dtype=torch.long, device=device)

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

        if not torch.all((iatomic_numbers - jatomic_numbers) <= 0):
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


import torch
import re
import logging
import sys
import inspect
from typing import Dict, Union, List, Optional
from e3nn import o3

# ================== Constants ==================
anglrMId = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}

# ================== Logging ==================
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s', stream=sys.stdout, force=True)
log = logging.getLogger("OrbitalMapper")


# ================== Helper Functions ==================
def irreps_from_l1l2(l1, l2, spinful, no_parity=False):
    """
    Generates Irreps for the interaction between angular momenta l1 and l2.
    Follows the DeepH style: Spatial x Spin.
    """
    mul = 1
    p = 1
    if not no_parity:
        p = (-1) ** (l1 + l2)

    # 1. Spatial part: |l1 - l2| to |l1 + l2|
    l_min = abs(l1 - l2)
    l_max = l1 + l2
    required_ls = range(l_min, l_max + 1)
    base_irreps_list = [(mul, (l, p)) for l in required_ls]
    required_irreps = o3.Irreps(base_irreps_list)

    required_irreps_full = required_irreps

    # 2. SOC Spin Expansion (Spatial x 1/2 x 1/2 -> Spatial x 0 + Spatial x 1)
    if spinful:
        extra_irreps = []
        for _, ir in required_irreps:
            l_base = ir.l
            # Spin 1 component addition (l-1, l, l+1)
            s_min = abs(l_base - 1)
            s_max = l_base + 1
            for l_s in range(s_min, s_max + 1):
                extra_irreps.append((mul, (l_s, p)))
        required_irreps_full = required_irreps + o3.Irreps(extra_irreps)

    return required_irreps_full


# ================== OrbitalMapper Class ==================
class OrbitalMapper(BondMapper):
    def __init__(
            self,
            basis: Dict[str, Union[List[str], str]],
            chemical_symbol_to_type: Optional[Dict[str, int]] = None,
            method: str = "e3tb",
            device: Union[str, torch.device] = torch.device("cpu"),
            # SOC Flags (Primarily for e3tb)
            has_soc: bool = False,
            soc_complex_doubling: bool = True,
            nextham_uureal_mask: bool = False,  # <--- 新增
    ):
        """
        Maps orbital pairs to feature indices (Reduced Matrix Elements).
        Supports both 'e3tb' (DeepH-like) and 'sktb' (Slater-Koster) methods.
        """

        # --- Stack Inspection for Logging Caller ---
        try:
            # stack[0] is here, stack[1] is the caller
            stack = inspect.stack()
            if len(stack) > 1:
                caller = stack[1]
                log.info(
                    f"OrbitalMapper initialized from: {caller.filename} (Line {caller.lineno}) via function '{caller.function}'")
            else:
                log.info("OrbitalMapper initialized (caller unknown)")
        except Exception as e:
            log.warning(f"Could not inspect call stack for OrbitalMapper: {e}")
        # --------------------------------------------

        if chemical_symbol_to_type is not None:
            assert set(basis.keys()) == set(chemical_symbol_to_type.keys())
            # [FIX] BondMapper defined the arg as 'chemical_symbols_to_type' (plural symbols)
            # We map our 'chemical_symbol_to_type' (singular) to it.
            super(OrbitalMapper, self).__init__(chemical_symbols_to_type=chemical_symbol_to_type, device=device)
        else:
            super(OrbitalMapper, self).__init__(chemical_symbols=list(basis.keys()), device=device)

        self.basis = basis
        self.method = method
        self.device = device
        self.has_soc = has_soc
        self.soc_complex_doubling = soc_complex_doubling if has_soc else False
        self.nextham_uureal_mask = nextham_uureal_mask
        if self.method not in ["e3tb", "sktb"]:
            raise ValueError(f"Unknown method {self.method}, only 'e3tb' and 'sktb' are supported.")

        # ==========================================
        # 1. Unified Basis Parsing (Super Basis Logic)
        # ==========================================
        # Parse string format basis (e.g., "2s2p") into list format (["1s", "2s", "1p", "2p"])
        if isinstance(self.basis[self.type_names[0]], str):
            basis_parsed = {k: [] for k in self.type_names}
            for ib in self.basis.keys():
                val = self.basis[ib]
                for io in ["s", "p", "d", "f", "g", "h"]:
                    # Find number before orbital letter, e.g., '2' in '2s'
                    match = re.search(r'(\d+)' + io, val)
                    if match:
                        count = int(match.group(1))
                        basis_parsed[ib].extend([str(i) + io for i in range(1, count + 1)])
            self.basis = basis_parsed

        # Count max orbitals per type across all atoms (Super Basis Count)
        nb = len(self.type_names)
        orbtype_count = {"s": 0, "p": 0, "d": 0, "f": 0, "g": 0, "h": 0}

        # Count per atom first
        orbtype_count_per_atom = {k: [0] * nb for k in orbtype_count}
        for ib, bt in enumerate(self.type_names):
            for io in self.basis[bt]:
                orb = re.findall(r'[A-Za-z]', io)[0]
                orbtype_count_per_atom[orb][ib] += 1

        # Take the maximum count to form the Super Basis
        for ko in orbtype_count.keys():
            orbtype_count[ko] = max(orbtype_count_per_atom[ko])
        self.orbtype_count = orbtype_count

        # Construct Full Basis list (e.g., [1s, 2s, 3s, 1p, ...])
        full_basis = []
        full_basis_norb = 0
        for io in ["s", "p", "d", "f", "g", "h"]:
            count = orbtype_count[io]
            if count > 0:
                full_basis.extend([str(i) + io for i in range(1, count + 1)])
                full_basis_norb += (2 * anglrMId[io] + 1) * count

        self.full_basis = full_basis
        self.full_basis_norb = full_basis_norb

        # ==========================================
        # 2. Reduced Matrix Element (RME) Calculation
        # ==========================================
        if self.method == "e3tb":
            if not self.has_soc:
                # [Non-SOC]: Compressed storage (Upper triangle including diagonal blocks)
                # Count = (Full_N^2 + Sum(Onsite_Block_Diags)) / 2
                total_onsite_block_elements = 0
                for ko in orbtype_count.keys():
                    total_onsite_block_elements += orbtype_count[ko] * (2 * anglrMId[ko] + 1) ** 2
                self.reduced_matrix_element = int((self.full_basis_norb ** 2 + total_onsite_block_elements) / 2)
            else:
                # [SOC]: Full matrix storage
                # Count = N^2 * 4 (Spin) * 2 (if Complex Doubling)
                factor = 4 * (2 if self.soc_complex_doubling else 1)
                self.reduced_matrix_element = self.full_basis_norb ** 2 * factor

        else:
            # [SKTB]: Strict restoration of SKTB counting logic
            # Count = Sum of SK integrals (min(l,l')+1) for all pairs in upper triangle.
            rme_count = 0
            types = ["s", "p", "d", "f", "g", "h"]
            for i, io in enumerate(types):
                if orbtype_count[io] == 0: continue
                for j in range(i, len(types)):
                    jo = types[j]
                    if orbtype_count[jo] == 0: continue

                    il, jl = anglrMId[io], anglrMId[jo]
                    n_sk = min(il, jl) + 1  # Number of SK integrals

                    # Total pairs between type i and type j
                    pair_hops = orbtype_count[io] * orbtype_count[jo] * n_sk

                    # If diagonal block type (e.g. s-s), we only store upper triangle
                    if io == jo:
                        # (N*N + N)/2 logic, derived as:
                        pair_hops += orbtype_count[io] * n_sk
                        pair_hops = int(pair_hops / 2)

                    rme_count += pair_hops

            self.reduced_matrix_element = rme_count

            # SKTB specific: Onsite Energy and SOC parameter counts
            self.n_onsite_Es = 0
            self.n_onsite_socLs = 0
            for ko in orbtype_count.keys():
                cnt = orbtype_count[ko]
                # Onsite Energies: (N^2 + N)/2 -> Upper triangle of onsite block
                self.n_onsite_Es += int(0.5 * (cnt ** 2 + cnt))
                # Onsite SOC Strengths: 1 parameter per orbital shell
                self.n_onsite_socLs += cnt

            self.n_onsite_Es = int(self.n_onsite_Es)

        # ==========================================
        # 3. Maps & Masks Initialization
        # ==========================================

        # 3.1 Sort atom-specific basis to match standard order
        for ib in self.basis.keys():
            self.basis[ib] = sorted(
                self.basis[ib],
                key=lambda s: (anglrMId[re.findall(r"[a-z]", s)[0]],
                               re.findall(r"[1-9*]", s)[0] if re.findall(r"[1-9*]", s) else '0')
            )

        # 3.2 Map Basis <-> Full Basis
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

        self.full_basis_to_basis = {}
        for at, maps in self.basis_to_full_basis.items():
            self.full_basis_to_basis[at] = {v: k for k, v in maps.items()}

        # 3.3 Mask to Basis (Super Basis Mask)
        self.mask_to_basis = torch.zeros(len(self.type_names), self.full_basis_norb, device=self.device,
                                         dtype=torch.bool)
        for ib in self.basis.keys():
            ibasis = list(self.basis_to_full_basis[ib].values())
            ist = 0
            for io in self.full_basis:
                l = anglrMId[io[1]]
                if io in ibasis:
                    self.mask_to_basis[self.chemical_symbol_to_type[ib]][ist:ist + 2 * l + 1] = True
                ist += 2 * l + 1

        # 3.4 Initialize Orbpair Maps (Core Logic)
        self.get_orbpair_maps()

        # 3.5 Masks for RME (ERME/NRME)
        self.mask_to_erme = torch.zeros(len(self.bond_types), self.reduced_matrix_element, dtype=torch.bool,
                                        device=self.device)
        self.mask_to_nrme = torch.zeros(len(self.type_names), self.reduced_matrix_element, dtype=torch.bool,
                                        device=self.device)

        # Onsite blocks (NRME)
        for ib, bb in self.basis.items():
            for io in bb:
                iof = self.basis_to_full_basis[ib][io]
                for jo in bb:
                    jof = self.basis_to_full_basis[ib][jo]
                    # Note: For SKTB/Non-SOC E3TB, only upper triangle keys exist in maps
                    if self.orbpair_maps.get(iof + "-" + jof) is not None:
                        self.mask_to_nrme[self.chemical_symbol_to_type[ib]][self.orbpair_maps[iof + "-" + jof]] = True

        # Hopping blocks (ERME)
        for ib in self.bond_to_type.keys():
            a, b = ib.split("-")
            for io in self.basis[a]:
                iof = self.basis_to_full_basis[a][io]
                for jo in self.basis[b]:
                    jof = self.basis_to_full_basis[b][jo]

                    # Try forward lookup
                    if self.orbpair_maps.get(iof + "-" + jof) is not None:
                        self.mask_to_erme[self.bond_to_type[ib]][self.orbpair_maps[iof + "-" + jof]] = True
                    # If symmetric storage (SKTB or Non-SOC E3TB), try reverse lookup
                    elif (not self.has_soc) and self.orbpair_maps.get(jof + "-" + iof) is not None:
                        self.mask_to_erme[self.bond_to_type[ib]][self.orbpair_maps[jof + "-" + iof]] = True

        # Special Diagonal Mask for Non-SOC E3TB (SKTB handles this in get_skonsite_maps)
        if self.method == "e3tb" and not self.has_soc:
            self.mask_to_ndiag = torch.zeros(len(self.type_names), self.reduced_matrix_element, dtype=torch.bool,
                                             device=self.device)
            for ib, bb in self.basis.items():
                for io in bb:
                    iof = self.basis_to_full_basis[ib][io]
                    if self.orbpair_maps.get(iof + "-" + iof) is not None:
                        sli = self.orbpair_maps[iof + "-" + iof]
                        l = anglrMId[re.findall("[a-z]", iof)[0]]
                        indices = torch.arange(2 * l + 1, device=self.device)
                        indices = indices + indices * (2 * l + 1)
                        indices += sli.start
                        self.mask_to_ndiag[self.chemical_symbol_to_type[ib]][indices] = True

        if self.nextham_uureal_mask:
            self._apply_nextham_uureal_mask()

        log.info(
            f"Init OrbitalMapper: Method={self.method}, SOC={self.has_soc}, RME={self.reduced_matrix_element}, Basis={orbtype_count}")

    # ================== Map Generation Methods ==================
    def _apply_nextham_uureal_mask(self):
        """
        NextHAM-like mask:
        only supervise SOC 'uu block' + real-part.

        Assumption of layout (matches your get_irreps() style block_ir = block_ir + block_ir):
        - if soc_complex_doubling=True: first half is real, second half is imag
        - inside each (real/imag) half: spin blocks are contiguous and uu is the first block
          => uu_real occupies the first base_dim entries of each orbpair slice.
        """
        if not (self.method == "e3tb" and self.has_soc):
            log.warning("[OrbitalMapper] nextham_uureal_mask=True but (method!=e3tb or has_soc=False). Skip.")
            return

        uu_real_mask_1d = torch.zeros(
            self.reduced_matrix_element, dtype=torch.bool, device=self.device
        )

        for k, sli in self.orbpair_maps.items():
            io, jo = k.split("-")
            il = anglrMId[re.findall(r"[a-z]", io)[0]]
            jl = anglrMId[re.findall(r"[a-z]", jo)[0]]

            base_dim = (2 * il + 1) * (2 * jl + 1)
            # Keep only uu_real => first base_dim entries of this slice
            uu_real_mask_1d[sli.start: sli.start + base_dim] = True

        # cache for debug
        self.mask_uureal = uu_real_mask_1d

        # Intersect with existing (element/bond super-basis) masks
        uu_real_mask_nrme = uu_real_mask_1d.unsqueeze(0)  # [1, RME]
        uu_real_mask_erme = uu_real_mask_1d.unsqueeze(0)  # [1, RME]

        self.mask_to_nrme = self.mask_to_nrme & uu_real_mask_nrme
        self.mask_to_erme = self.mask_to_erme & uu_real_mask_erme

        log.info(
            f"[OrbitalMapper] Applied nextham_uureal_mask: kept "
            f"{int(uu_real_mask_1d.sum().item())}/{uu_real_mask_1d.numel()} dims (before element/bond filtering)."
        )

    def get_orbpairtype_maps(self):
        """
        Maps orbital type pairs (e.g., s-s, s-p) to RME slices.
        Logic variations:
        - SKTB: Uses min(l, l')+1 features, stores Upper Triangle only.
        - E3TB (Non-SOC): Uses (2l+1)(2l'+1) features, stores Upper Triangle only.
        - E3TB (SOC): Uses (2l+1)(2l'+1)*Factor features, stores Full Matrix.
        """
        self.orbpairtype_maps = {}
        ist = 0
        types = ["s", "p", "d", "f", "g", "h"]

        for i, io in enumerate(types):
            if self.orbtype_count[io] == 0: continue

            # SOC E3TB: Iterate j from 0 (Full Matrix)
            # SKTB / Non-SOC: Iterate j from i (Upper Triangle)
            start_j = 0 if (self.method == "e3tb" and self.has_soc) else i

            for j in range(start_j, len(types)):
                jo = types[j]
                if self.orbtype_count[jo] == 0: continue

                orb_pair = io + "-" + jo
                il, jl = anglrMId[io], anglrMId[jo]

                # Determine feature size per hop
                if self.method == "e3tb":
                    if self.has_soc:
                        factor = 4 * (2 if self.soc_complex_doubling else 1)
                        n_rme = (2 * il + 1) * (2 * jl + 1) * factor
                    else:
                        n_rme = (2 * il + 1) * (2 * jl + 1)
                else:
                    # SKTB logic
                    n_rme = min(il, jl) + 1

                # Calculate number of hops in this block
                numhops = self.orbtype_count[io] * self.orbtype_count[jo] * n_rme

                # Compress diagonal blocks (s-s, p-p) for symmetric storage methods
                is_symmetric_storage = (not self.has_soc) or (self.method == "sktb")
                if is_symmetric_storage and io == jo:
                    numhops += self.orbtype_count[jo] * n_rme
                    numhops = int(numhops / 2)

                self.orbpairtype_maps[orb_pair] = slice(ist, ist + numhops)
                ist += numhops

        return self.orbpairtype_maps

    def get_orbpair_maps(self):
        """
        Maps specific orbital instances (e.g., 1s-2p) to RME indices.
        """
        if hasattr(self, "orbpair_maps"): return self.orbpair_maps
        if not hasattr(self, "orbpairtype_maps"): self.get_orbpairtype_maps()

        self.orbpair_maps = {}

        # Iterate through Full Basis
        for i, io in enumerate(self.full_basis):
            # SOC E3TB: Full loop. Others: Upper Triangle loop.
            start_j = 0 if (self.method == "e3tb" and self.has_soc) else i

            for j in range(start_j, len(self.full_basis)):
                jo = self.full_basis[j]

                # Parse orbital info (e.g., io="1s" -> n=1, type=s)
                i_n = int(re.findall(r'\d+', io)[0])
                i_type = re.findall(r'[a-z]', io)[0]

                j_n = int(re.findall(r'\d+', jo)[0])
                j_type = re.findall(r'[a-z]', jo)[0]

                il, jl = anglrMId[i_type], anglrMId[j_type]

                # Retrieve start index for this type pair
                type_pair = f"{i_type}-{j_type}"
                if type_pair not in self.orbpairtype_maps: continue
                type_slice = self.orbpairtype_maps[type_pair]

                # Determine feature size
                if self.method == "e3tb":
                    if self.has_soc:
                        factor = 4 * (2 if self.soc_complex_doubling else 1)
                        n_feature = (2 * il + 1) * (2 * jl + 1) * factor
                    else:
                        n_feature = (2 * il + 1) * (2 * jl + 1)
                else:
                    n_feature = min(il, jl) + 1

                # Calculate Offset within the block
                is_symmetric_storage = (not self.has_soc) or (self.method == "sktb")

                if is_symmetric_storage and i_type == j_type:
                    # Diagonal Block Compression (Arithmetic Progression Offset)
                    # i_n, j_n are 1-based indices relative to their type
                    count = self.orbtype_count[j_type]
                    # Formula: (2*N + 2 - i) * (i - 1) / 2 + (j - i)
                    offset = ((2 * count + 2 - i_n) * (i_n - 1) // 2 + (j_n - i_n))
                else:
                    # Full Block Grid Offset: (row * cols + col)
                    count_j = self.orbtype_count[j_type]
                    offset = (i_n - 1) * count_j + (j_n - 1)

                start = int(type_slice.start + offset * n_feature)
                self.orbpair_maps[f"{io}-{jo}"] = slice(start, start + n_feature)

        return self.orbpair_maps

    # ================== SKTB Specific Methods (Restored) ==================

    def get_skonsite_maps(self):
        """SKTB: Maps orbital pairs to Onsite Energy indices."""
        assert self.method == "sktb", "Only sktb orbitalmapper have skonsite maps."

        if hasattr(self, "skonsite_maps"): return self.skonsite_maps
        if not hasattr(self, "skonsitetype_maps"): self.get_skonsitetype_maps()

        self.mask_diag = torch.zeros(self.n_onsite_Es, dtype=torch.bool, device=self.device)
        self.skonsite_maps = {}

        # Similar to get_orbpair_maps, but for Onsite Energy storage (Upper Triangle)
        for i, io in enumerate(self.full_basis):
            for j, jo in enumerate(self.full_basis[i:]):  # Upper Triangle
                i_n = int(re.findall(r'\d+', io)[0])
                i_type = re.findall(r'[a-z]', io)[0]
                j_n = int(re.findall(r'\d+', jo)[0])
                j_type = re.findall(r'[a-z]', jo)[0]

                # Onsite terms only exist between same orbital types (s-s, p-p, etc.)
                if i_type == j_type:
                    full_basis_pair = io + "-" + jo
                    type_start = self.skonsitetype_maps[i_type].start
                    count = self.orbtype_count[i_type]

                    # Offset calculation (Upper triangle arithmetic progression)
                    offset = ((2 * count + 2 - i_n) * (i_n - 1) // 2 + (j_n - i_n))
                    start = int(type_start + offset)

                    self.skonsite_maps[full_basis_pair] = slice(start, start + 1)
                    if io == jo:
                        self.mask_diag[start] = True

        return self.skonsite_maps

    def get_skonsitetype_maps(self):
        """SKTB: Maps orbital types to Onsite Energy blocks."""
        assert self.method == "sktb"
        self.skonsitetype_maps = {}
        ist = 0
        for io in ["s", "p", "d", "f", "g", "h"]:
            if self.orbtype_count[io] != 0:
                # Storage size: (N^2 + N)/2
                numonsites = int(0.5 * (self.orbtype_count[io] ** 2 + self.orbtype_count[io]))
                self.skonsitetype_maps[io] = slice(ist, ist + numonsites)
                ist += numonsites
        return self.skonsitetype_maps

    def get_sksoctype_maps(self):
        """SKTB: Maps orbital types to Onsite SOC blocks."""
        assert self.method == "sktb"
        self.sksoctype_maps = {}
        ist = 0
        for io in ["s", "p", "d", "f", "g", "h"]:
            if self.orbtype_count[io] != 0:
                # 1 SOC parameter per orbital shell (lambda)
                numonsites = self.orbtype_count[io]
                self.sksoctype_maps[io] = slice(ist, ist + numonsites)
                ist += numonsites
        return self.sksoctype_maps

    def get_sksoc_maps(self):
        """SKTB: Maps specific orbitals to Onsite SOC indices."""
        assert self.method == "sktb"
        if hasattr(self, "sksoc_maps"): return self.sksoc_maps
        if not hasattr(self, "sksoctype_maps"): self.get_sksoctype_maps()

        self.sksoc_maps = {}
        for io in self.full_basis:
            i_n = int(re.findall(r'\d+', io)[0])
            i_type = re.findall(r'[a-z]', io)[0]

            start = int(self.sksoctype_maps[i_type].start + (i_n - 1))
            self.sksoc_maps[io] = slice(start, start + 1)

        return self.sksoc_maps

    # ================== Common Utility Methods ==================

    def get_orbital_maps(self):
        """Generates slices for each atom's basis in the Hamiltonian."""
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

    def get_irreps(self, no_parity=False):
        """E3TB: Generates E3NN Irreps for edge features."""
        assert self.method == "e3tb", "Only support e3tb method for now."
        cache_key = (no_parity, self.has_soc, self.soc_complex_doubling)
        if hasattr(self, "_cached_irreps_key") and self._cached_irreps_key == cache_key:
            return self.orbpair_irreps
        self.no_parity = no_parity

        if not hasattr(self, "orbpairtype_maps"): self.get_orbpairtype_maps()

        irs = []
        factor = 1 if no_parity else -1

        for pair, sli in self.orbpairtype_maps.items():
            l1, l2 = anglrMId[pair[0]], anglrMId[pair[2]]

            # Calculate dimension of a single interaction block
            if self.has_soc:
                factor_dim = 4 * (2 if self.soc_complex_doubling else 1)
                block_dim = (2 * l1 + 1) * (2 * l2 + 1) * factor_dim
            else:
                block_dim = (2 * l1 + 1) * (2 * l2 + 1)

            # Calculate how many such blocks exist based on slice size
            num_blocks = int((sli.stop - sli.start) / block_dim)

            # Generate irreps for a single block
            block_ir = irreps_from_l1l2(l1, l2, spinful=self.has_soc, no_parity=no_parity)

            if self.has_soc and self.soc_complex_doubling:
                block_ir = block_ir + block_ir

            # Sum irreps for all blocks
            irs += [block_ir] * num_blocks

        # Flatten and sum
        final_irreps = o3.Irreps("")
        for ir in irs:
            final_irreps += ir

        self.orbpair_irreps = final_irreps
        self._cached_irreps_key = cache_key
        return self.orbpair_irreps

    def get_irreps_sim(self, no_parity=False):
        return self.get_irreps(no_parity=no_parity).sort()[0].simplify()

    def get_irreps_ess(self, no_parity=False):
        irp_e = []
        for mul, (l, p) in self.get_irreps_sim(no_parity=no_parity):
            if (-1) ** l == p:
                irp_e.append((mul, (l, p)))
        return o3.Irreps(irp_e)

    def __eq__(self, other):
        return (self.basis == other.basis and
                self.method == other.method and
                getattr(self, 'has_soc', False) == getattr(other, 'has_soc', False))
