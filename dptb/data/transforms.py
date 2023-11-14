from typing import Dict, Optional, Union, List
from dptb.data.AtomicDataDict import Type
from dptb.utils.tools import get_uniq_symbol
from dptb.utils.constants import anglrMId
import re
import warnings

import torch

import ase.data

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
    ):
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
                size=(1 + self._max_Z - self._min_Z,), fill_value=-1, dtype=torch.long
            )
            for sym, type in self.chemical_symbol_to_type.items():
                Z_to_index[ase.data.atomic_numbers[sym] - self._min_Z] = type
            self._Z_to_index = Z_to_index
            self._index_to_Z = torch.zeros(
                size=(len(self.chemical_symbol_to_type),), dtype=torch.long
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

        return self._Z_to_index.to(device=atomic_numbers.device)[
            atomic_numbers - self._min_Z
        ]

    def untransform(self, atom_types):
        """Transform atom types back into atomic numbers"""
        return self._index_to_Z[atom_types].to(device=atom_types.device)

    @property
    def has_chemical_symbols(self) -> bool:
        return self.chemical_symbol_to_type is not None

    @staticmethod
    def format(
        data: list, type_names: List[str], element_formatter: str = ".6f"
    ) -> str:
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
            chemical_symbols_to_type:Union[Dict[str, int], None]=None
            ):
        super(BondMapper, self).__init__(chemical_symbol_to_type=chemical_symbols_to_type, chemical_symbols=chemical_symbols)

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
                size=(len(self._Z_to_index), len(self._Z_to_index)), fill_value=-1, dtype=torch.long
            )
        ZZ_to_reduced_index = torch.full(
                size=(len(self._Z_to_index), len(self._Z_to_index)), fill_value=-1, dtype=torch.long
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
                size=(len(self.bond_to_type),2), dtype=torch.long
            )
        self._reduced_index_to_ZZ = torch.zeros(
                size=(len(self.reduced_bond_to_type),2), dtype=torch.long
            )
        
        for abond, aidx in self.bond_to_type.items():
            asym, bsym = abond.split("-")
            self._index_to_ZZ[aidx] = torch.tensor([ase.data.atomic_numbers[asym], ase.data.atomic_numbers[bsym]], dtype=torch.long)

        for abond, aidx in self.reduced_bond_to_type.items():
            asym, bsym = abond.split("-")
            self._reduced_index_to_ZZ = torch.tensor([ase.data.atomic_numbers[asym], ase.data.atomic_numbers[bsym]], dtype=torch.long)


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

        return self._ZZ_to_index.to(device=iatomic_numbers.device)[
            iatomic_numbers - self._min_Z, jatomic_numbers - self._min_Z
        ]
    
    def transform_reduced_bond(self, iatomic_numbers, jatomic_numbers):
        
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

        return self._ZZ_to_reduced_index.to(device=iatomic_numbers.device)[
            iatomic_numbers - self._min_Z, jatomic_numbers - self._min_Z
        ]
    
    def untransform_atom(self, atom_types):
        """Transform atom types back into atomic numbers"""
        return self.untransform(atom_types)
    
    def untransform_bond(self, bond_types):
        """Transform bond types back into atomic numbers"""
        return self._index_to_ZZ[bond_types].to(device=bond_types.device)
    
    def untransform_reduced_bond(self, bond_types):
        """Transform reduced bond types back into atomic numbers"""
        return self._reduced_index_to_ZZ[bond_types].to(device=bond_types.device)
    
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
                self.reduced_bond_to_type is not None
            ), "Atomic numbers provided but there is no chemical_symbols/chemical_symbol_to_type mapping!"
            atomic_numbers = data[AtomicDataDict.ATOMIC_NUMBERS_KEY]

            assert (
                AtomicDataDict.EDGE_INDEX_KEY in data
            ), "The bond type mapper need a EDGE index as input."

            data[AtomicDataDict.EDGE_TYPE_KEY] = \
                self.transform_reduced_bond(
                    atomic_numbers[data[AtomicDataDict.EDGE_INDEX_KEY][0]],
                    atomic_numbers[data[AtomicDataDict.EDGE_INDEX_KEY][1]]
                    )
        else:
            if types_required:
                raise KeyError(
                    "Data doesn't contain any atom type information (EDGE_TYPE_KEY or ATOMIC_NUMBERS_KEY)"
                )
        return super().__call__(data=data, types_required=types_required)
        

    
    

class OrbitalMapper(BondMapper):
    def __init__(
            self, 
            basis: Dict[str, Union[List[str], str]], 
            chemical_symbol_to_type: Optional[Dict[str, int]] = None,
            method: str ="e3tb"
            ):
        """_summary_

        Parameters
        ----------
        basis : dict
            the definition of the basis set, should be like:
            {"A":"2s2p3d1f", "B":"1s2f3d1f"} or
            {"A":["2s", "2p"], "B":["2s", "2p"]}
            when list, "2s" indicate a "s" orbital in the second shell.
            when str, "2s" indicates two s orbital, 
            "2s2p3d4f" is equivilent to ["1s","2s", "1p", "2p", "1d", "2d", "3d", "1f"]
        """
        if chemical_symbol_to_type is not None:
            assert set(basis.keys()) == set(chemical_symbol_to_type.keys())
            super(OrbitalMapper, self).__init__(chemical_symbol_to_type=chemical_symbol_to_type)
        else:
            super(OrbitalMapper, self).__init__(chemical_symbols=list(basis.keys()))

        self.basis = basis
        self.method = method

        if self.method not in ["e3tb", "sktb"]:
            raise ValueError

        if isinstance(self.basis[self.type_names[0]], str):
            orbtype_count = {"s":0, "p":0, "d":0, "f":0}
            orbs = map(lambda bs: re.findall(r'[1-9]+[A-Za-z]', bs), self.basis.values())
            for ib in orbs:
                for io in ib:
                    if int(io[0]) > orbtype_count[io[1]]:
                        orbtype_count[io[1]] = int(io[0])
            # split into list basis
            basis = {k:[] for k in self.type_names}
            for ib in self.basis.keys():
                for io in ["s", "p", "d", "f"]:
                    if io in self.basis[ib]:
                        basis[ib].extend([str(i)+io for i in range(1, int(re.findall(r'[1-9]+'+io, self.basis[ib])[0][0])+1)])
            self.basis = basis

        elif isinstance(self.basis[self.type_names[0]], list):
            nb = len(self.type_names)
            orbtype_count = {"s":[0]*nb, "p":[0]*nb, "d":[0]*nb, "f":[0]*nb}
            for ib, bt in enumerate(self.type_names):
                for io in self.basis[bt]:
                    orb = re.findall(r'[A-Za-z]', io)[0]
                    orbtype_count[orb][ib] += 1
            
            for ko in orbtype_count.keys():
                orbtype_count[ko] = max(orbtype_count[ko])

        self.orbtype_count = orbtype_count
        self.full_basis_norb = 1 * orbtype_count["s"] + 3 * orbtype_count["p"] + 5 * orbtype_count["d"] + 7 * orbtype_count["f"]


        if self.method == "e3tb":
            self.edge_reduced_matrix_element = self.full_basis_norb ** 2
            self.node_reduced_matrix_element = int(((orbtype_count["s"] + 9 * orbtype_count["p"] + 25 * orbtype_count["d"] + 49 * orbtype_count["f"]) + \
                                                    self.edge_reduced_matrix_element)/2)
        else:
            self.edge_reduced_matrix_element = (
                1 * orbtype_count["s"] * orbtype_count["s"] + \
                2 * orbtype_count["s"] * orbtype_count["p"] + \
                2 * orbtype_count["s"] * orbtype_count["d"] + \
                2 * orbtype_count["s"] * orbtype_count["f"]
                ) + \
            2 * (
                1 * orbtype_count["p"] * orbtype_count["p"] + \
                2 * orbtype_count["p"] * orbtype_count["d"] + \
                2 * orbtype_count["p"] * orbtype_count["f"]
                ) + \
            3 * (
                1 * orbtype_count["d"] * orbtype_count["d"] + \
                2 * orbtype_count["d"] * orbtype_count["f"]
                ) + \
            4 * (orbtype_count["f"] * orbtype_count["f"])
            
            self.node_reduced_matrix_element = orbtype_count["s"] + orbtype_count["p"] + orbtype_count["d"] + orbtype_count["f"]
                                     
        

        # sort the basis
        for ib in self.basis.keys():
            self.basis[ib] = sorted(
                self.basis[ib], 
                key=lambda s: (anglrMId[re.findall(r"[a-z]",s)[0]], re.findall(r"[1-9*]",s)[0])
                )

        # TODO: get full basis set
        full_basis = []
        for io in ["s", "p", "d", "f"]:
            full_basis = full_basis + [str(i)+io for i in range(1, orbtype_count[io]+1)]
        self.full_basis = full_basis

        # TODO: get the mapping from list basis to full basis
        self.basis_to_full_basis = {}
        self.atom_norb = torch.zeros(len(self.type_names), dtype=torch.long)
        for ib in self.basis.keys():
            count_dict = {"s":0, "p":0, "d":0, "f":0}
            self.basis_to_full_basis.setdefault(ib, {})
            for o in self.basis[ib]:
                io = re.findall(r"[a-z]", o)[0]
                l = anglrMId[io]
                count_dict[io] += 1
                self.atom_norb[self.chemical_symbol_to_type[ib]] += 2*l+1

                self.basis_to_full_basis[ib][o] = str(count_dict[io])+io
        
        # Get the mask for mapping from full basis to atom specific basis
        self.mask_to_basis = torch.zeros(len(self.type_names), self.full_basis_norb, dtype=torch.bool)
        for ib in self.basis.keys():
            ibasis = list(self.basis_to_full_basis[ib].values())
            ist = 0
            for io in self.full_basis:
                l = anglrMId[io[1]]
                if io in ibasis:
                    self.mask_to_basis[self.chemical_symbol_to_type[ib]][ist:ist+2*l+1] = True
                
                ist += 2*l+1
            
        assert (self.mask_to_basis.sum(dim=1).int()-self.atom_norb).abs().sum() <= 1e-6

            


    def get_pairtype_maps(self):
        """
        The function `get_pairtype_maps` creates a mapping of orbital pair types, such as s-s, "s-p",
        to slices based on the number of hops between them.
        :return: a dictionary called `pairtype_map`.
        """
        
        self.pairtype_maps = {}
        ist = 0
        for io in ["s", "p", "d", "f"]:
            if self.orbtype_count[io] != 0:
                for jo in ["s", "p", "d", "f"]:
                    if self.orbtype_count[jo] != 0:
                        orb_pair = io+"-"+jo
                        il, jl = anglrMId[io], anglrMId[jo]
                        if self.method == "e3tb":
                            n_rme = (2*il+1) * (2*jl+1)
                        else:
                            n_rme = min(il, jl)+1
                        numhops =  self.orbtype_count[io] * self.orbtype_count[jo] * n_rme
                        self.pairtype_maps[orb_pair] = slice(ist, ist+numhops)

                        ist += numhops

        return self.pairtype_maps
    
    def get_pair_maps(self):

        # here we have the map from basis to full basis, but to define a map between basis pair to full basis pair,
        # one need to consider the id of the full basis pairs. Specifically, if we want to know the position where
        # "s*-2s" lies, we map it to the pair in full basis as "1s-2s", but we need to know the id of "1s-2s" in the 
        # features vector. For a full basis have three s: [1s, 2s, 3s], it will have 9 s features. Therefore, we need
        # to build a map from the full basis pair to the position in the vector.

        # We define the feature vector should look like [1s-1s, 1s-2s, 1s-3s, 2s-1s, 2s-2s, 2s-3s, 3s-1s, 3s-2s, 3s-3s,...]
        # it is sorted by the index of the left basis first, then the right basis. Therefore, we can build a map:

        # to do so we need the pair type maps first
        if hasattr(self, "pair_maps"):
            return self.pair_maps
        
        if not hasattr(self, "pairtype_maps"):
            self.pairtype_maps = self.get_pairtype_maps()
        self.pair_maps = {}
        for io in self.full_basis:
            for jo in self.full_basis:
                full_basis_pair = io+"-"+jo
                ir, jr = int(full_basis_pair[0]), int(full_basis_pair[3])
                iio, jjo = full_basis_pair[1], full_basis_pair[4]

                if self.method == "e3tb":
                    n_feature = (2*anglrMId[iio]+1) * (2*anglrMId[jjo]+1)
                else:
                    n_feature = min(anglrMId[iio], anglrMId[jjo])+1
                

                start = self.pairtype_maps[iio+"-"+jjo].start + \
                    n_feature * ((ir-1)*self.orbtype_count[jjo]+(jr-1))
                
                self.pair_maps[io+"-"+jo] = slice(start, start+n_feature)

        return self.pair_maps
    
    def get_node_maps(self):

        if hasattr(self, "node_maps"):
            return self.node_maps

        if not hasattr(self, "nodetype_maps"):
            self.get_nodetype_maps()
        
        self.node_maps = {}
        for i, io in enumerate(self.full_basis):
            for jo in self.full_basis[i:]:
                full_basis_pair = io+"-"+jo
                ir, jr = int(full_basis_pair[0]), int(full_basis_pair[3])
                iio, jjo = full_basis_pair[1], full_basis_pair[4]

                if self.method == "e3tb":
                    n_feature = (2*anglrMId[iio]+1) * (2*anglrMId[jjo]+1)
                    if iio == jjo:
                        start = self.nodetype_maps[iio+"-"+jjo].start + \
                            n_feature * ((2*self.orbtype_count[jjo]+2-ir) * (ir-1) / 2 + (jr - ir))
                    else:
                        start = self.nodetype_maps[iio+"-"+jjo].start + \
                            n_feature * ((ir-1)*self.orbtype_count[jjo]+(jr-1))
                    start = int(start)
                    self.node_maps[io+"-"+jo] = slice(start, start+n_feature)
                else:
                    if io == jo:
                        start = int(self.nodetype_maps[iio+"-"+jjo].start + (ir-1))
                        self.node_maps[io+"-"+jo] = slice(start, start+1)

        return self.node_maps

    def get_nodetype_maps(self):
        self.nodetype_maps = {}
        ist = 0

        for i, io in enumerate(["s", "p", "d", "f"]):
            if self.orbtype_count[io] != 0:
                for jo in ["s", "p", "d", "f"][i:]:
                    if self.orbtype_count[jo] != 0:
                        orb_pair = io+"-"+jo
                        il, jl = anglrMId[io], anglrMId[jo]
                        if self.method == "e3tb":
                            numonsites =  self.orbtype_count[io] * self.orbtype_count[jo] * (2*il+1) * (2*jl+1)
                            if io == jo:
                                numonsites +=  self.orbtype_count[jo] * (2*il+1) * (2*jl+1)
                                numonsites = int(numonsites / 2)
                        else:
                            if io == jo:
                                numonsites = self.orbtype_count[io]
                            else:
                                numonsites = 0

                        self.nodetype_maps[orb_pair] = slice(ist, ist+numonsites)

                        ist += numonsites


        return self.nodetype_maps

        
        # also need to think if we modify as this, how can we add extra basis when fitting.
        