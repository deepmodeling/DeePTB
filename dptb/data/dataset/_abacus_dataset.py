from typing import Dict, Any, List, Callable, Union, Optional
import os
import numpy as np
import h5py

import torch

from .. import (
    AtomicData,
    AtomicDataDict,
)
from ..transforms import TypeMapper, OrbitalMapper
from ._base_datasets import AtomicDataset
from dptb.nn.hamiltonian import E3Hamiltonian
from dptb.data.interfaces.ham_to_feature import ham_block_to_feature

orbitalLId = {0:"s", 1:"p", 2:"d", 3:"f"}

class ABACUSDataset(AtomicDataset):

    def __init__(
            self,
            root: str,
            key_mapping: Dict[str, str] = {
            "pos": AtomicDataDict.POSITIONS_KEY,
            "energy": AtomicDataDict.TOTAL_ENERGY_KEY,
            "atomic_numbers": AtomicDataDict.ATOMIC_NUMBERS_KEY,
            "kpoints": AtomicDataDict.KPOINT_KEY,
            "eigenvalues": AtomicDataDict.ENERGY_EIGENVALUE_KEY,
        },
        preprocess_path: str = None,
        h5file_names: Optional[str] = None,
        AtomicData_options: Dict[str, Any] = {},
        type_mapper: Optional[TypeMapper] = None,
    ):
        super().__init__(root=root, type_mapper=type_mapper)
        self.key_mapping = key_mapping
        self.key_list = list(key_mapping.keys())
        self.value_list = list(key_mapping.values())
        self.file_names = h5file_names
        self.preprocess_path = preprocess_path

        self.AtomicData_options = AtomicData_options
        # self.r_max = AtomicData_options["r_max"]
        # self.er_max = AtomicData_options["er_max"]
        # self.oer_max = AtomicData_options["oer_max"]
        # self.pbc = AtomicData_options["pbc"]

        self.index = None
        self.num_examples = len(h5file_names)

    def get(self, idx):
        file_name = self.file_names[idx]
        file = os.path.join(self.preprocess_path, file_name)
        data = h5py.File(file, "r")

        atomic_data = AtomicData.from_points(
            pos = data["pos"][:],
            cell = data["cell"][:],
            atomic_numbers = data["atomic_numbers"][:],
            **self.AtomicData_options,
        )

        if data["hamiltonian_blocks"]:
            basis = {}
            for key, value in data["basis"].items(): 
                basis[key] = [(f"{i+1}" + orbitalLId[l]) for i, l in enumerate(value)]
            idp = OrbitalMapper(basis)
            e3 = E3Hamiltonian(idp=idp, decompose=True)
            ham_block_to_feature(atomic_data, idp, data.get("hamiltonian_blocks", False), data.get("overlap_blocks", False))
            with torch.no_grad():
                atomic_data = e3(atomic_data.to_dict())
            atomic_data = AtomicData.from_dict(atomic_data)
        if data.get("eigenvalue") and data.get("kpoint"):
            atomic_data[AtomicDataDict.KPOINT_KEY] = torch.as_tensor(data["kpoint"][:], dtype=torch.get_default_dtype())
            atomic_data[AtomicDataDict.ENERGY_EIGENVALUE_KEY] = torch.as_tensor(data["eigenvalue"][:], dtype=torch.get_default_dtype())

        return atomic_data
    
    def len(self) -> int:
        return self.num_examples