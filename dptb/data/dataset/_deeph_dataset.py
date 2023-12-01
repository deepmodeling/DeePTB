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
from dptb.data.interfaces.ham_to_feature import openmx_to_deeptb

orbitalLId = {0:"s", 1:"p", 2:"d", 3:"f"}

class DeePHE3Dataset(AtomicDataset):

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
        subdir_names: Optional[str] = None,
        AtomicData_options: Dict[str, Any] = {},
        type_mapper: Optional[TypeMapper] = None,
    ):
        super().__init__(root=root, type_mapper=type_mapper)
        self.key_mapping = key_mapping
        self.key_list = list(key_mapping.keys())
        self.value_list = list(key_mapping.values())
        self.subdir_names = subdir_names
        self.preprocess_path = preprocess_path

        self.AtomicData_options = AtomicData_options
        # self.r_max = AtomicData_options["r_max"]
        # self.er_max = AtomicData_options["er_max"]
        # self.oer_max = AtomicData_options["oer_max"]
        # self.pbc = AtomicData_options["pbc"]

        self.index = None
        self.num_examples = len(subdir_names)

    def get(self, idx):
        file_name = self.subdir_names[idx]
        file = os.path.join(self.preprocess_path, file_name)

        if os.path.exists(os.path.join(file, "AtomicData.pth")):
            atomic_data = torch.load(os.path.join(file, "AtomicData.pth"))
        else:
            atomic_data = AtomicData.from_points(
                pos = np.loadtxt(os.path.join(file, "site_positions.dat")).T,
                cell = np.loadtxt(os.path.join(file, "lat.dat")).T,
                atomic_numbers = np.loadtxt(os.path.join(file, "element.dat")),
                **self.AtomicData_options,
            )

            idp = self.type_mapper
            e3 = E3Hamiltonian(idp=idp, decompose=True)

            openmx_to_deeptb(atomic_data, idp, os.path.join(file, "./hamiltonians.h5"))
            with torch.no_grad():
                atomic_data = e3(atomic_data.to_dict())
            atomic_data = AtomicData.from_dict(atomic_data)

            torch.save(atomic_data, os.path.join(file, "AtomicData.pth"))

        return atomic_data
    
    def len(self) -> int:
        return self.num_examples