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
from ._base_datasets import AtomicDataset, AtomicInMemoryDataset
#from dptb.nn.hamiltonian import E3Hamiltonian
from dptb.data.interfaces.ham_to_feature import block_to_feature

orbitalLId = {0:"s", 1:"p", 2:"d", 3:"f"}

def _abacus_h5_reader(h5file_path, AtomicData_options):
    data = h5py.File(h5file_path, "r")
    atomic_data = AtomicData.from_points(
        pos = data["pos"][:],
        cell = data["cell"][:],
        atomic_numbers = data["atomic_numbers"][:],
        **AtomicData_options,
    )
    if "hamiltonian_blocks" in data:
        basis = {}
        for key, value in data["basis"].items(): 
            basis[key] = [(f"{i+1}" + orbitalLId[l]) for i, l in enumerate(value)]
        idp = OrbitalMapper(basis)
        # e3 = E3Hamiltonian(idp=idp, decompose=True)
        block_to_feature(atomic_data, idp, data.get("hamiltonian_blocks", False), data.get("overlap_blocks", False))
        # with torch.no_grad():
        #     atomic_data = e3(atomic_data.to_dict())
        # atomic_data = AtomicData.from_dict(atomic_data)

    if "eigenvalues" in data and "kpionts" in data:
        atomic_data[AtomicDataDict.KPOINT_KEY] = torch.as_tensor(data["kpoints"][:], dtype=torch.get_default_dtype())
        atomic_data[AtomicDataDict.ENERGY_EIGENVALUE_KEY] = torch.as_tensor(data["eigenvalues"][:], dtype=torch.get_default_dtype())
    return atomic_data

# Lazy loading class, built for large dataset.

class ABACUSDataset(AtomicDataset):

    def __init__(
            self,
            root: str,
            preprocess_dir: str,
            AtomicData_options: Dict[str, Any] = {},
            type_mapper: Optional[TypeMapper] = None,
    ):
        super().__init__(root=root, type_mapper=type_mapper)
        self.preprocess_dir = preprocess_dir
        self.file_name = np.loadtxt(os.path.join(self.preprocess_dir, 'AtomicData_file.txt'), dtype=str)
        self.AtomicData_options = AtomicData_options
        self.num_examples = len(self.file_name)

    def get(self, idx):
        name = self.file_name[idx]
        h5_file = os.path.join(self.preprocess_dir, name)
        atomic_data = _abacus_h5_reader(h5_file, self.AtomicData_options)
        return atomic_data
    
    def len(self) -> int:
        return self.num_examples
    
# In memory version.

class ABACUSInMemoryDataset(AtomicInMemoryDataset):

    def __init__(
            self,
            root: str,
            preprocess_dir: str,
            url: Optional[str] = None,
            AtomicData_options: Dict[str, Any] = {},
            include_frames: Optional[List[int]] = None,
            type_mapper: TypeMapper = None,
    ):
        self.preprocess_dir = preprocess_dir
        self.file_name = np.loadtxt(os.path.join(self.preprocess_dir, 'AtomicData_file.txt'), dtype=str)

        super(ABACUSInMemoryDataset, self).__init__(
            file_name=self.file_name,
            url=url,
            root=root,
            AtomicData_options=AtomicData_options,
            include_frames=include_frames,
            type_mapper=type_mapper,
        )

    def get_data(self):
        data = []
        for name in self.file_name:
            h5_file = os.path.join(self.preprocess_dir, name)
            data.append(_abacus_h5_reader(h5_file, self.AtomicData_options))
        return data
    
    @property
    def raw_file_names(self):
        return "AtomicData.h5"

    @property
    def raw_dir(self):
        return self.root