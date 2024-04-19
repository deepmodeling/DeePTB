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
from ._base_datasets import AtomicInMemoryDataset
from dptb.nn.hamiltonian import E3Hamiltonian
from dptb.data.interfaces.ham_to_feature import block_to_feature
from dptb.data.interfaces.abacus import recursive_parse

orbitalLId = {0:"s", 1:"p", 2:"d", 3:"f"}

def _abacus_h5_reader(h5file_path, AtomicData_options):
    data = h5py.File(h5file_path, "r")
    atomic_data = AtomicData.from_points(
        pos = data["pos"][:],
        cell = data["cell"][:],
        atomic_numbers = data["atomic_numbers"][:],
        **AtomicData_options,
    )
    if data["hamiltonian_blocks"]:
        basis = {}
        for key, value in data["basis"].items(): 
            basis[key] = [(f"{i+1}" + orbitalLId[l]) for i, l in enumerate(value)]
        idp = OrbitalMapper(basis)
        # e3 = E3Hamiltonian(idp=idp, decompose=True)
        block_to_feature(atomic_data, idp, data.get("hamiltonian_blocks", False), data.get("overlap_blocks", False))
        # with torch.no_grad():
        #     atomic_data = e3(atomic_data.to_dict())
        # atomic_data = AtomicData.from_dict(atomic_data)

    if data.get("eigenvalue") and data.get("kpoint"):
        atomic_data[AtomicDataDict.KPOINT_KEY] = torch.as_tensor(data["kpoint"][:], dtype=torch.get_default_dtype())
        atomic_data[AtomicDataDict.ENERGY_EIGENVALUE_KEY] = torch.as_tensor(data["eigenvalue"][:], dtype=torch.get_default_dtype())
    return atomic_data


class ABACUSInMemoryDataset(AtomicInMemoryDataset):

    def __init__(
        self,
        root: str,
        abacus_args: Dict[str, Union[str,bool]] = {
            "input_dir": None,
            "preprocess_dir": None,
            "only_overlap": False, 
            "get_Ham": False, 
            "add_overlap": False, 
            "get_eigenvalues": False,
        },
        file_name: Optional[str] = None,
        url: Optional[str] = None,
        AtomicData_options: Dict[str, Any] = {},
        include_frames: Optional[List[int]] = None,
        type_mapper: TypeMapper = None,
        key_mapping: Dict[str, str] = {
            "pos": AtomicDataDict.POSITIONS_KEY,
            "energy": AtomicDataDict.TOTAL_ENERGY_KEY,
            "atomic_numbers": AtomicDataDict.ATOMIC_NUMBERS_KEY,
            "kpoints": AtomicDataDict.KPOINT_KEY,
            "eigenvalues": AtomicDataDict.ENERGY_EIGENVALUE_KEY,
        },
    ):
        if file_name is not None: 
            self.file_name = file_name
        else:
            self.abacus_args = abacus_args
            assert self.abacus_args.get("input_dir") is not None, "ABACUS calculation results MUST be provided."
            if self.abacus_args.get("preprocess_dir") is None:
                print("Creating new preprocess dictionary...")
                os.mkdir(os.path.join(root, "preprocess"))
                self.abacus_args["preprocess_dir"] = os.path.join(root, "preprocess")
            self.key_mapping = key_mapping

            print("Begin parsing ABACUS output...")
            h5_filenames = recursive_parse(**self.abacus_args)
            self.file_name = h5_filenames
            print("Finished parsing ABACUS output.")

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
        for h5_file in self.file_name:
            data.append(_abacus_h5_reader(h5_file, self.AtomicData_options))
        return data
    
    @property
    def raw_file_names(self):
        return "AtomicData.h5"

    @property
    def raw_dir(self):
        return self.root