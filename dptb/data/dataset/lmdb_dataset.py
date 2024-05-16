import numpy as np
from typing import Tuple, Dict, Any, List, Callable, Union, Optional

import torch

from dptb.utils.torch_geometric import Batch, Dataset
from dptb.utils.tools import download_url, extract_zip

import dptb
import os
from dptb.data import (
    AtomicData,
    AtomicDataDict,
    _NODE_FIELDS,
    _EDGE_FIELDS,
    _GRAPH_FIELDS,
)
from dptb.utils.batch_ops import bincount
from dptb.utils.regressor import solver
from dptb.utils.savenload import atomic_write
from ..transforms import TypeMapper
from ._base_datasets import AtomicDataset
import lmdb
from dptb.data.interfaces.ham_to_feature import block_to_feature
import pickle

class LMDBDataset(AtomicDataset):
    def __init__(
        self,
        root: str,
        info_files: Dict[str, Dict],
        url: Optional[str] = None,
        include_frames: Optional[List[int]] = None,
        type_mapper: TypeMapper = None,
        get_Hamiltonian: bool = False,
        get_overlap: bool = False,
        get_DM: bool = False,
        get_eigenvalues: bool = False,
    ):
        # TO DO, this may be simplified
        # See if a subclass defines some inputs
        self.url = getattr(type(self), "URL", url)
        self.include_frames = include_frames
        self.info_files = info_files

        for file in self.info_files.keys():
            info = info_files[file]
            assert "AtomicData_options" in info
            AtomicData_options = info["AtomicData_options"]
            assert "r_max" in AtomicData_options
            assert "pbc" in AtomicData_options

        self.data = None

        # !!! don't delete this block.
        # otherwise the inherent children class
        # will ignore the download function here
        class_type = type(self)
        if class_type != LMDBDataset:
            if "download" not in self.__class__.__dict__:
                class_type.download = LMDBDataset.download

        # Initialize the InMemoryDataset, which runs download and process
        # See https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-in-memory-datasets
        # Then pre-process the data if disk files are not found
        super().__init__(root=root, type_mapper=type_mapper)
        if self.data is None:
            self.data, include_frames = torch.load(self.processed_paths[0])
            if not np.all(include_frames == self.include_frames):
                raise ValueError(
                    f"the include_frames is changed. "
                    f"please delete the processed folder and rerun {self.processed_paths[0]}"
                )
            
        self.get_Hamiltonian = get_Hamiltonian
        self.get_overlap = get_overlap
        self.get_DM = get_DM
        self.get_eigenvalues = get_eigenvalues
        assert not get_Hamiltonian * get_DM, "Hamiltonian and Density Matrix can only loaded one at a time, for which will occupy the same attribute in the AtomicData."

    def len(self):
        if self.data is None:
            return 0
        return self.data.num_graphs

    def download(self):
        if (not hasattr(self, "url")) or (self.url is None):
            # Don't download, assume present. Later could have FileNotFound if the files don't actually exist
            pass
        else:
            download_path = download_url(self.url, self.raw_dir)
            if download_path.endswith(".zip"):
                extract_zip(download_path, self.raw_dir)

    def get(self, idx):
        db_env = lmdb.open(os.path.join(self.root), readonly=True, lock=False)
        with db_env.begin() as txn:
            data_dict = txn.get(int(idx).to_bytes(length=4, byteorder='big'))
            data_dict = pickle.loads(data_dict)
            cell, rcell, pos, atomic_numbers, basis = \
                np.frombuffer(data_dict['cell'], np.float32), \
                np.frombuffer(data_dict['rcell'], np.float32), \
                np.frombuffer(data_dict['pos'], np.float32), \
                np.frombuffer(data_dict['atomic_numbers'], np.int32), \
                np.formatter(data_dict['basis'], np.float32)
            pos = pos.reshape(-1, 3)
            
            if self.get_Hamiltonian:
                hamiltonians = pickle.loads(data_dict["hamiltonians"])
                kk, vv = hamiltonians.keys(), hamiltonians.values()
                vv = map(lambda x: np.frombuffer(x, np.float32), vv)
                hamiltonians = dict(zip(kk, vv))
                del kk
                del vv
            if self.get_overlap:
                overlap = pickle.loads(data_dict["overlap"])
                kk, vv = overlap.keys(), overlap.values()
                vv = map(lambda x: np.frombuffer(x, np.float32), vv)
                overlap = dict(zip(kk, vv))
                del kk
                del vv
            if self.get_DM:
                DM = pickle.loads(data_dict["DM"])
                kk, vv = DM.keys(), DM.values()
                vv = map(lambda x: np.frombuffer(x, np.float32), vv)
                DM = dict(zip(kk, vv))
                del kk
                del vv

            if self.get_eigenvalues:
                eigenvalues = np.frombuffer(data_dict["eigenvalues"], np.float32)
                kpoints = np.fromnuffer(data_dict["kpoints"], np.float32)
        
        db_env.close()
        atomicdata = AtomicData.from_points(
            pos=pos,
            cell=cell,
            atomic_number=atomic_numbers
        )

        # transform blocks to atomicdata features
        
        return atomicdata