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
        info: dict,
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
        self.info = info # there should be one info file for one LMDB Dataset

        assert "r_max" in info
            

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
        self.get_Hamiltonian = get_Hamiltonian
        self.get_overlap = get_overlap
        self.get_DM = get_DM
        self.get_eigenvalues = get_eigenvalues
        assert not get_Hamiltonian * get_DM, "Hamiltonian and Density Matrix can only loaded one at a time, for which will occupy the same attribute in the AtomicData."


        db_env = lmdb.open(os.path.join(self.root), readonly=True, lock=False)
        with db_env.begin() as txn:
            self.num_graphs = txn.stat()['entries']
        db_env.close()

    def len(self):
        return self.num_graphs
    
    @property
    def raw_file_names(self):
        # TODO: this is not implemented.
        return "Null"

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
            cell, pos, atomic_numbers = \
                data_dict[AtomicDataDict.CELL_KEY], \
                data_dict[AtomicDataDict.POSITIONS_KEY], \
                data_dict[AtomicDataDict.ATOMIC_NUMBERS_KEY]
            
            pbc = data_dict[AtomicDataDict.PBC_KEY]

            
            if self.get_Hamiltonian:
                blocks = data_dict["hamiltonian"]
                # kk, vv = blocks.keys(), blocks.values()
                # vv = map(lambda x: np.frombuffer(x, np.float32).reshape, vv)
                # blocks = dict(zip(kk, vv))
                # del kk
                # del vv

            if self.get_overlap:
                overlap = data_dict["overlaps"]
                # kk, vv = overlap.keys(), overlap.values()
                # vv = map(lambda x: np.frombuffer(x, np.float32), vv)
                # overlap = dict(zip(kk, vv))
                # del kk
                # del vv
            else:
                overlap = False

            if self.get_DM:
                blocks = data_dict["DM"]
                # kk, vv = blocks.keys(), blocks.values()
                # vv = map(lambda x: np.frombuffer(x, np.float32), vv)
                # blocks = dict(zip(kk, vv))
                # del kk
                # del vv

            if not (self.get_Hamiltonian or self.get_DM):
                blocks = False
        
        db_env.close()
        atomicdata = AtomicData.from_points(
            pos=pos.reshape(-1,3),
            cell=cell.reshape(3,3),
            atomic_numbers=atomic_numbers,
            pbc=pbc,
            **self.info
        )

        # transform blocks to atomicdata features
        if self.get_Hamiltonian or self.get_DM or self.get_overlap:
            block_to_feature(atomicdata, self.type_mapper, blocks, overlap)
        

        return atomicdata