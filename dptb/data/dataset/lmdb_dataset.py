import numpy as np
from typing import Tuple, Dict, Any, List, Callable, Union, Optional

import torch

from dptb.utils.torch_geometric import Batch, Dataset
from dptb.utils.tools import download_url, extract_zip

import dptb
import os
import os.path as osp
from dptb.data import (
    AtomicData,
    AtomicDataDict,
    _NODE_FIELDS,
    _EDGE_FIELDS,
    _GRAPH_FIELDS,
)
from tqdm import tqdm
from dptb.utils.batch_ops import bincount
from dptb.utils.regressor import solver
from dptb.utils.savenload import atomic_write
from ..transforms import TypeMapper
from ._base_datasets import AtomicDataset
from dptb.nn.hamiltonian import E3Hamiltonian
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
        orthogonal: bool = False,
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
        super().__init__(root=root, type_mapper=type_mapper) # the type_mapper will be called in getitem in PyG data class
        self.get_Hamiltonian = get_Hamiltonian
        self.get_overlap = get_overlap
        self.get_DM = get_DM
        self.get_eigenvalues = get_eigenvalues
        self.orthogonal = orthogonal
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
        # need to give a valid path so the download would not be triggered
        return ["data.mdb", "lock.mdb"]
    
    @property
    def raw_dir(self):
        return self.root

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
                overlap = data_dict["overlap"]
                # kk, vv = overlap.keys(), overlap.values()
                # vv = map(lambda x: np.frombuffer(x, np.float32), vv)
                # overlap = dict(zip(kk, vv))
                # del kk
                # del vv
            else:
                overlap = False

            if self.get_DM:
                blocks = data_dict["density_matrix"]
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
            block_to_feature(atomicdata, self.type_mapper, blocks, overlap, self.orthogonal)
        

        return atomicdata
    
    def E3statistics(self):
        assert self.transform is not None
        idp = self.transform

        if self.data[AtomicDataDict.EDGE_FEATURES_KEY].abs().sum() < 1e-7:
            return None
        
        e3h = E3Hamiltonian(basis=idp.basis, decompose=True)

        sorted_irreps = idp.orbpair_irreps.sort()[0].simplify()
        n_scalar = sorted_irreps[0].mul if sorted_irreps[0].ir.l == 0 else 0

        # init a count dict of atom species
        count_tp = {}
        for at, tp in idp.chemical_symbol_to_type.items():
            count_tp[tp] = 0

        for idx in tqdm(range(self.len()), desc="Collecting E3 irreps statistics: "):
            atomicdata = idp(self.get(idx=idx))

            # calculate norm & mean
            node_norm_ave = torch.ones(len(idp.chemical_symbol_to_type), idp.orbpair_irreps.num_irreps)
            node_square_ave = torch.zeros(len(idp.chemical_symbol_to_type), idp.orbpair_irreps.num_irreps)
            node_norm_std = torch.zeros(len(idp.chemical_symbol_to_type), idp.orbpair_irreps.num_irreps)
            node_scalar_ave = torch.ones(len(idp.chemical_symbol_to_type), n_scalar)
            node_scalar_square_ave = torch.zeros(len(idp.chemical_symbol_to_type), n_scalar)
            node_scalar_std = torch.zeros(len(idp.chemical_symbol_to_type), n_scalar)
            edge_norm_ave = torch.ones(len(idp.chemical_symbol_to_type), idp.orbpair_irreps.num_irreps)
            edge_norm_std = torch.zeros(len(idp.chemical_symbol_to_type), idp.orbpair_irreps.num_irreps)
            edge_scalar_ave = torch.ones(len(idp.chemical_symbol_to_type), n_scalar)
            edge_scalar_std = torch.zeros(len(idp.chemical_symbol_to_type), n_scalar)

            onsite_mask = idp.mask_to_nrme[atomicdata[AtomicDataDict.ATOM_TYPE_KEY].flatten()]

            for at, tp in idp.chemical_symbol_to_type.items():
                at_mask = onsite_mask[atomicdata[AtomicDataDict.ATOM_TYPE_KEY]].flatten().eq(tp)
                n_at = at_mask.shape[0]
                
                if n_at > 0:
                    at_onsite = atomicdata[AtomicDataDict.NODE_FEATURES_KEY][at_mask]
                    for ir, s in enumerate(idp.orbpair_irreps.slices()):
                        sub_tensor = at_onsite[at][:, s]
                        if sub_tensor.shape[-1] == 1:
                            count_scalar += 1
                        if not torch.isnan(sub_tensor).all():
                            norms = torch.norm(sub_tensor, p=2, dim=1)
                            # we do a running avg and var here
                            node_norm_ave[tp][ir] = (node_norm_ave[tp][ir] * count_tp[tp] + norms.mean(dim=0)) / (count_tp[tp] + n_at)
                            node_square_ave[tp][ir] = (node_square_ave[tp][ir] * count_tp[tp] + norms.mean(dim=0)**2) / (count_tp[tp] + n_at)
                            node_norm_std[tp][ir] = (count_tp[tp] + n_at) / (count_tp[tp] + n_at - 1) * (node_square_ave[tp][ir] - node_norm_ave[tp][ir]**2)

                            if sub_tensor.shape[-1] == 1:
                                # is scalar
                                node_scalar_ave[tp][ir] = (node_scalar_ave[tp][ir] * count_tp[tp] + sub_tensor.mean()) / (count_tp[tp] + n_at)
                                node_scalar_square_ave[tp][ir] = (node_scalar_square_ave[tp][ir] * count_tp[tp] + sub_tensor.mean()**2) / (count_tp[tp] + n_at)
                                node_scalar_std[tp][ir] = (count_tp[tp] + n_at) / (count_tp[tp] + n_at - 1) * (node_scalar_square_ave[tp][ir] - node_scalar_ave[tp][ir]**2)

                    count_tp[tp] += n_at
            
            # edge statistics
            hopping_mask = idp.mask_to_erme[atomicdata[AtomicDataDict.EDGE_TYPE_KEY].flatten()]
            for bt, tp in idp.bond_to_type.items():
                bt_mask = hopping_mask[atomicdata[AtomicDataDict.EDGE_TYPE_KEY]].flatten().eq(tp)
                n_bt = bt_mask.shape[0]

                if n_bt > 0:
                    bt_hopping = atomicdata[AtomicDataDict.EDGE_FEATURES_KEY][bt_mask]
                    for ir, s in enumerate(idp.orbpair_irreps.slices()):
                        sub_tensor = bt_hopping[bt][:, s]
                        if sub_tensor.shape[-1] == 1:
                            count_scalar += 1
                    
                pass




        stats = {}
        stats["node"] =  self._E3nodespecies_stat()
        stats["edge"] = self._E3edgespecies_stat()

        return stats