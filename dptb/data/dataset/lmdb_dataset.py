import numpy as np
from typing import Tuple, Dict, Any, List, Callable, Union, Optional

import torch
from dptb.utils.tools import download_url, extract_zip

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
        info_files: dict,
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
        self.info_files = info_files # there should be one info file for one LMDB Dataset
            

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


        self.num_graphs = 0
        self.file_map = []
        self.index_map = []
        for file in self.info_files.keys():
            db_env = lmdb.open(os.path.join(self.root, file), readonly=True, lock=False)
            with db_env.begin() as txn:
                self.num_graphs += txn.stat()['entries']
                self.file_map += [file] * txn.stat()['entries']
                self.index_map += list(range(txn.stat()['entries']))
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
        db_env = lmdb.open(os.path.join(self.root, self.file_map[idx]), readonly=True, lock=False)
        with db_env.begin() as txn:
            data_dict = txn.get(self.index_map[int(idx)].to_bytes(length=4, byteorder='big'))
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
            **self.info_files[self.file_map[idx]]
        )

        # transform blocks to atomicdata features
        if self.get_Hamiltonian or self.get_DM or self.get_overlap:
            block_to_feature(atomicdata, self.type_mapper, blocks, overlap, self.orthogonal)
        
        return atomicdata
    
    def E3statistics(self, model: torch.nn.Module=None):

        if not self.get_Hamiltonian and not self.get_DM:
            return None
        
        assert self.transform is not None
        idp = self.transform
        
        e3h = E3Hamiltonian(basis=idp.basis, decompose=True)
        idp.get_irreps()
        sorted_irreps = idp.orbpair_irreps.sort()[0].simplify()
        n_scalar = sorted_irreps[0].mul if sorted_irreps[0].ir.l == 0 else 0

        # init a count dict of atom species
        count_at = {}
        for at, tp in idp.chemical_symbol_to_type.items():
            count_at[tp] = 0

        count_bt = {}
        for bt, tp in idp.bond_to_type.items():
            count_bt[tp] = 0

        # calculate norm & mean
        node_norm_ave = torch.zeros(len(idp.chemical_symbol_to_type), idp.orbpair_irreps.num_irreps)
        node_square_ave = torch.zeros(len(idp.chemical_symbol_to_type), idp.orbpair_irreps.num_irreps)
        node_norm_std = torch.ones(len(idp.chemical_symbol_to_type), idp.orbpair_irreps.num_irreps)
        node_scalar_ave = torch.zeros(len(idp.chemical_symbol_to_type), n_scalar)
        node_scalar_square_ave = torch.zeros(len(idp.chemical_symbol_to_type), n_scalar)
        node_scalar_std = torch.ones(len(idp.chemical_symbol_to_type), n_scalar)
        edge_norm_ave = torch.zeros(len(idp.bond_types), idp.orbpair_irreps.num_irreps)
        edge_square_ave = torch.zeros(len(idp.bond_types), idp.orbpair_irreps.num_irreps)
        edge_norm_std = torch.ones(len(idp.bond_types), idp.orbpair_irreps.num_irreps)
        edge_scalar_ave = torch.zeros(len(idp.bond_types), n_scalar)
        edge_scalar_square_ave = torch.zeros(len(idp.bond_types), n_scalar)
        edge_scalar_std = torch.ones(len(idp.bond_types), n_scalar)

        for idx in tqdm(range(self.len()), desc="Collecting E3 irreps statistics: "):
            with torch.no_grad():
                atomicdata = idp(self.get(idx=idx)).to_dict()
                if atomicdata[AtomicDataDict.EDGE_FEATURES_KEY].abs().sum() < 1e-7:
                    continue
                atomicdata = e3h(atomicdata)

                subcount_at = {}
                for at, tp in idp.chemical_symbol_to_type.items():
                    subcount_at[tp] = 0

                subcount_bt = {}
                for bt, tp in idp.bond_to_type.items():
                    subcount_bt[tp] = 0

                onsite_mask = idp.mask_to_nrme[atomicdata[AtomicDataDict.ATOM_TYPE_KEY].flatten()]

                for at, tp in idp.chemical_symbol_to_type.items():
                    count_scalar = 0
                    at_mask = onsite_mask[atomicdata[AtomicDataDict.ATOM_TYPE_KEY].flatten().eq(tp)]
                    n_at = at_mask.shape[0]
                    
                    if n_at > 0:
                        at_onsite = atomicdata[AtomicDataDict.NODE_FEATURES_KEY][atomicdata[AtomicDataDict.ATOM_TYPE_KEY].flatten().eq(tp)]
                        for ir, s in enumerate(idp.orbpair_irreps.slices()):
                            sub_tensor = at_onsite[:, s]
                            if sub_tensor.shape[-1] == 1:
                                count_scalar += 1
                            norms = torch.norm(sub_tensor, p=2, dim=1)
                            # we do a running avg and var here
                            node_norm_ave[tp][ir] = (node_norm_ave[tp][ir] * count_at[tp] + norms.sum(dim=0)) / (count_at[tp] + n_at)
                            node_square_ave[tp][ir] = (node_square_ave[tp][ir] * count_at[tp] + (norms**2).sum(dim=0)) / (count_at[tp] + n_at)
                            if count_at[tp] + n_at > 1:
                                node_norm_std[tp][ir] = torch.nan_to_num(torch.sqrt((count_at[tp] + n_at) / (count_at[tp] + n_at - 1) * (node_square_ave[tp][ir] - node_norm_ave[tp][ir]**2)), nan=0.0)
                            else:
                                node_norm_std[tp][ir] = 1.0

                            if sub_tensor.shape[-1] == 1:
                                # is scalar
                                node_scalar_ave[tp][count_scalar-1] = (node_scalar_ave[tp][count_scalar-1] * count_at[tp] + sub_tensor.sum()) / (count_at[tp] + n_at)
                                node_scalar_square_ave[tp][count_scalar-1] = (node_scalar_square_ave[tp][count_scalar-1] * count_at[tp] + (sub_tensor**2).sum()) / (count_at[tp] + n_at)
                                if count_at[tp] + n_at > 1:
                                    node_scalar_std[tp][count_scalar-1] = torch.nan_to_num(torch.sqrt((count_at[tp] + n_at) / (count_at[tp] + n_at - 1) * (node_scalar_square_ave[tp][count_scalar-1] - node_scalar_ave[tp][count_scalar-1]**2)), nan=0.0)
                                else:    
                                    node_scalar_std[tp][count_scalar-1] = 1.0
                        subcount_at[tp] = n_at
                        count_at[tp] += n_at
                assert sum(subcount_at.values()) == atomicdata[AtomicDataDict.POSITIONS_KEY].shape[0]
                
                # edge statistics
                hopping_mask = idp.mask_to_erme[atomicdata[AtomicDataDict.EDGE_TYPE_KEY].flatten()]
                for bt, tp in idp.bond_to_type.items():
                    count_scalar = 0
                    bt_mask = hopping_mask[atomicdata[AtomicDataDict.EDGE_TYPE_KEY].flatten().eq(tp)]
                    n_bt = bt_mask.shape[0]

                    if n_bt > 0:
                        bt_hopping = atomicdata[AtomicDataDict.EDGE_FEATURES_KEY][atomicdata[AtomicDataDict.EDGE_TYPE_KEY].flatten().eq(tp)]
                        for ir, s in enumerate(idp.orbpair_irreps.slices()):
                            sub_tensor = bt_hopping[:, s]
                            if sub_tensor.shape[-1] == 1:
                                count_scalar += 1

                            norms = torch.norm(sub_tensor, p=2, dim=1)
                            # we do a running avg and var here
                            edge_norm_ave[tp][ir] = (edge_norm_ave[tp][ir] * count_bt[tp] + norms.sum(dim=0)) / (count_bt[tp] + n_bt)
                            edge_square_ave[tp][ir] = (edge_square_ave[tp][ir] * count_bt[tp] + (norms**2).sum(dim=0)) / (count_bt[tp] + n_bt)
                            if count_bt[tp] + n_bt > 1:
                                edge_norm_std[tp][ir] = torch.nan_to_num(torch.sqrt((count_bt[tp] + n_bt) / (count_bt[tp] + n_bt - 1) * (edge_square_ave[tp][ir] - edge_norm_ave[tp][ir]**2)), nan=0.0)
                            else:
                                edge_norm_std[tp][ir] = 1.0
                            if sub_tensor.shape[-1] == 1:
                                # is scalar
                                edge_scalar_ave[tp][count_scalar-1] = (edge_scalar_ave[tp][count_scalar-1] * count_bt[tp] + sub_tensor.sum()) / (count_bt[tp] + n_bt)
                                edge_scalar_square_ave[tp][count_scalar-1] = (edge_scalar_square_ave[tp][count_scalar-1] * count_bt[tp] + (sub_tensor**2).sum()) / (count_bt[tp] + n_bt)
                                if count_bt[tp] + n_bt > 1:
                                    edge_scalar_std[tp][count_scalar-1] = torch.nan_to_num(torch.sqrt((count_bt[tp] + n_bt) / (count_bt[tp] + n_bt - 1) * (edge_scalar_square_ave[tp][count_scalar-1] - edge_scalar_ave[tp][count_scalar-1]**2)), nan=0.0)
                                else:
                                    edge_scalar_std[tp][count_scalar-1] = 1.0
                                    
                        subcount_bt[tp] = n_bt
                        count_bt[tp] += n_bt
                assert sum(subcount_bt.values()) == atomicdata[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
                    
        stats = {}
        stats["node"] = {
            "norm_ave": node_norm_ave,
            "norm_std": node_norm_std,
            "scalar_ave": node_scalar_ave,
            "scalar_std": node_scalar_std
        }
        stats["edge"] = {
            "norm_ave": edge_norm_ave,
            "norm_std": edge_norm_std,
            "scalar_ave": edge_scalar_ave,
            "scalar_std": edge_scalar_std,
        }

        if model is not None:
            # initilize the model param with statistics
            scalar_mask = torch.BoolTensor([ir.dim==1 for ir in model.idp.orbpair_irreps])
            node_shifts = stats["node"]["scalar_ave"]
            node_scales = stats["node"]["norm_ave"]
            node_scales[:,scalar_mask] = stats["node"]["scalar_std"]

            edge_shifts = stats["edge"]["scalar_ave"]
            edge_scales = stats["edge"]["norm_ave"]
            edge_scales[:,scalar_mask] = stats["edge"]["scalar_std"]
            model.node_prediction_h.set_scale_shift(scales=node_scales, shifts=node_shifts)
            model.edge_prediction_h.set_scale_shift(scales=edge_scales, shifts=edge_shifts)

        return stats