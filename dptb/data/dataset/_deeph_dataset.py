from typing import Dict, Any, List, Callable, Union, Optional
import os
import numpy as np
import h5py

import torch

from .. import (
    AtomicData,
    AtomicDataDict,
)
from ..AtomicDataDict import with_edge_vectors
from ..transforms import TypeMapper, OrbitalMapper
from ._base_datasets import AtomicDataset, AtomicInMemoryDataset
from dptb.nn.hamiltonian import E3Hamiltonian
from dptb.data.interfaces.ham_to_feature import openmx_to_deeptb
from tqdm import tqdm

orbitalLId = {0:"s", 1:"p", 2:"d", 3:"f"}

class DeePHE3Dataset(AtomicInMemoryDataset):
    def __init__(
            self,
            root: str,
            info_files: Dict[str, Dict],
            url: Optional[str] = None,                    # seems useless but can't be remove
            include_frames: Optional[List[int]] = None,   # maybe support in future
            type_mapper: TypeMapper = None,
            get_Hamiltonian: bool = False,
            get_eigenvalues: bool = False,
    ):
        
        self.root = root
        self.url = url
        self.info_files = info_files
        # The following flags are stored to label dataset.
        self.get_Hamiltonian = get_Hamiltonian
        self.get_eigenvalues = get_eigenvalues
        
        # load all data files            
        self.raw_data = []
        self.data_options = {}
        for file in self.info_files.keys():
            # get the info here
            info = info_files[file]
            assert "r_max" in info
            assert "pbc" in info
            subdata = os.path.join(self.root, file)
            self.raw_data.append(subdata)
            self.data_options[subdata] = info

        # The AtomicData_options is never used here.
        # Because we always return a list of AtomicData object in `get_data()`.
        # That is, AtomicInMemoryDataset will not use AtomicData_options to build any AtomicData here.
        super().__init__(
            file_name=None, # this seems not important too.
            url=url,
            root=root,
            AtomicData_options={},  # we do not pass anything here.
            include_frames=include_frames,
            type_mapper=type_mapper,
        )

    def get_data(self):
        all_data = []
        for subpath in tqdm(self.raw_data, desc="Loading data"):
            # the type_mapper here is loaded in PyG `dataset` type as `transform` attritube
            # so the OrbitalMapper can be accessed by self.transform here
            info = self.data_options[subpath]
            atomic_data = AtomicData.from_points(
                pos = np.loadtxt(os.path.join(subpath, "site_positions.dat")).T,
                cell = np.loadtxt(os.path.join(subpath, "lat.dat")).T,
                atomic_numbers = np.loadtxt(os.path.join(subpath, "element.dat")),
                pbc = info["pbc"],
                r_max=info["r_max"],
                er_max=info.get("er_max", None),
                oer_max=info.get("oer_max", None)
            )
            idp = self.type_mapper
            openmx_to_deeptb(atomic_data, idp, os.path.join(subpath, "./hamiltonians.h5"))
            all_data.append(atomic_data)
        return all_data
    
    @property
    def raw_file_names(self):
        # TODO: this is not implemented.
        return "Null"

    @property
    def raw_dir(self):
        # TODO: this is not implemented.
        return self.root

    def E3statistics(self, decay=False):
        assert self.transform is not None
        idp = self.transform

        if self.data[AtomicDataDict.EDGE_FEATURES_KEY].abs().sum() < 1e-7:
            return None
        
        typed_dataset = idp(self.data.clone().to_dict())
        e3h = E3Hamiltonian(basis=idp.basis, decompose=True)
        with torch.no_grad():
            typed_dataset = e3h(typed_dataset)

        stats = {}
        stats["node"] =  self._E3nodespecies_stat(typed_dataset=typed_dataset)
        stats["edge"] = self._E3edgespecies_stat(typed_dataset=typed_dataset, decay=decay)

        return stats
    
    def _E3edgespecies_stat(self, typed_dataset, decay):
        # we get the bond type marked dataset first
        idp = self.transform
        typed_dataset = typed_dataset

        idp.get_irreps(no_parity=False)
        irrep_slices = idp.orbpair_irreps.slices()

        features = typed_dataset["edge_features"]
        hopping_block_mask = idp.mask_to_erme[typed_dataset["edge_type"].flatten()]
        typed_hopping = {}
        for bt, tp in idp.bond_to_type.items():
            hopping_tp_mask = hopping_block_mask[typed_dataset["edge_type"].flatten().eq(tp)]
            hopping_tp = features[typed_dataset["edge_type"].flatten().eq(tp)]
            filtered_vecs = torch.where(hopping_tp_mask, hopping_tp, torch.tensor(float('nan')))
            typed_hopping[bt] = filtered_vecs

        sorted_irreps = idp.orbpair_irreps.sort()[0].simplify()
        n_scalar = sorted_irreps[0].mul if sorted_irreps[0].ir.l == 0 else 0
        
        # calculate norm & mean
        typed_norm = {}
        typed_norm_ave = torch.ones(len(idp.bond_to_type), idp.orbpair_irreps.num_irreps)
        typed_norm_std = torch.zeros(len(idp.bond_to_type), idp.orbpair_irreps.num_irreps)
        typed_scalar_ave = torch.ones(len(idp.bond_to_type), n_scalar)
        typed_scalar_std = torch.zeros(len(idp.bond_to_type), n_scalar)
        for bt, tp in idp.bond_to_type.items():
            norms_per_irrep = []
            count_scalar = 0
            for ir, s in enumerate(irrep_slices):
                sub_tensor = typed_hopping[bt][:, s]
                # dump the nan blocks here
                if sub_tensor.shape[-1] == 1:
                    count_scalar += 1
                if not torch.isnan(sub_tensor).all():
                    # update the mean and ave
                    norms = torch.norm(sub_tensor, p=2, dim=1) # shape: [n_edge]
                    if sub_tensor.shape[-1] == 1:
                        # it's a scalar
                        typed_scalar_ave[tp][count_scalar-1] = sub_tensor.mean()
                        typed_scalar_std[tp][count_scalar-1] = sub_tensor.std()
                    typed_norm_ave[tp][ir] = norms.mean()
                    typed_norm_std[tp][ir] = norms.std()
                else:
                    norms = torch.ones_like(sub_tensor[:, 0])

                if decay:
                    norms_per_irrep.append(norms)

            assert count_scalar <= n_scalar
            # shape of typed_norm: (n_irreps, n_edges)

            if decay:
                typed_norm[bt] = torch.stack(norms_per_irrep)

        edge_stats = {
            "norm_ave": typed_norm_ave,
            "norm_std": typed_norm_std,
            "scalar_ave": typed_scalar_ave,
            "scalar_std": typed_scalar_std,
        }
        
        if decay:
            typed_dataset = with_edge_vectors(typed_dataset)
            decay = {}
            for bt, tp in idp.bond_to_type.items():
                decay_bt = {}
                lengths_bt = typed_dataset["edge_lengths"][typed_dataset["edge_type"].flatten().eq(tp)]
                sorted_lengths, indices = lengths_bt.sort() # from small to large
                # sort the norms by irrep l
                sorted_norms = typed_norm[bt][idp.orbpair_irreps.sort().inv, :]
                # sort the norms by edge length
                sorted_norms = sorted_norms[:, indices]
                decay_bt["edge_length"] = sorted_lengths
                decay_bt["norm_decay"] = sorted_norms
                decay[bt] = decay_bt
            
            edge_stats["decay"] = decay
        
        return edge_stats

    def _E3nodespecies_stat(self, typed_dataset):
        # we get the type marked dataset first
        idp = self.transform
        typed_dataset = typed_dataset

        idp.get_irreps(no_parity=False)
        irrep_slices = idp.orbpair_irreps.slices()

        sorted_irreps = idp.orbpair_irreps.sort()[0].simplify()
        n_scalar = sorted_irreps[0].mul if sorted_irreps[0].ir.l == 0 else 0

        features = typed_dataset["node_features"]
        onsite_block_mask = idp.mask_to_nrme[typed_dataset["atom_types"].flatten()]
        typed_onsite = {}
        for at, tp in idp.chemical_symbol_to_type.items():
            onsite_tp_mask = onsite_block_mask[typed_dataset["atom_types"].flatten().eq(tp)]
            onsite_tp = features[typed_dataset["atom_types"].flatten().eq(tp)]
            filtered_vecs = torch.where(onsite_tp_mask, onsite_tp, torch.tensor(float('nan')))
            typed_onsite[at] = filtered_vecs
        
        # calculate norm & mean
        typed_norm_ave = torch.ones(len(idp.chemical_symbol_to_type), idp.orbpair_irreps.num_irreps)
        typed_norm_std = torch.zeros(len(idp.chemical_symbol_to_type), idp.orbpair_irreps.num_irreps)
        typed_scalar_ave = torch.ones(len(idp.chemical_symbol_to_type), n_scalar)
        typed_scalar_std = torch.zeros(len(idp.chemical_symbol_to_type), n_scalar)
        for at, tp in idp.chemical_symbol_to_type.items():
            count_scalar = 0
            for ir, s in enumerate(irrep_slices):
                sub_tensor = typed_onsite[at][:, s]
                # dump the nan blocks here
                if sub_tensor.shape[-1] == 1:
                    count_scalar += 1
                if not torch.isnan(sub_tensor).all():

                    norms = torch.norm(sub_tensor, p=2, dim=1)
                    typed_norm_ave[tp][ir] = norms.mean()
                    if norms.numel() > 1:
                        typed_norm_std[tp][ir] = norms.std()
                    else:
                        typed_norm_std[tp][ir] = 1.0
                    if s.stop - s.start == 1:
                        # it's a scalar
                        typed_scalar_ave[tp][count_scalar-1] = sub_tensor.mean()
                        if sub_tensor.numel() > 1:
                            typed_scalar_std[tp][count_scalar-1] = sub_tensor.std()
                        else:
                            typed_scalar_std[tp][count_scalar-1] = 1.0

        edge_stats = {
            "norm_ave": typed_norm_ave,
            "norm_std": typed_norm_std,
            "scalar_ave": typed_scalar_ave,
            "scalar_std": typed_scalar_std,
        }

        return edge_stats