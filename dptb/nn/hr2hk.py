


import torch
from dptb.utils.constants import h_all_types, anglrMId, atomic_num_dict, atomic_num_dict_r
from typing import Tuple, Union, Dict
from dptb.data.transforms import OrbitalMapper
from dptb.data import AtomicDataDict
import re

class HR2HK(torch.nn.Module):
    def __init__(
            self, 
            basis: Dict[str, Union[str, list], None]=None,
            idp: Union[OrbitalMapper, None]=None,
            edge_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
            node_field: str = AtomicDataDict.NODE_FEATURES_KEY,
            out_field: str = AtomicDataDict.HAMILTONIAN_KEY,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu")
            ):
        super(HR2HK, self).__init__()

        assert basis is not None or idp is not None, "Either basis or idp should be provided."

        self.dtype = dtype
        self.device = device
        if self.basis is None:
            self.idp = OrbitalMapper(basis, method="e3tb")
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            self.idp = idp
            self.basis = self.idp.basis


        self.idp.get_nodetype_maps()
        self.idp.get_pairtype_maps()

        self.edge_field = edge_field
        self.node_field = node_field
        self.out_field = out_field

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        # construct bond wise hamiltonian block from obital pair wise node/edge features
        orbpair_hopping = data[self.edge_field]
        orbpair_onsite = data[self.node_field]
        bondwise_hopping = torch.zeros_like(orbpair_hopping).reshape(-1, self.idp.full_basis_norb, self.idp.full_basis_norb)
        onsite_block = torch.zeros_like(orbpair_onsite).reshape(-1, self.idp.full_basis_norb, self.idp.full_basis_norb)

        ist = 0
        for i,iorb in enumerate(self.idp.full_basis):
            jst = 0
            li = anglrMId(re.findall(r"[a-zA-Z]+", iorb)[0])
            for j,jorb in enumerate(self.idp.full_basis):
                orbpair = iorb + "-" + jorb
                lj = anglrMId(re.findall(r"[a-zA-Z]+", jorb)[0])
                bondwise_hopping[:,ist:ist+2*li+1,jst:jst+2*lj+1] = orbpair_hopping[:,self.idp.pair_maps[orbpair]].reshape(-1, 2*li+1, 2*lj+1)
                
                if i <= j:
                    onsite_block[:,ist:ist+2*li+1,jst:jst+2*lj+1] = orbpair_onsite[:,self.idp.node_maps[orbpair]].reshape(-1, 2*li+1, 2*lj+1)
                else:
                    onsite_block[:,ist:ist+2*li+1,jst:jst+2*lj+1] = onsite_block[:,jst:jst+2*lj+1,ist:ist+2*li+1].transpose(1,2)
                jst += 2*lj+1
            ist += 2*li+1


        # R2K procedure can be done for all kpoint at once, try to implement this.
        all_norb = sum([data[AtomicDataDict.ATOM_TYPE_KEY].eq(self.idp.transform_atom(atomic_num_dict[ia])).sum() * self.idp.atom_norb[ia] for ia in self.idp.basis.keys()])
        block = torch.zeros(data[AtomicDataDict.KPOINT_KEY].shaoe[0], all_norb, all_norb, dtype=self.dtype, device=self.device)

        atom_id_to_indices = {}
        ist = 0
        for i, oblock in enumerate(onsite_block):
            mask = self.idp.mask_to_basis[atomic_num_dict_r[data[AtomicDataDict.ATOMIC_NUMBERS_KEY][i]]]
            masked_oblock = oblock[mask][:,mask]
            block[:,ist:ist+masked_oblock.shape[0],ist:ist+masked_oblock.shape[1]] = 0.5 * masked_oblock.squeeze(0)
            atom_id_to_indices[i] = slice(ist, ist+masked_oblock.shape[0])
            ist += masked_oblock.shape[0]
        
        for i, hblock in enumerate(bondwise_hopping):
            iatom = data[AtomicDataDict.EDGE_INDEX_KEY][0][i]
            jatom = data[AtomicDataDict.EDGE_INDEX_KEY][1][i]
            iatom_indices = atom_id_to_indices[iatom]
            jatom_indices = atom_id_to_indices[jatom]
            imask = self.idp.mask_to_basis[atomic_num_dict_r[data[AtomicDataDict.ATOMIC_NUMBERS_KEY][iatom]]]
            jmask = self.idp.mask_to_basis[atomic_num_dict_r[data[AtomicDataDict.ATOMIC_NUMBERS_KEY][jatom]]]
            masked_hblock = hblock[imask][:,jmask]
            block[:,iatom_indices,jatom_indices] = masked_hblock.squeeze(0) * torch.exp(-1j * 2 * torch.pi * data[AtomicDataDict.KPOINT_KEY] @ data[AtomicDataDict.EDGE_CELL_SHIFT_KEY][i]).reshape(-1,1,1)

        block = block + block.transpose(1,2).conj()
        block.contiguous()

        data[self.out_field] = block

        return data

