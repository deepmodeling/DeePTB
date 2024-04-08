import torch
from dptb.utils.constants import h_all_types, anglrMId, atomic_num_dict, atomic_num_dict_r
from typing import Tuple, Union, Dict
from dptb.data.transforms import OrbitalMapper
from dptb.data import AtomicDataDict
import re

class HR2HK(torch.nn.Module):
    def __init__(
            self, 
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            edge_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
            node_field: str = AtomicDataDict.NODE_FEATURES_KEY,
            out_field: str = AtomicDataDict.HAMILTONIAN_KEY,
            overlap: bool = False,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
            ):
        super(HR2HK, self).__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype
        self.device = device
        self.overlap = overlap

        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb", device=self.device)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            assert idp.method == "e3tb", "The method of idp should be e3tb."
            self.idp = idp
        
        self.basis = self.idp.basis
        self.idp.get_orbpair_maps()

        self.edge_field = edge_field
        self.node_field = node_field
        self.out_field = out_field

        self.atom_norbs = []

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        # construct bond wise hamiltonian block from obital pair wise node/edge features
        # we assume the edge feature have the similar format as the node feature, which is reduced from orbitals index oj-oi with j>i

        orbpair_hopping = data[self.edge_field]
        orbpair_onsite = data.get(self.node_field)
        bondwise_hopping = torch.zeros((len(orbpair_hopping), self.idp.full_basis_norb, self.idp.full_basis_norb), dtype=self.dtype, device=self.device)
        bondwise_hopping.to(self.device)
        bondwise_hopping.type(self.dtype)
        onsite_block = torch.zeros((len(data[AtomicDataDict.ATOM_TYPE_KEY]), self.idp.full_basis_norb, self.idp.full_basis_norb,), dtype=self.dtype, device=self.device)

        ist = 0
        for i,iorb in enumerate(self.idp.full_basis):
            jst = 0
            li = anglrMId[re.findall(r"[a-zA-Z]+", iorb)[0]]
            for j,jorb in enumerate(self.idp.full_basis):
                orbpair = iorb + "-" + jorb
                lj = anglrMId[re.findall(r"[a-zA-Z]+", jorb)[0]]
                
                # constructing hopping blocks
                if iorb == jorb:
                    factor = 0.5  # for diagonal elements, we need to divide by 2 for later conjugate transpose operation to construct the whole Hamiltonian
                else:
                    factor = 1.0

                if i <= j:
                    bondwise_hopping[:,ist:ist+2*li+1,jst:jst+2*lj+1] = factor * orbpair_hopping[:,self.idp.orbpair_maps[orbpair]].reshape(-1, 2*li+1, 2*lj+1)


                # constructing onsite blocks
                if self.overlap:
                    if iorb == jorb:
                        onsite_block[:, ist:ist+2*li+1, jst:jst+2*lj+1] = factor * torch.eye(2*li+1, dtype=self.dtype, device=self.device).reshape(1, 2*li+1, 2*lj+1).repeat(onsite_block.shape[0], 1, 1)
                else:
                    if i <= j:
                        onsite_block[:,ist:ist+2*li+1,jst:jst+2*lj+1] = factor * orbpair_onsite[:,self.idp.orbpair_maps[orbpair]].reshape(-1, 2*li+1, 2*lj+1)

                jst += 2*lj+1
            ist += 2*li+1
        self.onsite_block = onsite_block
        self.bondwise_hopping = bondwise_hopping


        # R2K procedure can be done for all kpoint at once.
        all_norb = self.idp.atom_norb[data[AtomicDataDict.ATOM_TYPE_KEY]].sum()
        block = torch.zeros(data[AtomicDataDict.KPOINT_KEY].shape[0], all_norb, all_norb, dtype=self.dtype, device=self.device)
        block = torch.complex(block, torch.zeros_like(block))

        atom_id_to_indices = {}
        ist = 0
        for i, oblock in enumerate(onsite_block):
            mask = self.idp.mask_to_basis[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()[i]]
            masked_oblock = oblock[mask][:,mask]
            block[:,ist:ist+masked_oblock.shape[0],ist:ist+masked_oblock.shape[1]] = masked_oblock.squeeze(0)
            atom_id_to_indices[i] = slice(ist, ist+masked_oblock.shape[0])
            self.atom_norbs.append(masked_oblock.shape[0])
            ist += masked_oblock.shape[0]

        for i, hblock in enumerate(bondwise_hopping):
            iatom = data[AtomicDataDict.EDGE_INDEX_KEY][0][i]
            jatom = data[AtomicDataDict.EDGE_INDEX_KEY][1][i]
            iatom_indices = atom_id_to_indices[int(iatom)]
            jatom_indices = atom_id_to_indices[int(jatom)]
            imask = self.idp.mask_to_basis[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()[iatom]]
            jmask = self.idp.mask_to_basis[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()[jatom]]
            masked_hblock = hblock[imask][:,jmask]

            block[:,iatom_indices,jatom_indices] += masked_hblock.squeeze(0).type_as(block) * \
                torch.exp(-1j * 2 * torch.pi * (data[AtomicDataDict.KPOINT_KEY] @ data[AtomicDataDict.EDGE_CELL_SHIFT_KEY][i])).reshape(-1,1,1)

        block = block + block.transpose(1,2).conj()
        block = block.contiguous()

        data[self.out_field] = block

        return data
    