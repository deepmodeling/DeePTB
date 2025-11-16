import torch
from dptb.utils.constants import h_all_types, anglrMId, atomic_num_dict, atomic_num_dict_r
from typing import Tuple, Union, Dict
from dptb.data.transforms import OrbitalMapper
from dptb.data import AtomicDataDict
import re
from dptb.utils.tools import float2comlex


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
        self.ctype = float2comlex(dtype)

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
        self.idp.get_orbpair_soc_maps()

        self.edge_field = edge_field
        self.node_field = node_field
        self.out_field = out_field

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        # construct bond wise hamiltonian block from obital pair wise node/edge features
        # we assume the edge feature have the similar format as the node feature, which is reduced from orbitals index oj-oi with j>i
        
        orbpair_hopping = data[self.edge_field]
        orbpair_onsite = data.get(self.node_field)
        bondwise_hopping = torch.zeros((len(orbpair_hopping), self.idp.full_basis_norb, self.idp.full_basis_norb), dtype=self.dtype, device=self.device)
        bondwise_hopping.to(self.device)
        bondwise_hopping.type(self.dtype)
        onsite_block = torch.zeros((len(data[AtomicDataDict.ATOM_TYPE_KEY]), self.idp.full_basis_norb, self.idp.full_basis_norb,), dtype=self.dtype, device=self.device)
        kpoints = data[AtomicDataDict.KPOINT_KEY]
        if kpoints.is_nested:
            assert kpoints.size(0) == 1
            kpoints = kpoints[0]

        soc = data.get(AtomicDataDict.NODE_SOC_SWITCH_KEY, False)
        if isinstance(soc, torch.Tensor):
            soc = soc.all()
        if soc:
            if self.overlap:
                raise NotImplementedError("Overlap is not implemented for SOC.")
            
            orbpair_soc = data[AtomicDataDict.NODE_SOC_KEY]
            soc_upup_block = torch.zeros((len(data[AtomicDataDict.ATOM_TYPE_KEY]), self.idp.full_basis_norb, self.idp.full_basis_norb), dtype=self.ctype, device=self.device)
            soc_updn_block = torch.zeros((len(data[AtomicDataDict.ATOM_TYPE_KEY]), self.idp.full_basis_norb, self.idp.full_basis_norb), dtype=self.ctype, device=self.device)

        ist = 0
        for i,iorb in enumerate(self.idp.full_basis):
            jst = 0
            li = anglrMId[re.findall(r"[a-zA-Z]+", iorb)[0]]
            for j,jorb in enumerate(self.idp.full_basis):
                orbpair = iorb + "-" + jorb
                lj = anglrMId[re.findall(r"[a-zA-Z]+", jorb)[0]]
                
                # constructing hopping blocks
                if iorb == jorb:
                    factor = 0.5
                else:
                    factor = 1.0

                if i <= j:
                    bondwise_hopping[:,ist:ist+2*li+1,jst:jst+2*lj+1] = factor * orbpair_hopping[:,self.idp.orbpair_maps[orbpair]].reshape(-1, 2*li+1, 2*lj+1)


                # constructing onsite blocks
                if self.overlap:
                    # if iorb == jorb:
                    #     onsite_block[:, ist:ist+2*li+1, jst:jst+2*lj+1] = factor * torch.eye(2*li+1, dtype=self.dtype, device=self.device).reshape(1, 2*li+1, 2*lj+1).repeat(onsite_block.shape[0], 1, 1)
                    if i <= j:
                        onsite_block[:,ist:ist+2*li+1,jst:jst+2*lj+1] = factor * orbpair_onsite[:,self.idp.orbpair_maps[orbpair]].reshape(-1, 2*li+1, 2*lj+1)
                else:
                    if i <= j:
                        onsite_block[:,ist:ist+2*li+1,jst:jst+2*lj+1] = factor * orbpair_onsite[:,self.idp.orbpair_maps[orbpair]].reshape(-1, 2*li+1, 2*lj+1)
                    
                    if soc and i==j:
                        soc_updn_tmp = orbpair_soc[:,self.idp.orbpair_soc_maps[orbpair]].reshape(-1, 2*li+1, 2*(2*lj+1))
                        soc_upup_block[:,ist:ist+2*li+1,jst:jst+2*lj+1] = soc_updn_tmp[:, :2*li+1,:2*lj+1]
                        soc_updn_block[:,ist:ist+2*li+1,jst:jst+2*lj+1] = soc_updn_tmp[:, :2*li+1,2*lj+1:]
                
                jst += 2*lj+1
            ist += 2*li+1
        self.onsite_block = onsite_block
        self.bondwise_hopping = bondwise_hopping
        if soc:
            self.soc_upup_block = soc_upup_block
            self.soc_updn_block = soc_updn_block

        # R2K procedure can be done for all kpoint at once.
        all_norb = self.idp.atom_norb[data[AtomicDataDict.ATOM_TYPE_KEY]].sum()
        block = torch.zeros(kpoints.shape[0], all_norb, all_norb, dtype=self.ctype, device=self.device)
        # block = torch.complex(block, torch.zeros_like(block))
        # if data[AtomicDataDict.NODE_SOC_SWITCH_KEY].all():
        #     block_uu = torch.zeros(data[AtomicDataDict.KPOINT_KEY].shape[0], all_norb, all_norb, dtype=self.ctype, device=self.device)
        #     block_ud = torch.zeros(data[AtomicDataDict.KPOINT_KEY].shape[0], all_norb, all_norb, dtype=self.ctype, device=self.device)
        atom_id_to_indices = {}
        ist = 0
        for i, oblock in enumerate(onsite_block):
            mask = self.idp.mask_to_basis[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()[i]]
            masked_oblock = oblock[mask][:,mask]
            block[:,ist:ist+masked_oblock.shape[0],ist:ist+masked_oblock.shape[1]] = masked_oblock.squeeze(0)
            atom_id_to_indices[i] = slice(ist, ist+masked_oblock.shape[0])
            ist += masked_oblock.shape[0]
        
        # if data[AtomicDataDict.NODE_SOC_SWITCH_KEY].all():
        #     ist = 0
        #     for i, soc_block in enumerate(soc_upup_block):
        #         mask = self.idp.mask_to_basis[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()[i]]
        #         masked_soc_block = soc_block[mask][:,mask]
        #         block_uu[:,ist:ist+masked_soc_block.shape[0],ist:ist+masked_soc_block.shape[1]] = masked_soc_block.squeeze(0)
        #         ist += masked_soc_block.shape[0]
        #     ist = 0
        #     for i, soc_block in enumerate(soc_updn_block):
        #         mask = self.idp.mask_to_basis[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()[i]]
        #         masked_soc_block = soc_block[mask][:,mask]
        #         block_ud[:,ist:ist+masked_soc_block.shape[0],ist:ist+masked_soc_block.shape[1]] = masked_soc_block.squeeze(0)
        #         ist += masked_soc_block.shape[0]

        for i, hblock in enumerate(bondwise_hopping):
            iatom = data[AtomicDataDict.EDGE_INDEX_KEY][0][i]
            jatom = data[AtomicDataDict.EDGE_INDEX_KEY][1][i]
            iatom_indices = atom_id_to_indices[int(iatom)]
            jatom_indices = atom_id_to_indices[int(jatom)]
            imask = self.idp.mask_to_basis[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()[iatom]]
            jmask = self.idp.mask_to_basis[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()[jatom]]
            masked_hblock = hblock[imask][:,jmask]

            block[:,iatom_indices,jatom_indices] += masked_hblock.squeeze(0).type_as(block) * \
                torch.exp(-1j * 2 * torch.pi * (kpoints @ data[AtomicDataDict.EDGE_CELL_SHIFT_KEY][i])).reshape(-1,1,1)

        block = block + block.transpose(1,2).conj()
        block = block.contiguous()
        
        if soc:
            HK_SOC = torch.zeros(kpoints.shape[0], 2*all_norb, 2*all_norb, dtype=self.ctype, device=self.device)
            #HK_SOC[:,:all_norb,:all_norb] = block + block_uu
            #HK_SOC[:,:all_norb,all_norb:] = block_ud
            #HK_SOC[:,all_norb:,:all_norb] = block_ud.conj()
            #HK_SOC[:,all_norb:,all_norb:] = block + block_uu.conj()
            ist = 0
            assert len(soc_upup_block) == len(soc_updn_block)
            for i in range(len(soc_upup_block)):
                assert soc_upup_block[i].shape == soc_updn_block[i].shape
                mask = self.idp.mask_to_basis[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()[i]]
                masked_soc_upup_block = soc_upup_block[i][mask][:,mask]
                masked_soc_updn_block = soc_updn_block[i][mask][:,mask]
                HK_SOC[:,ist:ist+masked_soc_upup_block.shape[0],ist:ist+masked_soc_upup_block.shape[1]] = masked_soc_upup_block.squeeze(0)
                HK_SOC[:,ist:ist+masked_soc_updn_block.shape[0],ist+all_norb:ist+all_norb+masked_soc_updn_block.shape[1]] = masked_soc_updn_block.squeeze(0)
                assert masked_soc_upup_block.shape[0] == masked_soc_upup_block.shape[1]
                assert masked_soc_upup_block.shape[0] == masked_soc_updn_block.shape[0]
                
                ist += masked_soc_upup_block.shape[0]

            HK_SOC[:,all_norb:,:all_norb] = HK_SOC[:,:all_norb,all_norb:].conj()   
            HK_SOC[:,all_norb:,all_norb:] = HK_SOC[:,:all_norb,:all_norb].conj()  + block
            HK_SOC[:,:all_norb,:all_norb] = HK_SOC[:,:all_norb,:all_norb] + block  

            data[self.out_field] = HK_SOC
        else:
            data[self.out_field] = block

        return data


class HR2HK_Gamma_Only(torch.nn.Module):
    def __init__(
            self,
            basis: Dict[str, Union[str, list]] = None,
            idp: Union[OrbitalMapper, None] = None,
            edge_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
            node_field: str = AtomicDataDict.NODE_FEATURES_KEY,
            out_field: str = AtomicDataDict.HAMILTONIAN_KEY,
            overlap: bool = False,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(HR2HK_Gamma_Only, self).__init__()

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
        self.idp.get_orbpair_soc_maps()

        self.edge_field = edge_field
        self.node_field = node_field
        self.out_field = out_field

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """Gamma-point only version for isolated systems."""

        # Build orbital pair Hamiltonian blocks
        orbpair_hopping = data[self.edge_field]
        orbpair_onsite = data.get(self.node_field)
        bondwise_hopping = torch.zeros(
            (len(orbpair_hopping), self.idp.full_basis_norb, self.idp.full_basis_norb),
            dtype=self.dtype, device=self.device
        )
        onsite_block = torch.zeros(
            (len(data[AtomicDataDict.ATOM_TYPE_KEY]), self.idp.full_basis_norb, self.idp.full_basis_norb),
            dtype=self.dtype, device=self.device
        )

        ist = 0
        for i, iorb in enumerate(self.idp.full_basis):
            jst = 0
            li = anglrMId[re.findall(r"[a-zA-Z]+", iorb)[0]]
            for j, jorb in enumerate(self.idp.full_basis):
                orbpair = iorb + "-" + jorb
                lj = anglrMId[re.findall(r"[a-zA-Z]+", jorb)[0]]

                if iorb == jorb:
                    factor = 0.5
                else:
                    factor = 1.0

                if i <= j:
                    bondwise_hopping[:, ist:ist + 2 * li + 1, jst:jst + 2 * lj + 1] = factor * orbpair_hopping[:,
                                                                                               self.idp.orbpair_maps[
                                                                                                   orbpair]].reshape(-1,
                                                                                                                     2 * li + 1,
                                                                                                                     2 * lj + 1)

                if i <= j and orbpair_onsite is not None:
                    onsite_block[:, ist:ist + 2 * li + 1, jst:jst + 2 * lj + 1] = factor * orbpair_onsite[:,
                                                                                           self.idp.orbpair_maps[
                                                                                               orbpair]].reshape(-1,
                                                                                                                 2 * li + 1,
                                                                                                                 2 * lj + 1)

                jst += 2 * lj + 1
            ist += 2 * li + 1

        # Gamma-point processing: use real matrix
        all_norb = int(self.idp.atom_norb[data[AtomicDataDict.ATOM_TYPE_KEY]].sum().item())
        block = torch.zeros(1, all_norb, all_norb, dtype=self.dtype, device=self.device)

        atom_types_flat = data[AtomicDataDict.ATOM_TYPE_KEY].flatten()

        # Pre-compute atomic orbital slices and masks
        atom_slices = [0]
        atom_masks = []
        atom_types_int = []

        for i in range(atom_types_flat.shape[0]):
            atype = int(atom_types_flat[i].item()) if isinstance(atom_types_flat[i], torch.Tensor) else int(
                atom_types_flat[i])
            atom_types_int.append(atype)
            mask = self.idp.mask_to_basis[atype]
            atom_masks.append(mask)
            norb_this_atom = int(mask.sum().item())
            atom_slices.append(atom_slices[-1] + norb_this_atom)

        # Group atoms by type for batch processing - onsite part
        type_to_atoms = {}
        for i, atype in enumerate(atom_types_int):
            if atype not in type_to_atoms:
                type_to_atoms[atype] = []
            type_to_atoms[atype].append(i)

        for atype, atom_indices in type_to_atoms.items():
            mask = atom_masks[atom_indices[0]]
            for atom_idx in atom_indices:
                start_orb = atom_slices[atom_idx]
                end_orb = atom_slices[atom_idx + 1]
                norb = end_orb - start_orb

                if norb > 0:
                    oblock = onsite_block[atom_idx]
                    masked_oblock = oblock[mask][:, mask]
                    block[0, start_orb:end_orb, start_orb:end_orb] = masked_oblock

        # Create atomic index mapping
        atom_id_to_indices = {}
        for i in range(len(atom_slices) - 1):
            atom_id_to_indices[i] = slice(atom_slices[i], atom_slices[i + 1])

        # Process hopping terms: Gamma-point optimized version (no phase factors)
        edge_idx0 = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_idx1 = data[AtomicDataDict.EDGE_INDEX_KEY][1]

        # Group edges by atomic type pairs for batch processing
        edge_atom_types = []
        for i in range(len(edge_idx0)):
            iatom = int(edge_idx0[i].item()) if isinstance(edge_idx0[i], torch.Tensor) else int(edge_idx0[i])
            jatom = int(edge_idx1[i].item()) if isinstance(edge_idx1[i], torch.Tensor) else int(edge_idx1[i])
            edge_atom_types.append((iatom, jatom, atom_types_int[iatom], atom_types_int[jatom]))

        type_pair_to_edges = {}
        for edge_idx, (iatom, jatom, itype, jtype) in enumerate(edge_atom_types):
            type_pair = (itype, jtype)
            if type_pair not in type_pair_to_edges:
                type_pair_to_edges[type_pair] = []
            type_pair_to_edges[type_pair].append((edge_idx, iatom, jatom))

        # Process each type pair batch
        block2d = block[0]  # (all_norb, all_norb), 实数类型

        for (itype, jtype), edges_info in type_pair_to_edges.items():
            imask = self.idp.mask_to_basis[itype]
            jmask = self.idp.mask_to_basis[jtype]

            edge_indices = [e[0] for e in edges_info]
            iatoms = [e[1] for e in edges_info]
            jatoms = [e[2] for e in edges_info]

            # Batch extract masked hopping blocks
            hblocks = bondwise_hopping[edge_indices]
            masked_hblocks = hblocks[:, imask][:, :, jmask]

            # Write all blocks of this type pair at once
            for idx, (iatom, jatom) in enumerate(zip(iatoms, jatoms)):
                i_slice = atom_id_to_indices[iatom]
                j_slice = atom_id_to_indices[jatom]
                block2d[i_slice, j_slice] += masked_hblocks[idx]

        # Hermitianize
        block = block + block.transpose(1, 2)
        block = block.contiguous()

        data[self.out_field] = block[0]
        return data