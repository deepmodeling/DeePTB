import torch
import torch.nn as nn
from dptb.utils.constants import anglrMId
from typing import Union, Dict, Optional
from dptb.data.transforms import OrbitalMapper
from dptb.data import AtomicDataDict
import re
import math
from dptb.utils.tools import float2comlex


# ================== Helper Functions ==================

def recover_complex_tensor(tensor: torch.Tensor, is_complex_doubling: bool = True) -> torch.Tensor:
    if tensor.is_complex():
        return tensor
    if is_complex_doubling:
        if tensor.shape[-1] % 2 != 0:
            raise ValueError(f"Tensor shape {tensor.shape} last dim must be even for complex doubling.")
        n_real = tensor.shape[-1] // 2
        return torch.complex(tensor[..., :n_real], tensor[..., n_real:])
    return tensor.cfloat()


def _take_idp_tensor(
    tensor: torch.Tensor,
    index,
    result_device=None,
):
    if torch.is_tensor(index):
        tensor = tensor.to(device=index.device)
        out = tensor[index]
    else:
        out = tensor[index]

    if result_device is not None and torch.is_tensor(out):
        out = out.to(device=result_device)
    return out


# ================== HR2HK Module ==================

class HR2HK(nn.Module):
    def __init__(
            self,
            basis: Optional[Dict[str, Union[str, list]]] = None,
            idp: Optional[OrbitalMapper] = None,
            edge_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
            node_field: str = AtomicDataDict.NODE_FEATURES_KEY,
            out_field: str = AtomicDataDict.HAMILTONIAN_KEY,
            overlap: bool = False,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
            derivative: bool = False,
            out_derivative_field: str = AtomicDataDict.HAMILTONIAN_DERIV_KEY,
            gauge: bool = False
    ):
        super(HR2HK, self).__init__()

        # --- Basic Configuration ---
        if derivative:
            gauge = True
        self.gauge = gauge
        self.derivative = derivative

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype
        self.device = device
        self.overlap = overlap

        if hasattr(float2comlex, '__call__'):
            self.ctype = float2comlex(dtype)
        else:
            self.ctype = torch.complex64 if dtype == torch.float32 else torch.complex128

        self.edge_field = edge_field
        self.node_field = node_field
        self.out_field = out_field
        self.out_derivative_field = out_derivative_field

        # --- OrbitalMapper ---
        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb", device=self.device)
            if idp is not None:
                assert idp == self.idp, "The basis of provided idp and basis arg should be the same."
        else:
            assert idp is not None, "Either basis or idp (OrbitalMapper) should be provided."
            self.idp = idp

        self.basis = self.idp.basis
        if not hasattr(self.idp, "orbpair_maps"):
            self.idp.get_orbpair_maps()

        # --- Mode Detection ---
        self.has_soc = getattr(self.idp, "has_soc", False)
        self.soc_complex_doubling = getattr(self.idp, "soc_complex_doubling", self.has_soc)
        self.method = getattr(self.idp, "method", "e3tb")

        # S 矩阵强制使用 scalar 逻辑（因为 IO 中 Overlap 被存储为 scalar），H 矩阵如果是 SOC 则是 Full SOC
        self.is_full_soc = (self.method == "e3tb") and self.has_soc and (not self.overlap)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if self.gauge:
            data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)

        if self.is_full_soc:
            return self._forward_full_soc(data)
        else:
            return self._forward_scalar(data)

    def _forward_full_soc(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # Full SOC logic remains unchanged
        orbpair_hopping = data[self.edge_field]
        orbpair_onsite = data.get(self.node_field)
        kpoints = data[AtomicDataDict.KPOINT_KEY]
        if kpoints.is_nested: kpoints = kpoints[0]
        atom_types = data[AtomicDataDict.ATOM_TYPE_KEY]
        if atom_types.dim() > 1: atom_types = atom_types.flatten()
        num_atoms = len(atom_types)
        num_edges = orbpair_hopping.shape[0]

        local_spatial_dim = self.idp.full_basis_norb
        local_full_dim = local_spatial_dim * 2
        N_loc = local_spatial_dim
        bondwise_hopping = torch.zeros((num_edges, local_full_dim, local_full_dim),
                                       dtype=self.ctype, device=self.device)
        onsite_block = None
        if orbpair_onsite is not None:
            onsite_block = torch.zeros((num_atoms, local_full_dim, local_full_dim),
                                       dtype=self.ctype, device=self.device)

        def fill_blocks(target_tensor, features):
            ist = 0
            for i, iorb in enumerate(self.idp.full_basis):
                jst = 0
                li = anglrMId[re.findall(r"[a-zA-Z]", iorb)[0]]
                dim_i = 2 * li + 1
                for j, jorb in enumerate(self.idp.full_basis):
                    lj = anglrMId[re.findall(r"[a-zA-Z]", jorb)[0]]
                    dim_j = 2 * lj + 1
                    orbpair = iorb + "-" + jorb
                    if orbpair not in self.idp.orbpair_maps:
                        jst += dim_j
                        continue
                    sli = self.idp.orbpair_maps[orbpair]
                    val = features[:, sli]
                    val = recover_complex_tensor(val, self.soc_complex_doubling)

                    val_reshaped = val.view(-1, 4, dim_i, dim_j)
                    uu, ud, du, dd = val_reshaped[:, 0], val_reshaped[:, 1], val_reshaped[:, 2], val_reshaped[:, 3]
                    factor = 0.5

                    n_fill = val.shape[0]
                    target_tensor[:n_fill, ist:ist + dim_i, jst:jst + dim_j] = uu * factor
                    target_tensor[:n_fill, ist:ist + dim_i,
                    jst + local_spatial_dim:jst + local_spatial_dim + dim_j] = ud * factor
                    target_tensor[:n_fill, ist + local_spatial_dim:ist + local_spatial_dim + dim_i,
                    jst:jst + dim_j] = du * factor
                    target_tensor[:n_fill, ist + local_spatial_dim:ist + local_spatial_dim + dim_i,
                    jst + local_spatial_dim:jst + local_spatial_dim + dim_j] = dd * factor
                    jst += dim_j
                ist += dim_i

        fill_blocks(bondwise_hopping, orbpair_hopping)
        if onsite_block is not None:
            fill_blocks(onsite_block, orbpair_onsite)

        spatial_norb_total = int(_take_idp_tensor(
            self.idp.atom_norb,
            atom_types,
            result_device=data[self.node_field].device,
        ).sum().item())
        final_dim = spatial_norb_total * 2
        num_k = kpoints.shape[0]
        HK = torch.zeros(num_k, final_dim, final_dim, dtype=self.ctype, device=self.device)
        dHK = None if not self.derivative else torch.zeros(num_k, final_dim, final_dim, 3, dtype=self.ctype,
                                                           device=self.device)

        atom_slices = []
        curr = 0
        for at in atom_types:
            norb = int(_take_idp_tensor(
                self.idp.atom_norb,
                at,
                result_device=data[self.node_field].device,
            ).item())
            atom_slices.append(slice(curr, curr + norb))
            curr += norb

        def get_spin_slices(atom_idx):
            sl_spatial = atom_slices[atom_idx]
            offset = spatial_norb_total
            return sl_spatial, slice(sl_spatial.start + offset, sl_spatial.stop + offset)

        if onsite_block is not None:
            for i in range(num_atoms):
                mask = _take_idp_tensor(
                    self.idp.mask_to_basis,
                    atom_types[i],
                    result_device=onsite_block.device,
                )
                oblock = onsite_block[i]
                N_loc = local_spatial_dim
                uu = oblock[:N_loc, :N_loc][mask][:, mask]
                ud = oblock[:N_loc, N_loc:][mask][:, mask]
                du = oblock[N_loc:, :N_loc][mask][:, mask]
                dd = oblock[N_loc:, N_loc:][mask][:, mask]
                sl_up, sl_dn = get_spin_slices(i)
                HK[:, sl_up, sl_up] += uu.unsqueeze(0)
                HK[:, sl_up, sl_dn] += ud.unsqueeze(0)
                HK[:, sl_dn, sl_up] += du.unsqueeze(0)
                HK[:, sl_dn, sl_dn] += dd.unsqueeze(0)

        edge_indices = data[AtomicDataDict.EDGE_INDEX_KEY]
        src_list = edge_indices[0].cpu().numpy()
        dst_list = edge_indices[1].cpu().numpy()

        for k_edge in range(num_edges):
            u, v = int(src_list[k_edge]), int(dst_list[k_edge])
            hblock = bondwise_hopping[k_edge]
            mask_u = _take_idp_tensor(
                self.idp.mask_to_basis,
                atom_types[u],
                result_device=hblock.device,
            )
            mask_v = _take_idp_tensor(
                self.idp.mask_to_basis,
                atom_types[v],
                result_device=hblock.device,
            )

            uu = hblock[:N_loc, :N_loc][mask_u][:, mask_v]
            ud = hblock[:N_loc, N_loc:][mask_u][:, mask_v]
            du = hblock[N_loc:, :N_loc][mask_u][:, mask_v]
            dd = hblock[N_loc:, N_loc:][mask_u][:, mask_v]
            sl_u_up, sl_u_dn = get_spin_slices(u)
            sl_v_up, sl_v_dn = get_spin_slices(v)

            if self.gauge:
                edge_vec = data[AtomicDataDict.EDGE_VECTORS_KEY][k_edge]
                dot_prod = kpoints @ data[AtomicDataDict.CELL_KEY].inverse().T @ edge_vec
                phase = torch.exp(-1j * 2 * math.pi * dot_prod).reshape(-1, 1, 1)
                if self.derivative:
                    deriv_factor = (-1.0j * edge_vec).reshape(1, 1, 1, 3) * phase.unsqueeze(-1)
            else:
                shift = data[AtomicDataDict.EDGE_CELL_SHIFT_KEY][k_edge]
                phase = torch.exp(-1j * 2 * math.pi * (kpoints @ shift)).reshape(-1, 1, 1)

            HK[:, sl_u_up, sl_v_up] += uu.unsqueeze(0) * phase
            HK[:, sl_u_up, sl_v_dn] += ud.unsqueeze(0) * phase
            HK[:, sl_u_dn, sl_v_up] += du.unsqueeze(0) * phase
            HK[:, sl_u_dn, sl_v_dn] += dd.unsqueeze(0) * phase

            if self.derivative:
                d_p = deriv_factor
                dHK[:, sl_u_up, sl_v_up, :] += uu.unsqueeze(0).unsqueeze(-1) * d_p
                dHK[:, sl_u_up, sl_v_dn, :] += ud.unsqueeze(0).unsqueeze(-1) * d_p
                dHK[:, sl_u_dn, sl_v_up, :] += du.unsqueeze(0).unsqueeze(-1) * d_p
                dHK[:, sl_u_dn, sl_v_dn, :] += dd.unsqueeze(0).unsqueeze(-1) * d_p

        HK = HK + HK.transpose(1, 2).conj()
        if self.derivative:
            for alpha in range(3):
                dHK[..., alpha] = dHK[..., alpha] + dHK[..., alpha].transpose(1, 2).conj()
            data[self.out_derivative_field] = dHK

        data[self.out_field] = HK
        return data

    def _forward_scalar(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """
        Handles Scalar (Non-SOC) construction.
        Crucially handles the case where input features might be SOC-sized (4 blocks)
        but we only want the scalar part (e.g. Overlap in SOC calculation).
        """
        if self.gauge:
            data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)

        orbpair_hopping = data[self.edge_field]
        orbpair_onsite = data.get(self.node_field)

        kpoints = data[AtomicDataDict.KPOINT_KEY]
        if kpoints.is_nested: kpoints = kpoints[0]

        atom_types = data[AtomicDataDict.ATOM_TYPE_KEY]
        if atom_types.dim() > 1: atom_types = atom_types.flatten()
        num_atoms = len(atom_types)

        num_edges = orbpair_hopping.shape[0]
        local_dim = self.idp.full_basis_norb

        bondwise_hopping = torch.zeros((num_edges, local_dim, local_dim),
                                       dtype=self.ctype, device=self.device)
        onsite_block = None
        if orbpair_onsite is not None:
            onsite_block = torch.zeros((num_atoms, local_dim, local_dim),
                                       dtype=self.ctype, device=self.device)

        ist = 0
        for i, iorb in enumerate(self.idp.full_basis):
            jst = 0
            li = anglrMId[re.findall(r"[a-zA-Z]", iorb)[0]]
            dim_i = 2 * li + 1

            for j, jorb in enumerate(self.idp.full_basis):
                lj = anglrMId[re.findall(r"[a-zA-Z]", jorb)[0]]
                dim_j = 2 * lj + 1
                orbpair = iorb + "-" + jorb

                if orbpair in self.idp.orbpair_maps:
                    # 对于 Overlap 矩阵，通常存储上三角或下三角
                    # 逻辑保持：只处理 i <= j
                    if i <= j:
                        sli = self.idp.orbpair_maps[orbpair]

                        val_hop = orbpair_hopping[:, sli]
                        val_hop = recover_complex_tensor(val_hop, self.soc_complex_doubling)

                        # [核心修复] 处理 SOC 特征到 Scalar 的映射
                        # 如果 idp 是 SOC 的，特征里有 4 个块 [UU, UD, DU, DD]
                        # 而我们只想要 UU (Overlap is spin-diagonal)
                        if self.has_soc:
                            # 1. 显式 Reshape 拆分 4 个自旋块
                            val_hop = val_hop.view(-1, 4, dim_i, dim_j)
                            # 2. 取第一个块 (UU)
                            val_hop = val_hop[:, 0]
                            # 结果维度: [Edges, dim_i, dim_j]
                        else:
                            # 正常的 Scalar 特征
                            val_hop = val_hop.view(-1, dim_i, dim_j)

                        if i == j:
                            factor = 0.5
                        else:
                            factor = 1.0

                        n_fill = val_hop.shape[0]
                        bondwise_hopping[:n_fill, ist:ist + dim_i, jst:jst + dim_j] = val_hop * factor

                        if onsite_block is not None:
                            val_on = orbpair_onsite[:, sli]
                            val_on = recover_complex_tensor(val_on, self.soc_complex_doubling)
                            if self.has_soc:
                                val_on = val_on.view(-1, 4, dim_i, dim_j)
                                val_on = val_on[:, 0]
                            else:
                                val_on = val_on.view(-1, dim_i, dim_j)
                            onsite_block[:, ist:ist + dim_i, jst:jst + dim_j] = val_on * factor

                jst += dim_j
            ist += dim_i

        # 4. Global Assembly (Sparse -> Dense)
        global_norb = int(_take_idp_tensor(
            self.idp.atom_norb,
            atom_types,
            result_device=data[self.node_field].device,
        ).sum().item())
        HK = torch.zeros(kpoints.shape[0], global_norb, global_norb, dtype=self.ctype, device=self.device)
        dHK = None if not self.derivative else torch.zeros(kpoints.shape[0], global_norb, global_norb, 3,
                                                           dtype=self.ctype, device=self.device)

        atom_slices = []
        curr = 0
        for at in atom_types:
            norb = int(_take_idp_tensor(
                self.idp.atom_norb,
                at,
                result_device=data[self.node_field].device,
            ).item())
            atom_slices.append(slice(curr, curr + norb))
            curr += norb

        # 4.1 Onsite Assembly
        if onsite_block is not None:
            for i in range(num_atoms):
                mask = _take_idp_tensor(
                    self.idp.mask_to_basis,
                    atom_types[i],
                    result_device=onsite_block.device,
                )
                oblock = onsite_block[i][mask][:, mask]
                sl = atom_slices[i]
                HK[:, sl, sl] += oblock.unsqueeze(0)

        # 4.2 Hopping Assembly
        edge_idx = data[AtomicDataDict.EDGE_INDEX_KEY]
        src_list = edge_idx[0].cpu().numpy()
        dst_list = edge_idx[1].cpu().numpy()

        for k in range(num_edges):
            u, v = int(src_list[k]), int(dst_list[k])

            mask_u = _take_idp_tensor(
                self.idp.mask_to_basis,
                atom_types[u],
                result_device=bondwise_hopping.device,
            )
            mask_v = _take_idp_tensor(
                self.idp.mask_to_basis,
                atom_types[v],
                result_device=bondwise_hopping.device,
            )

            hblock = bondwise_hopping[k][mask_u][:, mask_v]

            sl_u = atom_slices[u]
            sl_v = atom_slices[v]

            if self.gauge:
                edge_vec = data[AtomicDataDict.EDGE_VECTORS_KEY][k]
                phase = torch.exp(
                    -1j * 2 * math.pi * (kpoints @ data[AtomicDataDict.CELL_KEY].inverse().T @ edge_vec)).reshape(-1, 1,
                                                                                                                  1)
                if self.derivative:
                    deriv_factor = (-1.0j * edge_vec).reshape(1, 1, 1, 3) * phase.unsqueeze(-1)
            else:
                shift = data[AtomicDataDict.EDGE_CELL_SHIFT_KEY][k]
                phase = torch.exp(-1j * 2 * math.pi * (kpoints @ shift)).reshape(-1, 1, 1)

            HK[:, sl_u, sl_v] += hblock.unsqueeze(0) * phase

            if self.derivative:
                dHK[:, sl_u, sl_v, :] += hblock.unsqueeze(0).unsqueeze(-1) * deriv_factor

        # 5. Symmetrization (Hermitian)
        HK = HK + HK.transpose(1, 2).conj()

        if self.derivative:
            for alpha in range(3):
                dHK[..., alpha] = dHK[..., alpha] + dHK[..., alpha].transpose(1, 2).conj()
            data[self.out_derivative_field] = dHK

        data[self.out_field] = HK
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
        all_norb = int(_take_idp_tensor(
            self.idp.atom_norb,
            data[AtomicDataDict.ATOM_TYPE_KEY],
            result_device=data[self.node_field].device,
        ).sum().item())
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
            mask = _take_idp_tensor(
                self.idp.mask_to_basis,
                atype,
                result_device=onsite_block.device,
            )
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
            imask = _take_idp_tensor(
                self.idp.mask_to_basis,
                itype,
                result_device=bondwise_hopping.device,
            )
            jmask = _take_idp_tensor(
                self.idp.mask_to_basis,
                jtype,
                result_device=bondwise_hopping.device,
            )

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
