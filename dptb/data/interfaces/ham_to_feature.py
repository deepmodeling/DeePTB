from .. import _keys
import ase
import numpy as np
import torch
import re
import e3nn.o3 as o3
import h5py
import logging
from dptb.utils.constants import OPENMX2DeePTB
from dptb.data import AtomicData, AtomicDataDict

log = logging.getLogger(__name__)

import torch
import numpy as np
import ase.data
# 假设你的项目中包含这个 key 定义文件，如果没有请替换为字符串常量


def block_to_feature(data, idp, blocks=False, overlap_blocks=False, orthogonal=False):
    """
    将 Hamiltonian/Overlap blocks 转换为模型可训练的 features。
    针对 SOC 进行了向量化优化，支持 Spinful H + Reduced Overlap 的混合输入。
    """

    # --- 1. 初始化与检查 ---
    has_ham = (blocks is not False) and (blocks is not None)
    has_ovp = (overlap_blocks is not False) and (overlap_blocks is not None)
    assert has_ham or has_ovp, "Feature blocks (Hamiltonian) and Overlap blocks are both missing."

    # 确保 map 已构建
    idp.get_orbital_maps()
    idp.get_orbpair_maps()

    # 获取 SOC 设置
    has_soc = bool(getattr(idp, "has_soc", False))
    soc_complex_doubling = bool(getattr(idp, "soc_complex_doubling", True))

    # 确定输出 Feature 的数据类型
    # 如果开启 SOC 且 Double Complex，输出通常为 Float (Real cat Imag)
    # 否则保持与输入一致或由 PyTorch 默认决定
    if has_soc and soc_complex_doubling:
        feature_dtype = torch.get_default_dtype()  # usually float32/float64
    else:
        # 如果不开 doubling，保留复数类型
        feature_dtype = torch.complex64 if has_soc else torch.get_default_dtype()

    # 确定设备 (CPU/GPU)
    device = None
    if isinstance(data, dict):
        pos = data.get("pos", None)
        device = pos.device if isinstance(pos, torch.Tensor) else None
    elif hasattr(data, "pos"):  # AtomicData
        device = data.pos.device

    # --- 2. 核心辅助函数 (Vectorized) ---

    def _to_torch(x):
        """将 Numpy array 转为 Tensor，处理 device 和 dtype"""
        if isinstance(x, torch.Tensor):
            t = x
        elif isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
        else:
            raise TypeError(f"Block data must be Tensor or ndarray, got {type(x)}")

        # 统一转为 Complex64 (如果包含复数) 或 Float32，避免 Double 精度不必要的显存占用
        if t.is_complex():
            t = t.to(torch.complex64)
        elif t.is_floating_point():
            t = t.to(torch.float32)

        return t.to(device=device) if device is not None else t

    def _pack_soc_tensor(tensor_data, slice_i, slice_j, norb_i, norb_j):
        """
        [核心] 向量化提取 SOC 子块。
        支持输入: (Batch, Rows, Cols) 或 (Rows, Cols)
        自动识别: Full SOC (2N, 2N) 或 Reduced Overlap (N, N)
        """
        # 1. 统一维度为 (Batch, Rows, Cols)
        is_batched = tensor_data.dim() == 3
        if not is_batched:
            tensor_data = tensor_data.unsqueeze(0)

        B, R, C = tensor_data.shape

        # 计算切片长度
        di = slice_i.stop - slice_i.start
        dj = slice_j.stop - slice_j.start

        # 2. 根据输入矩阵尺寸判断数据结构
        # 假设 ABACUS 的 parse_matrix 输出了 Spin-Block Major 格式:
        # [0:N, 0:N] -> UU, [0:N, N:2N] -> UD, [N:2N, 0:N] -> DU, [N:2N, N:2N] -> DD

        if R == 2 * norb_i and C == 2 * norb_j:
            # === Case A: Full SOC Matrix (Hamiltonian or Full Overlap) ===
            # 提取 4 个 Spin Block
            # 使用 reshape/transpose 可能会更通用，但直接切片最快且符合 Spin-Block 假设

            # (Batch, di, dj)
            uu = tensor_data[:, slice_i, slice_j]
            ud = tensor_data[:, slice_i, slice(slice_j.start + norb_j, slice_j.stop + norb_j)]
            du = tensor_data[:, slice(slice_i.start + norb_i, slice_i.stop + norb_i), slice_j]
            dd = tensor_data[:, slice(slice_i.start + norb_i, slice_i.stop + norb_i),
                 slice(slice_j.start + norb_j, slice_j.stop + norb_j)]

        elif R == norb_i and C == norb_j:
            # === Case B: Reduced Overlap (Spinless / Diagonal) ===
            # 这种情况常出现在 SOC 计算的 Overlap 中，只存储了空间部分
            base = tensor_data[:, slice_i, slice_j]
            zeros = torch.zeros_like(base)

            uu = base
            dd = base
            ud = zeros
            du = zeros
        else:
            raise ValueError(
                f"SOC shape mismatch. Atom/Bond needs ({2 * norb_i},{2 * norb_j}) or ({norb_i},{norb_j}), got ({R},{C}).")

        # 3. Flatten 空间维度 (Batch, di, dj) -> (Batch, di*dj)
        uu = uu.reshape(B, -1)
        ud = ud.reshape(B, -1)
        du = du.reshape(B, -1)
        dd = dd.reshape(B, -1)

        # 4. 拼接 Spin 分量 (顺序: UU, UD, DU, DD) -> (Batch, 4*di*dj)
        spin_cat = torch.cat([uu, ud, du, dd], dim=1)

        # 5. 处理复数双倍化 (Re, Im)
        if soc_complex_doubling:
            # 即使输入全是实数(如 Overlap)，SOC 模式下通常也需要输出 Re/Im 两个通道
            if spin_cat.is_complex():
                out = torch.cat([spin_cat.real, spin_cat.imag], dim=1)
            else:
                out = torch.cat([spin_cat, torch.zeros_like(spin_cat)], dim=1)

            # 确保转为 float
            out = out.to(dtype=feature_dtype)
        else:
            # 保持复数
            out = spin_cat.to(dtype=feature_dtype)

        # 如果输入不是 batched，去掉 batch 维
        if not is_batched:
            out = out.squeeze(0)

        return out

    def _pack_nosoc_tensor(tensor_data, slice_i, slice_j):
        """非 SOC 情况的简单提取"""
        is_batched = tensor_data.dim() == 3
        if not is_batched:
            tensor_data = tensor_data.unsqueeze(0)

        B = tensor_data.shape[0]
        val = tensor_data[:, slice_i, slice_j].reshape(B, -1)

        if val.is_complex() and not feature_dtype.is_complex:
            val = val.real

        val = val.to(dtype=feature_dtype)

        if not is_batched:
            val = val.squeeze(0)
        return val

    # --- 3. 准备数据 ---

    # 兼容 Data(PyG) 和 Dict
    if isinstance(data, dict):
        if _keys.ATOMIC_NUMBERS_KEY not in data or data[_keys.ATOMIC_NUMBERS_KEY] is None:
            data[_keys.ATOMIC_NUMBERS_KEY] = idp.untransform(data[_keys.ATOM_TYPE_KEY])
        atomic_numbers = data[_keys.ATOMIC_NUMBERS_KEY]
    else:
        if not hasattr(data, _keys.ATOMIC_NUMBERS_KEY):
            setattr(data, _keys.ATOMIC_NUMBERS_KEY, idp.untransform(data[_keys.ATOM_TYPE_KEY]))
        atomic_numbers = data[_keys.ATOMIC_NUMBERS_KEY]

    # 确定 block key 的起始索引 (0-based or 1-based)
    check_source = blocks if has_ham else overlap_blocks
    start_id = 0 if ("0_0_0_0_0" in check_source) else 1

    # --- 4. 处理 Onsite Features ---
    onsite_ham = []
    onsite_ovp = []

    num_atoms = len(atomic_numbers)

    for atom in range(num_atoms):
        # 获取原子信息
        z = int(atomic_numbers[atom])
        symbol = ase.data.chemical_symbols[z]
        basis_list = idp.basis[symbol]
        # 获取该原子的空间轨道数 (Spatial Norbs)
        spatial_norb = int(idp.norbs[symbol])

        block_idx_str = f"{atom + start_id}_{atom + start_id}_0_0_0"

        # 预分配特征容器
        ham_vec = torch.zeros(idp.reduced_matrix_element, dtype=feature_dtype, device=device) if has_ham else None
        ovp_vec = torch.zeros(idp.reduced_matrix_element, dtype=feature_dtype, device=device) if (
                    has_ovp and not orthogonal) else None

        # 读取 Block
        blk_h = _to_torch(blocks[block_idx_str][:]) if has_ham else None
        blk_s = _to_torch(overlap_blocks[block_idx_str][:]) if (has_ovp and not orthogonal) else None

        # 遍历轨道对 (Basis Pair)
        for i, basis_i in enumerate(basis_list):
            slice_i = idp.orbital_maps[symbol][basis_i]

            # SOC 全矩阵遍历 (j from 0)，Non-SOC 上三角 (j from i)
            start_j = 0 if has_soc else i

            for basis_j in basis_list[start_j:]:
                slice_j = idp.orbital_maps[symbol][basis_j]

                # 获取 Feature 切片位置
                full_i = idp.basis_to_full_basis[symbol][basis_i]
                full_j = idp.basis_to_full_basis[symbol][basis_j]
                pair_key = f"{full_i}-{full_j}"

                feat_slice = idp.orbpair_maps.get(pair_key)
                if feat_slice is None: continue

                # 填充 Hamiltonian
                if has_ham:
                    if has_soc:
                        val = _pack_soc_tensor(blk_h, slice_i, slice_j, spatial_norb, spatial_norb)
                    else:
                        val = _pack_nosoc_tensor(blk_h, slice_i, slice_j)

                    # 尺寸检查 (Debugging friendly)
                    expected_len = feat_slice.stop - feat_slice.start
                    if val.numel() != expected_len:
                        raise ValueError(
                            f"Onsite Ham size mismatch atom {atom}: got {val.numel()}, expected {expected_len} ({pair_key})")

                    ham_vec[feat_slice] = val

                # 填充 Overlap
                if has_ovp and not orthogonal:
                    if has_soc:
                        val = _pack_soc_tensor(blk_s, slice_i, slice_j, spatial_norb, spatial_norb)
                    else:
                        val = _pack_nosoc_tensor(blk_s, slice_i, slice_j)

                    if val.numel() != (feat_slice.stop - feat_slice.start):
                        raise ValueError(f"Onsite Ovp size mismatch atom {atom} ({pair_key})")

                    ovp_vec[feat_slice] = val

        if has_ham: onsite_ham.append(ham_vec)
        if has_ovp and not orthogonal: onsite_ovp.append(ovp_vec)

    # --- 5. 处理 Edge Features (Vectorized) ---

    edge_index = data[_keys.EDGE_INDEX_KEY]
    edge_shift = data[_keys.EDGE_CELL_SHIFT_KEY]
    # 获取每一条边的类型索引
    edge_types_idx = idp.transform_bond(*data[_keys.ATOMIC_NUMBERS_KEY][edge_index]).flatten()

    num_edges = edge_index.shape[1]

    # 预分配 Edge 特征矩阵 (Num_Edges, Feature_Dim)
    edge_feat_h = torch.zeros((num_edges, idp.reduced_matrix_element), dtype=feature_dtype,
                              device=device) if has_ham else None
    edge_feat_s = torch.zeros((num_edges, idp.reduced_matrix_element), dtype=feature_dtype,
                              device=device) if has_ovp else None

    # 按 Bond Type 批量处理
    for bt_idx in range(len(idp.bond_types)):

        # 找到属于该 Bond Type 的所有边
        mask = edge_types_idx.eq(bt_idx)
        if not mask.any(): continue

        # 获取该类型的元数据
        sym_i, sym_j = idp.bond_types[bt_idx].split("-")
        basis_i = idp.basis[sym_i]
        basis_j = idp.basis[sym_j]
        norb_i = int(idp.norbs[sym_i])
        norb_j = int(idp.norbs[sym_j])

        # 获取这些边的子集索引
        sub_edge_index = edge_index[:, mask]
        sub_edge_shift = edge_shift[mask]

        # 构建 Key 列表
        ijR = torch.cat([sub_edge_index.T + start_id, sub_edge_shift], dim=1).int().tolist()
        rev_ijR = torch.cat([sub_edge_index[[1, 0]].T + start_id, -sub_edge_shift], dim=1).int().tolist()

        keys_ijR = ['_'.join(map(str, x)) for x in ijR]
        keys_rev = ['_'.join(map(str, x)) for x in rev_ijR]

        # === 批量读取 Block ===
        # 这里必须用 Python 循环读取 IO，但只针对该类型的边，且后续处理是向量化的

        if has_ham:
            batch_list = []
            for k, rk in zip(keys_ijR, keys_rev):
                if k in blocks:
                    batch_list.append(blocks[k][:])
                elif rk in blocks:
                    # 厄米共轭 (Hermitian Conjugate)
                    val = blocks[rk][:].T
                    if np.iscomplexobj(val): val = val.conj()
                    batch_list.append(val)
                else:
                    # 缺失补零
                    shape = (2 * norb_i, 2 * norb_j) if has_soc else (norb_i, norb_j)
                    dtype = np.complex64 if has_soc else np.float32
                    batch_list.append(np.zeros(shape, dtype=dtype))

            # 堆叠成 Tensor (N_sub_edges, Rows, Cols)
            # 这一步将 Python list 转为 Tensor，后续全是 Tensor 操作
            batch_h_tensor = _to_torch(np.stack(batch_list, axis=0))

        if has_ovp:
            batch_list = []
            for k, rk in zip(keys_ijR, keys_rev):
                if k in overlap_blocks:
                    batch_list.append(overlap_blocks[k][:])
                elif rk in overlap_blocks:
                    val = overlap_blocks[rk][:].T
                    if np.iscomplexobj(val): val = val.conj()
                    batch_list.append(val)
                else:
                    # Overlap 缺省时可能是 Spinless 的 Shape，先用 norb_i
                    # 后续 _pack_soc_tensor 会处理
                    # 安全起见，如果 reduced, shape=(N,N), 如果 full, shape=(2N,2N)
                    # 这里先给 (N,N) 浮点零，如果后续需要 broadcasting 也能处理
                    batch_list.append(np.zeros((norb_i, norb_j), dtype=np.float32))

            batch_s_tensor = _to_torch(np.stack(batch_list, axis=0))

        # === 向量化填充特征 ===
        for orb_i in basis_i:
            slice_i = idp.orbital_maps[sym_i][orb_i]
            full_i = idp.basis_to_full_basis[sym_i][orb_i]

            for orb_j in basis_j:
                slice_j = idp.orbital_maps[sym_j][orb_j]
                full_j = idp.basis_to_full_basis[sym_j][orb_j]

                # 检查是否为上三角 (仅 Non-SOC)
                is_upper = idp.full_basis.index(full_i) <= idp.full_basis.index(full_j)
                if (not has_soc) and (not is_upper): continue

                # 获取 Feature Slice
                pair_key = f"{full_i}-{full_j}"
                feat_slice = idp.orbpair_maps.get(pair_key)
                if feat_slice is None: continue

                # 计算特征
                if has_ham:
                    if has_soc:
                        # (Batch, Feature_Len)
                        vecs = _pack_soc_tensor(batch_h_tensor, slice_i, slice_j, norb_i, norb_j)
                    else:
                        vecs = _pack_nosoc_tensor(batch_h_tensor, slice_i, slice_j)

                    # 批量赋值 (Mask Indexing)
                    # 注意: vecs 的第0维对应 mask 中为 True 的数量
                    edge_feat_h[mask, feat_slice] = vecs

                if has_ovp:
                    if has_soc:
                        vecs = _pack_soc_tensor(batch_s_tensor, slice_i, slice_j, norb_i, norb_j)
                    else:
                        vecs = _pack_nosoc_tensor(batch_s_tensor, slice_i, slice_j)

                    edge_feat_s[mask, feat_slice] = vecs

    # --- 6. 结果写回 ---
    if has_ham:
        data[_keys.NODE_FEATURES_KEY] = torch.stack(onsite_ham, dim=0)
        data[_keys.EDGE_FEATURES_KEY] = edge_feat_h

    if has_ovp:
        if not orthogonal:
            data[_keys.NODE_OVERLAP_KEY] = torch.stack(onsite_ovp, dim=0)
        data[_keys.EDGE_OVERLAP_KEY] = edge_feat_s

    return data


import torch
import numpy as np
import ase.data
import logging

# 配置 log
log = logging.getLogger(__name__)


def feature_to_block(data, idp, overlap: bool = False):
    """
    将模型预测的 features 转换回 Hamiltonian/Overlap blocks。

    修复:
    1. IndexError: 确保 mask 是 1D 的，兼容 [N, 1] 形状的 atomic_numbers。
    2. AttributeError: 显式初始化 idp map。
    3. 逻辑: 使用与 block_to_feature 对称的 Masking+Slicing 方案 (支持 SOC)。
    """

    # --- 0. 确保 idp 内部状态已初始化 ---
    if not hasattr(idp, "norbs") or not idp.norbs:
        idp.get_orbital_maps()
    if not hasattr(idp, "orbpair_maps") or not idp.orbpair_maps:
        idp.get_orbpair_maps()

    # --- 1. 准备数据 Key ---
    if overlap:
        node_key = getattr(_keys, 'NODE_OVERLAP_KEY', 'node_overlap')
        edge_key = getattr(_keys, 'EDGE_OVERLAP_KEY', 'edge_overlap')
    else:
        node_key = getattr(_keys, 'NODE_FEATURES_KEY', 'node_features')
        edge_key = getattr(_keys, 'EDGE_FEATURES_KEY', 'edge_features')

    # 检查数据
    if node_key not in data and edge_key not in data:
        log.warning("No features found in data for block conversion.")
        return {}

    node_features = data.get(node_key, None)
    edge_features = data.get(edge_key, None)

    # 确定设备
    device = node_features.device if node_features is not None else \
        (edge_features.device if edge_features is not None else torch.device('cpu'))

    # --- [关键修复] 获取并展平原子序数 ---
    if _keys.ATOMIC_NUMBERS_KEY in data:
        atomic_numbers = data[_keys.ATOMIC_NUMBERS_KEY]
    else:
        atomic_numbers = idp.untransform(data[_keys.ATOM_TYPE_KEY])

    # 确保转为 Numpy 且展平为 1D 数组 [N]
    if isinstance(atomic_numbers, torch.Tensor):
        atomic_numbers = atomic_numbers.cpu().numpy()

    # Force Flatten: 解决 [N, 1] 导致的 IndexError
    atomic_numbers = atomic_numbers.flatten()

    # SOC 设置
    has_soc = bool(getattr(idp, "has_soc", False))
    soc_complex_doubling = bool(getattr(idp, "soc_complex_doubling", True))

    blocks = {}
    start_id = 0

    # --- 2. 核心还原函数 ---
    def _unpack_tensor(vecs, rows, cols):
        B = vecs.shape[0]
        # A. Complex Doubling
        if soc_complex_doubling:
            n_real = vecs.shape[1] // 2
            val = torch.complex(vecs[:, :n_real], vecs[:, n_real:])
        else:
            val = vecs

        # B. SOC Reshape
        if has_soc:
            spatial_size = rows * cols
            uu = val[:, 0:spatial_size].reshape(B, rows, cols)
            ud = val[:, spatial_size:2 * spatial_size].reshape(B, rows, cols)
            du = val[:, 2 * spatial_size:3 * spatial_size].reshape(B, rows, cols)
            dd = val[:, 3 * spatial_size:].reshape(B, rows, cols)

            row1 = torch.cat([uu, ud], dim=2)
            row2 = torch.cat([du, dd], dim=2)
            mat = torch.cat([row1, row2], dim=1)
        else:
            mat = val.reshape(B, rows, cols)
        return mat

    # --- 3. 处理 Onsite Blocks ---
    if node_features is not None:
        unique_zs = np.unique(atomic_numbers)

        for z in unique_zs:
            symbol = ase.data.chemical_symbols[int(z)]
            # 获取该元素所有原子的 Mask (1D)
            atom_mask = (atomic_numbers == z)

            # 使用 Mask 索引 (此时 Mask 是一维的，不会报错)
            mask_tensor = torch.tensor(atom_mask, device=device)
            sub_feats = node_features[mask_tensor]

            num_atoms_sub = sub_feats.shape[0]
            if num_atoms_sub == 0: continue

            spatial_norb = int(idp.norbs[symbol])
            basis_list = idp.basis[symbol]

            mat_dim = 2 * spatial_norb if has_soc else spatial_norb
            is_complex_out = has_soc or sub_feats.is_complex() or soc_complex_doubling
            dtype_out = torch.complex64 if is_complex_out else torch.float32

            batch_blocks = torch.zeros((num_atoms_sub, mat_dim, mat_dim), dtype=dtype_out, device=device)

            for i, basis_i in enumerate(basis_list):
                slice_i = idp.orbital_maps[symbol][basis_i]
                start_j = 0 if has_soc else i

                for basis_j in basis_list[start_j:]:
                    slice_j = idp.orbital_maps[symbol][basis_j]
                    full_i = idp.basis_to_full_basis[symbol][basis_i]
                    full_j = idp.basis_to_full_basis[symbol][basis_j]

                    feat_slice = idp.orbpair_maps.get(f"{full_i}-{full_j}")
                    if feat_slice is None: continue

                    vecs = sub_feats[:, feat_slice]
                    ni = slice_i.stop - slice_i.start
                    nj = slice_j.stop - slice_j.start
                    block_part = _unpack_tensor(vecs, ni, nj)

                    if has_soc:
                        sp_h, sp_w = ni, nj
                        batch_blocks[:, slice_i, slice_j] = block_part[:, :sp_h, :sp_w]
                        batch_blocks[:, slice_i,
                        slice(slice_j.start + spatial_norb, slice_j.stop + spatial_norb)] = block_part[:, :sp_h, sp_w:]
                        batch_blocks[:, slice(slice_i.start + spatial_norb, slice_i.stop + spatial_norb),
                        slice_j] = block_part[:, sp_h:, :sp_w]
                        batch_blocks[:, slice(slice_i.start + spatial_norb, slice_i.stop + spatial_norb),
                        slice(slice_j.start + spatial_norb, slice_j.stop + spatial_norb)] = block_part[:, sp_h:, sp_w:]
                    else:
                        batch_blocks[:, slice_i, slice_j] = block_part
                        if i != basis_list.index(basis_j):
                            if block_part.is_complex():
                                batch_blocks[:, slice_j, slice_i] = block_part.transpose(-1, -2).conj()
                            else:
                                batch_blocks[:, slice_j, slice_i] = block_part.transpose(-1, -2)

            global_indices = np.where(atom_mask)[0]
            batch_blocks_np = batch_blocks.detach().cpu().numpy()

            for local_idx, global_idx in enumerate(global_indices):
                blocks[f"{global_idx + start_id}_{global_idx + start_id}_0_0_0"] = batch_blocks_np[local_idx]

    # --- 4. 处理 Edge Blocks ---
    if edge_features is not None and _keys.EDGE_INDEX_KEY in data:
        edge_index = data[_keys.EDGE_INDEX_KEY]
        if edge_index.shape[1] > 0:
            edge_shift = data[_keys.EDGE_CELL_SHIFT_KEY]
            src_atoms = atomic_numbers[edge_index[0].cpu()]
            dst_atoms = atomic_numbers[edge_index[1].cpu()]
            edge_types_idx = idp.transform_bond(src_atoms, dst_atoms).flatten()

            if isinstance(edge_types_idx, torch.Tensor):
                edge_types_idx = edge_types_idx.cpu().numpy()

            for bt_idx in range(len(idp.bond_types)):
                mask = (edge_types_idx == bt_idx)
                if not np.any(mask): continue

                mask_torch = torch.tensor(mask, device=device)
                sub_feats = edge_features[mask_torch]
                num_edges_sub = sub_feats.shape[0]

                sym_i, sym_j = idp.bond_types[bt_idx].split("-")
                basis_i, basis_j = idp.basis[sym_i], idp.basis[sym_j]
                norb_i, norb_j = int(idp.norbs[sym_i]), int(idp.norbs[sym_j])

                rows, cols = (2 * norb_i, 2 * norb_j) if has_soc else (norb_i, norb_j)
                is_complex_out = has_soc or sub_feats.is_complex() or soc_complex_doubling
                dtype_out = torch.complex64 if is_complex_out else torch.float32

                batch_blocks = torch.zeros((num_edges_sub, rows, cols), dtype=dtype_out, device=device)

                for orb_i in basis_i:
                    slice_i = idp.orbital_maps[sym_i][orb_i]
                    full_i = idp.basis_to_full_basis[sym_i][orb_i]
                    for orb_j in basis_j:
                        slice_j = idp.orbital_maps[sym_j][orb_j]
                        full_j = idp.basis_to_full_basis[sym_j][orb_j]

                        is_upper = idp.full_basis.index(full_i) <= idp.full_basis.index(full_j)
                        if (not has_soc) and (not is_upper): continue

                        feat_slice = idp.orbpair_maps.get(f"{full_i}-{full_j}")
                        if feat_slice is None: continue

                        vecs = sub_feats[:, feat_slice]
                        ni, nj = slice_i.stop - slice_i.start, slice_j.stop - slice_j.start
                        block_part = _unpack_tensor(vecs, ni, nj)

                        if has_soc:
                            sp_h, sp_w = ni, nj
                            batch_blocks[:, slice_i, slice_j] = block_part[:, :sp_h, :sp_w]
                            batch_blocks[:, slice_i, slice(slice_j.start + norb_j, slice_j.stop + norb_j)] = block_part[
                                                                                                             :, :sp_h,
                                                                                                             sp_w:]
                            batch_blocks[:, slice(slice_i.start + norb_i, slice_i.stop + norb_i), slice_j] = block_part[
                                                                                                             :, sp_h:,
                                                                                                             :sp_w]
                            batch_blocks[:, slice(slice_i.start + norb_i, slice_i.stop + norb_i),
                            slice(slice_j.start + norb_j, slice_j.stop + norb_j)] = block_part[:, sp_h:, sp_w:]
                        else:
                            batch_blocks[:, slice_i, slice_j] = block_part

                batch_blocks_np = batch_blocks.detach().cpu().numpy()
                sub_edge_index = edge_index[:, mask_torch].cpu().numpy()
                sub_edge_shift = edge_shift[mask_torch].cpu().numpy()

                for k in range(num_edges_sub):
                    u, v = sub_edge_index[0, k], sub_edge_index[1, k]
                    r = sub_edge_shift[k]
                    idx_str = f"{u + start_id}_{v + start_id}_{int(r[0])}_{int(r[1])}_{int(r[2])}"
                    if idx_str in blocks:
                        blocks[idx_str] += batch_blocks_np[k]
                    else:
                        blocks[idx_str] = batch_blocks_np[k]

    log.info(f"Converted {len(blocks)} blocks (Overlap={overlap}).")
    return blocks


def openmx_to_deeptb(data, idp, openmx_hpath):
    # Hamiltonian_blocks should be a h5 group in the current version
    Us_openmx2wiki = OPENMX2DeePTB
    # init_rot_mat
    rot_blocks = {}
    for asym, orbs in idp.basis.items():
        b = [Us_openmx2wiki[re.findall(r"[A-Za-z]", orb)[0]] for orb in orbs]
        rot_blocks[asym] = torch.block_diag(*b)

    Hamiltonian_blocks = h5py.File(openmx_hpath, 'r')
    
    onsite_ham = []
    edge_ham = []

    idp.get_orbital_maps()
    idp.get_orbpair_maps()

    atomic_numbers = data[_keys.ATOMIC_NUMBERS_KEY]

    # onsite features
    for atom in range(len(atomic_numbers)):
        block_index = str([0, 0, 0, atom+1, atom+1])
        try:
            block = Hamiltonian_blocks[block_index][:]
        except:
            raise IndexError("Hamiltonian block for onsite not found, check Hamiltonian file.")

        symbol = ase.data.chemical_symbols[atomic_numbers[atom]]
        block = rot_blocks[symbol] @ block @ rot_blocks[symbol].T
        basis_list = idp.basis[symbol]
        onsite_out = np.zeros(idp.reduced_matrix_element)

        for index, basis_i in enumerate(basis_list):
            slice_i = idp.orbital_maps[symbol][basis_i]  
            for basis_j in basis_list[index:]:
                slice_j = idp.orbital_maps[symbol][basis_j]
                block_ij = block[slice_i, slice_j]
                full_basis_i = idp.basis_to_full_basis[symbol][basis_i]
                full_basis_j = idp.basis_to_full_basis[symbol][basis_j]

                # fill onsite vector
                pair_ij = full_basis_i + "-" + full_basis_j
                feature_slice = idp.orbpair_maps[pair_ij]
                onsite_out[feature_slice] = block_ij.flatten()

        onsite_ham.append(onsite_out)
        #onsite_ham = np.array(onsite_ham)

    # edge features
    edge_index = data[_keys.EDGE_INDEX_KEY]
    edge_cell_shift = data[_keys.EDGE_CELL_SHIFT_KEY]

    for atom_i, atom_j, R_shift in zip(edge_index[0], edge_index[1], edge_cell_shift):
        block_index = str(list(R_shift.int().numpy())+[int(atom_i)+1, int(atom_j)+1])

        symbol_i = ase.data.chemical_symbols[atomic_numbers[atom_i]]
        symbol_j = ase.data.chemical_symbols[atomic_numbers[atom_j]]

        block = Hamiltonian_blocks.get(block_index, 0)
        if block == 0:
            block = torch.zeros(idp.norbs[symbol_i], idp.norbs[symbol_j])
            log.warning("Hamiltonian block for hopping {} not found, r_cut may be too big for input R.".format(block_index))
        else:
            block = block[:]

        block = rot_blocks[symbol_i] @ block @ rot_blocks[symbol_j].T
        basis_i_list = idp.basis[symbol_i]
        basis_j_list = idp.basis[symbol_j]
        hopping_out = np.zeros(idp.reduced_matrix_element)

        for basis_i in basis_i_list:
            slice_i = idp.orbital_maps[symbol_i][basis_i]
            for basis_j in basis_j_list:
                slice_j = idp.orbital_maps[symbol_j][basis_j]
                block_ij = block[slice_i, slice_j]
                full_basis_i = idp.basis_to_full_basis[symbol_i][basis_i]
                full_basis_j = idp.basis_to_full_basis[symbol_j][basis_j]

                if idp.full_basis.index(full_basis_i) <= idp.full_basis.index(full_basis_j):
                    # fill hopping vector
                    pair_ij = full_basis_i + "-" + full_basis_j
                    feature_slice = idp.orbpair_maps[pair_ij]
                    hopping_out[feature_slice] = block_ij.flatten()

        edge_ham.append(hopping_out)

    data[_keys.NODE_FEATURES_KEY] = torch.as_tensor(np.array(onsite_ham), dtype=torch.get_default_dtype())
    data[_keys.EDGE_FEATURES_KEY] = torch.as_tensor(np.array(edge_ham), dtype=torch.get_default_dtype())
    Hamiltonian_blocks.close()