import torch
import os
from typing import Optional, List, Union, Dict
from torch_scatter import scatter_mean, scatter_add
from e3nn import o3
from e3nn.o3 import Linear, SphericalHarmonics, FullyConnectedTensorProduct

from dptb.data import AtomicDataDict, _keys
from dptb.nn.embedding.emb import Embedding
from dptb.data.transforms import OrbitalMapper
from dptb.data.AtomicDataDict import with_edge_vectors, with_batch

# 复用 lem.py 中的组件
from .lem import InitLayer, Layer

# 导入本地 tensor product 接口
from dptb.nn import tensor_product as tp


# =============================================================================
# 1. 惯性张量正则化工具
# =============================================================================
class InertialCanonicalizer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def get_canonical_orientation(self, pos: torch.Tensor, batch: torch.Tensor):
        """
        计算每个 batch 的旋转矩阵 Q，使得 pos_can = (pos - center) @ Q
        """
        device = pos.device
        dtype = pos.dtype

        # 1. 计算质心并去中心化
        center = scatter_mean(pos, batch, dim=0)
        pos_centered = pos - center[batch]

        # 2. 计算惯性张量 I = sum(r^2 * I - r * r.T)
        r_sq = (pos_centered ** 2).sum(dim=-1, keepdim=True)
        outer = torch.bmm(pos_centered.unsqueeze(2), pos_centered.unsqueeze(1))
        I_atom = r_sq.unsqueeze(-1) * torch.eye(3, device=device, dtype=dtype) - outer
        I_mol = scatter_add(I_atom, batch, dim=0)

        # 3. 特征分解 (Batch Eigen Decomposition)
        # torch.linalg.eigh 默认升序排列，符合 Inertial Transformer 要求
        L, Q = torch.linalg.eigh(I_mol)

        # 4. 锚点消歧 & 强制右手系 (Anchor Disambiguation)
        # 使用 In-place clone 避免梯度问题
        Q_final = Q.clone()
        num_graphs = Q.shape[0]

        # 循环处理每个图的消歧（Batch 较小，循环开销可忽略）
        for b in range(num_graphs):
            mask = (batch == b)
            p_local = pos_centered[mask]
            q_local = Q_final[b]

            # 投影
            p_proj = p_local @ q_local

            # 检查 X, Y 轴方向
            for axis in [0, 1]:
                for i in range(p_proj.shape[0]):
                    val = p_proj[i, axis]
                    if torch.abs(val) > 1e-4:
                        if val < 0:
                            q_local[:, axis] = -q_local[:, axis]
                        break

            # 强制右手系
            q_local[:, 2] = torch.linalg.cross(q_local[:, 0], q_local[:, 1])
            Q_final[b] = q_local

        return Q_final


# =============================================================================
# 2. LemFrame 主类
# =============================================================================
@Embedding.register("lem_frame")
class LemFrame(torch.nn.Module):
    def __init__(
            self,
            basis: Dict[str, Union[str, list]] = None,
            idp: Union[OrbitalMapper, None] = None,
            # required params
            n_layers: int = 3,
            n_radial_basis: int = 10,
            r_max: float = 5.0,
            irreps_hidden: o3.Irreps = None,
            avg_num_neighbors: Optional[float] = None,
            # cutoffs
            r_start_cos_ratio: float = 0.8,
            norm_eps: float = 1e-8,
            PolynomialCutoff_p: float = 6,
            cutoff_type: str = "polynomial",
            # general hyperparameters:
            env_embed_multiplicity: int = 32,
            sh_normalized: bool = True,
            sh_normalization: str = "component",
            # tp parameters:
            tp_radial_emb: bool = False,
            tp_radial_channels: list = [128, 128],
            # MLP parameters:
            latent_channels: list = [128, 128],
            latent_dim: int = 128,
            edge_one_hot_dim: int = 128,
            use_out_onehot_tp: bool = True,
            use_layer_onehot_tp: bool = True,
            res_update: bool = True,
            res_update_ratios: Optional[List[float]] = None,
            res_update_ratios_learnable: bool = False,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
            universal: Optional[bool] = False,
            use_interpolation_out: Optional[bool] = True,
            **kwargs,
    ):
        super(LemFrame, self).__init__()

        irreps_hidden = o3.Irreps(irreps_hidden)
        lmax = irreps_hidden.lmax

        if isinstance(dtype, str): dtype = getattr(torch, dtype)
        self.dtype = dtype
        if isinstance(device, str): device = torch.device(device)
        self.device = device

        # --- 1. 初始化正则化器 ---
        self.canonicalizer = InertialCanonicalizer()

        # --- 2. 基础设置 ---
        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb")
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp

        self.basis = self.idp.basis
        self.idp.get_irreps(no_parity=False)
        self.n_atom = 95 if universal else len(self.basis.keys())

        # --- 3. 预加载 Wigner-D 资源 (Local PT) ---
        self._init_wigner_resources()

        # --- 4. 构建网络层 (复用 Lem 逻辑) ---
        irreps_sh = o3.Irreps([(1, (i, (-1) ** i)) for i in range(lmax + 1)])
        orbpair_irreps = self.idp.orbpair_irreps.sort()[0].simplify()

        # Check irreps
        irreps_out_check = []
        for mul, ir1 in irreps_hidden:
            for _, ir2 in orbpair_irreps:
                irreps_out_check += [o3.Irrep(str(irr)) for irr in ir1 * ir2]
        irreps_out_check = o3.Irreps(irreps_out_check).sort()[0].simplify()
        assert all(ir in irreps_out_check for _, ir in orbpair_irreps), "Hidden irreps insufficient."

        self.sh = SphericalHarmonics(irreps_sh, sh_normalized, sh_normalization)

        # Import OneHot locally to avoid circular dependencies if any, or assume imported
        from dptb.nn.type_encode.one_hot import OneHotAtomEncoding, OneHotEdgeEmbedding
        self.onehot = OneHotAtomEncoding(num_types=self.n_atom, set_features=False, idp=self.idp, universal=universal)
        self.edge_one_hot = OneHotEdgeEmbedding(num_types=self.n_atom, idp=self.idp, universal=universal,
                                                d_emb=edge_one_hot_dim)

        self.init_layer = InitLayer(
            idp=self.idp, num_types=self.n_atom, n_radial_basis=n_radial_basis, r_max=r_max,
            irreps_sh=irreps_sh, avg_num_neighbors=avg_num_neighbors, env_embed_multiplicity=env_embed_multiplicity,
            two_body_latent_channels=latent_channels, latent_dim=latent_dim, r_start_cos_ratio=r_start_cos_ratio,
            PolynomialCutoff_p=PolynomialCutoff_p, cutoff_type=cutoff_type, edge_one_hot_dim=edge_one_hot_dim,
            device=device, dtype=dtype, norm_eps=norm_eps
        )

        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            irreps_in = self.init_layer.irreps_out if i == 0 else irreps_hidden

            if i == n_layers - 1:
                irreps_out = orbpair_irreps
                use_interp = use_interpolation_out
            else:
                irreps_out = irreps_hidden
                use_interp = False

            self.layers.append(Layer(
                num_types=self.n_atom, avg_num_neighbors=avg_num_neighbors,
                irreps_in=irreps_in, irreps_out=irreps_out,
                tp_radial_emb=tp_radial_emb, tp_radial_channels=tp_radial_channels,
                use_layer_onehot_tp=use_layer_onehot_tp, edge_one_hot_dim=edge_one_hot_dim,
                latent_channels=latent_channels, latent_dim=latent_dim,
                res_update=res_update, res_update_ratios=res_update_ratios,
                res_update_ratios_learnable=res_update_ratios_learnable,
                dtype=dtype, device=device, use_interpolation_tp=use_interp, norm_eps=norm_eps
            ))

        # Output layers
        self.use_out_onehot_tp = use_out_onehot_tp
        if self.use_out_onehot_tp:
            self.out_node_ele_tp = FullyConnectedTensorProduct(
                irreps_in1=self.layers[-1].irreps_out, irreps_in2='95x0e', irreps_out=self.idp.orbpair_irreps,
            )
            self.out_edge_ele_tp = FullyConnectedTensorProduct(
                irreps_in1=self.layers[-1].irreps_out, irreps_in2=f'{edge_one_hot_dim}x0e',
                irreps_out=self.idp.orbpair_irreps,
            )
        self.out_edge = Linear(self.layers[-1].irreps_out, self.idp.orbpair_irreps, shared_weights=True,
                               internal_weights=True, biases=True)
        self.out_node = Linear(self.layers[-1].irreps_out, self.idp.orbpair_irreps, shared_weights=True,
                               internal_weights=True, biases=True)

    def _init_wigner_resources(self):
        """加载并注册 Wigner-D 计算资源"""
        tp_path = os.path.dirname(tp.__file__)
        try:
            Jd = torch.load(os.path.join(tp_path, "Jd.pt"), weights_only=False)
            idx_data = torch.load(os.path.join(tp_path, "z_rot_indices_lmax12.pt"), weights_only=False)
        except FileNotFoundError:
            # Fallback for development environment
            Jd = torch.load("Jd.pt", weights_only=False)
            idx_data = torch.load("z_rot_indices_lmax12.pt", weights_only=False)

        # Determine L max
        l_list = [l for (_, (l, _)) in self.idp.orbpair_irreps]
        self.l_max = max(l_list) if l_list else 0

        # Register buffers to ensure correct device/dtype
        for l, tensor in enumerate(Jd):
            if l <= self.l_max:
                self.register_buffer(f"Jd_{l}", tensor)

        # Register indices (non-trainable)
        self.register_buffer("idx_sizes", idx_data["sizes"][:self.l_max + 1])
        self.register_buffer("idx_offsets", idx_data["offsets"][:self.l_max + 1])
        self.register_buffer("idx_mask", idx_data["mask"][:self.l_max + 1])
        self.register_buffer("idx_freq", idx_data["freq"][:self.l_max + 1])
        self.register_buffer("idx_reversed_inds", idx_data["reversed_inds"][:self.l_max + 1])

        # Pre-calculate slicing indices for output assembly
        pool_offsets = {}
        curr = 0
        for l in range(self.l_max + 1):
            pool_offsets[l] = curr
            curr += (2 * l + 1)

        self.block_slices = []
        self.target_dim = self.idp.orbpair_irreps.dim
        current_target_idx = 0
        for mul, (l, p) in self.idp.orbpair_irreps:
            for _ in range(mul):
                start_in_pool = pool_offsets[l]
                size = 2 * l + 1
                self.block_slices.append(
                    (start_in_pool, start_in_pool + size, current_target_idx, current_target_idx + size))
                current_target_idx += size

    def _get_wigner_D_matrix(self, Q):
        """使用本地资源计算 Wigner-D 矩阵"""
        # 1. Matrix -> Euler (使用 e3nn 纯几何工具，安全)
        alpha, beta, gamma = o3.matrix_to_angles(Q)

        # 2. Prepare Jd list
        Jd_list = [getattr(self, f"Jd_{l}") for l in range(self.l_max + 1)]

        # 3. Local Batch Calculation (Mimic batch_wigner_D)
        N = alpha.shape[0]
        D_total = self.idx_sizes.sum().item()
        dtype = Q.dtype
        device = Q.device

        # Reconstruct J_full locally using buffers
        J_full_small = torch.zeros(D_total, D_total, device=device, dtype=dtype)
        for l in range(self.l_max + 1):
            start = self.idx_offsets[l]
            J_full_small[start:start + 2 * l + 1, start:start + 2 * l + 1] = Jd_list[l]

        J_full = J_full_small.unsqueeze(0).expand(N, -1, -1)
        angle_stack = torch.cat([alpha, beta, gamma], dim=0)

        # Call tp.build_z_rot_multi
        Xa, Xb, Xc = tp.build_z_rot_multi(
            angle_stack, self.idx_mask, self.idx_freq,
            self.idx_reversed_inds, self.idx_offsets, self.idx_sizes
        )

        # Full pool
        D_pool = Xa @ J_full @ Xb @ J_full @ Xc

        # 4. Assemble Target
        D_final = torch.zeros(N, self.target_dim, self.target_dim, device=device, dtype=dtype)
        for (ps, pe, ts, te) in self.block_slices:
            D_final[:, ts:te, ts:te] = D_pool[:, ps:pe, ps:pe]

        return D_final

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # ============================================================
        # 1. PRE-PROCESSING: Canonicalization
        # ============================================================
        pos = data[_keys.POSITIONS_KEY]
        batch = data[_keys.BATCH_KEY] if _keys.BATCH_KEY in data else torch.zeros(pos.shape[0], dtype=torch.long,
                                                                                  device=pos.device)

        # Calculate Q and rotate pos
        Q = self.canonicalizer.get_canonical_orientation(pos, batch)

        center = scatter_mean(pos, batch, dim=0)
        pos_centered = pos - center[batch]
        # Rotate to canonical frame: pos_can = (pos - center) @ Q
        # Q is [Batch, 3, 3], pos_centered is [N, 3]
        # Broadcast Q to N atoms
        pos_can = torch.bmm(pos_centered.unsqueeze(1), Q[batch]).squeeze(1)

        # Update data with canonical positions for GNN
        data[_keys.POSITIONS_KEY] = pos_can
        # ============================================================

        # --- Standard Lem Forward ---
        data = with_edge_vectors(data, with_lengths=True)
        data = with_batch(data)

        edge_index = data[_keys.EDGE_INDEX_KEY]
        edge_vector = data[_keys.EDGE_VECTORS_KEY]
        edge_sh = self.sh(data[_keys.EDGE_VECTORS_KEY][:, [1, 2, 0]])
        edge_length = data[_keys.EDGE_LENGTH_KEY]

        data = self.onehot(data)
        edge_one_hot = self.edge_one_hot(data)
        node_one_hot = data[_keys.NODE_ATTRS_KEY]
        atom_type = data[_keys.ATOM_TYPE_KEY].flatten()
        bond_type = data[_keys.EDGE_TYPE_KEY].flatten()

        num_nodes_total = node_one_hot.shape[0]

        latents, node_features, edge_features, cutoff_coeffs, active_edges = self.init_layer(edge_index, atom_type,
                                                                                             bond_type, edge_sh,
                                                                                             edge_length, edge_one_hot)

        n_active_nodes = node_features.shape[0]
        safe_node_one_hot = node_one_hot[:n_active_nodes] if n_active_nodes < num_nodes_total else node_one_hot
        edge_one_hot = edge_one_hot[active_edges]
        data[_keys.EDGE_OVERLAP_KEY] = latents

        wigner_D_all = None
        for idx, layer in enumerate(self.layers):
            latents, node_features, edge_features, wigner_D_all = layer(
                latents, node_features, edge_features, safe_node_one_hot,
                edge_index, edge_vector, atom_type, cutoff_coeffs, active_edges,
                edge_one_hot, wigner_D_all
            )

        if node_features.shape[0] < num_nodes_total:
            pad_num = num_nodes_total - node_features.shape[0]
            pad = torch.zeros(pad_num, node_features.shape[1], device=node_features.device, dtype=node_features.dtype)
            node_features = torch.cat([node_features, pad], dim=0)

        out_node_features = self.out_node(node_features)
        out_edge_features = self.out_edge(edge_features)

        if self.use_out_onehot_tp:
            out_node_features = out_node_features + self.out_node_ele_tp(node_features, node_one_hot)
            out_edge_features = out_edge_features + self.out_edge_ele_tp(edge_features, edge_one_hot)

        # ============================================================
        # 2. POST-PROCESSING: Restoration
        # ============================================================
        # Get Wigner-D matrix for the transformation Q
        # Note: Q transforms Global Basis -> Canonical Basis
        # To restore Canonical Features -> Global Features, we rotate by Q
        # F_global = D(Q) @ F_can

        D_matrix = self._get_wigner_D_matrix(Q)  # [Batch, Dim, Dim]

        # Node restoration
        D_nodes = D_matrix[batch]  # Broadcast to nodes
        out_node_features = torch.einsum('nij,nj->ni', D_nodes, out_node_features)

        # Edge restoration
        active_edge_index_src = edge_index[0, active_edges]
        active_edge_batch = batch[active_edge_index_src]
        D_edges = D_matrix[active_edge_batch]
        out_edge_features = torch.einsum('nij,nj->ni', D_edges, out_edge_features)
        # ============================================================

        data[_keys.NODE_FEATURES_KEY] = out_node_features
        data[_keys.EDGE_FEATURES_KEY] = torch.zeros(edge_index.shape[1], self.idp.orbpair_irreps.dim, dtype=self.dtype,
                                                    device=self.device)
        data[_keys.EDGE_FEATURES_KEY] = torch.index_copy(data[_keys.EDGE_FEATURES_KEY], 0, active_edges,
                                                         out_edge_features)

        return data

    @property
    def out_edge_irreps(self):
        return self.idp.orbpair_irreps

    @property
    def out_node_irreps(self):
        return self.idp.orbpair_irreps