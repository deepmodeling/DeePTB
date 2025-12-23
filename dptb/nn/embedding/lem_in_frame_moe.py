from typing import Optional, List, Union, Dict
import os
import torch
from torch_runstats.scatter import scatter
from torch_scatter import scatter_mean
from e3nn import o3
from e3nn.nn import Gate
from e3nn.o3 import (
    Linear,
    SphericalHarmonics,
    FullyConnectedTensorProduct,
    TensorProduct,
    xyz_to_angles,
)

from dptb.data import AtomicDataDict, _keys
from dptb.data.AtomicDataDict import with_edge_vectors, with_batch
from dptb.data.transforms import OrbitalMapper
from dptb.nn.embedding.emb import Embedding
from dptb.nn.norm import SeperableLayerNorm
from dptb.nn.base import ScalarMLPFunction
from dptb.nn.type_encode.one_hot import OneHotAtomEncoding, OneHotEdgeEmbedding

# === 关键修改：从 moe 模块导入 SO2_Linear 和 MoE 相关组件 ===
from dptb.nn.tensor_product_moe import SO2_Linear, MOLEGlobals, MOLERouter
from dptb.nn.tensor_product import batch_wigner_D, _Jd, rotate_vector
from dptb.nn.rescale import E3ElementLinear

# === 复用部分 ===
from .lem_in_frame import InitLayer


class UpdateNodeInFrameMoE(torch.nn.Module):
    """
    Keep-in-Frame 版本的 UpdateNode，集成了 MoE 逻辑。
    """

    def __init__(
        self,
        node_irreps_in: o3.Irreps,
        edge_irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        latent_dim: int,
        node_one_hot_dim: int,
        norm_eps: float = 1e-8,
        radial_emb: bool = False,
        radial_channels: list = [128, 128],
        res_update: bool = True,
        use_layer_onehot_tp: bool = True,
        use_interpolation_tp: bool = False,
        res_update_ratios: Optional[List[float]] = None,
        res_update_ratios_learnable: bool = False,
        avg_num_neighbors: Optional[float] = None,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
        # === SO2 flags ===
        tp_rotate_in: bool = True,
        tp_rotate_out: bool = True,
        # === Global flags ===
        ln_flag: bool = True,
        in_frame_flag: bool = True,
        onehot_mode: str = "FullTP",
        # === MoE params ===
        num_experts: int = 8,
    ):
        super(UpdateNodeInFrameMoE, self).__init__()

        self.node_irreps_in = o3.Irreps(node_irreps_in)
        self.edge_irreps_in = o3.Irreps(edge_irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.node_one_hot_dim = node_one_hot_dim
        self.dtype = dtype
        self.device = device
        self.res_update = res_update

        self.tp_rotate_in = tp_rotate_in
        self.tp_rotate_out = tp_rotate_out
        self.in_frame_flag = in_frame_flag
        self.use_layer_onehot_tp = use_layer_onehot_tp
        self.onehot_mode = onehot_mode.lower()

        self.register_buffer(
            "env_sum_normalizations",
            torch.as_tensor(avg_num_neighbors).rsqrt(),
        )

        # ---- LayerNorm with flag ----
        if ln_flag:
            self.sln_n = SeperableLayerNorm(
                irreps=self.node_irreps_in,
                eps=norm_eps,
                affine=True,
                normalization="component",
                std_balance_degrees=True,
                dtype=self.dtype,
                device=self.device,
            )
            self.sln_e = SeperableLayerNorm(
                irreps=self.edge_irreps_in,
                eps=norm_eps,
                affine=True,
                normalization="component",
                std_balance_degrees=True,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.sln_n = torch.nn.Identity()
            self.sln_e = torch.nn.Identity()

        self._env_weighter = E3ElementLinear(
            irreps_in=self.irreps_out,
            dtype=dtype,
            device=device,
        )
        assert self.irreps_out[0].ir.l == 0

        self.env_embed_mlps = ScalarMLPFunction(
            mlp_input_dimension=latent_dim,
            mlp_latent_dimensions=[],
            mlp_output_dimension=self._env_weighter.weight_numel,
        )

        irreps_scalar = o3.Irreps([(mul, ir) for mul, ir in self.irreps_out if ir.l == 0]).simplify()
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_out if ir.l > 0]).simplify()
        irreps_gates = o3.Irreps([(mul, (0, 1)) for mul, _ in irreps_gated]).simplify()
        act = {1: torch.nn.functional.silu, -1: torch.tanh}
        act_gates = {1: torch.sigmoid, -1: torch.tanh}

        self.activation = Gate(
            irreps_scalar,
            [act[ir.p] for _, ir in irreps_scalar],
            irreps_gates,
            [act_gates[ir.p] for _, ir in irreps_gates],
            irreps_gated,
        )

        # === 关键修改：使用 MoE 版本的 SO2_Linear ===
        self.tp = SO2_Linear(
            irreps_in=self.node_irreps_in + self.edge_irreps_in + self.node_irreps_in,
            irreps_out=self.activation.irreps_in,
            latent_dim=latent_dim,
            radial_emb=radial_emb,
            radial_channels=radial_channels,
            use_interpolation=use_interpolation_tp,
            rotate_in=tp_rotate_in,
            rotate_out=tp_rotate_out,
            num_experts=num_experts,  # 传入 experts 数量
        )

        self.lin_post = Linear(
            self.activation.irreps_out,
            self.irreps_out,
            shared_weights=True,
            internal_weights=True,
            biases=True,
        )

        if res_update_ratios is None:
            res_update_params = torch.zeros(1)
        else:
            res_update_ratios = torch.as_tensor(res_update_ratios, dtype=torch.get_default_dtype())
            res_update_params = torch.special.logit(res_update_ratios)
            res_update_params.clamp_(-6.0, 6.0)

        if res_update_ratios_learnable:
            self._res_update_params = torch.nn.Parameter(res_update_params)
        else:
            self.register_buffer("_res_update_params", res_update_params)

        # ---- one-hot 调节的 TP ----
        if self.use_layer_onehot_tp:
            mode = self.onehot_mode
            if mode == "fulltp":
                self.node_onehot_tp = FullyConnectedTensorProduct(
                    irreps_in1=self.irreps_out,
                    irreps_in2=f"{self.node_one_hot_dim}x0e",
                    irreps_out=self.irreps_out,
                )
            elif mode == "elementtp":
                instructions = []
                for i, (mul, ir) in enumerate(self.irreps_out):
                    instructions.append((i, 0, i, "uvu", True))
                self.node_onehot_tp = TensorProduct(
                    irreps_in1=self.irreps_out,
                    irreps_in2=f"{self.node_one_hot_dim}x0e",
                    irreps_out=self.irreps_out,
                    instructions=instructions,
                )
            else:
                raise ValueError(f"Unknown onehot_mode={onehot_mode!r}")

        self.use_identity_res = ((self.node_irreps_in == self.irreps_out) and res_update)
        if not self.use_identity_res and res_update:
            self.linear_res = Linear(
                self.node_irreps_in,
                self.irreps_out,
                shared_weights=True,
                internal_weights=True,
                biases=True,
            )

    def forward(
        self,
        latents: torch.Tensor,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        atom_type: torch.Tensor,
        node_onehot: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vector: torch.Tensor,
        active_edges: torch.Tensor,
        wigner_D_all: Optional[torch.Tensor],
        mole_globals: MOLEGlobals,  # === 关键修改：接收 MOE 全局信息 ===
    ):
        edge_center = edge_index[0]
        edge_neighbor = edge_index[1]

        norm_node_features = self.sln_n(node_features)
        norm_edge_features = self.sln_e(edge_features)

        # 1. 计算 / 复用 Wigner-D
        if wigner_D_all is None:
            l_max = max(self.node_irreps_in.lmax, self.edge_irreps_in.lmax, self.irreps_out.lmax)
            if l_max > 0:
                angle = xyz_to_angles(edge_vector[active_edges][:, [1, 2, 0]])
                wigner_D_all = batch_wigner_D(
                    l_max,
                    angle[0],
                    angle[1],
                    torch.zeros_like(angle[0]),
                    _Jd,
                )

        # 2. 准备 TP 输入
        center_node = norm_node_features[edge_center[active_edges]]
        neighbor_node = norm_node_features[edge_neighbor[active_edges]]

        if self.in_frame_flag and (not self.tp_rotate_in):
            center_node = rotate_vector(center_node, self.node_irreps_in, wigner_D_all, back=False)
            neighbor_node = rotate_vector(neighbor_node, self.node_irreps_in, wigner_D_all, back=False)

        edge_input = torch.cat([center_node, norm_edge_features, neighbor_node], dim=-1)

        # 3. SO2 Tensor Product (with MoE)
        # === 关键修改：传递 mole_globals ===
        edge_messages, wigner_D_all = self.tp(
            edge_input,
            edge_vector[active_edges],
            mole_globals, 
            latents[active_edges],
            wigner_D_all,
        )

        edge_messages = self.activation(edge_messages)
        edge_messages = self.lin_post(edge_messages)

        # 4. 节点更新
        msg_for_node = edge_messages
        if self.in_frame_flag and (not self.tp_rotate_out):
            msg_for_node = rotate_vector(edge_messages, self.irreps_out, wigner_D_all, back=True)

        weights = self.env_embed_mlps(latents[active_edges])
        edge_messages_weighted = self._env_weighter(msg_for_node, weights)

        aggregated_node_messages = scatter(edge_messages_weighted, edge_center[active_edges], dim=0)

        if self.env_sum_normalizations.ndim < 1:
            norm_const = self.env_sum_normalizations
        else:
            norm_const = self.env_sum_normalizations[atom_type.flatten()].unsqueeze(-1)

        new_node_features = aggregated_node_messages * norm_const

        if self.res_update:
            update_coefficients = self._res_update_params.sigmoid()
            coefficient_old = torch.rsqrt(update_coefficients.square() + 1)
            coefficient_new = update_coefficients * coefficient_old
            if self.use_identity_res:
                node_features = coefficient_old * node_features + coefficient_new * new_node_features
            else:
                node_features = coefficient_old * self.linear_res(node_features) + coefficient_new * new_node_features
        else:
            node_features = new_node_features

        if self.use_layer_onehot_tp:
            onehot_tune_node_feat = self.node_onehot_tp(node_features, node_onehot)
            node_features = node_features + onehot_tune_node_feat

        return node_features, edge_messages, wigner_D_all


@Embedding.register("lem_in_frame_moe")
class LemInFrameMoE(torch.nn.Module):
    """
    LemInFrame + MoE (Mixture of Experts)
    轻量化的 Lem 结构 (No Update Edge Step) 结合 MoE 的路由机制。
    """

    def __init__(
        self,
        basis: Dict[str, Union[str, list]] = None,
        idp: Union[OrbitalMapper, None] = None,
        n_layers: int = 3,
        n_radial_basis: int = 10,
        r_max: float = 5.0,
        irreps_hidden: o3.Irreps = None,
        avg_num_neighbors: Optional[float] = None,
        r_start_cos_ratio: float = 0.8,
        norm_eps: float = 1e-8,
        PolynomialCutoff_p: float = 6,
        cutoff_type: str = "polynomial",
        env_embed_multiplicity: int = 32,
        sh_normalized: bool = True,
        sh_normalization: str = "component",
        tp_radial_emb: bool = False,
        tp_radial_channels: list = [128, 128],
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
        prune_edges_by_cutoff: bool = True,
        prune_log_path: Optional[str] = None,
        # ---- Flags ----
        ln_flag: bool = True,
        in_frame_flag: bool = True,
        onehot_mode: str = "FullTP",
        # ---- MoE params ----
        num_experts: int = 8,
        **kwargs,
    ):
        super(LemInFrameMoE, self).__init__()

        self.prune_log_path = prune_log_path
        if self.prune_log_path and os.path.exists(self.prune_log_path):
            try:
                os.remove(self.prune_log_path)
            except Exception:
                pass

        irreps_hidden = o3.Irreps(irreps_hidden)
        lmax = irreps_hidden.lmax
        self.num_experts = num_experts
        print(f'LemInFrameMoE initialized. num_experts: {self.num_experts}, in_frame_flag: {in_frame_flag}')

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.ln_flag = ln_flag
        self.in_frame_flag = in_frame_flag
        self.onehot_mode = onehot_mode

        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb")
            if idp is not None:
                assert idp == self.idp
        else:
            assert idp is not None
            self.idp = idp

        self.basis = self.idp.basis
        self.idp.get_irreps(no_parity=False)

        if universal:
            self.n_atom = 95
        else:
            self.n_atom = len(self.basis.keys())

        irreps_sh = o3.Irreps([(1, (i, (-1) ** i)) for i in range(lmax + 1)])
        orbpair_irreps = self.idp.orbpair_irreps.sort()[0].simplify()

        irreps_out_check = []
        for mul, ir1 in irreps_hidden:
            for _, ir2 in orbpair_irreps:
                irreps_out_check += [o3.Irrep(str(irr)) for irr in ir1 * ir2]
        irreps_out_check = o3.Irreps(irreps_out_check).sort()[0].simplify()
        assert all(ir in irreps_out_check for _, ir in orbpair_irreps)

        self.sh = SphericalHarmonics(irreps_sh, sh_normalized, sh_normalization)
        self.onehot = OneHotAtomEncoding(
            num_types=self.n_atom, set_features=False, idp=self.idp, universal=universal
        )
        self.edge_one_hot = OneHotEdgeEmbedding(
            num_types=self.n_atom,
            idp=self.idp,
            universal=universal,
            d_emb=edge_one_hot_dim,
        )

        # === 关键修改：初始化 Router ===
        self.router = MOLERouter(in_features=self.n_atom, num_experts=num_experts)

        self.init_layer = InitLayer(
            idp=self.idp,
            num_types=self.n_atom,
            n_radial_basis=n_radial_basis,
            r_max=r_max,
            irreps_sh=irreps_sh,
            avg_num_neighbors=avg_num_neighbors,
            env_embed_multiplicity=env_embed_multiplicity,
            two_body_latent_channels=latent_channels,
            latent_dim=latent_dim,
            r_start_cos_ratio=r_start_cos_ratio,
            PolynomialCutoff_p=PolynomialCutoff_p,
            cutoff_type=cutoff_type,
            device=device,
            dtype=dtype,
            edge_one_hot_dim=edge_one_hot_dim,
            norm_eps=norm_eps,
            prune_edges_by_cutoff=prune_edges_by_cutoff,
            ln_flag=ln_flag,
        )

        self.layers = torch.nn.ModuleList()
        current_irreps = self.init_layer.irreps_out

        for i in range(n_layers):
            if i == 0:
                irreps_in_layer = self.init_layer.irreps_out
            else:
                irreps_in_layer = irreps_hidden

            # 旋转逻辑 (Keep-in-Frame)
            if self.in_frame_flag:
                if i == 0:
                    rotate_in = True
                    rotate_out = False
                else:
                    rotate_in = False
                    if i == n_layers - 1:
                        rotate_out = True
                    else:
                        rotate_out = False
            else:
                rotate_in = True
                rotate_out = True

            if i == n_layers - 1:
                irreps_out_layer = orbpair_irreps.sort()[0].simplify()
                use_interpolation_tp = bool(use_interpolation_out)
            else:
                irreps_out_layer = irreps_hidden
                use_interpolation_tp = False

            # === 关键修改：使用 UpdateNodeInFrameMoE ===
            self.layers.append(
                UpdateNodeInFrameMoE(
                    node_irreps_in=irreps_in_layer,
                    edge_irreps_in=irreps_in_layer,
                    irreps_out=irreps_out_layer,
                    latent_dim=latent_dim,
                    node_one_hot_dim=self.n_atom,
                    norm_eps=norm_eps,
                    radial_emb=tp_radial_emb,
                    radial_channels=tp_radial_channels,
                    res_update=res_update,
                    use_layer_onehot_tp=use_layer_onehot_tp,
                    use_interpolation_tp=use_interpolation_tp,
                    res_update_ratios=res_update_ratios,
                    res_update_ratios_learnable=res_update_ratios_learnable,
                    avg_num_neighbors=avg_num_neighbors,
                    dtype=dtype,
                    device=device,
                    tp_rotate_in=rotate_in,
                    tp_rotate_out=rotate_out,
                    ln_flag=ln_flag,
                    in_frame_flag=in_frame_flag,
                    onehot_mode=onehot_mode,
                    num_experts=num_experts, # Pass param
                )
            )

            current_irreps = irreps_out_layer
            if use_interpolation_tp:
                print(f"Use interpolation SO2 layer in layer {i}")

        self.node_irreps_out = current_irreps
        self.edge_irreps_out = current_irreps

        self.use_out_onehot_tp = use_out_onehot_tp
        if self.use_out_onehot_tp:
            self.out_node_ele_tp = FullyConnectedTensorProduct(
                irreps_in1=self.node_irreps_out,
                irreps_in2=f"{self.n_atom}x0e",
                irreps_out=self.idp.orbpair_irreps,
            )
            self.out_edge_ele_tp = FullyConnectedTensorProduct(
                irreps_in1=self.edge_irreps_out,
                irreps_in2=f"{edge_one_hot_dim}x0e",
                irreps_out=self.idp.orbpair_irreps,
            )

        self.out_node = Linear(
            self.node_irreps_out,
            self.idp.orbpair_irreps,
            shared_weights=True,
            internal_weights=True,
            biases=True,
        )
        self.out_edge = Linear(
            self.edge_irreps_out,
            self.idp.orbpair_irreps,
            shared_weights=True,
            internal_weights=True,
            biases=True,
        )

    @property
    def out_edge_irreps(self):
        return self.idp.orbpair_irreps

    @property
    def out_node_irreps(self):
        return self.idp.orbpair_irreps

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = with_edge_vectors(data, with_lengths=True)
        data = with_batch(data)

        edge_index = data[_keys.EDGE_INDEX_KEY]
        edge_vector = data[_keys.EDGE_VECTORS_KEY]
        edge_sh = self.sh(data[_keys.EDGE_VECTORS_KEY][:, [1, 2, 0]])
        edge_length = data[_keys.EDGE_LENGTH_KEY]

        data = self.onehot(data)
        edge_one_hot_all = self.edge_one_hot(data)
        node_one_hot = data[_keys.NODE_ATTRS_KEY]
        atom_type = data[_keys.ATOM_TYPE_KEY].flatten()
        bond_type = data[_keys.EDGE_TYPE_KEY].flatten()
        batch = data[_keys.BATCH_KEY]

        # === 关键修改：MOLE 路由逻辑 (步骤1: 计算路由系数) ===
        # 1. 计算每个 system 的全局特征 (Node One-Hot 的平均)
        global_feat = scatter_mean(node_one_hot, batch, dim=0)  # [Batch, n_atom]
        # 2. 计算路由系数
        coeffs = self.router(global_feat)  # [Batch, num_experts]

        num_nodes_total = node_one_hot.shape[0]

        latents, node_features, edge_features, cutoff_coeffs, active_edges = self.init_layer(
            edge_index,
            atom_type,
            bond_type,
            edge_sh,
            edge_length,
            edge_one_hot_all,
        )

        if self.prune_log_path is not None:
            with open(self.prune_log_path, "a") as f:
                f.write(f"{edge_index.shape[1]},{active_edges.shape[0]}\n")

        # === 关键修改：MOLE 路由逻辑 (步骤2: 计算 Active Edges 的 Batch Size) ===
        # 因为 SO2_Linear 在 active_edges 上操作，我们需要知道每个 graph 有多少条 active edges
        edge_batch = batch[edge_index[0][active_edges]]  # 找出每条 active edge 属于哪个 graph
        num_systems = batch.max().item() + 1
        edge_sizes = torch.bincount(edge_batch, minlength=num_systems)
        
        # 3. 构建 MOLEGlobals
        mole_globals = MOLEGlobals(coefficients=coeffs, sizes=edge_sizes)

        # 准备数据
        n_active_nodes = node_features.shape[0]
        safe_node_one_hot = (
            node_one_hot[:n_active_nodes] if n_active_nodes < num_nodes_total else node_one_hot
        )
        edge_one_hot = edge_one_hot_all[active_edges]
        data[_keys.EDGE_OVERLAP_KEY] = latents

        wigner_D_all = None
        for layer in self.layers:
            # === 关键修改：传递 mole_globals 到 Layer ===
            node_features, edge_features, wigner_D_all = layer(
                latents,
                node_features,
                edge_features,
                atom_type,
                safe_node_one_hot,
                edge_index,
                edge_vector,
                active_edges,
                wigner_D_all,
                mole_globals, # Pass globals
            )

        if node_features.shape[0] < num_nodes_total:
            pad = torch.zeros(
                num_nodes_total - node_features.shape[0],
                node_features.shape[1],
                device=node_features.device,
                dtype=node_features.dtype,
            )
            node_features = torch.cat([node_features, pad], dim=0)

        out_node_features = self.out_node(node_features)
        out_edge_features = self.out_edge(edge_features)

        if self.use_out_onehot_tp:
            out_node_features = out_node_features + self.out_node_ele_tp(node_features, node_one_hot)
            out_edge_features = out_edge_features + self.out_edge_ele_tp(edge_features, edge_one_hot)

        data[_keys.NODE_FEATURES_KEY] = out_node_features
        edge_feat_full = torch.zeros(
            edge_index.shape[1],
            self.idp.orbpair_irreps.dim,
            dtype=self.dtype,
            device=self.device,
        )
        edge_feat_full = torch.index_copy(edge_feat_full, 0, active_edges, out_edge_features)
        data[_keys.EDGE_FEATURES_KEY] = edge_feat_full

        return data