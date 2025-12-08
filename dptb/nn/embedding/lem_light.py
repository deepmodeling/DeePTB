from typing import Optional, List, Union, Dict
import os
import math

import torch
from torch_runstats.scatter import scatter
from e3nn import o3
from e3nn.nn import Gate
from e3nn.o3 import Linear, SphericalHarmonics, FullyConnectedTensorProduct, TensorProduct

from dptb.nn.embedding.emb import Embedding
from dptb.data import AtomicDataDict, _keys
from dptb.data.AtomicDataDict import with_edge_vectors, with_batch
from dptb.data.transforms import OrbitalMapper

from ..type_encode.one_hot import OneHotAtomEncoding, OneHotEdgeEmbedding
from ..base import ScalarMLPFunction
from dptb.nn.rescale import E3ElementLinear
from dptb.nn.tensor_product import SO2_Linear
# === 1. 引入 SeperableLayerNorm ===
from dptb.nn.norm import SeperableLayerNorm

# 复用原 lem.py 中的 InitLayer
from .lem import InitLayer


class UpdateNode(torch.nn.Module):
    """
    仅节点更新版本的 UpdateNode，同时显式更新 edge feature：
    - 使用 [中心节点, 旧边特征, 邻居节点] 做张量积生成 per-edge 的 edge_messages；
    - edge_messages 既作为新的 edge feature 保存，又通过加权聚合更新节点特征。
    """

    def __init__(
            self,
            node_irreps_in: o3.Irreps,
            edge_irreps_in: o3.Irreps,
            irreps_out: o3.Irreps,
            latent_dim: int,
            # === 2. 接收 node_one_hot_dim 参数 ===
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
    ):
        super(UpdateNode, self).__init__()

        self.node_irreps_in = o3.Irreps(node_irreps_in)
        self.edge_irreps_in = o3.Irreps(edge_irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.node_one_hot_dim = node_one_hot_dim

        self.dtype = dtype
        self.device = device
        self.res_update = res_update

        # 按平均邻居数做归一化
        self.register_buffer(
            "env_sum_normalizations",
            torch.as_tensor(avg_num_neighbors).rsqrt(),
        )

        # === 3. 定义 Normalization 层 ===
        self.sln_n = SeperableLayerNorm(
            irreps=self.node_irreps_in,
            eps=norm_eps,
            affine=True,
            normalization='component',
            std_balance_degrees=True,
            dtype=self.dtype,
            device=self.device
        )

        self.sln_e = SeperableLayerNorm(
            irreps=self.edge_irreps_in,
            eps=norm_eps,
            affine=True,
            normalization='component',
            std_balance_degrees=True,
            dtype=self.dtype,
            device=self.device
        )

        # 将 per-edge 消息（edge_messages）映射到用于聚合到节点的权重空间
        self._env_weighter = E3ElementLinear(
            irreps_in=self.irreps_out,
            dtype=dtype,
            device=device,
        )

        assert self.irreps_out[0].ir.l == 0

        # 生成 attention-like 权重
        self.env_embed_mlps = ScalarMLPFunction(
            mlp_input_dimension=latent_dim,
            mlp_latent_dimensions=[],
            mlp_output_dimension=self._env_weighter.weight_numel,
        )

        # Gate 激活拆分 scalar / gated 部分
        irreps_scalar = o3.Irreps(
            [(mul, ir) for mul, ir in self.irreps_out if ir.l == 0]
        ).simplify()
        irreps_gated = o3.Irreps(
            [(mul, ir) for mul, ir in self.irreps_out if ir.l > 0]
        ).simplify()
        irreps_gates = o3.Irreps(
            [(mul, (0, 1)) for mul, _ in irreps_gated]
        ).simplify()

        act = {1: torch.nn.functional.silu, -1: torch.tanh}
        act_gates = {1: torch.sigmoid, -1: torch.tanh}

        self.activation = Gate(
            irreps_scalar,
            [act[ir.p] for _, ir in irreps_scalar],
            irreps_gates,
            [act_gates[ir.p] for _, ir in irreps_gates],
            irreps_gated,
        )

        # 核心张量积：参考 UpdateEdge 的设计，用 [node_i, edge, node_j]
        self.tp = SO2_Linear(
            irreps_in=self.node_irreps_in + self.edge_irreps_in + self.node_irreps_in,
            irreps_out=self.activation.irreps_in,
            latent_dim=latent_dim,
            radial_emb=radial_emb,
            radial_channels=radial_channels,
            use_interpolation=use_interpolation_tp,
        )

        self.lin_post = Linear(
            self.activation.irreps_out,
            self.irreps_out,
            shared_weights=True,
            internal_weights=True,
            biases=True,
        )

        # 残差权重参数（只用于节点）
        if res_update_ratios is None:
            res_update_params = torch.zeros(1)
        else:
            res_update_ratios = torch.as_tensor(
                res_update_ratios, dtype=torch.get_default_dtype()
            )
            assert torch.all(res_update_ratios > 0.0)
            assert torch.all(res_update_ratios < 1.0)
            res_update_params = torch.special.logit(res_update_ratios)
            res_update_params.clamp_(-6.0, 6.0)

        if res_update_ratios_learnable:
            self._res_update_params = torch.nn.Parameter(res_update_params)
        else:
            self.register_buffer("_res_update_params", res_update_params)

        # 每层对节点做 one-hot 调整
        self.use_layer_onehot_tp = use_layer_onehot_tp
        if use_layer_onehot_tp:
            instructions = []
            for i, (mul, ir) in enumerate(self.irreps_out):
                instructions.append((i, 0, i, "uvu", True))

            # === 4. 使用动态维度构建 TensorProduct ===
            self.node_onehot_tp = TensorProduct(
                irreps_in1=self.irreps_out,
                irreps_in2=f"{self.node_one_hot_dim}x0e",  # 修复硬编码 "95x0e"
                irreps_out=self.irreps_out,
                instructions=instructions,
            )

        self.use_identity_res = (
                                        self.node_irreps_in == self.irreps_out
                                ) and res_update
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
    ):
        edge_center = edge_index[0]
        edge_neighbor = edge_index[1]

        # === 5. 应用 SeperableLayerNorm (Fix Stability) ===
        normed_nodes = self.sln_n(node_features)
        normed_edges = self.sln_e(edge_features)

        # 使用归一化后的特征进行 TP 计算
        center_node_feat = normed_nodes[edge_center[active_edges]]
        neighbor_node_feat = normed_nodes[edge_neighbor[active_edges]]
        edge_input = torch.cat(
            [center_node_feat, normed_edges, neighbor_node_feat], dim=-1
        )

        edge_messages, wigner_D_all = self.tp(
            edge_input,
            edge_vector[active_edges],
            latents[active_edges],
            wigner_D_all,
        )

        edge_messages = self.activation(edge_messages)
        edge_messages = self.lin_post(edge_messages)

        # === 注意：后续聚合和残差使用原始 latents 和 features (未被 SLN 处理的 residuals) ===

        # === 2. 使用 latent 生成权重，聚合 edge_messages 到节点 ===
        weights = self.env_embed_mlps(latents[active_edges])
        # 对 edge_messages 做元素线性变换，相当于 attention 权重
        edge_messages_weighted = self._env_weighter(edge_messages, weights)

        aggregated_node_messages = scatter(
            edge_messages_weighted,
            edge_center[active_edges],
            dim=0,
        )

        # 邻居数归一化
        if self.env_sum_normalizations.ndim < 1:
            norm_const = self.env_sum_normalizations
        else:
            norm_const = self.env_sum_normalizations[
                atom_type.flatten()
            ].unsqueeze(-1)

        new_node_features = aggregated_node_messages * norm_const

        # === 3. 节点残差更新 ===
        if self.res_update:
            update_coefficients = self._res_update_params.sigmoid()
            coefficient_old = torch.rsqrt(
                update_coefficients.square() + 1
            )
            coefficient_new = update_coefficients * coefficient_old

            if self.use_identity_res:
                node_features = (
                        coefficient_old * node_features
                        + coefficient_new * new_node_features
                )
            else:
                node_features = (
                        coefficient_old * self.linear_res(node_features)
                        + coefficient_new * new_node_features
                )
        else:
            node_features = new_node_features

        # === 4. 节点 one-hot 调整 ===
        if self.use_layer_onehot_tp:
            onehot_tune_node_feat = self.node_onehot_tp(
                node_features, node_onehot
            )
            node_features = node_features + onehot_tune_node_feat

        # 注意：此处不对 edge_messages 做残差，直接作为新的边特征使用
        return node_features, edge_messages, wigner_D_all


@Embedding.register("lem_light")
class LemLight(torch.nn.Module):
    """
    仅使用 UpdateNode 的轻量版 Lem：
    - InitLayer 初始化 latents / node_features / edge_features；
    - 每一层的 UpdateNode 同时更新 node_features 和 edge_features（message）；
    - 输出接口与原 Lem 保持一致。
    """

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
            # 与 Lem 一致的裁边参数
            prune_edges_by_cutoff: bool = True,
            prune_log_path: Optional[str] = None,
            **kwargs,
    ):
        super(LemLight, self).__init__()

        # 裁边日志
        self.prune_log_path = prune_log_path
        if self.prune_log_path and os.path.exists(self.prune_log_path):
            try:
                os.remove(self.prune_log_path)
            except:
                pass

        irreps_hidden = o3.Irreps(irreps_hidden)
        lmax = irreps_hidden.lmax

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # basis / idp
        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb")
            if idp is not None:
                assert (
                        idp == self.idp
                ), "The basis of idp and basis should be the same."
        else:
            assert (
                    idp is not None
            ), "Either basis or idp should be provided."
            self.idp = idp

        self.basis = self.idp.basis
        self.idp.get_irreps(no_parity=False)

        # === 动态确定 atom types 数量 ===
        if universal:
            self.n_atom = 95
        else:
            self.n_atom = len(self.basis.keys())

        irreps_sh = o3.Irreps(
            [(1, (i, (-1) ** i)) for i in range(lmax + 1)]
        )
        orbpair_irreps = self.idp.orbpair_irreps.sort()[0].simplify()

        # 检查 hidden irreps 至少覆盖哈密顿量所需 irreps
        irreps_out_check = []
        for mul, ir1 in irreps_hidden:
            for _, ir2 in orbpair_irreps:
                irreps_out_check += [
                    o3.Irrep(str(irr)) for irr in ir1 * ir2
                ]
        irreps_out_check = o3.Irreps(irreps_out_check).sort()[0].simplify()
        assert all(ir in irreps_out_check for _, ir in orbpair_irreps), (
            "hidden irreps should at least cover all the required irreps "
            f"in the hamiltonian data {orbpair_irreps}"
        )

        # SH & one-hot 编码
        self.sh = SphericalHarmonics(
            irreps_sh, sh_normalized, sh_normalization
        )
        self.onehot = OneHotAtomEncoding(
            num_types=self.n_atom,
            set_features=False,
            idp=self.idp,
            universal=universal,
        )
        self.edge_one_hot = OneHotEdgeEmbedding(
            num_types=self.n_atom,
            idp=self.idp,
            universal=universal,
            d_emb=edge_one_hot_dim,
        )

        # InitLayer：从 .lem 复用
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
        )

        # 仅堆叠 UpdateNode 层：节点/边 irreps 在每层中共享
        self.layers = torch.nn.ModuleList()
        current_irreps = self.init_layer.irreps_out

        for i in range(n_layers):
            if i == 0:
                irreps_in_layer = self.init_layer.irreps_out
            else:
                irreps_in_layer = irreps_hidden

            if i == n_layers - 1:
                irreps_out_layer = orbpair_irreps.sort()[0].simplify()
                use_interpolation_tp = bool(use_interpolation_out)
            else:
                irreps_out_layer = irreps_hidden
                use_interpolation_tp = False

            self.layers.append(
                UpdateNode(
                    node_irreps_in=irreps_in_layer,
                    edge_irreps_in=irreps_in_layer,
                    irreps_out=irreps_out_layer,
                    latent_dim=latent_dim,
                    # === 6. 传入正确的 one-hot 维度 (Fix Crash) ===
                    node_one_hot_dim=self.n_atom,
                    norm_eps=norm_eps,
                    radial_emb=tp_radial_emb,
                    radial_channels=tp_radial_channels,
                    res_update=res_update,
                    use_layer_onehot_tp=use_layer_onehot_tp,
                    use_interpolation_tp=use_interpolation_tp,
                    res_update_ratios=res_update_ratios,
                    res_update_ratios_learnable=(
                        res_update_ratios_learnable
                    ),
                    avg_num_neighbors=avg_num_neighbors,
                    dtype=dtype,
                    device=device,
                )
            )

            current_irreps = irreps_out_layer

            if use_interpolation_tp:
                print(f"Use interpolation SO2 layer in layer {i}")

        self.node_irreps_out = current_irreps
        self.edge_irreps_out = current_irreps

        # 输出层
        self.use_out_onehot_tp = use_out_onehot_tp
        if self.use_out_onehot_tp:
            # === 7. 动态输出层 TP 维度 (Fix Crash) ===
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
        # 边向量 & batch
        data = with_edge_vectors(data, with_lengths=True)
        data = with_batch(data)

        edge_index = data[_keys.EDGE_INDEX_KEY]
        edge_vector = data[_keys.EDGE_VECTORS_KEY]
        edge_sh = self.sh(
            data[_keys.EDGE_VECTORS_KEY][:, [1, 2, 0]]
        )
        edge_length = data[_keys.EDGE_LENGTH_KEY]

        # One-hot
        data = self.onehot(data)
        edge_one_hot_all = self.edge_one_hot(data)
        node_one_hot = data[_keys.NODE_ATTRS_KEY]
        atom_type = data[_keys.ATOM_TYPE_KEY].flatten()
        bond_type = data[_keys.EDGE_TYPE_KEY].flatten()

        num_nodes_total = node_one_hot.shape[0]

        # 初始化：latents, node_features, edge_features（仅活跃边）
        latents, node_features, edge_features, cutoff_coeffs, active_edges = self.init_layer(
            edge_index,
            atom_type,
            bond_type,
            edge_sh,
            edge_length,
            edge_one_hot_all,
        )

        # 可选裁边统计
        if self.prune_log_path is not None:
            total_edges = edge_index.shape[1]
            active_count = active_edges.shape[0]
            with open(self.prune_log_path, "a") as f:
                f.write(f"{total_edges},{active_count}\n")

        # 只对“活跃的”节点用 one-hot（避免越界）
        n_active_nodes = node_features.shape[0]
        if n_active_nodes < num_nodes_total:
            safe_node_one_hot = node_one_hot[:n_active_nodes]
        else:
            safe_node_one_hot = node_one_hot

        # 只保留活跃边 one-hot，用于最终输出
        edge_one_hot = edge_one_hot_all[active_edges]

        # 保存 latents（与原 Lem 一致）
        data[_keys.EDGE_OVERLAP_KEY] = latents

        wigner_D_all = None
        for layer in self.layers:
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
            )

        # 对被裁掉、没有邻居的尾部节点补零
        if node_features.shape[0] < num_nodes_total:
            pad_num = num_nodes_total - node_features.shape[0]
            pad = torch.zeros(
                pad_num,
                node_features.shape[1],
                device=node_features.device,
                dtype=node_features.dtype,
            )
            node_features = torch.cat([node_features, pad], dim=0)

        # 输出映射到最终 orbpair_irreps
        out_node_features = self.out_node(node_features)
        out_edge_features = self.out_edge(edge_features)

        if self.use_out_onehot_tp:
            # 节点 one-hot 调整
            out_node_features = out_node_features + self.out_node_ele_tp(
                node_features, node_one_hot
            )
            # 边 one-hot 调整（仅活跃边）
            out_edge_features = out_edge_features + self.out_edge_ele_tp(
                edge_features, edge_one_hot
            )

        data[_keys.NODE_FEATURES_KEY] = out_node_features

        # 将活跃边特征 scatter 回完整边列表
        edge_feat_full = torch.zeros(
            edge_index.shape[1],
            self.idp.orbpair_irreps.dim,
            dtype=self.dtype,
            device=self.device,
        )
        edge_feat_full = torch.index_copy(
            edge_feat_full, 0, active_edges, out_edge_features
        )
        data[_keys.EDGE_FEATURES_KEY] = edge_feat_full

        return data