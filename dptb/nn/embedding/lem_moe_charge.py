from typing import Optional, List, Union, Dict
import math
import torch
import logging
from torch_runstats.scatter import scatter
from torch_scatter import scatter_mean
from e3nn import o3
from dptb.data import AtomicDataDict, _keys
from dptb.nn.embedding.emb import Embedding
from dptb.data.AtomicDataDict import with_edge_vectors, with_batch
from dptb.data.transforms import OrbitalMapper

# 导入 MoE 组件
from dptb.nn.tensor_product_moe import MOLERouter, MOLEGlobals

# 导入两个基础骨架模型
# 假设 LemMoE 在 .lem_moe 中，LemInFrameMoE 在 .lem_in_frame_moe 中
from .lem_moe import LemMoE
from .lem_in_frame_moe import LemInFrameMoE

log = logging.getLogger(__name__)


# ==========================================
# 1. Charge Embedding Module (Helper Classes)
# ==========================================

class ResidualMLP(torch.nn.Module):
    def __init__(self, num_features, num_residual, activation="swish", bias=True, zero_init=True):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()

        act_cls = torch.nn.SiLU if activation == "swish" else torch.nn.ReLU

        for _ in range(num_residual):
            self.activations.append(act_cls())
            block = torch.nn.Sequential(
                torch.nn.Linear(num_features, num_features, bias=bias),
                act_cls(),
                torch.nn.Linear(num_features, num_features, bias=bias),
            )
            if zero_init:
                torch.nn.init.zeros_(block[-1].weight)
                if bias:
                    torch.nn.init.zeros_(block[-1].bias)
            self.layers.append(block)

    def forward(self, x):
        for act, layer in zip(self.activations, self.layers):
            x = x + layer(act(x))
        return x


class ElectronicEmbedding(torch.nn.Module):
    """
    x: 形状 [N, F]，这里的 F = 零阶 (l=0) 标量通道维度
    E: 形状 [B]，每个 batch 的总电荷
    batch_seg: 形状 [N]，节点所属 batch
    """

    def __init__(self, num_features: int, num_residual: int = 1, activation: str = "swish"):
        super().__init__()
        self.num_features = num_features

        self.linear_q = torch.nn.Linear(num_features, num_features)
        self.linear_k = torch.nn.Linear(2, num_features, bias=False)
        self.linear_v = torch.nn.Linear(2, num_features, bias=False)

        self.resblock = ResidualMLP(
            num_features=num_features,
            num_residual=num_residual,
            activation=activation,
            zero_init=True,
            bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.orthogonal_(self.linear_k.weight)
        torch.nn.init.orthogonal_(self.linear_v.weight)
        torch.nn.init.orthogonal_(self.linear_q.weight)
        torch.nn.init.zeros_(self.linear_q.bias)

    def forward(self, x, E, num_batch, batch_seg, eps: float = 1e-8):
        # x: [N, F]
        E = E.to(dtype=x.dtype)

        # query: 来自节点零阶标量特征
        q = self.linear_q(x)  # [N, F]

        # 构造成 [Q, -Q] 的 2 维特征
        e = torch.nn.functional.relu(torch.stack([E, -E], dim=-1))  # [B, 2]

        # 归一化后映射到 k，未归一化映射到 v
        enorm = torch.maximum(e, torch.ones_like(e))
        k = self.linear_k(e / enorm)[batch_seg]  # [N, F]
        v = self.linear_v(e)[batch_seg]  # [N, F]

        # 注意力
        dot = torch.sum(k * q, dim=-1) / (k.shape[-1] ** 0.5)  # [N]
        a = torch.nn.functional.softplus(dot)  # [N]

        # 归一化注意力系数
        anorm = torch.zeros(num_batch, device=x.device, dtype=x.dtype).index_add(0, batch_seg, a)  # [B]
        out = self.resblock((a / (anorm[batch_seg] + eps)).unsqueeze(-1) * v)  # [N, F]

        return out


# ==========================================
# 2. LemMoECharge (Main Class)
# ==========================================

@Embedding.register("lem_moe_charge")
class LemMoECharge(LemInFrameMoE):
    """
    支持 Charge Embedding 的 MoE 模型。
    通过 backbone 参数选择底层实现：
    - "lem_in_frame_moe" (默认): 更轻量，无 edge update，支持 keep-in-frame。
    - "lem_moe": 原始重型版本，带 edge update。
    """

    def __init__(
            self,
            # Charge Params
            num_charge_residual: int = 1,
            charge_activation: str = "swish",
            charge_embedding_dim: Optional[int] = None,
            # Backbone Selection
            backbone: str = "lem_in_frame_moe",  # Options: "lem_in_frame_moe", "lem_moe"
            # LemMoE / MOE Params
            top_k_experts: int = 1,
            **kwargs,
    ):
        self.backbone = backbone

        # 1. 根据 backbone 选择初始化逻辑
        if backbone == "lem_moe":
            # 显式调用 LemMoE 的初始化
            LemMoE.__init__(self, **kwargs)
        else:
            # 默认调用父类 (LemInFrameMoE) 的初始化
            # 注意：LemInFrameMoE 支持 in_frame_flag, ln_flag 等特有参数，会在 kwargs 中传递
            super().__init__(**kwargs)

        # 2. 自动获取零阶 (l=0) 标量通道的维度
        scalar_irreps = o3.Irreps(
            [(mul, ir) for mul, ir in self.init_layer.irreps_out if ir.l == 0]
        ).simplify()

        if scalar_irreps.dim == 0:
            raise ValueError(
                "init_layer.irreps_out 中没有任何 l=0 分量，无法为电荷构造嵌入。"
            )

        self.scalar_dim = scalar_irreps.dim

        # 3. 构建电子 / 电荷嵌入模块
        self.elec_emb = ElectronicEmbedding(
            num_features=self.scalar_dim,
            num_residual=num_charge_residual,
            activation=charge_activation,
        )

        # 4. 覆盖 Router
        # Router 输入维度 = 零阶标量 + 电荷嵌入（拼接，所以是 2 * scalar_dim）
        router_in_features = 2 * self.scalar_dim

        if hasattr(self, "num_experts"):
            num_experts = self.num_experts
        else:
            num_experts = kwargs.get("num_experts", 8)

        # 重新初始化 Router，覆盖掉 backbone 中创建的那个
        self.router = MOLERouter(
            in_features=router_in_features,
            num_experts=num_experts,
            top_k=top_k_experts,
        )

        self._logged_charge_info = False

        log.info(
            f"[LemMoECharge] Backbone: {self.backbone}, "
            f"scalar_dim={self.scalar_dim}, "
            f"router_in_features={router_in_features}, "
            f"top_k_experts={top_k_experts}"
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # --- 通用前处理 ---
        data = with_edge_vectors(data, with_lengths=True)
        data = with_batch(data)

        edge_index = data[_keys.EDGE_INDEX_KEY]
        edge_vector = data[_keys.EDGE_VECTORS_KEY]
        edge_sh = self.sh(data[_keys.EDGE_VECTORS_KEY][:, [1, 2, 0]])
        edge_length = data[_keys.EDGE_LENGTH_KEY]

        data = self.onehot(data)
        node_one_hot = data[_keys.NODE_ATTRS_KEY]
        edge_one_hot_all = self.edge_one_hot(data)  # 获取所有边的 onehot

        atom_type = data[_keys.ATOM_TYPE_KEY].flatten()
        bond_type = data[_keys.EDGE_TYPE_KEY].flatten()
        batch = data[_keys.BATCH_KEY]

        # 计算 batch 数
        if batch.numel() > 0:
            num_batch = batch.max().item() + 1
        else:
            num_batch = 1
            batch = torch.zeros(
                node_one_hot.shape[0],
                dtype=torch.long,
                device=self.device,
            )

        # --- InitLayer ---
        num_nodes_total = node_one_hot.shape[0]
        latents, node_features, edge_features, cutoff_coeffs, active_edges = self.init_layer(
            edge_index, atom_type, bond_type, edge_sh, edge_length, edge_one_hot_all
        )

        n_active_nodes = node_features.shape[0]
        if n_active_nodes < num_nodes_total:
            safe_node_one_hot = node_one_hot[:n_active_nodes]
        else:
            safe_node_one_hot = node_one_hot

        edge_one_hot = edge_one_hot_all[active_edges]  # 筛选 active edges
        batch_nodes = batch[:n_active_nodes]

        # ============================================================
        # Charge Logic (通用)
        # ============================================================
        total_charge = data.get("charge") if "charge" in data else data.get("Q")

        if not self._logged_charge_info:
            if total_charge is not None:
                log.info("[LemMoECharge] Dataset contains charge info.")
            else:
                log.warning("[LemMoECharge] No charge info. Using dummy charges.")
            self._logged_charge_info = True

        if total_charge is None:
            total_charge = torch.randint(
                low=0, high=4, size=(num_batch,), device=self.device,
            ).to(dtype=self.dtype) - 1.0

        # 取出零阶部分 -> 计算 Embedding -> 加回
        node_scalar_orig = node_features[:, :self.scalar_dim]
        q_emb = self.elec_emb(node_scalar_orig, total_charge, num_batch, batch_nodes)
        node_scalar_updated = node_scalar_orig + q_emb

        if node_features.shape[1] > self.scalar_dim:
            node_features = torch.cat(
                [node_scalar_updated, node_features[:, self.scalar_dim:]], dim=-1
            )
        else:
            node_features = node_scalar_updated

        # ============================================================
        # MOE Router (通用)
        # ============================================================
        # 输入为 [scalar || q_emb]
        router_node_feat = torch.cat([node_scalar_updated, q_emb], dim=-1)
        global_feat = scatter_mean(router_node_feat, batch_nodes, dim=0)

        # 计算系数
        coeffs = self.router(global_feat)

        # 构建 Globals
        edge_batch = batch[edge_index[0][active_edges]]
        num_systems = batch.max().item() + 1 if batch.numel() > 0 else 1
        edge_sizes = torch.bincount(edge_batch, minlength=num_systems)
        mole_globals = MOLEGlobals(coefficients=coeffs, sizes=edge_sizes)

        # ============================================================
        # Layer Loop (根据 Backbone 分支)
        # ============================================================
        data[_keys.EDGE_OVERLAP_KEY] = latents
        wigner_D_all = None

        if self.backbone == "lem_moe":
            # --- Legacy LemMoE Loop (Has UpdateEdge, returns latents) ---
            for layer in self.layers:
                latents, node_features, edge_features, wigner_D_all = layer(
                    latents,
                    node_features,
                    edge_features,
                    safe_node_one_hot,
                    edge_index,
                    edge_vector,
                    atom_type,
                    cutoff_coeffs,
                    active_edges,
                    edge_one_hot,
                    wigner_D_all,
                    mole_globals
                )
        else:
            # --- LemInFrameMoE Loop (No UpdateEdge step, In-Frame logic) ---
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
                    mole_globals
                )

        # --- Output Block (通用) ---
        if node_features.shape[0] < num_nodes_total:
            pad_num = num_nodes_total - node_features.shape[0]
            pad = torch.zeros(
                pad_num, node_features.shape[1],
                device=node_features.device, dtype=node_features.dtype,
            )
            node_features = torch.cat([node_features, pad], dim=0)

        out_node_features = self.out_node(node_features)
        out_edge_features = self.out_edge(edge_features)

        if self.use_out_onehot_tp:
            node_one_hot_full = data[_keys.NODE_ATTRS_KEY]
            out_node_features = out_node_features + self.out_node_ele_tp(
                node_features, node_one_hot_full
            )
            out_edge_features = out_edge_features + self.out_edge_ele_tp(
                edge_features, edge_one_hot
            )

        data[_keys.NODE_FEATURES_KEY] = out_node_features
        data[_keys.EDGE_FEATURES_KEY] = torch.zeros(
            edge_index.shape[1],
            self.idp.orbpair_irreps.dim,
            dtype=self.dtype,
            device=self.device,
        )
        data[_keys.EDGE_FEATURES_KEY] = torch.index_copy(
            data[_keys.EDGE_FEATURES_KEY], 0, active_edges, out_edge_features
        )

        return data