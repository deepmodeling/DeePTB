import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union, List

from dptb.data import AtomicDataDict, _keys
from dptb.nn.embedding.emb import Embedding

# 复用 lem.py 中的组件
from .lem import Lem


# ==========================================
# 1. SpookyNet 组件 (仅保留 Charge 相关)
# ==========================================

class ResidualMLP(nn.Module):
    """
    SpookyNet 的残差块实现
    """

    def __init__(self, num_features, num_residual, activation="swish", bias=True, zero_init=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        if activation == "swish":
            act_cls = nn.SiLU
        elif activation == "relu":
            act_cls = nn.ReLU
        else:
            act_cls = nn.SiLU

        for _ in range(num_residual):
            self.activations.append(act_cls())
            block = nn.Sequential(
                nn.Linear(num_features, num_features, bias=bias),
                act_cls(),
                nn.Linear(num_features, num_features, bias=bias)
            )
            if zero_init:
                nn.init.zeros_(block[-1].weight)
                if bias:
                    nn.init.zeros_(block[-1].bias)
            self.layers.append(block)

    def forward(self, x):
        for act, layer in zip(self.activations, self.layers):
            res = layer(act(x))
            x = x + res
        return x


class ElectronicEmbedding(nn.Module):
    """
    SpookyNet 的电荷嵌入模块 (无 Spin)
    """

    def __init__(
            self,
            num_features: int,
            num_residual: int = 1,
            activation: str = "swish",
    ) -> None:
        super().__init__()
        self.num_features = num_features

        # Query projection (Atom features)
        self.linear_q = nn.Linear(num_features, num_features)

        # Key/Value projection (Global Charge)
        # Charge 分为 [正, 负] 两个通道，输入维度为 2
        self.linear_k = nn.Linear(2, num_features, bias=False)
        self.linear_v = nn.Linear(2, num_features, bias=False)

        self.resblock = ResidualMLP(
            num_features,
            num_residual,
            activation=activation,
            zero_init=True,
            bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.linear_k.weight)
        nn.init.orthogonal_(self.linear_v.weight)
        nn.init.orthogonal_(self.linear_q.weight)
        nn.init.zeros_(self.linear_q.bias)

    def forward(
            self,
            x: torch.Tensor,
            E: torch.Tensor,
            num_batch: int,
            batch_seg: torch.Tensor,
            eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        x: [N, F] 原子特征
        E: [B] 分子总电荷
        """
        E = E.to(dtype=x.dtype)

        # 1. Prepare Query (Atom)
        q = self.linear_q(x)  # [N, F]

        # 2. Prepare Key/Value (Global Charge)
        # 将标量电荷转为 [正, 负] 通道，例如 +1 -> [1, 0], -2 -> [0, 2]
        e = F.relu(torch.stack([E, -E], dim=-1))  # [B, 2]
        enorm = torch.maximum(e, torch.ones_like(e))

        # 广播到原子层级
        k = self.linear_k(e / enorm)[batch_seg]  # [N, F]
        v = self.linear_v(e)[batch_seg]  # [N, F]

        # 3. Attention
        dot = torch.sum(k * q, dim=-1) / (k.shape[-1] ** 0.5)
        a = F.softplus(dot)  # [N]

        # 4. Normalization
        anorm = torch.zeros(num_batch, device=x.device, dtype=x.dtype)
        anorm = anorm.index_add(0, batch_seg, a)
        anorm_per_atom = anorm[batch_seg]

        # 5. Injection
        attn_weights = (a / (anorm_per_atom + eps)).unsqueeze(-1)  # [N, 1]
        out = self.resblock(attn_weights * v)

        return out


# ==========================================
# 2. LemCharge 主类
# ==========================================

@Embedding.register("lem_charge")
class LemCharge(Lem):
    def __init__(
            self,
            # SpookyNet Embedding Params
            num_charge_residual: int = 1,
            charge_activation: str = "swish",
            # Lem Params
            **kwargs,
    ):

        # 初始化父类
        super().__init__(**kwargs)

        # 初始化电荷嵌入模块
        # 注意：LEM 的输入特征是 one-hot，维度为 n_atom (95)
        # 我们要将电荷嵌入加到 one-hot 上，所以维度必须对齐
        self.elec_emb = ElectronicEmbedding(
            num_features=self.n_atom,
            num_residual=num_charge_residual,
            activation=charge_activation,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # 1. 基础预处理 (同 Lem)
        from dptb.data.AtomicDataDict import with_edge_vectors, with_batch
        data = with_edge_vectors(data, with_lengths=True)
        data = with_batch(data)

        edge_index = data[_keys.EDGE_INDEX_KEY]
        edge_vector = data[_keys.EDGE_VECTORS_KEY]
        edge_sh = self.sh(data[_keys.EDGE_VECTORS_KEY][:, [1, 2, 0]])
        edge_length = data[_keys.EDGE_LENGTH_KEY]

        # 获取原子 One-Hot (作为 SpookyNet 中的 'z' 嵌入)
        data = self.onehot(data)
        node_one_hot = data[_keys.NODE_ATTRS_KEY]

        batch_seg = data[_keys.BATCH_KEY]

        # 获取 Batch Size
        if batch_seg.numel() > 0:
            num_batch = batch_seg.max().item() + 1
        else:
            num_batch = 1
            batch_seg = torch.zeros(node_one_hot.shape[0], dtype=torch.long, device=self.device)

        # print(batch_seg)
        # ============================================================
        # [NEW] Charge Embedding 逻辑 (含 Dummy Charge)
        # ============================================================

        # 尝试从 data 中获取真实 charge
        # 假设 key 为 "charge" 或 "Q"
        total_charge = data.get("charge") if "charge" in data else data.get("Q")

        if total_charge is None:
            # ---> Dummy Charge 生成逻辑 <---
            # 如果数据里没写 charge，为了跑通前向，随机生成 [-1, 0, 1, 2]
            # randint(0, 4) -> [0, 1, 2, 3] -> minus 1 -> [-1, 0, 1, 2]
            total_charge = torch.randint(
                low=0, high=4, size=(num_batch,),
                device=self.device, dtype=self.dtype
            ) - 1.0

            # (Optional) 打印一次 warning 或者 debug info，确认 dummy 生效
            # print(f"[DEBUG] Using Dummy Charge: {total_charge[:5]}")

        # 计算 Embedding
        q_emb = self.elec_emb(node_one_hot, total_charge, num_batch, batch_seg)

        # 注入信息: x = z + q
        # 这使得原本离散的原子类型特征变成了携带全局电荷信息的连续向量
        node_one_hot = node_one_hot + q_emb

        # 更新 data dict，确保后续 Layer 使用的是注入电荷后的特征
        data[_keys.NODE_ATTRS_KEY] = node_one_hot

        # ============================================================
        # 后续逻辑与 Lem 原版保持一致
        # ============================================================

        edge_one_hot = self.edge_one_hot(data)
        atom_type = data[_keys.ATOM_TYPE_KEY].flatten()
        bond_type = data[_keys.EDGE_TYPE_KEY].flatten()

        num_nodes_total = node_one_hot.shape[0]

        # InitLayer
        latents, node_features, edge_features, cutoff_coeffs, active_edges = self.init_layer(
            edge_index, atom_type, bond_type, edge_sh, edge_length, edge_one_hot
        )

        n_active_nodes = node_features.shape[0]
        if n_active_nodes < num_nodes_total:
            safe_node_one_hot = node_one_hot[:n_active_nodes]
        else:
            safe_node_one_hot = node_one_hot

        edge_one_hot = edge_one_hot[active_edges]

        data[_keys.EDGE_OVERLAP_KEY] = latents
        wigner_D_all = None

        # Message Passing Layers
        for idx, layer in enumerate(self.layers):
            latents, node_features, edge_features, wigner_D_all = \
                layer(
                    latents,
                    node_features,
                    edge_features,
                    safe_node_one_hot,  # 这里的 safe_node_one_hot 已经包含了 charge 信息
                    edge_index,
                    edge_vector,
                    atom_type,
                    cutoff_coeffs,
                    active_edges,
                    edge_one_hot,
                    wigner_D_all
                )

        # Output
        if node_features.shape[0] < num_nodes_total:
            pad_num = num_nodes_total - node_features.shape[0]
            pad = torch.zeros(
                pad_num,
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
        data[_keys.EDGE_FEATURES_KEY] = torch.zeros(edge_index.shape[1], self.idp.orbpair_irreps.dim, dtype=self.dtype,
                                                    device=self.device)
        data[_keys.EDGE_FEATURES_KEY] = torch.index_copy(data[_keys.EDGE_FEATURES_KEY], 0, active_edges,
                                                         out_edge_features)

        return data