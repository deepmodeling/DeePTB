from typing import Optional, List, Union, Dict
import math
import torch
import logging  # [修改 1] 导入 logging
from torch_runstats.scatter import scatter
from torch_scatter import scatter_mean
from e3nn import o3
from dptb.data import AtomicDataDict, _keys
from dptb.nn.embedding.emb import Embedding
from dptb.data.AtomicDataDict import with_edge_vectors, with_batch
from dptb.data.transforms import OrbitalMapper

from .lem_moe import LemMoE, InitLayer, Layer
log = logging.getLogger(__name__)


# ==========================================
# 1. Charge Embedding Module (Inline)
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
                torch.nn.Linear(num_features, num_features, bias=bias)
            )
            if zero_init:
                torch.nn.init.zeros_(block[-1].weight)
                if bias: torch.nn.init.zeros_(block[-1].bias)
            self.layers.append(block)

    def forward(self, x):
        for act, layer in zip(self.activations, self.layers):
            x = x + layer(act(x))
        return x


class ElectronicEmbedding(torch.nn.Module):
    def __init__(self, num_features: int, num_residual: int = 1, activation: str = "swish"):
        super().__init__()
        self.num_features = num_features
        self.linear_q = torch.nn.Linear(num_features, num_features)
        self.linear_k = torch.nn.Linear(2, num_features, bias=False)
        self.linear_v = torch.nn.Linear(2, num_features, bias=False)
        self.resblock = ResidualMLP(num_features, num_residual, activation, zero_init=True, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.orthogonal_(self.linear_k.weight)
        torch.nn.init.orthogonal_(self.linear_v.weight)
        torch.nn.init.orthogonal_(self.linear_q.weight)
        torch.nn.init.zeros_(self.linear_q.bias)

    def forward(self, x, E, num_batch, batch_seg, eps=1e-8):
        E = E.to(dtype=x.dtype)
        q = self.linear_q(x)
        e = torch.nn.functional.relu(torch.stack([E, -E], dim=-1))
        enorm = torch.maximum(e, torch.ones_like(e))
        k = self.linear_k(e / enorm)[batch_seg]
        v = self.linear_v(e)[batch_seg]
        dot = torch.sum(k * q, dim=-1) / (k.shape[-1] ** 0.5)
        a = torch.nn.functional.softplus(dot)
        anorm = torch.zeros(num_batch, device=x.device, dtype=x.dtype).index_add(0, batch_seg, a)
        out = self.resblock((a / (anorm[batch_seg] + eps)).unsqueeze(-1) * v)
        return out


# ==========================================
# 2. LemMoECharge
# ==========================================

from dptb.nn.tensor_product_moe import MOLERouter, MOLEGlobals


@Embedding.register("lem_moe_charge")
class LemMoECharge(LemMoE):
    def __init__(
            self,
            # Charge Params
            num_charge_residual: int = 1,
            charge_activation: str = "swish",
            charge_embedding_dim: int = 32,
            # LemMoE Params
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.charge_embedding_dim = charge_embedding_dim
        self.elec_emb = ElectronicEmbedding(
            num_features=self.n_atom,
            num_residual=num_charge_residual,
            activation=charge_activation,
        )

        self._logged_charge_info = False

        print(f"[LemMoECharge] Charge Injection Strategy: Additive (x = z + q)")

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = with_edge_vectors(data, with_lengths=True)
        data = with_batch(data)

        edge_index = data[_keys.EDGE_INDEX_KEY]
        edge_vector = data[_keys.EDGE_VECTORS_KEY]
        edge_sh = self.sh(data[_keys.EDGE_VECTORS_KEY][:, [1, 2, 0]])
        edge_length = data[_keys.EDGE_LENGTH_KEY]

        data = self.onehot(data)
        node_one_hot = data[_keys.NODE_ATTRS_KEY]

        batch_seg = data[_keys.BATCH_KEY]
        if batch_seg.numel() > 0:
            num_batch = batch_seg.max().item() + 1
        else:
            num_batch = 1
            batch_seg = torch.zeros(node_one_hot.shape[0], dtype=torch.long, device=self.device)

        # ============================================================
        # [NEW] Charge Injection Logic
        # ============================================================
        total_charge = data.get("charge") if "charge" in data else data.get("Q")

        if not self._logged_charge_info:
            if total_charge is not None:
                log.info(f"✅ [LemMoECharge] Dataset contains charge info (found in batch).")
            else:
                log.warning(f"⚠️ [LemMoECharge] Dataset DOES NOT contain charge info. Using random dummy charges.")
            self._logged_charge_info = True

        # Dummy Charge Logic
        if total_charge is None:
            total_charge = torch.randint(
                low=0, high=4, size=(num_batch,),
                device=self.device, dtype=self.dtype
            ) - 1.0

        # 1. Compute Charge Embedding [N, n_atom]
        q_emb = self.elec_emb(node_one_hot, total_charge, num_batch, batch_seg)

        # 2. Inject into Node Features [N, n_atom]
        node_one_hot = node_one_hot + q_emb

        data[_keys.NODE_ATTRS_KEY] = node_one_hot

        edge_one_hot = self.edge_one_hot(data)
        atom_type = data[_keys.ATOM_TYPE_KEY].flatten()
        bond_type = data[_keys.EDGE_TYPE_KEY].flatten()
        batch = data[_keys.BATCH_KEY]

        global_feat = scatter_mean(node_one_hot, batch, dim=0)
        coeffs = self.router(global_feat)

        num_nodes_total = node_one_hot.shape[0]
        latents, node_features, edge_features, cutoff_coeffs, active_edges = self.init_layer(
            edge_index, atom_type, bond_type, edge_sh, edge_length, edge_one_hot
        )

        n_active_nodes = node_features.shape[0]
        if n_active_nodes < num_nodes_total:
            safe_node_one_hot = node_one_hot[:n_active_nodes]
        else:
            safe_node_one_hot = node_one_hot

        edge_one_hot = edge_one_hot[active_edges]

        edge_batch = batch[edge_index[0][active_edges]]
        num_systems = batch.max().item() + 1 if batch.numel() > 0 else 1
        edge_sizes = torch.bincount(edge_batch, minlength=num_systems)

        mole_globals = MOLEGlobals(coefficients=coeffs, sizes=edge_sizes)

        data[_keys.EDGE_OVERLAP_KEY] = latents
        wigner_D_all = None
        for idx, layer in enumerate(self.layers):
            latents, node_features, edge_features, wigner_D_all = \
                layer(
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