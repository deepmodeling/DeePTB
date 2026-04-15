from __future__ import annotations

from typing import Any

import torch
from torch_scatter import scatter_mean

from dptb.data import AtomicDataDict, _keys
from dptb.data.AtomicDataDict import with_batch, with_edge_vectors
from dptb.nn.embedding.emb import Embedding
from dptb.nn.tensor_product_moe_v3 import MOLEGlobals

from .lem_moe_v3 import LemMoEV3
from .lem_moe_v3_h0_helpers import H0InitLayer


@Embedding.register("lem_moe_v3_h0")
class LemMoEV3H0(LemMoEV3):
    def __init__(
        self,
        use_h0_init: bool = True,
        h0_node_key: str = _keys.NODE_H0_KEY,
        h0_edge_key: str = _keys.EDGE_H0_KEY,
        h0_node_mode: str = "direct",
        fallback_to_hamiltonian: bool = True,
        h0_fallback_to_hamiltonian: Any = None,
        fallback_node_key: str = _keys.NODE_FEATURES_KEY,
        fallback_edge_key: str = _keys.EDGE_FEATURES_KEY,
        h0_merge_mode: str = "replace",
        h0_self_edge_tol: float = 1e-8,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.use_h0_init = use_h0_init
        if h0_fallback_to_hamiltonian is not None:
            fallback_to_hamiltonian = bool(h0_fallback_to_hamiltonian)

        if self.use_h0_init:
            self.init_layer = H0InitLayer(
                base_init=self.init_layer,
                h0_node_key=h0_node_key,
                h0_edge_key=h0_edge_key,
                h0_node_mode=h0_node_mode,
                fallback_to_hamiltonian=fallback_to_hamiltonian,
                fallback_node_key=fallback_node_key,
                fallback_edge_key=fallback_edge_key,
                merge_mode=h0_merge_mode,
                self_edge_tol=h0_self_edge_tol,
                dtype=self.dtype,
                device=self.device,
            )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if not self.use_h0_init:
            return super().forward(data)

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
        batch = data[_keys.BATCH_KEY]

        global_feat = scatter_mean(node_one_hot, batch, dim=0)
        coeffs, monitor_val, expert_load_cv = self.router(global_feat)
        data["mean_max_prob"] = monitor_val
        data["expert_load_cv"] = expert_load_cv

        num_nodes_total = node_one_hot.shape[0]
        latents, node_features, edge_features, cutoff_coeffs, active_edges = self.init_layer(
            data,
            edge_index,
            atom_type,
            bond_type,
            edge_sh,
            edge_length,
            edge_one_hot,
        )

        if node_features.shape[0] < num_nodes_total:
            safe_node_one_hot = node_one_hot[: node_features.shape[0]]
        else:
            safe_node_one_hot = node_one_hot

        edge_one_hot = edge_one_hot[active_edges]
        edge_batch = batch[edge_index[0][active_edges]]
        num_systems = batch.max().item() + 1
        edge_sizes = torch.bincount(edge_batch, minlength=num_systems)
        mole_globals = MOLEGlobals(coefficients=coeffs, sizes=edge_sizes)

        data[_keys.EDGE_OVERLAP_KEY] = latents
        wigner_D_all = None
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
                mole_globals,
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
        data[_keys.EDGE_FEATURES_KEY] = torch.zeros(
            edge_index.shape[1],
            self.idp.orbpair_irreps.dim,
            dtype=self.dtype,
            device=self.device,
        )
        data[_keys.EDGE_FEATURES_KEY] = torch.index_copy(
            data[_keys.EDGE_FEATURES_KEY],
            0,
            active_edges,
            out_edge_features,
        )
        return data
