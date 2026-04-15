from __future__ import annotations

import logging
from typing import Optional, Sequence, Union

import torch
from e3nn.o3 import Linear
from torch_runstats.scatter import scatter

from dptb.data import AtomicDataDict, _keys
from dptb.data.AtomicData import register_fields


# Keep H0 tensors as first-class node/edge fields once the dataset starts
# emitting them. This is a no-op for the current fallback-only workflow.
register_fields(
    node_fields=[_keys.NODE_H0_KEY],
    edge_fields=[_keys.EDGE_H0_KEY],
)


log = logging.getLogger(__name__)


def _prepare_source_tensor(
    tensor: Optional[torch.Tensor],
    expected_dim: int,
    dtype: Union[str, torch.dtype],
    device: Union[str, torch.device],
    key: str,
    label: str,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None

    if torch.is_complex(tensor):
        log.warning("%s source `%s` is complex; only the real part is used.", label, key)
        tensor = tensor.real

    tensor = tensor.to(device=device, dtype=dtype)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[-1] != expected_dim:
        log.warning(
            "Skip %s source `%s` because last dim %s != expected %s.",
            label,
            key,
            tensor.shape[-1],
            expected_dim,
        )
        return None
    return tensor


def _get_feature_source(
    data: AtomicDataDict.Type,
    candidate_keys: Sequence[str],
    expected_dim: int,
    dtype: Union[str, torch.dtype],
    device: Union[str, torch.device],
    label: str,
) -> Optional[torch.Tensor]:
    for key in candidate_keys:
        tensor = _prepare_source_tensor(
            data.get(key, None),
            expected_dim=expected_dim,
            dtype=dtype,
            device=device,
            key=key,
            label=label,
        )
        if tensor is not None:
            return tensor
    return None


class H0InitLayer(torch.nn.Module):
    def __init__(
        self,
        base_init: torch.nn.Module,
        h0_node_key: str = _keys.NODE_H0_KEY,
        h0_edge_key: str = _keys.EDGE_H0_KEY,
        h0_node_mode: str = "direct",
        fallback_to_hamiltonian: bool = True,
        fallback_node_key: str = _keys.NODE_FEATURES_KEY,
        fallback_edge_key: str = _keys.EDGE_FEATURES_KEY,
        merge_mode: str = "replace",
        self_edge_tol: float = 1e-8,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super().__init__()
        if h0_node_mode not in {"direct", "self_edge"}:
            raise ValueError(f"Unsupported h0_node_mode={h0_node_mode!r}")
        if merge_mode not in {"replace", "add"}:
            raise ValueError(f"Unsupported merge_mode={merge_mode!r}")

        self.base_init = base_init
        self.idp = base_init.idp
        self.irreps_out = base_init.irreps_out
        self.h0_irreps = self.idp.orbpair_irreps.sort()[0].simplify()
        self.h0_dim = self.h0_irreps.dim
        self.h0_node_key = h0_node_key
        self.h0_edge_key = h0_edge_key
        self.h0_node_mode = h0_node_mode
        self.fallback_to_hamiltonian = fallback_to_hamiltonian
        self.fallback_node_key = fallback_node_key
        self.fallback_edge_key = fallback_edge_key
        self.merge_mode = merge_mode
        self.self_edge_tol = self_edge_tol
        self.dtype = dtype
        self.device = device

        self.node_projector = Linear(
            irreps_in=self.h0_irreps,
            irreps_out=self.irreps_out,
            shared_weights=True,
            internal_weights=True,
            biases=True,
        )
        self.edge_projector = Linear(
            irreps_in=self.h0_irreps,
            irreps_out=self.irreps_out,
            shared_weights=True,
            internal_weights=True,
            biases=True,
        )

    def _candidate_keys(
        self,
        primary_key: str,
        feature_fallback_key: Optional[str],
        hamiltonian_key: Optional[str],
    ) -> list[str]:
        keys: list[str] = []
        for key in [primary_key]:
            if key and key not in keys:
                keys.append(key)

        if self.fallback_to_hamiltonian:
            for key in [hamiltonian_key, feature_fallback_key]:
                if key and key not in keys:
                    keys.append(key)

        return keys

    def _align_feature_rows(
        self,
        tensor: torch.Tensor,
        n_rows: int,
    ) -> torch.Tensor:
        if tensor.shape[0] == n_rows:
            return tensor
        if tensor.shape[0] > n_rows:
            return tensor[:n_rows]

        pad = torch.zeros(
            n_rows - tensor.shape[0],
            tensor.shape[1],
            device=tensor.device,
            dtype=tensor.dtype,
        )
        return torch.cat([tensor, pad], dim=0)

    def _merge_features(
        self,
        base_features: torch.Tensor,
        h0_features: torch.Tensor,
    ) -> torch.Tensor:
        if self.merge_mode == "replace":
            return h0_features

        n_rows = max(base_features.shape[0], h0_features.shape[0])
        base_features = self._align_feature_rows(base_features, n_rows)
        h0_features = self._align_feature_rows(h0_features, n_rows)
        return base_features + h0_features

    def _mask_node_source(
        self,
        node_source: torch.Tensor,
        atom_type: torch.Tensor,
    ) -> torch.Tensor:
        mask = self.idp.mask_to_nrme[atom_type.flatten()].to(device=node_source.device)
        return node_source * mask.to(dtype=node_source.dtype)

    def _mask_edge_source(
        self,
        edge_source: torch.Tensor,
        bond_type: torch.Tensor,
    ) -> torch.Tensor:
        mask = self.idp.mask_to_erme[bond_type.flatten()].to(device=edge_source.device)
        return edge_source * mask.to(dtype=edge_source.dtype)

    def _node_from_self_edge(
        self,
        edge_features: torch.Tensor,
        data: AtomicDataDict.Type,
        edge_index: torch.Tensor,
        edge_length: torch.Tensor,
        active_edges: torch.Tensor,
        atom_type: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        active_src = edge_index[0][active_edges]
        active_dst = edge_index[1][active_edges]
        active_len = edge_length[active_edges]
        self_mask = torch.logical_and(active_src == active_dst, active_len <= self.self_edge_tol)

        if self_mask.any():
            node_features = scatter(
                edge_features[self_mask],
                active_src[self_mask],
                dim=0,
                dim_size=atom_type.numel(),
            )
            return node_features

        return None

    def forward(
        self,
        data: AtomicDataDict.Type,
        edge_index: torch.Tensor,
        atom_type: torch.Tensor,
        bond_type: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_length: torch.Tensor,
        edge_one_hot: torch.Tensor,
    ):
        latents, base_node_features, base_edge_features, cutoff_coeffs, active_edges = self.base_init(
            edge_index,
            atom_type,
            bond_type,
            edge_sh,
            edge_length,
            edge_one_hot,
        )

        edge_source = _get_feature_source(
            data=data,
            candidate_keys=self._candidate_keys(
                self.h0_edge_key,
                self.fallback_edge_key,
                _keys.EDGE_HAMILTONIAN_KEY,
            ),
            expected_dim=self.h0_dim,
            dtype=self.dtype,
            device=self.device,
            label="edge H0",
        )
        if edge_source is None:
            log.warning(
                "No usable edge H0 source found; falling back to the original InitLayer output."
            )
            return latents, base_node_features, base_edge_features, cutoff_coeffs, active_edges

        edge_source = self._mask_edge_source(edge_source, bond_type)
        edge_features_h0 = self.edge_projector(edge_source[active_edges])
        edge_features = self._merge_features(base_edge_features, edge_features_h0)

        if self.h0_node_mode == "self_edge":
            node_features_h0 = self._node_from_self_edge(
                edge_features=edge_features_h0,
                data=data,
                edge_index=edge_index,
                edge_length=edge_length,
                active_edges=active_edges,
                atom_type=atom_type,
            )
        else:
            node_source = _get_feature_source(
                data=data,
                candidate_keys=self._candidate_keys(
                    self.h0_node_key,
                    self.fallback_node_key,
                    _keys.NODE_HAMILTONIAN_KEY,
                ),
                expected_dim=self.h0_dim,
                dtype=self.dtype,
                device=self.device,
                label="node H0",
            )
            node_features_h0 = None
            if node_source is not None:
                node_source = self._mask_node_source(node_source, atom_type)
                node_features_h0 = self.node_projector(node_source)

        if node_features_h0 is None:
            node_source = _get_feature_source(
                data=data,
                candidate_keys=self._candidate_keys(
                    self.h0_node_key,
                    self.fallback_node_key,
                    _keys.NODE_HAMILTONIAN_KEY,
                ),
                expected_dim=self.h0_dim,
                dtype=self.dtype,
                device=self.device,
                label="node H0",
            )
            if node_source is None:
                log.warning(
                    "No usable node H0 source found; falling back to the original node init."
                )
                node_features = base_node_features
            else:
                node_source = self._mask_node_source(node_source, atom_type)
                node_features_h0 = self.node_projector(node_source)
                node_features = self._merge_features(base_node_features, node_features_h0)
        else:
            node_features = self._merge_features(base_node_features, node_features_h0)

        return latents, node_features, edge_features, cutoff_coeffs, active_edges
