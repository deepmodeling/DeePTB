from typing import Optional, List, Union, Dict
import math
import os
import logging

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
# Removed cutoff imports
from dptb.nn.radial_basis import BesselBasis
from dptb.nn.rescale import E3ElementLinear
from dptb.nn.norm import SeperableLayerNorm
from dptb.nn.base import ScalarMLPFunction
from dptb.nn.type_encode.one_hot import OneHotAtomEncoding, OneHotEdgeEmbedding

from dptb.nn.tensor_product import (
    SO2_Linear,
    batch_wigner_D,
    _Jd,
    rotate_vector,
)

# 初始化 logger
logger = logging.getLogger(__name__)


def get_l0_indices(irreps: o3.Irreps):
    """获取 irreps 中所有 l=0 (标量) 部分在展平向量中的索引。"""
    indices = []
    offset = 0
    for mul, ir in irreps:
        dim = mul * ir.dim
        if ir.l == 0:
            indices.extend(range(offset, offset + dim))
        offset += dim
    return torch.tensor(indices, dtype=torch.long)


# =============================================================================
# Helper: MP Active Edge Mask
# =============================================================================
def get_mp_edge_mask(
        edge_length: torch.Tensor,
        bond_type: torch.Tensor,
        idp: OrbitalMapper,
        mp_cutoff: Union[float, Dict[str, float], None],
        device: torch.device
) -> torch.Tensor:
    """
    根据 mp_cutoff 判断哪些边属于 Active Message Passing 边。
    """
    if mp_cutoff is None:
        return torch.ones(edge_length.shape[0], dtype=torch.bool, device=device)

    if isinstance(mp_cutoff, (float, int)):
        return edge_length < mp_cutoff

    if isinstance(mp_cutoff, dict):
        mask = torch.zeros(edge_length.shape[0], dtype=torch.bool, device=device)
        for bond_str, type_idx in idp.bond_to_type.items():
            atoms = bond_str.split("-")
            if len(atoms) == 2:
                r_i = mp_cutoff.get(atoms[0])
                r_j = mp_cutoff.get(atoms[1])

                if r_i is not None and r_j is not None:
                    cutoff_val = 0.5 * (r_i + r_j)
                    type_mask = (bond_type == type_idx)
                    mask |= (type_mask & (edge_length < cutoff_val))
        return mask

    raise TypeError(f"mp_cutoff type {type(mp_cutoff)} not supported.")


class InitLayer(torch.nn.Module):
    def __init__(
            self,
            idp,
            num_types: int,
            n_radial_basis: int,
            r_max: float,
            avg_num_neighbors: Optional[float] = None,
            irreps_sh: o3.Irreps = None,
            env_embed_multiplicity: int = 32,
            two_body_latent_channels: list = [128, 128],
            latent_dim: int = 128,
            norm_eps: float = 1e-8,
            edge_one_hot_dim: int = 128,
            device: Union[str, torch.device] = torch.device("cpu"),
            dtype: Union[str, torch.dtype] = torch.float32,
            ln_flag: bool = True,
    ):
        super(InitLayer, self).__init__()

        SCALAR = o3.Irrep("0e")
        self.num_types = num_types
        self.device = device
        self.dtype = dtype

        self.bessel = BesselBasis(r_max=r_max, num_basis=n_radial_basis, trainable=True)

        self.irreps_out = o3.Irreps([(env_embed_multiplicity, ir) for _, ir in irreps_sh])
        assert all(mul == 1 for mul, _ in irreps_sh)
        assert irreps_sh[0].ir == SCALAR

        self.register_buffer(
            "env_sum_normalizations",
            torch.as_tensor(avg_num_neighbors).rsqrt(),
        )

        self.two_body_latent = ScalarMLPFunction(
            mlp_input_dimension=(edge_one_hot_dim + n_radial_basis),
            mlp_output_dimension=latent_dim,
            mlp_latent_dimensions=two_body_latent_channels,
            mlp_nonlinearity="silu",
            mlp_initialization="uniform",
        )

        self._env_weighter = Linear(
            irreps_in=irreps_sh,
            irreps_out=self.irreps_out,
            internal_weights=False,
            shared_weights=False,
            path_normalization="element",
        )

        self.env_embed_mlp = ScalarMLPFunction(
            mlp_input_dimension=self.two_body_latent.out_features,
            mlp_output_dimension=self._env_weighter.weight_numel,
            mlp_latent_dimensions=[],
            mlp_nonlinearity=None,
            mlp_initialization="uniform",
        )

        if ln_flag:
            self.sln_n = SeperableLayerNorm(
                irreps=self.irreps_out,
                eps=norm_eps,
                affine=True,
                normalization="component",
                std_balance_degrees=True,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.sln_n = torch.nn.Identity()

    def forward(self, edge_index, atom_type, edge_sh, edge_length, edge_one_hot):
        edge_center = edge_index[0]
        edge_invariants = self.bessel(edge_length)

        raw_latents = self.two_body_latent(
            torch.cat([edge_one_hot, edge_invariants], dim=-1)
        )

        weights_e = self.env_embed_mlp(raw_latents)
        edge_features = self._env_weighter(edge_sh, weights_e)

        node_features = scatter(edge_features, edge_center, dim=0)
        norm_const = self.env_sum_normalizations
        if norm_const.ndim >= 1:
            norm_const = norm_const[atom_type.flatten()].unsqueeze(-1)
        node_features = self.sln_n(node_features * norm_const)

        return raw_latents, node_features, edge_features


class InactiveEdgeLayer(torch.nn.Module):
    def __init__(
            self,
            node_irreps_in: o3.Irreps,
            edge_irreps_in: o3.Irreps,
            irreps_out: o3.Irreps,
            latent_dim: int,
            norm_eps: float = 1e-8,
            radial_emb: bool = False,
            radial_channels: list = [128, 128],
            use_interpolation_tp: bool = False,
            dtype=torch.float32,
            device="cpu",
            ln_flag: bool = True,
            in_frame_flag: bool = True,
            tp_rotate_in: bool = True,
            tp_rotate_out: bool = True,
    ):
        super(InactiveEdgeLayer, self).__init__()
        self.node_irreps_in = o3.Irreps(node_irreps_in)
        self.edge_irreps_in = o3.Irreps(edge_irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.in_frame_flag = in_frame_flag
        self.tp_rotate_in = tp_rotate_in
        self.tp_rotate_out = tp_rotate_out

        if ln_flag:
            self.sln_n = SeperableLayerNorm(
                irreps=self.node_irreps_in, eps=norm_eps, affine=True, normalization="component",
                std_balance_degrees=True, dtype=dtype, device=device
            )
            self.sln_e = SeperableLayerNorm(
                irreps=self.edge_irreps_in, eps=norm_eps, affine=True, normalization="component",
                std_balance_degrees=True, dtype=dtype, device=device
            )
        else:
            self.sln_n = torch.nn.Identity()
            self.sln_e = torch.nn.Identity()

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

        self.tp = SO2_Linear(
            irreps_in=self.node_irreps_in + self.edge_irreps_in + self.node_irreps_in,
            irreps_out=self.activation.irreps_in,
            latent_dim=latent_dim,
            radial_emb=radial_emb,
            radial_channels=radial_channels,
            use_interpolation=use_interpolation_tp,
            rotate_in=tp_rotate_in,
            rotate_out=tp_rotate_out,
        )

        self.lin_post = Linear(
            self.activation.irreps_out,
            self.irreps_out,
            shared_weights=True, internal_weights=True, biases=True
        )

    def forward(
            self,
            latents: torch.Tensor,
            node_features: torch.Tensor,
            edge_features: torch.Tensor,
            edge_index: torch.Tensor,
            edge_vector: torch.Tensor,
            active_edges: torch.Tensor,
            wigner_D_all: Optional[torch.Tensor],
    ):
        edge_center = edge_index[0]
        edge_neighbor = edge_index[1]

        norm_node_features = self.sln_n(node_features)
        norm_edge_features = self.sln_e(edge_features)

        center_node = norm_node_features[edge_center[active_edges]]
        neighbor_node = norm_node_features[edge_neighbor[active_edges]]

        if self.in_frame_flag and (not self.tp_rotate_in):
            center_node = rotate_vector(center_node, self.node_irreps_in, wigner_D_all, back=False)
            neighbor_node = rotate_vector(neighbor_node, self.node_irreps_in, wigner_D_all, back=False)

        edge_input = torch.cat([center_node, norm_edge_features, neighbor_node], dim=-1)

        # latents is Full Tensor, sliced by active_edges
        edge_messages, _ = self.tp(
            edge_input,
            edge_vector[active_edges],
            latents[active_edges],
            wigner_D_all,
        )

        edge_messages = self.activation(edge_messages)
        edge_messages = self.lin_post(edge_messages)

        if self.in_frame_flag and (not self.tp_rotate_out):
            edge_messages = rotate_vector(edge_messages, self.irreps_out, wigner_D_all, back=True)

        return edge_messages


class UpdateNodeInFrame(torch.nn.Module):
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
            tp_rotate_in: bool = True,
            tp_rotate_out: bool = True,
            ln_flag: bool = True,
            in_frame_flag: bool = True,
            onehot_mode: str = "FullTP",
            self_mix_flag: bool = False,
    ):
        super(UpdateNodeInFrame, self).__init__()
        self.node_irreps_in = o3.Irreps(node_irreps_in)
        self.edge_irreps_in = o3.Irreps(edge_irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.dtype = dtype
        self.device = device
        self.res_update = res_update
        self.tp_rotate_in = tp_rotate_in
        self.tp_rotate_out = tp_rotate_out
        self.in_frame_flag = in_frame_flag
        self.use_layer_onehot_tp = use_layer_onehot_tp
        self.self_mix_flag = self_mix_flag
        self.node_one_hot_dim = node_one_hot_dim

        self.register_buffer("env_sum_normalizations", torch.as_tensor(avg_num_neighbors).rsqrt())

        if ln_flag:
            self.sln_n = SeperableLayerNorm(self.node_irreps_in, eps=norm_eps, affine=True, dtype=dtype, device=device)
            self.sln_e = SeperableLayerNorm(self.edge_irreps_in, eps=norm_eps, affine=True, dtype=dtype, device=device)
        else:
            self.sln_n = torch.nn.Identity()
            self.sln_e = torch.nn.Identity()

        self._env_weighter = E3ElementLinear(self.irreps_out, dtype=dtype, device=device)

        self.env_embed_mlps = ScalarMLPFunction(
            mlp_input_dimension=latent_dim,
            mlp_latent_dimensions=[],
            mlp_output_dimension=self._env_weighter.weight_numel
        )

        irreps_scalar = o3.Irreps([(mul, ir) for mul, ir in self.irreps_out if ir.l == 0]).simplify()
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_out if ir.l > 0]).simplify()
        if len(irreps_gated) > 0:
            irreps_gates = o3.Irreps([(mul, (0, 1)) for mul, _ in irreps_gated]).simplify()
            self.activation = Gate(irreps_scalar, [torch.nn.functional.silu] * len(irreps_scalar), irreps_gates,
                                   [torch.sigmoid] * len(irreps_gates), irreps_gated)
        else:
            self.activation = Gate(irreps_scalar, [torch.nn.functional.silu] * len(irreps_scalar), o3.Irreps(""), [],
                                   o3.Irreps(""))

        self.tp = SO2_Linear(
            irreps_in=self.node_irreps_in + self.edge_irreps_in + self.node_irreps_in,
            irreps_out=self.activation.irreps_in,
            latent_dim=latent_dim,
            radial_emb=radial_emb,
            radial_channels=radial_channels,
            use_interpolation=use_interpolation_tp,
            rotate_in=tp_rotate_in,
            rotate_out=tp_rotate_out,
        )
        self.lin_post = Linear(self.activation.irreps_out, self.irreps_out, shared_weights=True, internal_weights=True,
                               biases=True)

        self.register_buffer("_res_update_params", torch.zeros(1))

        if self.use_layer_onehot_tp:
            if onehot_mode.lower() == "fulltp":
                self.node_onehot_tp = FullyConnectedTensorProduct(self.irreps_out, f"{self.node_one_hot_dim}x0e",
                                                                  self.irreps_out)
            else:
                instructions = [(i, 0, i, "uvu", True) for i, _ in enumerate(self.irreps_out)]
                self.node_onehot_tp = TensorProduct(self.irreps_out, f"{self.node_one_hot_dim}x0e", self.irreps_out,
                                                    instructions=instructions)

        if self.self_mix_flag:
            self.l0_indices = get_l0_indices(self.irreps_out)
            if len(self.l0_indices) > 0:
                irreps_scalars = o3.Irreps(f"{len(self.l0_indices)}x0e")
                instructions = [(i, 0, i, "uvu", True) for i, _ in enumerate(self.irreps_out)]
                self.self_mix_tp = TensorProduct(self.irreps_out, irreps_scalars, self.irreps_out,
                                                 instructions=instructions)
                self.self_mix_post_linear = Linear(self.irreps_out, self.irreps_out, internal_weights=True,
                                                   shared_weights=True)
            else:
                self.self_mix_flag = False

        self.use_identity_res = ((self.node_irreps_in == self.irreps_out) and res_update)
        if not self.use_identity_res and res_update:
            self.linear_res = Linear(self.node_irreps_in, self.irreps_out, shared_weights=True, internal_weights=True,
                                     biases=True)

    def forward(self, latents, node_features, edge_features, atom_type, node_onehot, edge_index, edge_vector,
                active_edges, wigner_D_all):
        edge_center, edge_neighbor = edge_index[0], edge_index[1]
        norm_node_features = self.sln_n(node_features)
        norm_edge_features = self.sln_e(edge_features)

        if wigner_D_all is None and (self.node_irreps_in.lmax > 0 or self.edge_irreps_in.lmax > 0):
            angle = xyz_to_angles(edge_vector[active_edges][:, [1, 2, 0]])
            wigner_D_all = batch_wigner_D(max(self.node_irreps_in.lmax, self.edge_irreps_in.lmax), angle[0], angle[1],
                                          torch.zeros_like(angle[0]), _Jd)

        center_node = norm_node_features[edge_center[active_edges]]
        neighbor_node = norm_node_features[edge_neighbor[active_edges]]

        if self.in_frame_flag and not self.tp_rotate_in:
            center_node = rotate_vector(center_node, self.node_irreps_in, wigner_D_all, back=False)
            neighbor_node = rotate_vector(neighbor_node, self.node_irreps_in, wigner_D_all, back=False)

        edge_input = torch.cat([center_node, norm_edge_features, neighbor_node], dim=-1)

        # latents is full tensor, active_edges slices it
        edge_messages, wigner_D_all = self.tp(edge_input, edge_vector[active_edges], latents[active_edges],
                                              wigner_D_all)
        edge_messages = self.lin_post(self.activation(edge_messages))

        msg_for_node = edge_messages
        if self.in_frame_flag and not self.tp_rotate_out:
            msg_for_node = rotate_vector(edge_messages, self.irreps_out, wigner_D_all, back=True)

        # latents slice for weights
        weights = self.env_embed_mlps(latents[active_edges])
        weighted_msg = self._env_weighter(msg_for_node, weights)
        agg_msg = scatter(weighted_msg, edge_center[active_edges], dim=0)

        norm_const = self.env_sum_normalizations if self.env_sum_normalizations.ndim < 1 else \
        self.env_sum_normalizations[atom_type.flatten()].unsqueeze(-1)
        new_node = agg_msg * norm_const

        if self.res_update:
            coeffs = self._res_update_params.sigmoid()
            c_old = torch.rsqrt(coeffs.square() + 1)
            c_new = coeffs * c_old
            node_features = c_old * (
                node_features if self.use_identity_res else self.linear_res(node_features)) + c_new * new_node
        else:
            node_features = new_node

        if self.use_layer_onehot_tp:
            node_features = node_features + self.node_onehot_tp(node_features, node_onehot)

        if self.self_mix_flag:
            scalars = node_features[:, self.l0_indices]
            node_features = node_features + self.self_mix_post_linear(self.self_mix_tp(node_features, scalars))

        return node_features, edge_messages, wigner_D_all


@Embedding.register("lem_cutoff")
class LemCutoff(torch.nn.Module):
    def __init__(
            self,
            basis: Dict[str, Union[str, list]] = None,
            idp: Union[OrbitalMapper, None] = None,
            n_layers: int = 3,
            n_radial_basis: int = 10,
            r_max: Union[float, Dict[str, float]] = 5.0,  # 仅作为 BesselBasis 参数
            irreps_hidden: o3.Irreps = None,
            avg_num_neighbors: Optional[float] = None,
            norm_eps: float = 1e-8,
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
            prune_log_path: Optional[str] = None,
            # ---- Flags ----
            mp_cutoff: Optional[Union[float, Dict[str, float]]] = None,
            ln_flag: bool = True,
            in_frame_flag: bool = True,
            onehot_mode: str = "FullTP",
            self_mix_flag: bool = False,
            **kwargs,
    ):
        super(LemCutoff, self).__init__()

        self.use_out_onehot_tp = use_out_onehot_tp
        self.use_layer_onehot_tp = use_layer_onehot_tp
        self.res_update = res_update
        self.prune_log_path = prune_log_path
        self.ln_flag = ln_flag
        self.in_frame_flag = in_frame_flag
        self.onehot_mode = onehot_mode
        self.self_mix_flag = self_mix_flag
        self.mp_cutoff = mp_cutoff

        if self.prune_log_path and os.path.exists(self.prune_log_path):
            try:
                os.remove(self.prune_log_path)
            except Exception:
                pass

        irreps_hidden = o3.Irreps(irreps_hidden)
        lmax = irreps_hidden.lmax
        if isinstance(dtype, str): dtype = getattr(torch, dtype)
        self.dtype = dtype
        if isinstance(device, str): device = torch.device(device)
        self.device = device

        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb")
            if idp is not None: assert idp == self.idp
        else:
            assert idp is not None
            self.idp = idp
        self.basis = self.idp.basis
        self.idp.get_irreps(no_parity=False)
        self.n_atom = 95 if universal else len(self.basis.keys())

        irreps_sh = o3.Irreps([(1, (i, (-1) ** i)) for i in range(lmax + 1)])
        orbpair_irreps = self.idp.orbpair_irreps.sort()[0].simplify()

        # 计算 Bessel Init 所需 r_max scalar
        if isinstance(r_max, dict):
            r_max_scalar = max(r_max.values())
        else:
            r_max_scalar = r_max

        self.sh = SphericalHarmonics(irreps_sh, sh_normalized, sh_normalization)
        self.onehot = OneHotAtomEncoding(num_types=self.n_atom, set_features=False, idp=self.idp, universal=universal)
        self.edge_one_hot = OneHotEdgeEmbedding(num_types=self.n_atom, idp=self.idp, universal=universal,
                                                d_emb=edge_one_hot_dim)

        self.init_layer = InitLayer(
            idp=self.idp,
            num_types=self.n_atom,
            n_radial_basis=n_radial_basis,
            r_max=r_max_scalar,
            irreps_sh=irreps_sh,
            avg_num_neighbors=avg_num_neighbors,
            env_embed_multiplicity=env_embed_multiplicity,
            two_body_latent_channels=latent_channels,
            latent_dim=latent_dim,
            edge_one_hot_dim=edge_one_hot_dim,
            norm_eps=norm_eps,
            ln_flag=ln_flag,
            device=device,
            dtype=dtype,
        )

        self.layers = torch.nn.ModuleList()
        current_irreps = self.init_layer.irreps_out

        for i in range(n_layers):
            irreps_in_layer = self.init_layer.irreps_out if i == 0 else irreps_hidden

            if self.in_frame_flag:
                rotate_in = (i == 0)
                rotate_out = (i == n_layers - 1)
            else:
                rotate_in = True
                rotate_out = True

            if i == n_layers - 1:
                irreps_out_layer = orbpair_irreps.sort()[0].simplify()
                use_interpolation_tp = bool(use_interpolation_out)
            else:
                irreps_out_layer = irreps_hidden
                use_interpolation_tp = False

            self.layers.append(
                UpdateNodeInFrame(
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
                    self_mix_flag=self_mix_flag,
                )
            )
            current_irreps = irreps_out_layer

        self.node_irreps_out = current_irreps
        self.edge_irreps_out = current_irreps

        if self.mp_cutoff is not None and n_layers > 1:
            target_inactive_irreps = irreps_hidden
            self.inactive_layer = InactiveEdgeLayer(
                node_irreps_in=self.init_layer.irreps_out,
                edge_irreps_in=self.init_layer.irreps_out,
                irreps_out=target_inactive_irreps,
                latent_dim=latent_dim,
                norm_eps=norm_eps,
                radial_emb=tp_radial_emb,
                radial_channels=tp_radial_channels,
                use_interpolation_tp=False,
                dtype=dtype,
                device=device,
                ln_flag=ln_flag,
                in_frame_flag=in_frame_flag,
                tp_rotate_in=True,
                tp_rotate_out=True,
            )
        else:
            self.inactive_layer = None

        if self.use_out_onehot_tp:
            self.out_node_ele_tp = FullyConnectedTensorProduct(self.node_irreps_out, f"{self.n_atom}x0e",
                                                               self.idp.orbpair_irreps)
            self.out_edge_ele_tp = FullyConnectedTensorProduct(self.edge_irreps_out, f"{edge_one_hot_dim}x0e",
                                                               self.idp.orbpair_irreps)

        self.out_node = Linear(self.node_irreps_out, self.idp.orbpair_irreps, shared_weights=True,
                               internal_weights=True, biases=True)
        self.out_edge = Linear(self.edge_irreps_out, self.idp.orbpair_irreps, shared_weights=True,
                               internal_weights=True, biases=True)

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
        # N_total here means all edges passed in
        num_total = edge_index.shape[1]

        edge_vector = data[_keys.EDGE_VECTORS_KEY]
        edge_sh = self.sh(data[_keys.EDGE_VECTORS_KEY][:, [1, 2, 0]])
        edge_length = data[_keys.EDGE_LENGTH_KEY]

        data = self.onehot(data)
        edge_one_hot_all = self.edge_one_hot(data)
        node_one_hot = data[_keys.NODE_ATTRS_KEY]
        atom_type = data[_keys.ATOM_TYPE_KEY].flatten()
        bond_type = data[_keys.EDGE_TYPE_KEY].flatten()

        num_nodes_total = node_one_hot.shape[0]

        # 1. Run InitLayer (Get Raw Full Latents)
        # latents_full: (N_total, dim)
        latents_full, node_features, edge_features = self.init_layer(
            edge_index, atom_type, edge_sh, edge_length, edge_one_hot_all
        )

        init_node_features = node_features.clone()

        # 2. Determine MP Split
        # Default assumption: All edges are MP Active, Inactive branch is skipped.
        all_indices = torch.arange(num_total, device=self.device)
        mp_edges_indices = all_indices
        edge_features_mp = edge_features
        mp_mask = None

        run_inactive_branch = False
        inactive_edges_indices = None
        edge_features_inactive = None

        if self.mp_cutoff is not None and len(self.layers) > 1:
            mp_mask = get_mp_edge_mask(
                edge_length=edge_length,
                bond_type=bond_type,
                idp=self.idp,
                mp_cutoff=self.mp_cutoff,
                device=self.device
            )

            # 只有当确实存在 Inactive Edges (mask 不全为 True) 时，才启动 Inactive 分支
            # 如果 all active，保持上面的默认值
            if not mp_mask.all():
                run_inactive_branch = True
                mp_edges_indices = all_indices[mp_mask]
                inactive_edges_indices = all_indices[~mp_mask]

                edge_features_mp = edge_features[mp_mask]
                edge_features_inactive = edge_features[~mp_mask]

            # 如果 mp_mask.all() is True, run_inactive_branch 保持 False
            # mp_edges_indices = all, edge_features_mp = edge_features

        # --- Logging ---
        num_mp = mp_edges_indices.shape[0]

        if not hasattr(self, "_log_counter"): self._log_counter = 0
        if self._log_counter < 5:
            print(f"[LemCutoff] Edges: Total={num_total} -> MP(Active)={num_mp}")
            self._log_counter += 1

        if self.prune_log_path:
            with open(self.prune_log_path, "a") as f:
                f.write(f"{num_total},{num_mp}\n")
        # ----------------

        # 3. Partial MP Loop (Run on MP edges)
        mp_layers = self.layers[:-1]
        last_layer = self.layers[-1]

        safe_node_one_hot = node_one_hot[:node_features.shape[0]]
        wigner_D_all = None

        for layer in mp_layers:
            # Inputs: latents_full (N_total), mp_edges_indices
            node_features, edge_features_mp, wigner_D_all = layer(
                latents_full,
                node_features,
                edge_features_mp,
                atom_type,
                safe_node_one_hot,
                edge_index,
                edge_vector,
                mp_edges_indices,
                wigner_D_all,
            )

        # 4. Inactive Projection & Merge (Conditional)
        if run_inactive_branch and self.inactive_layer is not None:
            # Only enter here if there ARE inactive edges
            edge_features_inactive_out = self.inactive_layer(
                latents_full,
                init_node_features,
                edge_features_inactive,
                edge_index,
                edge_vector,
                inactive_edges_indices,
                None
            )

            # Merge
            final_dim = edge_features_mp.shape[-1]
            merged_edge_features = torch.zeros(
                num_total, final_dim,
                dtype=edge_features_mp.dtype, device=edge_features_mp.device
            )

            merged_edge_features[mp_mask] = edge_features_mp
            merged_edge_features[~mp_mask] = edge_features_inactive_out

            edge_features_for_last = merged_edge_features
            wigner_D_last = None
        else:
            # All active case
            edge_features_for_last = edge_features_mp
            wigner_D_last = wigner_D_all

        # 5. Final Layer (Run on All edges)
        node_features, edge_features_final, _ = last_layer(
            latents_full,
            node_features,
            edge_features_for_last,
            atom_type,
            safe_node_one_hot,
            edge_index,
            edge_vector,
            all_indices,
            wigner_D_last
        )

        # 6. Output
        if node_features.shape[0] < num_nodes_total:
            pad = torch.zeros(num_nodes_total - node_features.shape[0], node_features.shape[1], device=self.device,
                              dtype=self.dtype)
            node_features = torch.cat([node_features, pad], dim=0)

        out_node_features = self.out_node(node_features)
        out_edge_features = self.out_edge(edge_features_final)

        if self.use_out_onehot_tp:
            out_node_features = out_node_features + self.out_node_ele_tp(node_features, node_one_hot)
            out_edge_features = out_edge_features + self.out_edge_ele_tp(edge_features_final, edge_one_hot_all)

        data[_keys.NODE_FEATURES_KEY] = out_node_features
        data[_keys.EDGE_FEATURES_KEY] = out_edge_features
        data[_keys.EDGE_OVERLAP_KEY] = latents_full

        return data