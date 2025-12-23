from typing import Optional, List, Union, Dict, Iterable, Tuple
import math
import os

import torch
import torch.nn as nn
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
    Irreps,
)

from dptb.data import AtomicDataDict, _keys
from dptb.data.AtomicDataDict import with_edge_vectors, with_batch
from dptb.data.transforms import OrbitalMapper
from dptb.nn.embedding.emb import Embedding
from dptb.nn.cutoff import cosine_cutoff, polynomial_cutoff
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


# =============================================================================
# Helper Functions
# =============================================================================

def resolve_actfn(actfn: str):
    """Resolve activation function name to torch module."""
    actfn = actfn.lower()
    if actfn == "silu":
        return nn.SiLU()
    elif actfn == "relu":
        return nn.ReLU()
    elif actfn == "tanh":
        return nn.Tanh()
    elif actfn == "gelu":
        return nn.GELU()
    else:
        return nn.SiLU()


def create_gate(irreps: Irreps, act_scalars=None, act_gates=None) -> Tuple[Gate, Irreps]:
    """
    Helper to create an e3nn Gate and return the Gate module + its input irreps.
    Args:
        irreps: The target output irreps (scalars + gated vectors).
    Returns:
        gate: The Gate module.
        irreps_gate_in: The irreps required as input to this gate (includes extra gate scalars).
    """
    if act_scalars is None:
        act_scalars = {1: torch.nn.functional.silu, -1: torch.tanh}
    if act_gates is None:
        act_gates = {1: torch.sigmoid, -1: torch.tanh}

    irreps_scalar = o3.Irreps([(mul, ir) for mul, ir in irreps if ir.l == 0]).simplify()
    irreps_gated = o3.Irreps([(mul, ir) for mul, ir in irreps if ir.l > 0]).simplify()
    # Create gate scalars corresponding to the gated irreps
    irreps_gates = o3.Irreps([(mul, "0e") for mul, _ in irreps_gated]).simplify()

    gate = Gate(
        irreps_scalar,
        [act_scalars[ir.p] for _, ir in irreps_scalar],
        irreps_gates,
        [act_gates[ir.p] for _, ir in irreps_gates],
        irreps_gated,
    )
    return gate, gate.irreps_in



# =============================================================================
# InitLayer (No Change)
# =============================================================================

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
            r_start_cos_ratio: float = 0.8,
            norm_eps: float = 1e-8,
            PolynomialCutoff_p: float = 6,
            cutoff_type: str = "polynomial",
            edge_one_hot_dim: int = 128,
            device: Union[str, torch.device] = torch.device("cpu"),
            dtype: Union[str, torch.dtype] = torch.float32,
            prune_edges_by_cutoff: bool = True,
            ln_flag: bool = True,
    ):
        super(InitLayer, self).__init__()
        self.prune_edges_by_cutoff = prune_edges_by_cutoff

        SCALAR = o3.Irrep("0e")
        self.num_types = num_types
        if isinstance(r_max, float) or isinstance(r_max, int):
            self.r_max = torch.tensor(r_max, device=device, dtype=dtype)
            self.r_max_dict = None
        elif isinstance(r_max, dict):
            c_set = set(list(r_max.values()))
            self.r_max = torch.tensor(max(list(r_max.values())), device=device, dtype=dtype)
            if len(r_max) == 1 or len(c_set) == 1:
                self.r_max_dict = None
            else:
                self.r_max_dict = {}
                for k, v in r_max.items():
                    self.r_max_dict[k] = torch.tensor(v, device=device, dtype=dtype)
        else:
            raise TypeError("r_max should be either float, int or dict")

        self.idp = idp
        self.r_start_cos_ratio = r_start_cos_ratio
        self.polynomial_cutoff_p = PolynomialCutoff_p
        self.cutoff_type = cutoff_type
        self.device = device
        self.dtype = dtype
        self.irreps_out = o3.Irreps([(env_embed_multiplicity, ir) for _, ir in irreps_sh])

        assert all(mul == 1 for mul, _ in irreps_sh)
        assert irreps_sh[0].ir == SCALAR, "env_embed_irreps must start with scalars"

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

        self.bessel = BesselBasis(r_max=self.r_max, num_basis=n_radial_basis, trainable=True)

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

    def forward(self, edge_index, atom_type, bond_type, edge_sh, edge_length, edge_one_hot):
        edge_center = edge_index[0]
        edge_invariants = self.bessel(edge_length)

        if self.r_max_dict is None:
            if self.cutoff_type == "cosine":
                cutoff_coeffs = cosine_cutoff(
                    edge_length,
                    self.r_max.reshape(-1),
                    r_start_cos_ratio=self.r_start_cos_ratio,
                ).flatten()
            elif self.cutoff_type == "polynomial":
                cutoff_coeffs = polynomial_cutoff(
                    edge_length,
                    self.r_max.reshape(-1),
                    p=self.polynomial_cutoff_p,
                ).flatten()
            else:
                assert False, "Invalid cutoff type"
        else:
            cutoff_coeffs = torch.zeros(edge_index.shape[1], dtype=self.dtype, device=self.device)
            for bond, ty in self.idp.bond_to_type.items():
                mask = bond_type == ty
                index = mask.nonzero().squeeze(-1)
                if mask.any():
                    iatom, jatom = bond.split("-")
                    if self.cutoff_type == "cosine":
                        c_coeff = cosine_cutoff(
                            edge_length[mask],
                            0.5 * (self.r_max_dict[iatom] + self.r_max_dict[jatom]),
                            r_start_cos_ratio=self.r_start_cos_ratio,
                        ).flatten()
                    elif self.cutoff_type == "polynomial":
                        c_coeff = polynomial_cutoff(
                            edge_length[mask],
                            0.5 * (self.r_max_dict[iatom] + self.r_max_dict[jatom]),
                            p=self.polynomial_cutoff_p,
                        ).flatten()
                    else:
                        assert False, "Invalid cutoff type"
                    cutoff_coeffs = torch.index_copy(cutoff_coeffs, 0, index, c_coeff)

        if self.prune_edges_by_cutoff:
            prev_mask = cutoff_coeffs > 0
            active_edges = (cutoff_coeffs > 0).nonzero().squeeze(-1)
        else:
            prev_mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=self.device)
            active_edges = torch.arange(edge_index.shape[1], device=self.device)

        latents = torch.zeros(
            (edge_sh.shape[0], self.two_body_latent.out_features),
            dtype=edge_sh.dtype,
            device=edge_sh.device,
        )
        new_latents = self.two_body_latent(
            torch.cat([edge_one_hot[active_edges], edge_invariants[active_edges]], dim=-1)
        )
        latents = torch.index_copy(
            latents,
            0,
            active_edges,
            cutoff_coeffs[active_edges].unsqueeze(-1) * new_latents,
        )

        weights_e = self.env_embed_mlp(latents[prev_mask])
        edge_features = self._env_weighter(edge_sh[prev_mask], weights_e)

        node_features = scatter(edge_features, edge_center[active_edges], dim=0)

        if self.env_sum_normalizations.ndim < 1:
            norm_const = self.env_sum_normalizations
        else:
            norm_const = self.env_sum_normalizations[atom_type.flatten()].unsqueeze(-1)

        node_features = node_features * norm_const
        node_features = self.sln_n(node_features)

        return latents, node_features, edge_features, cutoff_coeffs, active_edges


# =============================================================================
# UpdateNodeInFrame
# =============================================================================


class UpdateNodeInFrame(torch.nn.Module):
    """
    Keep-in-Frame Update Node Layer.
    Updated:
      1. Implements Latent Update using edge features and edge one-hot.
      2. Mix Strategy uses default e3nn TensorProduct with custom instruction generation.
      3. Implements Gating mechanism for the Mix TensorProduct.
      4. Added SeperableLayerNorm after Mix TensorProduct for stability.
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
            # === global flags ===
            ln_flag: bool = True,
            in_frame_flag: bool = True,
            # === mix mode ===
            onehot_mode: str = "FullTP",
            # === Latent Update Params ===
            edge_one_hot_dim: int = 128,
            latent_channels: list = [128, 128],
    ):
        super(UpdateNodeInFrame, self).__init__()

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
        self.ln_flag = ln_flag

        self.register_buffer(
            "env_sum_normalizations",
            torch.as_tensor(avg_num_neighbors).rsqrt(),
        )

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

        # Main Activation (Gate)
        self.activation, irreps_tp_out = create_gate(self.irreps_out)

        self.tp = SO2_Linear(
            irreps_in=self.node_irreps_in + self.edge_irreps_in + self.node_irreps_in,
            irreps_out=irreps_tp_out,
            latent_dim=latent_dim,
            radial_emb=radial_emb,
            radial_channels=radial_channels,
            use_interpolation=use_interpolation_tp,
            rotate_in=tp_rotate_in,
            rotate_out=tp_rotate_out,
        )

        # ---- Latent Update MLPs (Stateful Update) ----
        self.ln_latents = torch.nn.LayerNorm(latent_dim)

        irreps_scalar_part = self.activation.irreps_scalars
        scalar_dim = irreps_scalar_part.dim

        self.latents_mlp_1 = ScalarMLPFunction(
            mlp_input_dimension=latent_dim + scalar_dim,
            mlp_output_dimension=latent_dim,
            mlp_latent_dimensions=latent_channels,
            mlp_nonlinearity="silu",
            mlp_initialization="uniform",
        )

        self.latents_mlp_2 = ScalarMLPFunction(
            mlp_input_dimension=latent_dim + edge_one_hot_dim,
            mlp_output_dimension=latent_dim,
            mlp_latent_dimensions=latent_channels,
            mlp_nonlinearity="silu",
            mlp_initialization="uniform",
        )
        # -----------------------------------------------

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

        # ---- FEATURE MIXING / ENRICHMENT BLOCK ----
        if self.use_layer_onehot_tp:
            mode = self.onehot_mode

            if mode in ["mix_uuw", "mix_uvw"]:
                # === Unified Bottleneck Architecture with Internal Weights ===

                # 1. Define Bottleneck Irreps
                muls = [mul for mul, _ in self.irreps_out]
                min_mul = min(muls)
                irreps_mix_in = o3.Irreps([(min_mul, ir) for _, ir in self.irreps_out])

                print(f"[{mode}] Configured with Bottleneck Architecture + Internal Weights.")

                # 2. Down-Projection Layers
                self.mix_lin_u = Linear(self.irreps_out, irreps_mix_in)
                self.mix_lin_v = Linear(self.irreps_out, irreps_mix_in)

                # 3. Define Gate for Mix layer
                self.mix_gate, irreps_mix_tp_out = create_gate(irreps_mix_in)

                # 4. Tensor Product Setup
                tp_mode_str = "uuw" if mode == "mix_uuw" else "uvw"

                instructions = []
                for i, (_, ir_in1) in enumerate(irreps_mix_in):
                    for j, (_, ir_in2) in enumerate(irreps_mix_in):
                        for k, (_, ir_out) in enumerate(irreps_mix_tp_out):
                            if ir_out in ir_in1 * ir_in2:
                                instructions.append((i, j, k, tp_mode_str, True))

                self.mix_tp = TensorProduct(
                    irreps_in1=irreps_mix_in,
                    irreps_in2=irreps_mix_in,
                    irreps_out=irreps_mix_tp_out,
                    instructions=instructions,
                    internal_weights=True,
                    shared_weights=True
                )

                # 5. [NEW] Stabilization Norm after TP
                if self.ln_flag:
                    self.sln_mix = SeperableLayerNorm(
                        irreps=irreps_mix_tp_out,  # Matches TP output / Gate input
                        eps=norm_eps,
                        affine=True,
                        normalization="component",
                        std_balance_degrees=True,
                        dtype=dtype,
                        device=device,
                    )
                else:
                    self.sln_mix = torch.nn.Identity()

                # 6. Up-Projection Layer
                self.mix_lin_back = Linear(self.mix_gate.irreps_out, self.irreps_out)

            elif mode == "fulltp":
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
                raise ValueError(
                    f"Unknown onehot_mode={onehot_mode!r}. Supported: 'mix_uuw', 'mix_uvw', 'fulltp', 'elementtp'"
                )

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
            edge_one_hot: torch.Tensor,
            wigner_D_all: Optional[torch.Tensor],
            cutoff_coeffs: Optional[torch.Tensor] = None,
    ):
        edge_center = edge_index[0]
        edge_neighbor = edge_index[1]

        norm_node_features = self.sln_n(node_features)
        norm_edge_features = self.sln_e(edge_features)

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

        center_node = norm_node_features[edge_center[active_edges]]
        neighbor_node = norm_node_features[edge_neighbor[active_edges]]

        if self.in_frame_flag and (not self.tp_rotate_in):
            center_node = rotate_vector(center_node, self.node_irreps_in, wigner_D_all, back=False)
            neighbor_node = rotate_vector(neighbor_node, self.node_irreps_in, wigner_D_all, back=False)

        edge_input = torch.cat([center_node, norm_edge_features, neighbor_node], dim=-1)

        edge_messages_raw, wigner_D_all = self.tp(
            edge_input,
            edge_vector[active_edges],
            latents[active_edges],
            wigner_D_all,
        )

        edge_messages = self.activation(edge_messages_raw)
        edge_messages = self.lin_post(edge_messages)

        # ---- LATENT UPDATE (Stateful) ----
        scalar_dim = self.activation.irreps_scalars.dim
        edge_scalars = edge_messages[:, :self.irreps_out[0].dim]

        new_latents = self.latents_mlp_1(torch.cat(
            [self.ln_latents(latents[active_edges]), edge_scalars], dim=-1
        ))

        new_latents = self.latents_mlp_2(torch.cat(
            [new_latents, edge_one_hot], dim=-1
        ))

        if cutoff_coeffs is not None:
            new_latents = cutoff_coeffs[active_edges].unsqueeze(-1) * new_latents
        # ----------------------------------

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

        # ---- APPLY MIXING ----
        if self.use_layer_onehot_tp:
            if self.onehot_mode in ["mix_uuw", "mix_uvw"]:
                new_node_features = new_node_features.detach()
                # 1. Project to Bottleneck
                x_u = self.mix_lin_u(new_node_features)
                x_v = self.mix_lin_v(new_node_features)

                # 2. Tensor Product
                mix_feat_raw = self.mix_tp(x_u, x_v)

                # 3. [NEW] Stabilization Norm
                mix_feat_raw = self.sln_mix(mix_feat_raw)

                # 4. Gate Activation
                mix_feat = self.mix_gate(mix_feat_raw)

                # 5. Project Back
                mix_feat = self.mix_lin_back(mix_feat)

                # 6. Residual Connection
                new_node_features = new_node_features + mix_feat

            else:
                onehot_tune_node_feat = self.node_onehot_tp(new_node_features, node_onehot)
                new_node_features = new_node_features + onehot_tune_node_feat

        # Residual Update for Nodes & Latents
        update_coefficients = self._res_update_params.sigmoid()
        coefficient_old = torch.rsqrt(update_coefficients.square() + 1)
        coefficient_new = update_coefficients * coefficient_old
        if self.res_update:
            if self.use_identity_res:
                node_features = coefficient_old * node_features + coefficient_new * new_node_features
            else:
                node_features = coefficient_old * self.linear_res(node_features) + coefficient_new * new_node_features

            latents = torch.index_copy(
                latents, 0, active_edges,
                coefficient_new * new_latents + coefficient_old * latents[active_edges]
            )
        else:
            node_features = new_node_features
            latents = torch.index_copy(
                latents, 0, active_edges, new_latents
            )

        return node_features, edge_messages, latents, wigner_D_all

# =============================================================================
# CartTensorMix2
# =============================================================================

class CartTensorMix2(nn.Module):
    def __init__(
            self,
            node_dim: int = 128,
            edge_irreps: Union[str, o3.Irreps, Iterable] = "128x0e + 64x1o + 32x2e",
            hidden_dim: int = 64,
            hidden_irreps: Union[str, o3.Irreps, Iterable] = "16x0e + 16x1o + 16x2e",
            order: int = 2,
            required_symmetry: str = "ij",
            vector_space: Optional[dict] = None,
            output_dim: int = 9,
            actfn: str = "silu",
            reduce_op: Optional[str] = "sum",
    ):
        super().__init__()
        self.irreps_in = edge_irreps if isinstance(edge_irreps, o3.Irreps) else o3.Irreps(edge_irreps)
        self.irreps_hidden = hidden_irreps if isinstance(hidden_irreps, o3.Irreps) else o3.Irreps(hidden_irreps)

        self.trace_out = (output_dim == 1 and order == 2)
        self.indices = required_symmetry.split("=")[0].replace("-", "")
        vec_space = vector_space if vector_space is not None else {i: "1o" for i in self.indices}

        if output_dim == 3 ** (order) or self.trace_out:
            rtp = o3.ReducedTensorProducts(formula=required_symmetry, **vec_space)
        else:
            raise NotImplementedError(
                f"Current CartTensorOut module does not support the required output dim={output_dim}, please check.")

        self.rtp = rtp
        self.irreps_out = self.rtp.irreps_out
        self.activation = resolve_actfn(actfn)
        self.reduce_op = reduce_op

        # 使用 get_feasible_tp 构建指令
        irreps_tp_out, instructions = get_feasible_tp(
            self.irreps_hidden, self.irreps_hidden, self.irreps_out, tp_mode="uuw"
        )

        self.tp = o3.TensorProduct(
            self.irreps_hidden,
            self.irreps_hidden,
            irreps_tp_out,  # 使用计算出的中间 Irreps
            instructions=instructions,
            internal_weights=False,
            shared_weights=False,
        )

        self.lin_out_U = o3.Linear(self.irreps_in, self.irreps_hidden, biases=False)
        self.lin_out_V = o3.Linear(self.irreps_in, self.irreps_hidden, biases=False)

        self.gate_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim, bias=True),
            self.activation,
            nn.Linear(hidden_dim, self.tp.weight_numel, bias=True)
        )

        if irreps_tp_out.sort()[0].simplify() == self.irreps_out.sort()[0].simplify():
            self.post_trans = False
        else:
            self.post_trans = True
            self.lin_out_post = o3.Linear(irreps_tp_out, self.irreps_out, biases=False)

        self.register_buffer("cartesian_index", torch.LongTensor([2, 0, 1]))

    def forward(
            self,
            batch: torch.Tensor,
            x_scalar: torch.Tensor,
            x_spherical: torch.Tensor,
    ) -> torch.Tensor:
        x_sph_U = self.lin_out_U(x_spherical)
        x_sph_V = self.lin_out_V(x_spherical)

        tp_weights = self.gate_mlp(x_scalar)
        x_tp = self.tp(x_sph_U, x_sph_V, tp_weights)

        atom_out = self.lin_out_post(x_tp) if self.post_trans else x_tp

        if self.reduce_op is not None:
            res_sph = scatter(atom_out, batch, dim=0, reduce=self.reduce_op)
        else:
            res_sph = atom_out

        Q = self.rtp.change_of_basis
        res_cart = res_sph @ Q.flatten(-len(self.indices))
        shape = list(res_sph.shape[:-1]) + list(Q.shape[1:])
        res_cart = res_cart.view(shape)

        if self.trace_out:
            res = torch.diagonal(res_cart, dim1=-2, dim2=-1).sum(dim=-1) / 3
            return res.unsqueeze(-1)
        else:
            for i_dim in range(1, len(self.indices) + 1):
                res_cart = torch.index_select(res_cart, dim=-i_dim, index=self.cartesian_index)
            return res_cart


# =============================================================================
# LemInFrame Main Model
# =============================================================================

@Embedding.register("lem_in_frame")
class LemInFrame(torch.nn.Module):
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
            # ---- flags ----
            ln_flag: bool = True,
            in_frame_flag: bool = True,
            onehot_mode: str = "FullTP",
            # ---- output flags ----
            output_mode: str = "linear",
            tensor_order: int = 2,
            tensor_symmetry: str = "ij",
            tensor_hidden_irreps: str = "32x0e + 32x1o + 32x2e",
            tensor_output_dim: int = 9,
            **kwargs,
    ):
        super(LemInFrame, self).__init__()

        self.prune_log_path = prune_log_path
        if self.prune_log_path and os.path.exists(self.prune_log_path):
            try:
                os.remove(self.prune_log_path)
            except Exception:
                pass

        irreps_hidden = o3.Irreps(irreps_hidden)
        lmax = irreps_hidden.lmax

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.ln_flag = ln_flag
        self.in_frame_flag = in_frame_flag
        self.onehot_mode = onehot_mode
        self.output_mode = output_mode.lower()
        print(f'Layer Update Mode (onehot_mode): {onehot_mode}')
        print(f'Model Output Mode (output_mode): {self.output_mode}')

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
                    # New params
                    edge_one_hot_dim=edge_one_hot_dim,
                    latent_channels=latent_channels,
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

        if self.output_mode == "mix2":
            scalar_dim_in = sum(mul for mul, ir in self.node_irreps_out if ir.l == 0)
            self.cart_tensor_mix = CartTensorMix2(
                node_dim=scalar_dim_in,
                edge_irreps=self.node_irreps_out,
                hidden_dim=64,
                hidden_irreps=tensor_hidden_irreps,
                order=tensor_order,
                required_symmetry=tensor_symmetry,
                output_dim=tensor_output_dim,
            )
        else:
            self.cart_tensor_mix = None

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

        n_active_nodes = node_features.shape[0]
        safe_node_one_hot = (
            node_one_hot[:n_active_nodes] if n_active_nodes < num_nodes_total else node_one_hot
        )
        edge_one_hot = edge_one_hot_all[active_edges]
        data[_keys.EDGE_OVERLAP_KEY] = latents

        wigner_D_all = None
        for layer in self.layers:
            # Updated to receive new latents and pass edge_one_hot
            node_features, edge_features, latents, wigner_D_all = layer(
                latents,
                node_features,
                edge_features,
                atom_type,
                safe_node_one_hot,
                edge_index,
                edge_vector,
                active_edges,
                edge_one_hot,  # Pass edge_one_hot
                wigner_D_all,
                cutoff_coeffs  # Pass cutoff for latent masking
            )

        if node_features.shape[0] < num_nodes_total:
            pad = torch.zeros(
                num_nodes_total - node_features.shape[0],
                node_features.shape[1],
                device=node_features.device,
                dtype=node_features.dtype,
            )
            node_features = torch.cat([node_features, pad], dim=0)

        if self.output_mode == "linear":
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

        elif self.output_mode == "mix2":
            scalar_indices = []
            start = 0
            for mul, ir in self.node_irreps_out:
                if ir.l == 0:
                    scalar_indices.extend(range(start, start + mul))
                start += mul * ir.dim

            x_scalar = node_features[:, scalar_indices]
            x_spherical = node_features
            batch = data[_keys.BATCH_KEY]

            tensor_out = self.cart_tensor_mix(batch, x_scalar, x_spherical)
            data["cart_tensor_output"] = tensor_out
            data[_keys.NODE_FEATURES_KEY] = node_features

        return data