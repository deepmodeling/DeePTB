import os
import time
import torch
from typing import Optional, List, Union, Dict
from concurrent.futures import ThreadPoolExecutor

from torch import nn
from torch_runstats.scatter import scatter
from e3nn import o3
from e3nn.nn import Gate
from e3nn.o3 import (
    Linear,
    SphericalHarmonics,
    FullyConnectedTensorProduct,
    TensorProduct,
    xyz_to_angles,
)

# Optional import for openequivariance
try:
    import openequivariance as oeq
except ImportError:
    oeq = None
    print("Warning: openequivariance not found. OEQ modules will fail.")

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


# ==============================================================================
# 1. Base Classes & Helpers
# ==============================================================================

def get_l0_indices(irreps: o3.Irreps):
    indices = []
    offset = 0
    for mul, ir in irreps:
        dim = mul * ir.dim
        if ir.l == 0:
            indices.extend(range(offset, offset + dim))
        offset += dim
    return torch.tensor(indices, dtype=torch.long)


def create_gate(irreps_out: o3.Irreps) -> Gate:
    irreps_scalar = o3.Irreps([(mul, ir) for mul, ir in irreps_out if ir.l == 0]).simplify()
    irreps_gated = o3.Irreps([(mul, ir) for mul, ir in irreps_out if ir.l > 0]).simplify()
    irreps_gates = o3.Irreps([(mul, (0, 1)) for mul, _ in irreps_gated]).simplify()

    act = {1: torch.nn.functional.silu, -1: torch.tanh}
    act_gates = {1: torch.sigmoid, -1: torch.tanh}

    return Gate(
        irreps_scalar,
        [act[ir.p] for _, ir in irreps_scalar],
        irreps_gates,
        [act_gates[ir.p] for _, ir in irreps_gates],
        irreps_gated,
    )


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
            polynomial_cutoff_p: float = 6,
            cutoff_type: str = "polynomial",
            edge_one_hot_dim: int = 128,
            device: Union[str, torch.device] = torch.device("cpu"),
            dtype: Union[str, torch.dtype] = torch.float32,
            prune_edges_by_cutoff: bool = True,
            ln_flag: bool = True,
    ):
        super(InitLayer, self).__init__()
        self.prune_edges_by_cutoff = prune_edges_by_cutoff
        self.num_types = num_types
        self.device = device
        self.dtype = dtype

        # Handle r_max
        if isinstance(r_max, (float, int)):
            self.r_max = torch.tensor(r_max, device=device, dtype=dtype)
            self.r_max_dict = None
        elif isinstance(r_max, dict):
            c_set = set(list(r_max.values()))
            self.r_max = torch.tensor(max(list(r_max.values())), device=device, dtype=dtype)
            if len(r_max) == 1 or len(c_set) == 1:
                self.r_max_dict = None
            else:
                self.r_max_dict = {k: torch.tensor(v, device=device, dtype=dtype) for k, v in r_max.items()}
        else:
            raise TypeError("r_max should be either float, int or dict")

        self.idp = idp
        self.r_start_cos_ratio = r_start_cos_ratio
        self.polynomial_cutoff_p = polynomial_cutoff_p
        self.cutoff_type = cutoff_type

        scalar_irrep = o3.Irrep("0e")
        self.irreps_out = o3.Irreps([(env_embed_multiplicity, ir) for _, ir in irreps_sh])

        assert all(mul == 1 for mul, _ in irreps_sh)
        assert irreps_sh[0].ir == scalar_irrep, "env_embed_irreps must start with scalars"

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

        # Calculate cutoff coefficients
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
                    r_max_val = 0.5 * (self.r_max_dict[iatom] + self.r_max_dict[jatom])

                    if self.cutoff_type == "cosine":
                        c_coeff = cosine_cutoff(
                            edge_length[mask],
                            r_max_val,
                            r_start_cos_ratio=self.r_start_cos_ratio,
                        ).flatten()
                    elif self.cutoff_type == "polynomial":
                        c_coeff = polynomial_cutoff(
                            edge_length[mask],
                            r_max_val,
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


class EAMP(torch.nn.Module):
    """
    Electrostatic-Aware Message-Passing (EAMP) Block.
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
            tp_rotate_in: bool = True,
            tp_rotate_out: bool = True,
            ln_flag: bool = True,
            in_frame_flag: bool = True,
            optimized_in_frame: bool = True,  # New Flag for optimization strategy
            onehot_mode: str = "FullTP",
            self_mix_flag: bool = False,
            self_mix_mode: str = "scalar_channelwise",
            self_mix_iter: int = 1,
            self_mix_type: str = "node",  # "node", "edge", or "all"
    ):
        super(EAMP, self).__init__()

        self.node_irreps_in = o3.Irreps(node_irreps_in)
        self.edge_irreps_in = o3.Irreps(edge_irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.node_one_hot_dim = node_one_hot_dim
        self.dtype = dtype
        self.device = device
        self.res_update = res_update
        self.norm_eps = norm_eps

        self.tp_rotate_in = tp_rotate_in
        self.tp_rotate_out = tp_rotate_out
        self.in_frame_flag = in_frame_flag
        self.optimized_in_frame = optimized_in_frame  # Store flag

        self.use_layer_onehot_tp = use_layer_onehot_tp
        self.onehot_mode = onehot_mode.lower()
        self.self_mix_flag = self_mix_flag
        self.self_mix_mode = self_mix_mode.lower()
        self.self_mix_iter = self_mix_iter
        self.self_mix_type = self_mix_type.lower()

        self.register_buffer(
            "env_sum_normalizations",
            torch.as_tensor(avg_num_neighbors).rsqrt(),
        )

        if ln_flag:
            self.sln_n = SeperableLayerNorm(
                irreps=self.node_irreps_in, eps=norm_eps, affine=True, normalization="component",
                std_balance_degrees=True, dtype=self.dtype, device=self.device,
            )
            self.sln_e = SeperableLayerNorm(
                irreps=self.edge_irreps_in, eps=norm_eps, affine=True, normalization="component",
                std_balance_degrees=True, dtype=self.dtype, device=self.device,
            )
        else:
            self.sln_n = torch.nn.Identity()
            self.sln_e = torch.nn.Identity()

        self._env_weighter = E3ElementLinear(
            irreps_in=self.irreps_out, dtype=dtype, device=device
        )
        assert self.irreps_out[0].ir.l == 0

        self.env_embed_mlps = ScalarMLPFunction(
            mlp_input_dimension=latent_dim,
            mlp_latent_dimensions=[],
            mlp_output_dimension=self._env_weighter.weight_numel,
        )

        # ---- Main Convolution Activation ----
        self.activation = create_gate(self.irreps_out)

        # === 核心逻辑：TP 初始化参数 ===
        # 如果 in_frame 且开启了 optimized_in_frame 优化，强制 TP 不转出（rotate_out=False）。
        # 否则（Legacy模式），遵循 tp_rotate_out 的指示。
        real_tp_rotate_out = tp_rotate_out
        if self.in_frame_flag and self.optimized_in_frame:
            real_tp_rotate_out = False

        self.tp = SO2_Linear(
            irreps_in=self.node_irreps_in + self.edge_irreps_in + self.node_irreps_in,
            irreps_out=self.activation.irreps_in,
            latent_dim=latent_dim,
            radial_emb=radial_emb,
            radial_channels=radial_channels,
            use_interpolation=use_interpolation_tp,
            rotate_in=tp_rotate_in,
            rotate_out=real_tp_rotate_out,
        )

        self.lin_post = Linear(
            self.activation.irreps_out,
            self.irreps_out,
            shared_weights=True, internal_weights=True, biases=True,
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

        # ---- One-Hot Update ----
        if self.use_layer_onehot_tp:
            self.node_onehot_gate = create_gate(self.irreps_out)

            if self.onehot_mode == "fulltp":
                self.node_onehot_tp = FullyConnectedTensorProduct(
                    irreps_in1=self.irreps_out,
                    irreps_in2=f"{self.node_one_hot_dim}x0e",
                    irreps_out=self.node_onehot_gate.irreps_in,
                )
            elif self.onehot_mode == "elementtp":
                self.node_onehot_tp = FullyConnectedTensorProduct(
                    irreps_in1=self.irreps_out,
                    irreps_in2=f"{self.node_one_hot_dim}x0e",
                    irreps_out=self.node_onehot_gate.irreps_in,
                )
            else:
                raise ValueError(f"Unknown onehot_mode={self.onehot_mode!r}")

            self.node_onehot_linear = Linear(
                self.node_onehot_gate.irreps_out,
                self.irreps_out,
                shared_weights=True, internal_weights=True, biases=True
            )

        # ---- Self-Mix Update (Node or Edge or Both) ----
        if self.self_mix_flag:
            l0_indices = get_l0_indices(self.irreps_out)
            self.register_buffer("l0_indices", l0_indices)

            self.node_mixer = None
            self.edge_mixer = None

            if "node" in self.self_mix_type or "all" in self.self_mix_type:
                self.node_mixer = self._build_mixer_module()

            if "edge" in self.self_mix_type or "all" in self.self_mix_type:
                self.edge_mixer = self._build_mixer_module()

        self.use_identity_res = ((self.node_irreps_in == self.irreps_out) and res_update)
        if not self.use_identity_res and res_update:
            self.linear_res = Linear(
                self.node_irreps_in,
                self.irreps_out,
                shared_weights=True, internal_weights=True, biases=True,
            )

    def _build_mixer_module(self):
        """Builds a single instance of the mixing layers (TPs, Gate, Linear, Norm)"""
        mixer = torch.nn.ModuleDict()

        # 1. Add Normalization Layer
        mixer["norm"] = SeperableLayerNorm(
            irreps=self.irreps_out,
            eps=self.norm_eps,
            affine=True,
            normalization="component",
            std_balance_degrees=True,
            dtype=self.dtype,
            device=self.device
        )

        l0_indices = self.l0_indices
        scalar_dim = len(l0_indices)

        gate = create_gate(self.irreps_out)
        mixer["gate"] = gate

        tps = nn.ModuleList()
        pre_gate_linear = None

        # --- TP Construction ---
        for _ in range(self.self_mix_iter):
            tp_layer = None

            if "scalar" in self.self_mix_mode:
                irreps_in2 = o3.Irreps(f"{scalar_dim}x0e")
                if "full" in self.self_mix_mode:
                    tp_layer = FullyConnectedTensorProduct(
                        irreps_in1=self.irreps_out,
                        irreps_in2=irreps_in2,
                        irreps_out=gate.irreps_in
                    )
                elif "channelwise" in self.self_mix_mode:
                    instructions = [(i, 0, i, "uvu", True) for i, _ in enumerate(self.irreps_out)]
                    tp_layer = TensorProduct(
                        irreps_in1=self.irreps_out,
                        irreps_in2=irreps_in2,
                        irreps_out=self.irreps_out,
                        instructions=instructions
                    )
                    if pre_gate_linear is None:
                        pre_gate_linear = Linear(
                            self.irreps_out, gate.irreps_in,
                            internal_weights=True, shared_weights=True
                        )
                else:
                    raise ValueError(f"Unknown scalar mode: {self.self_mix_mode}")

            elif "full_full" in self.self_mix_mode:
                irreps_in2 = self.irreps_out
                if "uvu" in self.self_mix_mode:
                    instructions = []
                    for i, (_, ir) in enumerate(self.irreps_out):
                        if ir in (ir * ir):
                            instructions.append((i, i, i, "uvu", True))
                    tp_layer = TensorProduct(
                        irreps_in1=self.irreps_out,
                        irreps_in2=irreps_in2,
                        irreps_out=self.irreps_out,
                        instructions=instructions
                    )
                elif "uuw" in self.self_mix_mode:
                    instructions = []
                    for i, (_, ir_in) in enumerate(self.irreps_out):
                        for k, (_, ir_out) in enumerate(self.irreps_out):
                            if ir_out in (ir_in * ir_in):
                                instructions.append((i, i, k, "uuw", True))
                    tp_layer = TensorProduct(
                        irreps_in1=self.irreps_out,
                        irreps_in2=irreps_in2,
                        irreps_out=self.irreps_out,
                        instructions=instructions
                    )
                else:
                    raise ValueError(f"Unknown full_full mode: {self.self_mix_mode}")

                if pre_gate_linear is None:
                    pre_gate_linear = Linear(
                        self.irreps_out, gate.irreps_in,
                        internal_weights=True, shared_weights=True
                    )
            else:
                raise ValueError(f"Unknown self_mix_mode: {self.self_mix_mode}")

            tps.append(tp_layer)

        mixer["tps"] = tps

        if pre_gate_linear is not None:
            mixer["pre_gate_linear"] = pre_gate_linear

        mixer["post_linear"] = Linear(
            gate.irreps_out,
            self.irreps_out,
            internal_weights=True, shared_weights=True
        )
        return mixer

    def _apply_self_mix(self, mixer, current_features):
        """Applies the iterative self-mix logic"""
        # 1. Normalization
        normed_features = mixer["norm"](current_features)

        input2 = None
        if "scalar" in self.self_mix_mode:
            input2 = normed_features[:, self.l0_indices]
        elif "full_full" in self.self_mix_mode:
            input2 = normed_features

        current_tp_feat = normed_features

        # 2. Iterative Tensor Product
        for i in range(self.self_mix_iter):
            tp_layer = mixer["tps"][i]
            current_tp_feat = tp_layer(current_tp_feat, input2)

        # Projection and Gate
        if "pre_gate_linear" in mixer:
            tp_out = mixer["pre_gate_linear"](current_tp_feat)
        else:
            tp_out = current_tp_feat

        gate_out = mixer["gate"](tp_out)
        mix_out = mixer["post_linear"](gate_out)

        return mix_out

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

        norm_node_features = self.sln_n(node_features)
        norm_edge_features = self.sln_e(edge_features)

        # ---- Bond aligned SO2 Layer ----
        if wigner_D_all is None:
            l_max = max(self.node_irreps_in.lmax, self.edge_irreps_in.lmax, self.irreps_out.lmax)
            if l_max > 0:
                angle = xyz_to_angles(edge_vector[active_edges][:, [1, 2, 0]])
                wigner_D_all = batch_wigner_D(
                    l_max, angle[0], angle[1], torch.zeros_like(angle[0]), _Jd,
                )

        center_node = norm_node_features[edge_center[active_edges]]
        neighbor_node = norm_node_features[edge_neighbor[active_edges]]

        if self.in_frame_flag and (not self.tp_rotate_in):
            center_node = rotate_vector(center_node, self.node_irreps_in, wigner_D_all, back=False)
            neighbor_node = rotate_vector(neighbor_node, self.node_irreps_in, wigner_D_all, back=False)

        edge_input = torch.cat([center_node, norm_edge_features, neighbor_node], dim=-1)

        edge_messages, wigner_D_all = self.tp(
            edge_input, edge_vector[active_edges], latents[active_edges], wigner_D_all,
        )

        # ==========================================================================================
        # Branching Logic: Optimized vs Legacy
        # ==========================================================================================

        if self.optimized_in_frame:
            # === OPTIMIZED SCHEME (Local-First Consistency) ===
            # 1. edge_messages guaranteed LOCAL here.
            # 2. Perform all non-linear operations in Local Frame.
            edge_messages = self.activation(edge_messages)
            edge_messages = self.lin_post(edge_messages)

            if self.self_mix_flag and self.edge_mixer is not None:
                edge_mix_out = self._apply_self_mix(self.edge_mixer, edge_messages)
                edge_messages = edge_messages + edge_mix_out

            # 3. Branching
            msg_for_node = edge_messages
            # Branch A: For Node Update (Must be Global, rotate from Local)
            if self.in_frame_flag:
                msg_for_node = rotate_vector(edge_messages, self.irreps_out, wigner_D_all, back=True)

            # Branch B: For Next Layer / Output
            if self.in_frame_flag and self.tp_rotate_out:
                # Last layer: Output must be Global. Reuse rotated msg.
                next_edge_features = msg_for_node
            else:
                # Intermediate: Output remains Local.
                next_edge_features = edge_messages

        else:
            # === LEGACY SCHEME (Mixed Consistency) ===
            # 1. Non-linear ops (State mixed: Global or Local depending on tp_rotate_out)
            edge_messages = self.activation(edge_messages)
            edge_messages = self.lin_post(edge_messages)

            if self.self_mix_flag and self.edge_mixer is not None:
                edge_mix_out = self._apply_self_mix(self.edge_mixer, edge_messages)
                edge_messages = edge_messages + edge_mix_out

            # 2. Prepare Node Msg
            msg_for_node = edge_messages
            if self.in_frame_flag and (not self.tp_rotate_out):
                msg_for_node = rotate_vector(edge_messages, self.irreps_out, wigner_D_all, back=True)

            # 3. Next Layer
            next_edge_features = edge_messages

        # ==========================================================================================
        # Common Node Update Logic
        # ==========================================================================================

        weights = self.env_embed_mlps(latents[active_edges])
        edge_messages_weighted = self._env_weighter(msg_for_node, weights)

        aggregated_node_messages = scatter(edge_messages_weighted, edge_center[active_edges], dim=0)

        if self.env_sum_normalizations.ndim < 1:
            norm_const = self.env_sum_normalizations
        else:
            norm_const = self.env_sum_normalizations[atom_type.flatten()].unsqueeze(-1)

        new_node_features = aggregated_node_messages * norm_const

        # Main Branch Residual Update
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

        # ---- NODE SELF-MIX UPDATE ----
        if self.self_mix_flag and self.node_mixer is not None:
            node_mix_out = self._apply_self_mix(self.node_mixer, node_features)
            node_features = node_features + node_mix_out

        # One-Hot Update
        if self.use_layer_onehot_tp:
            tp_out = self.node_onehot_tp(node_features, node_onehot)
            gate_out = self.node_onehot_gate(tp_out)
            linear_out = self.node_onehot_linear(gate_out)
            node_features = node_features + linear_out

        return node_features, next_edge_features, wigner_D_all


@Embedding.register("emoles")
class EMolES(torch.nn.Module):
    """
    EMOLES (Electrolyte Molecules Electronic Structure) is a geometric learning framework
    designed to achieve electrostatic awareness in polarizable systems via efficient
    high-order bond-aligned representations..
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
            polynomial_cutoff_p: float = 6,
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
            ln_flag: bool = True,
            in_frame_flag: bool = True,
            optimized_in_frame: bool = True, # Flag for optimization
            onehot_mode: str = "FullTP",
            self_mix_flag: bool = False,
            self_mix_mode: str = "scalar_channelwise",
            self_mix_iter: int = 2,
            self_mix_type: str = "node",
            **kwargs,
    ):
        super(EMolES, self).__init__()

        self.prune_log_path = prune_log_path
        if self.prune_log_path and os.path.exists(self.prune_log_path):
            try:
                os.remove(self.prune_log_path)
            except OSError:
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
        self.optimized_in_frame = optimized_in_frame
        self.onehot_mode = onehot_mode
        self.self_mix_flag = self_mix_flag
        self.self_mix_mode = self_mix_mode
        self.self_mix_iter = self_mix_iter
        self.self_mix_type = self_mix_type

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
            polynomial_cutoff_p=polynomial_cutoff_p,
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
                    rotate_in, rotate_out = True, False
                else:
                    rotate_in = False
                    rotate_out = (i == n_layers - 1)
            else:
                rotate_in, rotate_out = True, True

            if i == n_layers - 1:
                irreps_out_layer = orbpair_irreps.sort()[0].simplify()
                use_interpolation_tp = bool(use_interpolation_out)
            else:
                irreps_out_layer = irreps_hidden
                use_interpolation_tp = False

            self.layers.append(
                EAMP(
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
                    optimized_in_frame=optimized_in_frame,
                    onehot_mode=onehot_mode,
                    self_mix_flag=self_mix_flag,
                    self_mix_mode=self_mix_mode,
                    self_mix_iter=self_mix_iter,
                    self_mix_type=self_mix_type,
                )
            )

            current_irreps = irreps_out_layer

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


# ==============================================================================
# 2. OEQ Extensions
# ==============================================================================

def get_feasible_tp(
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        filter_irreps_out: o3.Irreps,
        tp_mode: str = "uvw",
        trainable: bool = True
):
    """Generate irreps_out and instructions for OpenEquivariance TP."""
    assert tp_mode in ["uvw", "uvu", "uvv", "uuw", "uuu", "uvuv"]
    irreps_mid = []
    instructions = []

    for i, (mul_1, ir_in1) in enumerate(irreps_in1):
        for j, (mul_2, ir_in2) in enumerate(irreps_in2):
            if tp_mode in ["uuw", "uuu"] and mul_1 != mul_2:
                continue

            for ir_out in ir_in1 * ir_in2:
                if ir_out in filter_irreps_out:
                    if tp_mode == "uvw":
                        mul_out = filter_irreps_out.count(ir_out)
                    elif tp_mode == "uvu":
                        mul_out = mul_1
                    elif tp_mode == "uvv":
                        mul_out = mul_2
                    elif tp_mode == "uuu":
                        mul_out = mul_1
                    elif tp_mode == "uuw":
                        mul_out = filter_irreps_out.count(ir_out)
                    elif tp_mode == "uvuv":
                        mul_out = mul_1 * mul_2
                    else:
                        raise NotImplementedError(f"Unsupported TP mode: {tp_mode}")

                    found_k = -1
                    for k, (m, ir) in enumerate(irreps_mid):
                        if ir == ir_out and m == mul_out:
                            found_k = k
                            break

                    if found_k == -1:
                        found_k = len(irreps_mid)
                        irreps_mid.append((mul_out, ir_out))

                    instructions.append((i, j, found_k, tp_mode, trainable))

    irreps_mid_obj = o3.Irreps(irreps_mid)

    final_instructions = []
    for ins in instructions:
        i_in1, i_in2, i_out, mode, train = ins
        alpha = 1.0
        final_instructions.append((i_in1, i_in2, i_out, mode, train, alpha))

    return irreps_mid_obj, final_instructions


class OEQTensorProduct(nn.Module):
    def __init__(
            self,
            irreps_in1: o3.Irreps,
            irreps_in2: o3.Irreps,
            irreps_out: o3.Irreps,
            tp_mode: str = "uvw",
            internal_weights: bool = True,
            shared_weights: bool = True
    ):
        super().__init__()
        if oeq is None:
            raise ImportError("OpenEquivariance not installed.")

        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)
        self.internal_weights_flag = internal_weights

        self.irreps_mid, instructions = get_feasible_tp(
            self.irreps_in1, self.irreps_in2, self.irreps_out, tp_mode=tp_mode
        )

        self.problem = oeq.TPProblem(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_mid,
            instructions,
            shared_weights=shared_weights,
            internal_weights=False
        )

        self.tp = oeq.TensorProduct(self.problem, torch_op=True)
        self.weight_numel = self.problem.weight_numel

        if self.internal_weights_flag and self.weight_numel > 0:
            self.weights = nn.Parameter(torch.randn(self.weight_numel))
            with torch.no_grad():
                self.weights.div_(self.weight_numel ** 0.5)
        else:
            self.register_parameter('weights', None)

        if self.irreps_mid != self.irreps_out:
            self.post_linear = o3.Linear(self.irreps_mid, self.irreps_out)
        else:
            self.post_linear = nn.Identity()

    def forward(self, x: torch.Tensor, y: torch.Tensor, weight: Optional[torch.Tensor] = None):
        w = self.weights if self.internal_weights_flag else weight
        if self.weight_numel > 0:
            out = self.tp(x, y, w)
        else:
            out = self.tp(x, y)
        out = self.post_linear(out)
        return out


class EAMPOpenequi(EAMP):
    """
    Inherits from EAMP, replaces E3NN TP with OEQ TP.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.use_layer_onehot_tp:
            tp_mode_map = {"fulltp": "uvw", "elementtp": "uvu"}
            self.node_onehot_tp = OEQTensorProduct(
                irreps_in1=self.irreps_out,
                irreps_in2=o3.Irreps(f"{self.node_one_hot_dim}x0e"),
                irreps_out=self.node_onehot_gate.irreps_in,
                tp_mode=tp_mode_map.get(self.onehot_mode, "uvw")
            )

    def _build_mixer_module(self):
        """Overrides parent method to use OEQ mixers."""
        mixer = torch.nn.ModuleDict()
        mixer["norm"] = SeperableLayerNorm(
            irreps=self.irreps_out,
            eps=self.norm_eps,
            affine=True,
            normalization="component",
            std_balance_degrees=True,
            dtype=self.dtype,
            device=self.device
        )

        l0_indices = self.l0_indices
        scalar_dim = len(l0_indices)

        gate = create_gate(self.irreps_out)
        mixer["gate"] = gate

        tps = nn.ModuleList()
        pre_gate_linear = None

        for _ in range(self.self_mix_iter):
            tp_layer = None

            if "scalar" in self.self_mix_mode:
                irreps_in2 = o3.Irreps(f"{scalar_dim}x0e")
                if "full" in self.self_mix_mode:
                    tp_layer = OEQTensorProduct(
                        irreps_in1=self.irreps_out,
                        irreps_in2=irreps_in2,
                        irreps_out=gate.irreps_in,
                        tp_mode="uvw"
                    )
                elif "channelwise" in self.self_mix_mode:
                    tp_layer = OEQTensorProduct(
                        irreps_in1=self.irreps_out,
                        irreps_in2=irreps_in2,
                        irreps_out=self.irreps_out,
                        tp_mode="uvu"
                    )
                    if pre_gate_linear is None:
                        pre_gate_linear = Linear(
                            self.irreps_out, gate.irreps_in,
                            internal_weights=True, shared_weights=True
                        )
                else:
                    raise ValueError(f"Unknown scalar mode: {self.self_mix_mode}")

            elif "full_full" in self.self_mix_mode:
                irreps_in2 = self.irreps_out
                if "uvu" in self.self_mix_mode:
                    tp_layer = OEQTensorProduct(
                        irreps_in1=self.irreps_out,
                        irreps_in2=irreps_in2,
                        irreps_out=self.irreps_out,
                        tp_mode="uvu"
                    )
                    if pre_gate_linear is None:
                        pre_gate_linear = Linear(
                            self.irreps_out, gate.irreps_in,
                            internal_weights=True, shared_weights=True
                        )
                elif "uuw" in self.self_mix_mode:
                    tp_layer = OEQTensorProduct(
                        irreps_in1=self.irreps_out,
                        irreps_in2=irreps_in2,
                        irreps_out=self.irreps_out,
                        tp_mode="uuw"
                    )
                    if pre_gate_linear is None:
                        pre_gate_linear = Linear(
                            self.irreps_out, gate.irreps_in,
                            internal_weights=True, shared_weights=True
                        )
                else:
                    raise ValueError(f"Unknown full_full mode: {self.self_mix_mode}")
            else:
                raise ValueError(f"Unknown self_mix_mode: {self.self_mix_mode}")

            tps.append(tp_layer)

        mixer["tps"] = tps
        if pre_gate_linear is not None:
            mixer["pre_gate_linear"] = pre_gate_linear

        mixer["post_linear"] = Linear(
            gate.irreps_out,
            self.irreps_out,
            internal_weights=True, shared_weights=True
        )
        return mixer


# ==============================================================================
# Helper Functions for Threading
# ==============================================================================

def _create_layer_worker(args):
    """Worker to instantiate layer in separate thread for parallel compilation."""
    idx, layer_kwargs = args
    t_start = time.time()
    layer = EAMPOpenequi(**layer_kwargs)
    duration = time.time() - t_start
    return idx, layer, duration


def _create_tp_worker(args):
    """Worker to instantiate TensorProduct in separate thread."""
    name, tp_kwargs = args
    t_start = time.time()
    tp = OEQTensorProduct(**tp_kwargs)
    duration = time.time() - t_start
    return name, tp, duration


@Embedding.register("emoles_openequi")
class EMolESOpenequi(EMolES):
    """
    EMolESOpenequi: EMolES with Openequivariance Tensor Product support.
    """
    def __init__(self, **kwargs):
        n_layers = kwargs.get('n_layers', 3)
        irreps_hidden = kwargs.get('irreps_hidden')
        use_interpolation_out = kwargs.get('use_interpolation_out', True)
        edge_one_hot_dim = kwargs.get('edge_one_hot_dim', 128)

        super().__init__(**kwargs)

        if oeq is None:
            raise ImportError("OpenEquivariance is not installed.")

        # Parallel compilation preparation
        self.layers = torch.nn.ModuleList([None] * n_layers)
        irreps_hidden_obj = o3.Irreps(irreps_hidden)
        orbpair_irreps = self.idp.orbpair_irreps.sort()[0].simplify()

        base_layer_kwargs = {
            "latent_dim": kwargs.get('latent_dim', 128),
            "norm_eps": kwargs.get('norm_eps', 1e-8),
            "radial_emb": kwargs.get('tp_radial_emb', False),
            "radial_channels": kwargs.get('tp_radial_channels', [128, 128]),
            "res_update": kwargs.get('res_update', True),
            "use_layer_onehot_tp": kwargs.get('use_layer_onehot_tp', True),
            "res_update_ratios": kwargs.get('res_update_ratios', None),
            "res_update_ratios_learnable": kwargs.get('res_update_ratios_learnable', False),
            "avg_num_neighbors": kwargs.get('avg_num_neighbors', None),
            "dtype": self.dtype,
            "device": self.device,
            "ln_flag": kwargs.get('ln_flag', True),
            "in_frame_flag": kwargs.get('in_frame_flag', True),
            "optimized_in_frame": kwargs.get('optimized_in_frame', True),
            "onehot_mode": kwargs.get('onehot_mode', "FullTP"),
            "self_mix_flag": kwargs.get('self_mix_flag', False),
            "self_mix_mode": kwargs.get('self_mix_mode', "scalar_channelwise"),
            "self_mix_iter": kwargs.get('self_mix_iter', 1),
            "self_mix_type": kwargs.get('self_mix_type', "node"),
        }

        tasks = []
        for i in range(n_layers):
            if i == 0:
                irreps_in_layer = self.init_layer.irreps_out
            else:
                irreps_in_layer = irreps_hidden_obj

            if self.in_frame_flag:
                rotate_in = (i == 0)
                rotate_out = (i == n_layers - 1)
            else:
                rotate_in, rotate_out = True, True

            if i == n_layers - 1:
                irreps_out_layer = orbpair_irreps
                use_interpolation_tp = bool(use_interpolation_out)
            else:
                irreps_out_layer = irreps_hidden_obj
                use_interpolation_tp = False

            current_kwargs = base_layer_kwargs.copy()
            current_kwargs.update({
                "node_irreps_in": irreps_in_layer,
                "edge_irreps_in": irreps_in_layer,
                "irreps_out": irreps_out_layer,
                "tp_rotate_in": rotate_in,
                "tp_rotate_out": rotate_out,
                "use_interpolation_tp": use_interpolation_tp,
                "node_one_hot_dim": self.n_atom,
            })
            tasks.append((i, current_kwargs))

        print(f"Starting parallel compilation for {n_layers} layers...")
        t_start_all = time.time()

        with ThreadPoolExecutor(max_workers=8) as executor:
            layer_futures = [executor.submit(_create_layer_worker, task) for task in tasks]

            tp_futures = []
            if self.use_out_onehot_tp:
                tp1_kwargs = {
                    "irreps_in1": self.node_irreps_out,
                    "irreps_in2": o3.Irreps(f"{self.n_atom}x0e"),
                    "irreps_out": self.idp.orbpair_irreps,
                    "tp_mode": "uvw"
                }
                tp2_kwargs = {
                    "irreps_in1": self.edge_irreps_out,
                    "irreps_in2": o3.Irreps(f"{edge_one_hot_dim}x0e"),
                    "irreps_out": self.idp.orbpair_irreps,
                    "tp_mode": "uvw"
                }
                tp_futures.append(executor.submit(_create_tp_worker, ("out_node_ele_tp", tp1_kwargs)))
                tp_futures.append(executor.submit(_create_tp_worker, ("out_edge_ele_tp", tp2_kwargs)))

            for future in layer_futures:
                idx, layer, duration = future.result()
                self.layers[idx] = layer

            for future in tp_futures:
                name, tp_module, duration = future.result()
                setattr(self, name, tp_module)

        print(f"Compilation finished in {time.time() - t_start_all:.2f}s")