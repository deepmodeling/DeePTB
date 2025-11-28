import torch
import math
from typing import Optional, List, Union, Dict, Tuple
from functools import partial
from torch import nn, LongTensor, Tensor
from torch.nn import functional as F
from torch_runstats.scatter import scatter
from torch_scatter import scatter_mean
from e3nn import o3
from e3nn.nn import Gate
from e3nn.o3 import Linear, SphericalHarmonics, FullyConnectedTensorProduct, TensorProduct
import e3nn

from dptb.data import AtomicDataDict, _keys
from dptb.nn.embedding.emb import Embedding
from ..radial_basis import BesselBasis
from ..base import ScalarMLPFunction
from dptb.data.transforms import OrbitalMapper
from ..type_encode.one_hot import OneHotAtomEncoding, OneHotEdgeEmbedding
from dptb.nn.norm import SeperableLayerNorm
from dptb.data.AtomicDataDict import with_edge_vectors, with_batch
from dptb.nn.cutoff import cosine_cutoff, polynomial_cutoff
from dptb.nn.rescale import E3ElementLinear
from dptb.nn.tensor_product import SO2_Linear

# Reuse original UpdateEdge
# from .lem_so2 import UpdateEdge


# ==========================================
# Helper Functions: Safety Wrappers
# ==========================================

def safe_norm(x, dim=-1, keepdim=False, eps=1e-8):
    """Calculates norm with protection against zero gradients."""
    return torch.norm(x, dim=dim, keepdim=keepdim).clamp(min=eps)


# ==========================================
# Helper Classes (Copied from previous context)
# ==========================================

class BaseMLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            activation: nn.Module = nn.SiLU(),
            norm: Optional[nn.Module] = None,
            residual: bool = False,
            last_act: bool = False,
    ) -> None:
        super(BaseMLP, self).__init__()
        self.residual = residual
        if residual:
            assert output_dim == input_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Identity() if norm is None else norm(hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim),
            nn.Identity() if norm is None else norm(output_dim),
            activation if last_act else nn.Identity()
        )

    def forward(self, x):
        out = self.mlp(x)
        if self.residual:
            out = x + out
        # Protection against exploding activations during inference
        if not self.training and torch.isnan(out).any():
            out = torch.nan_to_num(out)
        return out


class NodeColor(nn.Module):
    def __init__(self, hidden_dim, color_type='center_radius', max_ell=6, activation=nn.SiLU()):
        super().__init__()
        MLP = partial(BaseMLP, hidden_dim=hidden_dim, activation=activation)
        if color_type == 'mp':
            self.mlp_msg = MLP(input_dim=hidden_dim * 2 + 1, output_dim=hidden_dim)
            self.mlp_node_feat = MLP(input_dim=hidden_dim, output_dim=hidden_dim)
        elif color_type == 'center_radius':
            self.mlp_node_feat = MLP(input_dim=1, output_dim=hidden_dim)
        elif color_type == 'tp':
            sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
            self.spherical_harmonics = o3.SphericalHarmonics(
                sh_irreps, normalize=True, normalization="norm"
            )
            self.tp = o3.FullyConnectedTensorProduct(sh_irreps, sh_irreps, f'{max_ell + 1}x0e', shared_weights=False)

            self.mlp_sh_coff = MLP(input_dim=hidden_dim, output_dim=self.tp.weight_numel)
            self.mlp_node_feat = MLP(input_dim=max_ell + 1, output_dim=hidden_dim)
        self.color_type = color_type
        print(f'self.color_type: {self.color_type}')

    def forward(self, node_feat, node_pos, batch, edge_index=None):
        # Protection: handle empty batch
        if node_pos.shape[0] == 0:
            return torch.zeros_like(node_feat)

        center = scatter_mean(node_pos, batch, dim=0)
        # Protection: if scatter returns NaNs (unlikely but possible)
        center = torch.nan_to_num(center)
        pos = node_pos - center[batch]

        if self.color_type == 'mp':
            assert edge_index is not None
            row, col = edge_index
            diff = node_pos[row] - node_pos[col]
            # Use safe norm
            dist = safe_norm(diff, dim=1, keepdim=True)
            msg = torch.cat([node_feat[row], node_feat[col], dist], dim=1)
            msg = self.mlp_msg(msg)
            scalar = scatter_mean(msg, row, dim=0, dim_size=node_feat.size(0))
        elif self.color_type == 'center_radius':
            scalar = safe_norm(pos, dim=1, keepdim=True)
        elif self.color_type == 'tp':
            sh = self.spherical_harmonics(pos)
            global_sh = scatter_mean(sh, batch, dim=0)
            scalar = self.tp(sh, global_sh[batch], self.mlp_sh_coff(node_feat))
        else:
            raise NotImplementedError

        return self.mlp_node_feat(scalar)


class VirtualNode(nn.Module):
    def __init__(self, vn_channel=4, hidden_dim=64, activation=nn.SiLU()):
        super().__init__()
        self.vn_channel = vn_channel
        self.get_vn_pos = o3.FullyConnectedTensorProduct(
            '1x1o', '1x0e', f'{vn_channel}x1o', shared_weights=False
        )

        MLP = partial(BaseMLP, hidden_dim=hidden_dim, activation=activation)
        self.mlp_vec_coff = MLP(input_dim=hidden_dim, output_dim=self.get_vn_pos.weight_numel)

    def forward(self, node_feat, node_pos, batch):
        if node_pos.shape[0] == 0:
            return torch.zeros((0, self.vn_channel, 3), device=node_pos.device)

        center = scatter_mean(node_pos, batch, dim=0)
        pos = node_pos - center[batch]
        one = torch.ones([pos.size(0), 1], device=pos.device)

        vn_pos_disp = self.get_vn_pos(pos, one, self.mlp_vec_coff(node_feat))
        vn_pos_global = scatter_mean(vn_pos_disp, batch, dim=0)

        vn_pos_global = vn_pos_global.view(-1, self.vn_channel, 3)
        # Protection: Safe Division
        norm = safe_norm(vn_pos_global, dim=2, keepdim=True)
        vn_pos_global = vn_pos_global / norm

        vn_pos_global = vn_pos_global.view(-1, self.vn_channel * 3)

        vn_pos = vn_pos_global + center.repeat(1, self.vn_channel)

        return vn_pos


class NodeFeatByVN(nn.Module):
    def __init__(self, vn_channel=4, hidden_dim=64, activation=nn.SiLU()):
        super().__init__()
        self.vn_channel = vn_channel

        MLP = partial(BaseMLP, hidden_dim=hidden_dim, activation=activation)
        self.mlp_node_feat = MLP(input_dim=vn_channel ** 2, output_dim=hidden_dim)

    def forward(self, node_feat, node_pos, vn_pos, batch):
        if node_pos.shape[0] == 0:
            return torch.zeros((0, self.mlp_node_feat.mlp[0].out_features), device=node_pos.device)

        info_vec = node_pos.repeat(1, self.vn_channel) - vn_pos[batch]
        info_vec = info_vec.view(node_pos.size(0), self.vn_channel, 3)

        # cdist can be unstable if inputs are large/nan.
        info_scalar = torch.cdist(info_vec, info_vec).view(node_pos.size(0), self.vn_channel ** 2)

        # Protection: Safe Division
        norm = safe_norm(info_scalar, dim=1, keepdim=True)
        info_scalar = info_scalar / norm

        return self.mlp_node_feat(info_scalar)


# ==========================================
# Modified Components
# ==========================================

class InitLayerGlobal(torch.nn.Module):
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
            cpl_dim: int = 128,
            r_start_cos_ratio: float = 0.8,
            norm_eps: float = 1e-8,
            PolynomialCutoff_p: float = 6,
            cutoff_type: str = "polynomial",
            edge_one_hot_dim: int = 128,
            device: Union[str, torch.device] = torch.device("cpu"),
            dtype: Union[str, torch.dtype] = torch.float32,
    ):
        super(InitLayerGlobal, self).__init__()
        SCALAR = o3.Irrep("0e")
        self.num_types = num_types
        if isinstance(r_max, float) or isinstance(r_max, int):
            self.r_max = torch.tensor(r_max, device=device, dtype=dtype)
            self.r_max_dict = None
        elif isinstance(r_max, dict):
            self.r_max = torch.tensor(max(list(r_max.values())), device=device, dtype=dtype)
            self.r_max_dict = {k: torch.tensor(v, device=device, dtype=dtype) for k, v in r_max.items()}
        else:
            raise TypeError("r_max should be either float, int or dict")

        self.idp = idp
        self.r_start_cos_ratio = r_start_cos_ratio
        self.polynomial_cutoff_p = PolynomialCutoff_p
        self.cutoff_type = cutoff_type
        self.device = device
        self.dtype = dtype
        self.irreps_out = o3.Irreps([(env_embed_multiplicity, ir) for _, ir in irreps_sh])

        assert irreps_sh[0].ir == SCALAR

        self.register_buffer(
            "env_sum_normalizations",
            torch.as_tensor(avg_num_neighbors).rsqrt(),
        )

        self.two_body_latent = ScalarMLPFunction(
            mlp_input_dimension=(edge_one_hot_dim + n_radial_basis + cpl_dim * 2),
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

    def forward(self, edge_index, atom_type, bond_type, edge_sh, edge_length, edge_one_hot, msg_cpl):
        edge_center = edge_index[0]
        edge_neighbor = edge_index[1]

        # Protection: Edge length 0 can cause instability in Bessel or Cutoffs
        edge_length = torch.nan_to_num(edge_length)

        edge_invariants = self.bessel(edge_length)

        if self.r_max_dict is None:
            if self.cutoff_type == "cosine":
                cutoff_coeffs = cosine_cutoff(edge_length, self.r_max.reshape(-1),
                                              r_start_cos_ratio=self.r_start_cos_ratio).flatten()
            elif self.cutoff_type == "polynomial":
                cutoff_coeffs = polynomial_cutoff(edge_length, self.r_max.reshape(-1),
                                                  p=self.polynomial_cutoff_p).flatten()
        else:
            cutoff_coeffs = polynomial_cutoff(edge_length, self.r_max.reshape(-1), p=self.polynomial_cutoff_p).flatten()

        prev_mask = cutoff_coeffs > 0
        active_edges = (cutoff_coeffs > 0).nonzero().squeeze(-1)

        if active_edges.numel() == 0:
            # Handle case with no active edges to prevent crash
            dummy_latents = torch.zeros((edge_sh.shape[0], self.two_body_latent.out_features), dtype=edge_sh.dtype,
                                        device=edge_sh.device)
            dummy_node_feat = torch.zeros((atom_type.shape[0], self.irreps_out.dim), dtype=edge_sh.dtype,
                                          device=edge_sh.device)
            dummy_edge_feat = torch.zeros((0, self.irreps_out.dim), dtype=edge_sh.dtype, device=edge_sh.device)
            return dummy_latents, dummy_node_feat, dummy_edge_feat, cutoff_coeffs, active_edges

        cpl_pair = torch.cat([msg_cpl[edge_center[active_edges]], msg_cpl[edge_neighbor[active_edges]]], dim=-1)

        latents = torch.zeros(
            (edge_sh.shape[0], self.two_body_latent.out_features),
            dtype=edge_sh.dtype,
            device=edge_sh.device,
        )

        new_latents = self.two_body_latent(torch.cat([
            edge_one_hot[active_edges],
            edge_invariants[active_edges],
            cpl_pair
        ], dim=-1))

        latents = torch.index_copy(
            latents, 0, active_edges,
            cutoff_coeffs[active_edges].unsqueeze(-1) * new_latents
        )

        weights_e = self.env_embed_mlp(latents[prev_mask])

        edge_features = self._env_weighter(
            edge_sh[prev_mask], weights_e
        )

        node_features = scatter(
            edge_features,
            edge_center[active_edges],
            dim=0,
            dim_size=atom_type.shape[0]
        )

        if self.env_sum_normalizations.ndim < 1:
            norm_const = self.env_sum_normalizations
        else:
            norm_const = self.env_sum_normalizations[atom_type.flatten()].unsqueeze(-1)

        node_features = node_features * norm_const

        return latents, node_features, edge_features, cutoff_coeffs, active_edges


class UpdateNodeGlobal(torch.nn.Module):
    def __init__(
            self,
            edge_irreps_in: o3.Irreps,
            irreps_in: o3.Irreps,
            irreps_out: o3.Irreps,
            latent_dim: int,
            cpl_dim: int,
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
        super(UpdateNodeGlobal, self).__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.edge_irreps_in = edge_irreps_in
        self.dtype = dtype
        self.device = device
        self.res_update = res_update

        self.register_buffer("env_sum_normalizations", torch.as_tensor(avg_num_neighbors).rsqrt())

        self._env_weighter = E3ElementLinear(irreps_in=irreps_out, dtype=dtype, device=device)

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
            irreps_scalar, [act[ir.p] for _, ir in irreps_scalar],
            irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],
            irreps_gated
        )

        self.tp = SO2_Linear(
            irreps_in=self.irreps_in + self.edge_irreps_in,
            irreps_out=self.activation.irreps_in,
            latent_dim=latent_dim,
            radial_emb=radial_emb,
            radial_channels=radial_channels,
            use_interpolation=use_interpolation_tp
        )
        # print('update node:')
        # print(f'irreps_in {self.tp.irreps_in}')
        # print(f'irreps_out {self.tp.irreps_out}')
        # print(f'latent dim {latent_dim}')
        # print(f'radial_emb {radial_emb}')
        # print(f'radial_channels {radial_channels}')
        # print(f'use_interpolation_tp {use_interpolation_tp}')

        self.lin_post = Linear(self.activation.irreps_out, self.irreps_out, shared_weights=True, internal_weights=True,
                               biases=True)

        if res_update:
            self.linear_res = Linear(self.irreps_in, self.irreps_out, shared_weights=True, internal_weights=True,
                                     biases=True)

        # ==========================================
        # FIX: Manual Rescale Mechanism (Params: O(C))
        # ==========================================
        self.num_scalar_weights = sum(mul for mul, _ in self.irreps_out)

        scaling_indices = []
        current_idx = 0
        for mul, ir in self.irreps_out:
            for _ in range(mul):
                scaling_indices.extend([current_idx] * ir.dim)
                current_idx += 1

        self.register_buffer('scaling_indices', torch.tensor(scaling_indices, dtype=torch.long))

        MLP = partial(BaseMLP, hidden_dim=cpl_dim, activation=nn.SiLU())
        self.mlp_cpl_coff = MLP(input_dim=cpl_dim, output_dim=self.num_scalar_weights)

        self.get_scalar_from_irreps = Linear(self.irreps_out, f'{cpl_dim}x0e')
        self.mlp_msg_cpl = MLP(input_dim=2 * cpl_dim, output_dim=cpl_dim)

        # PROTECTION: LayerNorm for recurrent scalar update to prevent explosion
        self.cpl_norm = nn.LayerNorm(cpl_dim)
        # ==========================================

        if res_update_ratios is None:
            res_update_params = torch.zeros(1)
        else:
            res_update_params = torch.special.logit(torch.as_tensor(res_update_ratios, dtype=torch.get_default_dtype()))
            res_update_params.clamp_(-6.0, 6.0)

        if res_update_ratios_learnable:
            self._res_update_params = torch.nn.Parameter(res_update_params)
        else:
            self.register_buffer("_res_update_params", res_update_params)

        self.use_layer_onehot_tp = use_layer_onehot_tp
        if use_layer_onehot_tp:
            instructions = []
            for i, (mul, ir) in enumerate(self.irreps_out):
                instructions.append((i, 0, i, 'uvu', True))
            self.node_onehot_tp = TensorProduct(irreps_in1=self.irreps_out, irreps_in2=f'95x0e',
                                                irreps_out=self.irreps_out, instructions=instructions)

    def forward(self, latents, node_features, edge_features, atom_type, node_onehot, edge_index, edge_vector,
                active_edges, wigner_D_all, msg_cpl):
        edge_center = edge_index[0]

        if active_edges.numel() == 0:
            return node_features, msg_cpl

        new_node_features = node_features

        # Protection: Ensure vectors are not NaN before TP
        edge_vector_safe = torch.nan_to_num(edge_vector[active_edges])

        message, _ = self.tp(
            torch.cat([new_node_features[edge_center[active_edges]], edge_features], dim=-1),
            edge_vector_safe, latents[active_edges], wigner_D_all
        )

        message = self.activation(message)
        message = self.lin_post(message)

        weights = self.env_embed_mlps(latents[active_edges])
        new_node_features = scatter(
            self._env_weighter(message, weights),
            edge_center[active_edges],
            dim=0,
            dim_size=node_features.shape[0]
        )

        if self.env_sum_normalizations.ndim < 1:
            norm_const = self.env_sum_normalizations
        else:
            norm_const = self.env_sum_normalizations[atom_type.flatten()].unsqueeze(-1)

        new_node_features = new_node_features * norm_const

        # ==========================================
        # Apply Canonical Rescale (Diagonal / Elementwise)
        # ==========================================
        coeffs = self.mlp_cpl_coff(msg_cpl)

        # Broadcast to full feature dimension
        expanded_coeffs = coeffs[:, self.scaling_indices]

        # Element-wise multiplication (Equivariant Scaling)
        new_node_features = new_node_features * expanded_coeffs

        # Update msg_cpl
        scalars_from_irreps = self.get_scalar_from_irreps(new_node_features)
        update = self.mlp_msg_cpl(torch.cat([scalars_from_irreps, msg_cpl], dim=-1))
        msg_cpl = msg_cpl + update

        # PROTECTION: LayerNorm
        msg_cpl = self.cpl_norm(msg_cpl)
        # ==========================================

        if self.res_update:
            update_coefficients = self._res_update_params.sigmoid()
            coefficient_old = torch.rsqrt(update_coefficients.square() + 1)
            coefficient_new = update_coefficients * coefficient_old
            node_features = coefficient_new * new_node_features + coefficient_old * self.linear_res(node_features)
        else:
            node_features = new_node_features

        if self.use_layer_onehot_tp:
            onehot_tune_node_feat = self.node_onehot_tp(node_features, node_onehot)
            node_features = node_features + onehot_tune_node_feat

        return node_features, msg_cpl


class LayerGlobal(torch.nn.Module):
    def __init__(
            self,
            num_types: int,
            avg_num_neighbors: Optional[float] = None,
            irreps_in: o3.Irreps = None,
            irreps_out: o3.Irreps = None,
            tp_radial_emb: bool = False,
            tp_radial_channels: list = [128, 128],
            norm_eps: float = 1e-8,
            latent_channels: list = [128, 128],
            latent_dim: int = 128,
            cpl_dim: int = 128,
            res_update: bool = True,
            use_layer_onehot_tp: bool = True,
            use_interpolation_tp: bool = False,
            edge_one_hot_dim: int = 128,
            res_update_ratios: Optional[List[float]] = None,
            res_update_ratios_learnable: bool = False,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(LayerGlobal, self).__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out

        self.edge_update = UpdateEdge(
            node_irreps_in=self.irreps_in,
            num_types=num_types,
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            latent_dim=latent_dim,
            latent_channels=latent_channels,
            radial_emb=tp_radial_emb,
            radial_channels=tp_radial_channels,
            use_layer_onehot_tp=use_layer_onehot_tp,
            edge_one_hot_dim=edge_one_hot_dim,
            res_update=res_update,
            res_update_ratios=res_update_ratios,
            res_update_ratios_learnable=res_update_ratios_learnable,
            dtype=dtype,
            device=device,
            use_interpolation_tp=use_interpolation_tp,
            norm_eps=norm_eps
        )

        self.node_update = UpdateNodeGlobal(
            edge_irreps_in=self.edge_update.irreps_out,
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            latent_dim=latent_dim,
            cpl_dim=cpl_dim,
            radial_emb=tp_radial_emb,
            use_layer_onehot_tp=use_layer_onehot_tp,
            radial_channels=tp_radial_channels,
            res_update=res_update,
            res_update_ratios=res_update_ratios,
            res_update_ratios_learnable=res_update_ratios_learnable,
            avg_num_neighbors=avg_num_neighbors,
            dtype=dtype,
            device=device,
            use_interpolation_tp=use_interpolation_tp,
            norm_eps=norm_eps
        )

    def forward(self, latents, node_features, edge_features, node_onehot, edge_index, edge_vector, atom_type,
                cutoff_coeffs, active_edges, edge_one_hot, wigner_D_all, msg_cpl):
        edge_features, latents, wigner_D_all = self.edge_update(latents, node_features, node_onehot, edge_features,
                                                                edge_index, edge_vector, cutoff_coeffs, active_edges,
                                                                edge_one_hot, wigner_D_all)
        node_features, msg_cpl = self.node_update(latents, node_features, edge_features, atom_type, node_onehot,
                                                  edge_index, edge_vector, active_edges, wigner_D_all, msg_cpl)

        return latents, node_features, edge_features, wigner_D_all, msg_cpl


# ==========================================
# Main Model
# ==========================================

@Embedding.register("lem_global")
class LemGlobal(torch.nn.Module):
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
            # ========================
            # New Virtual Node Params
            # ========================
            vn_channel: int = 6,
            cpl_dim: int = 128,
            color_mode: str = 'tp',
            **kwargs,
    ):

        super(LemGlobal, self).__init__()

        irreps_hidden = o3.Irreps(irreps_hidden)
        lmax = irreps_hidden.lmax

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb")
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp

        self.latent_dim = latent_dim
        self.cpl_dim = cpl_dim

        self.basis = self.idp.basis
        self.idp.get_irreps(no_parity=False)
        if universal:
            self.n_atom = 95
        else:
            self.n_atom = len(self.basis.keys())

        irreps_sh = o3.Irreps([(1, (i, (-1) ** i)) for i in range(lmax + 1)])
        orbpair_irreps = self.idp.orbpair_irreps.sort()[0].simplify()

        irreps_out = []
        for mul, ir1 in irreps_hidden:
            for _, ir2 in orbpair_irreps:
                irreps_out += [o3.Irrep(str(irr)) for irr in ir1 * ir2]
        irreps_out = o3.Irreps(irreps_out).sort()[0].simplify()

        assert all(
            ir in irreps_out for _, ir in orbpair_irreps), "hidden irreps should at least cover all the reqired irreps"

        self.sh = SphericalHarmonics(
            irreps_sh, sh_normalized, sh_normalization
        )
        self.onehot = OneHotAtomEncoding(num_types=self.n_atom, set_features=False, idp=self.idp, universal=universal)
        self.edge_one_hot = OneHotEdgeEmbedding(num_types=self.n_atom, idp=self.idp, universal=universal,
                                                d_emb=edge_one_hot_dim)

        # ========================
        # Virtual Node Modules
        # ========================
        self.init_embed = nn.Linear(1, cpl_dim)
        self.node_color = NodeColor(hidden_dim=cpl_dim, color_type=color_mode)
        self.vn = VirtualNode(vn_channel=vn_channel, hidden_dim=cpl_dim)
        self.node_feat_by_vn = NodeFeatByVN(vn_channel=vn_channel, hidden_dim=cpl_dim)
        # ========================

        self.init_layer = InitLayerGlobal(
            idp=self.idp,
            num_types=self.n_atom,
            n_radial_basis=n_radial_basis,
            r_max=r_max,
            irreps_sh=irreps_sh,
            avg_num_neighbors=avg_num_neighbors,
            env_embed_multiplicity=env_embed_multiplicity,
            # MLP parameters:
            two_body_latent_channels=latent_channels,
            latent_dim=latent_dim,
            cpl_dim=cpl_dim,
            # cutoffs
            r_start_cos_ratio=r_start_cos_ratio,
            PolynomialCutoff_p=PolynomialCutoff_p,
            cutoff_type=cutoff_type,
            device=device,
            dtype=dtype,
            edge_one_hot_dim=edge_one_hot_dim,
            norm_eps=norm_eps
        )

        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                irreps_in = self.init_layer.irreps_out
            else:
                irreps_in = irreps_hidden

            if i == n_layers - 1:
                irreps_out_layer = orbpair_irreps.sort()[0].simplify()
                if use_interpolation_out:
                    use_interpolation_tp = True
            else:
                irreps_out_layer = irreps_hidden
                use_interpolation_tp = False

            self.layers.append(LayerGlobal(
                num_types=self.n_atom,
                avg_num_neighbors=avg_num_neighbors,
                irreps_in=irreps_in,
                irreps_out=irreps_out_layer,
                tp_radial_emb=tp_radial_emb,
                tp_radial_channels=tp_radial_channels,
                use_layer_onehot_tp=use_layer_onehot_tp,
                edge_one_hot_dim=edge_one_hot_dim,
                latent_channels=latent_channels,
                latent_dim=latent_dim,
                cpl_dim=cpl_dim,
                res_update=res_update,
                res_update_ratios=res_update_ratios,
                res_update_ratios_learnable=res_update_ratios_learnable,
                dtype=dtype,
                device=device,
                use_interpolation_tp=use_interpolation_tp
            ))
            if use_interpolation_tp:
                print(f'Use interpolation SO2 layer in layer {i}')

        self.use_out_onehot_tp = use_out_onehot_tp
        if self.use_out_onehot_tp:
            self.out_node_ele_tp = FullyConnectedTensorProduct(
                irreps_in1=self.layers[-1].irreps_out,
                irreps_in2='95x0e',
                irreps_out=self.idp.orbpair_irreps,
            )
            self.out_edge_ele_tp = FullyConnectedTensorProduct(
                irreps_in1=self.layers[-1].irreps_out,
                irreps_in2=f'{edge_one_hot_dim}x0e',
                irreps_out=self.idp.orbpair_irreps,
            )
        self.out_edge = Linear(self.layers[-1].irreps_out, self.idp.orbpair_irreps, shared_weights=True,
                               internal_weights=True, biases=True)
        self.out_node = Linear(self.layers[-1].irreps_out, self.idp.orbpair_irreps, shared_weights=True,
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
        edge_vector = data[_keys.EDGE_VECTORS_KEY]
        edge_sh = self.sh(data[_keys.EDGE_VECTORS_KEY][:, [1, 2, 0]])
        edge_length = data[_keys.EDGE_LENGTH_KEY]

        data = self.onehot(data)
        edge_one_hot = self.edge_one_hot(data)
        node_one_hot = data[_keys.NODE_ATTRS_KEY]
        atom_type = data[_keys.ATOM_TYPE_KEY].flatten()
        bond_type = data[_keys.EDGE_TYPE_KEY].flatten()
        batch = data[_keys.BATCH_KEY]

        # FIX: use standard 'pos' key (POS_KEY) if available, else string "pos"
        # Based on trace, use data["pos"]
        node_pos = data["pos"]

        # ==========================================
        # Virtual Node Pre-processing
        # ==========================================
        init_feat = self.init_embed(atom_type.float().unsqueeze(-1))
        colored_feat = init_feat + self.node_color(init_feat, node_pos, batch, edge_index)
        vn_pos = self.vn(colored_feat, node_pos, batch)
        msg_cpl = self.node_feat_by_vn(colored_feat, node_pos, vn_pos, batch)
        # ==========================================

        num_nodes_total = node_one_hot.shape[0]

        latents, node_features, edge_features, cutoff_coeffs, active_edges = self.init_layer(
            edge_index, atom_type, bond_type, edge_sh, edge_length, edge_one_hot, msg_cpl
        )

        n_active_nodes = node_features.shape[0]
        if n_active_nodes < num_nodes_total:
            safe_node_one_hot = node_one_hot[:n_active_nodes]
        else:
            safe_node_one_hot = node_one_hot

        edge_one_hot = edge_one_hot[active_edges]

        data[_keys.EDGE_OVERLAP_KEY] = latents
        wigner_D_all = None

        for idx, layer in enumerate(self.layers):
            latents, node_features, edge_features, wigner_D_all, msg_cpl = \
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
                    msg_cpl
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
        data[_keys.EDGE_FEATURES_KEY] = torch.zeros(edge_index.shape[1], self.idp.orbpair_irreps.dim, dtype=self.dtype, device=self.device)
        data[_keys.EDGE_FEATURES_KEY] = torch.index_copy(data[_keys.EDGE_FEATURES_KEY], 0, active_edges, out_edge_features)

        return data


@torch.jit.script
def ShiftedSoftPlus(x: torch.Tensor):
    return torch.nn.functional.softplus(x) - math.log(2.0)

class InitLayer(torch.nn.Module):
    def __init__(
            self,
            # required params
            idp,
            num_types: int,
            n_radial_basis: int,
            r_max: float,
            avg_num_neighbors: Optional[float] = None,
            irreps_sh: o3.Irreps=None,
            env_embed_multiplicity: int = 32,
            # MLP parameters:
            two_body_latent_channels: list=[128, 128],
            latent_dim: int=128,
            # cutoffs
            r_start_cos_ratio: float = 0.8,
            norm_eps: float = 1e-8,
            PolynomialCutoff_p: float = 6,
            cutoff_type: str = "polynomial",
            edge_one_hot_dim: int = 128,
            device: Union[str, torch.device] = torch.device("cpu"),
            dtype: Union[str, torch.dtype] = torch.float32,
    ):
        super(InitLayer, self).__init__()
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
                for k,v in r_max.items():
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

        assert all(mul==1 for mul, _ in irreps_sh)
        # env_embed_irreps = o3.Irreps([(1, ir) for _, ir in irreps_sh])
        assert (
            irreps_sh[0].ir == SCALAR
        ), "env_embed_irreps must start with scalars"

        self.register_buffer(
            "env_sum_normalizations",
            # dividing by sqrt(N)
            torch.as_tensor(avg_num_neighbors).rsqrt(),
        )

        # Node invariants for center and neighbor (chemistry)
        # Plus edge invariants for the edge (radius).
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
            path_normalization = "element", # if path normalization is element and input irreps has 1 mul, it should not have effect !
        )

        self.env_embed_mlp = ScalarMLPFunction(
                        mlp_input_dimension=self.two_body_latent.out_features,
                        mlp_output_dimension=self._env_weighter.weight_numel,
                        mlp_latent_dimensions=[],
                        mlp_nonlinearity=None,
                        mlp_initialization="uniform",
                    )

        self.bessel = BesselBasis(r_max=self.r_max, num_basis=n_radial_basis, trainable=True)


    def forward(self, edge_index, atom_type, bond_type, edge_sh, edge_length, edge_one_hot):
        edge_center = edge_index[0]

        edge_invariants = self.bessel(edge_length)

        # Vectorized precompute per layer cutoffs
        if self.r_max_dict is None:
            if self.cutoff_type == "cosine":
                cutoff_coeffs = cosine_cutoff(
                    edge_length,
                    self.r_max.reshape(-1),
                    r_start_cos_ratio=self.r_start_cos_ratio,
                ).flatten()

            elif self.cutoff_type == "polynomial":
                cutoff_coeffs = polynomial_cutoff(
                    edge_length, self.r_max.reshape(-1), p=self.polynomial_cutoff_p
                ).flatten()

            else:
                # This branch is unreachable (cutoff type is checked in __init__)
                # But TorchScript doesn't know that, so we need to make it explicitly
                # impossible to make it past so it doesn't throw
                # "cutoff_coeffs_all is not defined in the false branch"
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
                            0.5*(self.r_max_dict[iatom]+self.r_max_dict[jatom]),
                            r_start_cos_ratio=self.r_start_cos_ratio,
                        ).flatten()
                    elif self.cutoff_type == "polynomial":
                        c_coeff = polynomial_cutoff(
                            edge_length[mask],
                            0.5*(self.r_max_dict[iatom]+self.r_max_dict[jatom]),
                            p=self.polynomial_cutoff_p
                        ).flatten()

                    else:
                        # This branch is unreachable (cutoff type is checked in __init__)
                        # But TorchScript doesn't know that, so we need to make it explicitly
                        # impossible to make it past so it doesn't throw
                        # "cutoff_coeffs_all is not defined in the false branch"
                        assert False, "Invalid cutoff type"

                    cutoff_coeffs = torch.index_copy(cutoff_coeffs, 0, index, c_coeff)

        # Determine which edges are still in play
        prev_mask = cutoff_coeffs > 0
        active_edges = (cutoff_coeffs > 0).nonzero().squeeze(-1)

        # Compute latents
        latents = torch.zeros(
            (edge_sh.shape[0], self.two_body_latent.out_features),
            dtype=edge_sh.dtype,
            device=edge_sh.device,
        )

        new_latents = self.two_body_latent(torch.cat([
            edge_one_hot[active_edges],
            edge_invariants[active_edges],
        ], dim=-1))

        # Apply cutoff, which propagates through to everything else
        latents = torch.index_copy(
            latents, 0, active_edges,
            cutoff_coeffs[active_edges].unsqueeze(-1) * new_latents
            )

        weights_e = self.env_embed_mlp(latents[prev_mask])
        # features = self.bn(features)

        edge_features = self._env_weighter(
            edge_sh[prev_mask], weights_e
        )

        node_features = scatter(
            edge_features,
            edge_center[active_edges],
            dim=0,
        )

        if self.env_sum_normalizations.ndim < 1:
            norm_const = self.env_sum_normalizations
        else:
            norm_const = self.env_sum_normalizations[atom_type.flatten()].unsqueeze(-1)

        node_features = node_features * norm_const

        return latents, node_features, edge_features, cutoff_coeffs, active_edges # the radial embedding x and the sperical hidden V


class UpdateEdge(torch.nn.Module):
    def __init__(
        self,
        num_types,
        node_irreps_in: o3.Irreps,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        latent_dim: int,
        norm_eps: float = 1e-8,
        latent_channels: list=[128, 128],
        radial_emb: bool=False,
        radial_channels: list=[128, 128],
        res_update: bool = True,
        use_layer_onehot_tp: bool = True,
        use_interpolation_tp: bool = False,
        edge_one_hot_dim: int = 128,
        res_update_ratios: Optional[List[float]] = None,
        res_update_ratios_learnable: bool = False,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(UpdateEdge, self).__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.node_irreps_in = node_irreps_in
        self.dtype = dtype
        self.device = device
        self.res_update = res_update

        self._edge_weighter = E3ElementLinear(
                irreps_in=irreps_out,
                dtype=dtype,
                device=device,
            )

        self.edge_embed_mlps = ScalarMLPFunction(
                mlp_input_dimension=latent_dim,
                mlp_latent_dimensions=[],
                mlp_output_dimension=self._edge_weighter.weight_numel,
            )

        self.ln = torch.nn.LayerNorm(latent_dim)

        irreps_scalar = o3.Irreps([(mul, ir) for mul, ir in self.irreps_out if ir.l == 0]).simplify()
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_out if ir.l > 0]).simplify()
        irreps_gates = o3.Irreps([(mul, (0,1)) for mul, _ in irreps_gated]).simplify()
        act={1: torch.nn.functional.silu, -1: torch.tanh}
        act_gates={1: torch.sigmoid, -1: torch.tanh}

        self.activation = Gate(
            irreps_scalar, [act[ir.p] for _, ir in irreps_scalar],  # scalar
            irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
            irreps_gated  # gated tensors
        )

        self.tp = SO2_Linear(
            irreps_in=self.node_irreps_in+self.irreps_in+self.node_irreps_in,
            irreps_out=self.activation.irreps_in,
            latent_dim=latent_dim,
            radial_emb=radial_emb,
            radial_channels=radial_channels,
            use_interpolation=use_interpolation_tp
        )

        self.latents_mlp_1 = ScalarMLPFunction(
            mlp_input_dimension=latent_dim+self.irreps_out[0].dim,
            mlp_output_dimension=latent_dim,
            mlp_latent_dimensions=latent_channels,
            mlp_nonlinearity="silu",
            mlp_initialization="uniform",
        )

        self.latents_mlp_2 = ScalarMLPFunction(
            mlp_input_dimension=latent_dim+edge_one_hot_dim,
            mlp_output_dimension=latent_dim,
            mlp_latent_dimensions=latent_channels,
            mlp_nonlinearity="silu",
            mlp_initialization="uniform",
        )

        self.lin_post = Linear(
            self.activation.irreps_out,
            self.irreps_out,
            shared_weights=True,
            internal_weights=True,
            biases=True,
        )

        if res_update:
            self.linear_res = Linear(
                self.irreps_in,
                self.irreps_out,
                shared_weights=True,
                internal_weights=True,
                biases=True,
            )

        # - layer resnet update weights -
        if res_update_ratios is None:
            # We initialize to zeros, which under the sigmoid() become 0.5
            # so 1/2 * layer_1 + 1/4 * layer_2 + ...
            # note that the sigmoid of these are the factor _between_ layers
            # so the first entry is the ratio for the latent resnet of the first and second layers, etc.
            # e.g. if there are 3 layers, there are 2 ratios: l1:l2, l2:l3
            res_update_params = torch.zeros(1)
        else:
            res_update_ratios = torch.as_tensor(
                res_update_ratios, dtype=torch.get_default_dtype()
            )
            assert res_update_ratios > 0.0
            assert res_update_ratios < 1.0
            res_update_params = torch.special.logit(
                res_update_ratios
            )
            # The sigmoid is mostly saturated at ±6, keep it in a reasonable range
            res_update_params.clamp_(-6.0, 6.0)

        if res_update_ratios_learnable:
            self._res_update_params = torch.nn.Parameter(
                res_update_params
            )
        else:
            self.register_buffer(
                "_res_update_params", res_update_params
            )

        self.use_layer_onehot_tp = use_layer_onehot_tp
        if use_layer_onehot_tp:
            instructions = []
            for i, (mul, ir) in enumerate(self.irreps_out):
                instructions.append((i, 0, i, 'uvu', True))
            self.edge_onehot_tp = TensorProduct(
                irreps_in1=self.irreps_out,
                irreps_in2=f'{edge_one_hot_dim}x0e',
                irreps_out=self.irreps_out,
                instructions=instructions
            )


    def forward(self, latents, node_features, node_onehot, edge_features, edge_index, edge_vector, cutoff_coeffs, active_edges, edge_one_hot, wigner_D_all):
        active_edge_index = edge_index[:, active_edges]

        new_node_features = node_features
        new_edge_features, wigner_D_all = self.tp(
            torch.cat(
                [
                    new_node_features[active_edge_index[0]],
                    edge_features,
                    new_node_features[active_edge_index[1]]
                    ]
                , dim=-1), edge_vector[active_edges], latents[active_edges], wigner_D_all) # full_out_irreps

        new_edge_features = self.activation(new_edge_features)
        new_edge_features = self.lin_post(new_edge_features)

        scalars = new_edge_features[:, :self.irreps_out[0].dim]
        assert len(scalars.shape) == 2

        weights = self.edge_embed_mlps(latents[active_edges])
        new_edge_features = self._edge_weighter(new_edge_features, weights)

        # update latent

        new_latents = self.latents_mlp_1(torch.cat(
            [
                self.ln(latents[active_edges]),
                scalars,
            ], dim=-1))

        new_latents = self.latents_mlp_2(torch.cat(
            [
                new_latents,
                edge_one_hot,
            ], dim=-1))

        new_latents = cutoff_coeffs[active_edges].unsqueeze(-1) * new_latents

        if self.res_update:
            update_coefficients = self._res_update_params.sigmoid()
            coefficient_old = torch.rsqrt(update_coefficients.square() + 1)
            coefficient_new = update_coefficients * coefficient_old
            edge_features = coefficient_new * new_edge_features + coefficient_old * self.linear_res(edge_features)

            latents = torch.index_copy(
                latents, 0, active_edges,
                coefficient_new * new_latents + coefficient_old * latents[active_edges]
            )
        else:
            edge_features = new_edge_features
            latents = torch.index_copy(
                latents, 0, active_edges,
                new_latents
            )
        if self.use_layer_onehot_tp:
            onehot_tune_edge_feat = self.edge_onehot_tp(edge_features, edge_one_hot)
            edge_features = edge_features + onehot_tune_edge_feat

        return edge_features, latents, wigner_D_all
