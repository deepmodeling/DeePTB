import torch
import math
from typing import Optional, List, Union, Dict
from functools import partial
from torch import nn
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
from dptb.nn.embedding.from_deephe3.deephe3 import tp_path_exists
from dptb.nn.cutoff import cosine_cutoff, polynomial_cutoff
from dptb.nn.rescale import E3ElementLinear
from dptb.nn.tensor_product import SO2_Linear
from dptb.data.transforms import OrbitalMapper
from ..type_encode.one_hot import OneHotAtomEncoding, OneHotEdgeEmbedding
from dptb.data.AtomicDataDict import with_edge_vectors, with_batch

# Reuse original UpdateEdge and InitLayer and Layer
from .lem import UpdateEdge, InitLayer, Layer

# ============================================================================
# Helper Functions: Safety Wrappers
# ============================================================================

def safe_norm(x, dim=-1, keepdim=False, eps=1e-8):
    """Calculates norm with protection against zero gradients (NaNs)."""
    return torch.norm(x, dim=dim, keepdim=keepdim).clamp(min=eps)


def safe_normalize(x, dim=-1, eps=1e-8):
    """Safely normalize a vector."""
    return x / safe_norm(x, dim=dim, keepdim=True, eps=eps)


# ============================================================================
# Uni-EGNN (TFN-CPL) Core Components Patch (FIXED & PROTECTED)
# ============================================================================

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

        # PROTECTION: Check for NaNs during inference/val to prevent crash
        if not self.training and torch.isnan(out).any():
            out = torch.nan_to_num(out)

        return out


class VNInitial(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 64,
            vn_channel: int = 4,
            edge_attr_dim: int = 0,
            activation: nn.Module = nn.SiLU(),
            norm: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.vn_channel = vn_channel

        self.diff_vec_coff = e3nn.o3.FullyConnectedTensorProduct(
            '1x1o', '1x0e', f'{self.vn_channel}x1o', shared_weights=False
        )

        MLP = partial(BaseMLP, hidden_dim=hidden_dim, activation=activation, norm=norm)
        self.mlp_feat_init = MLP(input_dim=hidden_dim, output_dim=hidden_dim, last_act=True)
        self.mlp_msg = MLP(input_dim=2 * hidden_dim + edge_attr_dim + 4, output_dim=hidden_dim)
        self.mlp_diff_vel_coff = MLP(input_dim=hidden_dim, output_dim=self.diff_vec_coff.weight_numel)
        self.mlp_vn_feat = MLP(input_dim=hidden_dim, output_dim=hidden_dim)

    def forward(self, node_feat, node_pos, edge_index, edge_attr):
        vn_feat = self.mlp_feat_init(node_feat)
        vn_pos = node_pos.repeat(1, self.vn_channel)

        row, col = edge_index
        diff_pos = node_pos[row] - node_pos[col]

        # PROTECTION: Safe Normalize
        diff_vec_norm = safe_normalize(diff_pos, dim=-1)

        dist_sq = torch.sum(diff_pos ** 2, dim=-1, keepdim=True)
        ip = dist_sq.repeat(1, 4)

        one = torch.ones([diff_pos.size(0), 1], device=diff_pos.device)
        msg = torch.cat([node_feat[row], node_feat[col], edge_attr, ip], dim=1)
        msg = self.mlp_msg(msg)

        diff_vec_out = self.diff_vec_coff(diff_vec_norm, one, self.mlp_diff_vel_coff(msg))

        # scatter_mean handles dim_size, safe even if index is empty
        agg_feat = scatter_mean(src=msg, index=row, dim=0, dim_size=node_feat.size(0))
        agg_vec = scatter_mean(src=diff_vec_out, index=row, dim=0, dim_size=node_feat.size(0))

        vn_feat = vn_feat + self.mlp_vn_feat(agg_feat)
        vn_pos = vn_pos + agg_vec
        return vn_feat, vn_pos


class VNLayer(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            target_irreps: e3nn.o3.Irreps,
            vn_channel: int = 4,
            activation: nn.Module = nn.SiLU(),
            norm: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.vn_channel = vn_channel
        self.target_irreps = target_irreps

        self.scalar_indices = []
        self.scalar_dim = 0
        idx = 0
        for mul, ir in self.target_irreps:
            dim = mul * (2 * ir.l + 1)
            if ir.l == 0 and ir.p == 1:
                self.scalar_indices.append((idx, idx + dim))
                self.scalar_dim += dim
            idx += dim

        self.get_scalar = e3nn.o3.Linear(target_irreps, f'{self.scalar_dim}x0e')

        self.diff_pos_vr_coff = e3nn.o3.FullyConnectedTensorProduct(
            f'{self.vn_channel}x1o', '1x0e', '1x1o', shared_weights=False
        )
        self.diff_pos_rv_coff = e3nn.o3.FullyConnectedTensorProduct(
            f'{self.vn_channel}x1o', '1x0e', f'{self.vn_channel}x1o', shared_weights=False
        )

        MLP = partial(BaseMLP, hidden_dim=hidden_dim, activation=activation, norm=norm)
        self.mlp_com_msg = MLP(input_dim=self.scalar_dim + hidden_dim + vn_channel ** 2, output_dim=hidden_dim)

        self.mlp_diff_pos_vr_coff = MLP(input_dim=hidden_dim, output_dim=self.diff_pos_vr_coff.weight_numel)
        self.mlp_diff_pos_rv_coff = MLP(input_dim=hidden_dim, output_dim=self.diff_pos_rv_coff.weight_numel)

        self.mlp_feat = MLP(input_dim=hidden_dim, output_dim=hidden_dim)
        self.mlp_feat_to_target = MLP(input_dim=hidden_dim, output_dim=self.scalar_dim)
        self.mlp_vn_feat = MLP(input_dim=hidden_dim, output_dim=hidden_dim)

    def forward(self, node_feat, node_pos, vn_feat, vn_pos, edge_index):
        node_feat_scalar = self.get_scalar(node_feat)
        row, col = edge_index

        # Shape: [Num_Edges, VN_Channel, 3]
        diff_pos_vr = node_pos.repeat(1, self.vn_channel)[row] - vn_pos[col]

        # Inner Product (Scalar Invariant)
        ip = torch.einsum('bij,bkj->bik', diff_pos_vr.view(-1, self.vn_channel, 3),
                          diff_pos_vr.view(-1, self.vn_channel, 3)).view(-1, self.vn_channel ** 2)

        # PROTECTION: Safe Norm for normalization
        ip = ip / safe_norm(ip, dim=-1, keepdim=True)

        # PROTECTION: Safe Normalize for direction vectors
        # Reshape to handle the multi-channel normalization correctly
        diff_pos_vr_flat = diff_pos_vr.view(-1, 3)
        diff_pos_vr_norm = safe_normalize(diff_pos_vr_flat, dim=-1).view(diff_pos_vr.shape)

        diff_pos_rv_norm = -diff_pos_vr_norm

        com_msg_vr = torch.cat([node_feat_scalar[row], vn_feat[col], ip], dim=1)
        com_msg_vr = self.mlp_com_msg(com_msg_vr)

        # Flatten for TensorProduct (needs 2D inputs)
        diff_pos_vr_input = diff_pos_vr_norm.view(-1, self.vn_channel * 3)
        diff_pos_rv_input = diff_pos_rv_norm.view(-1, self.vn_channel * 3)

        # Use input device for ones tensor
        one_vr = torch.ones([diff_pos_vr_input.size(0), 1], device=diff_pos_vr_input.device)
        diff_pos_vr_update = self.diff_pos_vr_coff(diff_pos_vr_input, one_vr, self.mlp_diff_pos_vr_coff(com_msg_vr))

        # >>> FIX: device=diff_pos_rv_input.device (was diff_pos_rv.device) <<<
        one_rv = torch.ones([diff_pos_rv_input.size(0), 1], device=diff_pos_rv_input.device)
        diff_pos_rv_update = self.diff_pos_rv_coff(diff_pos_rv_input, one_rv, self.mlp_diff_pos_rv_coff(com_msg_vr))

        dim_size = node_feat.size(0)

        agg_feat_vr = scatter_mean(src=com_msg_vr, index=row, dim=0, dim_size=dim_size)
        agg_pos_vr = scatter_mean(src=diff_pos_vr_update, index=row, dim=0, dim_size=dim_size)
        agg_pos_rv = scatter_mean(src=diff_pos_rv_update, index=row, dim=0, dim_size=dim_size)

        delta_hidden = self.mlp_feat(agg_feat_vr)
        delta_scalar = self.mlp_feat_to_target(delta_hidden)

        delta_node_feat = torch.zeros_like(node_feat)
        scalar_ptr = 0
        current_idx = 0
        for mul, ir in self.target_irreps:
            length = mul * (2 * ir.l + 1)
            if ir.l == 0 and ir.p == 1:
                delta_node_feat[:, current_idx:current_idx + length] = delta_scalar[:, scalar_ptr:scalar_ptr + length]
                scalar_ptr += length
            current_idx += length

        node_features_out = node_feat + delta_node_feat

        node_pos_out = node_pos + agg_pos_vr
        vn_feat_out = vn_feat + self.mlp_vn_feat(agg_feat_vr)
        vn_pos_out = vn_pos + agg_pos_rv

        return node_features_out, node_pos_out, vn_feat_out, vn_pos_out


# ============================================================================
# Patched Lem Model
# ============================================================================

@Embedding.register("lem_local")
class LemLocal(torch.nn.Module):
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
            # >>> PATCH ARGS <<<
            vn_channel: int = 4,
            vn_hidden_dim: int = 128,
            # >>> END PATCH <<<
            **kwargs,
    ):

        super(LemLocal, self).__init__()

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

        self.basis = self.idp.basis
        self.idp.get_irreps(no_parity=False)
        if universal:
            self.n_atom = 95
        else:
            self.n_atom = len(self.basis.keys())

        irreps_sh = o3.Irreps([(1, (i, (-1) ** i)) for i in range(lmax + 1)])
        orbpair_irreps = self.idp.orbpair_irreps.sort()[0].simplify()

        # check if the irreps setting satisfied the requirement of idp
        irreps_out = []
        for mul, ir1 in irreps_hidden:
            for _, ir2 in orbpair_irreps:
                irreps_out += [o3.Irrep(str(irr)) for irr in ir1 * ir2]
        irreps_out = o3.Irreps(irreps_out).sort()[0].simplify()

        assert all(ir in irreps_out for _, ir in orbpair_irreps), "hidden irreps..."

        self.sh = SphericalHarmonics(
            irreps_sh, sh_normalized, sh_normalization
        )
        self.onehot = OneHotAtomEncoding(num_types=self.n_atom, set_features=False, idp=self.idp, universal=universal)
        self.edge_one_hot = OneHotEdgeEmbedding(num_types=self.n_atom, idp=self.idp, universal=universal,
                                                d_emb=edge_one_hot_dim)

        self.init_layer = InitLayer(
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

        # >>> PATCH START: Initialize VN Modules <<<
        self.vn_init = VNInitial(
            hidden_dim=vn_hidden_dim,
            vn_channel=vn_channel,
            edge_attr_dim=edge_one_hot_dim,
            activation=nn.SiLU()
        )
        self.emb_to_vn = nn.Linear(self.n_atom, vn_hidden_dim)

        self.vn_layers = torch.nn.ModuleList()
        # VNLayer 0: For InitLayer output
        self.vn_layers.append(VNLayer(
            hidden_dim=vn_hidden_dim,
            target_irreps=self.init_layer.irreps_out,
            vn_channel=vn_channel,
        ))
        # >>> PATCH END <<<

        for i in range(n_layers):
            if i == 0:
                irreps_in = self.init_layer.irreps_out
            else:
                irreps_in = irreps_hidden

            if i == n_layers - 1:
                irreps_out = orbpair_irreps.sort()[0].simplify()
                if use_interpolation_out:
                    use_interpolation_tp = True
            else:
                irreps_out = irreps_hidden
                use_interpolation_tp = False

            self.layers.append(Layer(
                num_types=self.n_atom,
                # required params
                avg_num_neighbors=avg_num_neighbors,
                irreps_in=irreps_in,
                irreps_out=irreps_out,
                tp_radial_emb=tp_radial_emb,
                tp_radial_channels=tp_radial_channels,
                use_layer_onehot_tp=use_layer_onehot_tp,
                edge_one_hot_dim=edge_one_hot_dim,
                # MLP parameters:
                latent_channels=latent_channels,
                latent_dim=latent_dim,
                res_update=res_update,
                res_update_ratios=res_update_ratios,
                res_update_ratios_learnable=res_update_ratios_learnable,
                dtype=dtype,
                device=device,
                use_interpolation_tp=use_interpolation_tp
            )
            )

            # >>> PATCH START: Add VNLayer for this layer <<<
            self.vn_layers.append(VNLayer(
                hidden_dim=vn_hidden_dim,
                target_irreps=irreps_out,
                vn_channel=vn_channel,
            ))
            # >>> PATCH END <<<

            if use_interpolation_tp:
                print(f'Use interpolation SO2 layer in layer {i}')

        # initilize output_layer
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
        # data = with_env_vectors(data, with_lengths=True)
        data = with_batch(data)

        edge_index = data[_keys.EDGE_INDEX_KEY]
        edge_vector = data[_keys.EDGE_VECTORS_KEY]

        # PROTECTION: Safe access to positions
        if _keys.POSITIONS_KEY in data:
            node_pos = data[_keys.POSITIONS_KEY]
        else:
            node_pos = data["pos"]  # Fallback standard PyG key

        edge_sh = self.sh(data[_keys.EDGE_VECTORS_KEY][:, [1, 2, 0]])

        # PROTECTION: Clean NaN in inputs
        edge_length = torch.nan_to_num(data[_keys.EDGE_LENGTH_KEY])

        data = self.onehot(data)
        edge_one_hot = self.edge_one_hot(data)
        node_one_hot = data[_keys.NODE_ATTRS_KEY]
        atom_type = data[_keys.ATOM_TYPE_KEY].flatten()
        bond_type = data[_keys.EDGE_TYPE_KEY].flatten()

        # 获取总节点数，用于后续兜底
        num_nodes_total = node_one_hot.shape[0]

        # >>> PATCH START: VN Init <<<
        vn_feat_in = self.emb_to_vn(node_one_hot)
        vn_feat, vn_pos = self.vn_init(vn_feat_in, node_pos, edge_index, edge_one_hot)
        node_pos_dyn = node_pos
        # >>> PATCH END <<<

        latents, node_features, edge_features, cutoff_coeffs, active_edges = self.init_layer(edge_index, atom_type,
                                                                                             bond_type, edge_sh,
                                                                                             edge_length, edge_one_hot)

        # >>> PATCH START: Emergency Injection for InitLayer <<<
        node_features, node_pos_dyn, vn_feat, vn_pos = self.vn_layers[0](
            node_features, node_pos_dyn, vn_feat, vn_pos, edge_index
        )
        # >>> PATCH END <<<

        n_active_nodes = node_features.shape[0]
        if n_active_nodes < num_nodes_total:
            safe_node_one_hot = node_one_hot[:n_active_nodes]
        else:
            safe_node_one_hot = node_one_hot

        edge_one_hot = edge_one_hot[active_edges]

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
                    wigner_D_all
                )

            # >>> PATCH START: Accompanied Update <<<
            node_features, node_pos_dyn, vn_feat, vn_pos = self.vn_layers[idx + 1](
                node_features, node_pos_dyn, vn_feat, vn_pos, edge_index
            )
            # >>> PATCH END <<<

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
            # node one-hot
            out_node_features = out_node_features + self.out_node_ele_tp(node_features, node_one_hot)
            # edge one-hot
            out_edge_features = out_edge_features + self.out_edge_ele_tp(edge_features, edge_one_hot)

        data[_keys.NODE_FEATURES_KEY] = out_node_features
        data[_keys.EDGE_FEATURES_KEY] = torch.zeros(edge_index.shape[1], self.idp.orbpair_irreps.dim, dtype=self.dtype,
                                                    device=self.device)
        data[_keys.EDGE_FEATURES_KEY] = torch.index_copy(data[_keys.EDGE_FEATURES_KEY], 0, active_edges,
                                                         out_edge_features)

        return data

# (InitLayer, UpdateNode, UpdateEdge, Layer definitions omitted, assumed unchanged)

@torch.jit.script
def ShiftedSoftPlus(x: torch.Tensor):
    return torch.nn.functional.softplus(x) - math.log(2.0)


class UpdateNode(torch.nn.Module):
    def __init__(
            self,
            edge_irreps_in: o3.Irreps,
            irreps_in: o3.Irreps,
            irreps_out: o3.Irreps,
            latent_dim: int,
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
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.edge_irreps_in = edge_irreps_in
        self.dtype = dtype
        self.device = device
        self.res_update = res_update

        self.register_buffer(
            "env_sum_normalizations",
            # dividing by sqrt(N)
            torch.as_tensor(avg_num_neighbors).rsqrt(),
        )

        self._env_weighter = E3ElementLinear(
            irreps_in=irreps_out,
            dtype=dtype,
            device=device,
        )

        assert irreps_out[0].ir.l == 0

        # here we adopt the graph attention's idea to generate the weights as the attention scores
        # self.latent_act = torch.nn.LeakyReLU()
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
            irreps_scalar, [act[ir.p] for _, ir in irreps_scalar],  # scalar
            irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
            irreps_gated  # gated tensors
        )

        self.tp = SO2_Linear(
            irreps_in=self.irreps_in + self.edge_irreps_in,
            irreps_out=self.activation.irreps_in,
            latent_dim=latent_dim,
            radial_emb=radial_emb,
            radial_channels=radial_channels,
            use_interpolation=use_interpolation_tp
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
            self.node_onehot_tp = TensorProduct(
                irreps_in1=self.irreps_out,
                irreps_in2=f'95x0e',
                irreps_out=self.irreps_out,
                instructions=instructions
            )

    def forward(self, latents, node_features, edge_features, atom_type, node_onehot, edge_index, edge_vector,
                active_edges, wigner_D_all):
        edge_center = edge_index[0]
        edge_neighbor = edge_index[1]

        new_node_features = node_features
        message, _ = self.tp(
            torch.cat(
                [new_node_features[edge_center[active_edges]], edge_features]
                , dim=-1), edge_vector[active_edges], latents[active_edges], wigner_D_all)  # full_out_irreps

        message = self.activation(message)
        message = self.lin_post(message)
        scalars = message[:, :self.irreps_out[0].dim]

        # get the attention scores
        # weights = self.env_embed_mlps(self.latent_act(latents[active_edges]))
        # weights = torch_geometric.utils.softmax(weights, edge_center[active_edges], num_nodes=node_features.shape[0])
        weights = self.env_embed_mlps(latents[active_edges])
        new_node_features = scatter(
            self._env_weighter(message, weights),
            edge_center[active_edges],
            dim=0,
        )

        if self.env_sum_normalizations.ndim < 1:
            norm_const = self.env_sum_normalizations
        else:
            norm_const = self.env_sum_normalizations[atom_type.flatten()].unsqueeze(-1)
        assert len(scalars.shape) == 2

        new_node_features = new_node_features * norm_const

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

        return node_features

