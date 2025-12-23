from typing import Optional, List, Union, Dict, Tuple
import math
import torch
import torch.nn as nn
from torch_runstats.scatter import scatter
from e3nn import o3
from e3nn.nn import Gate
from e3nn.o3 import (
    Linear,
    SphericalHarmonics,
    FullyConnectedTensorProduct,
)
try:
    import openequivariance as oeq  # Import OpenEquivariance
except:
    pass
from tqdm import tqdm

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
from .lem import InitLayer as InitLayer  # 假设原版在同路径

import os

os.environ["MAX_JOBS"] = "64"
# --- 工具函数 ---
def get_feasible_tp(
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        filter_irreps_out: o3.Irreps,
        tp_mode: str = "uvw",
        trainable: bool = True
):
    """
    Utility for cutomizing e3nn.TensorProduct instructions.
    Adapted for OpenEquivariance compatibility.
    """
    assert tp_mode in ["uvw", "uvu", "uvv", "uuw", "uuu", "uvuv"]
    irreps_mid = []
    instructions = []

    # 1. Generate path candidates
    for i, (mul_1, ir_in1) in enumerate(irreps_in1):
        for j, (mul_2, ir_in2) in enumerate(irreps_in2):
            for ir_out in ir_in1 * ir_in2:
                if ir_out in filter_irreps_out:
                    if tp_mode == "uvw":
                        mul_out = filter_irreps_out.count(ir_out)
                    elif tp_mode == "uvu":
                        mul_out = mul_1
                    elif tp_mode == "uvv":
                        mul_out = mul_2
                    elif tp_mode == "uuu":
                        assert mul_1 == mul_2
                        mul_out = mul_1
                    elif tp_mode == "uuw":
                        assert mul_1 == mul_2
                        mul_out = filter_irreps_out.count(ir_out)
                    elif tp_mode == "uvuv":
                        mul_out = mul_1 * mul_2
                    else:
                        raise NotImplementedError(f"Unsupported TP mode: {tp_mode}")

                    if (mul_out, ir_out) not in irreps_mid:
                        k = len(irreps_mid)
                        irreps_mid.append((mul_out, ir_out))
                    else:
                        k = irreps_mid.index((mul_out, ir_out))
                    instructions.append((i, j, k, tp_mode, trainable))

    irreps_mid = o3.Irreps(irreps_mid)

    # 2. Calculate normalization coefficients
    normalization_coefficients = []
    for ins in instructions:
        ins_i, ins_j, ins_k, ins_mode, _ = ins
        ins_dict = {
            "uvw": irreps_in1[ins_i].mul * irreps_in2[ins_j].mul,
            "uvu": irreps_in2[ins_j].mul,
            "uvv": irreps_in1[ins_i].mul,
            "uuw": irreps_in1[ins_i].mul,
            "uuu": 1,
            "uvuv": 1,
        }
        alpha = irreps_mid[ins_k].ir.dim
        x = sum([ins_dict[sub_ins[3]] for sub_ins in instructions if sub_ins[2] == ins_k])
        if x > 0.0:
            alpha /= x
        normalization_coefficients.append(math.sqrt(alpha))

    # 3. Sort irreps and remap instructions
    irreps_mid, p, _ = irreps_mid.sort()
    final_instructions = []
    for (i_in1, i_in2, i_out, mode, train), alpha in zip(instructions, normalization_coefficients):
        final_instructions.append((i_in1, i_in2, p[i_out], mode, train, alpha))

    return irreps_mid, final_instructions


class OEQFullyConnectedTP(nn.Module):
    def __init__(self, irreps_in1, irreps_in2, irreps_out):
        super().__init__()
        # 1. 使用第一份代码那个“安全”的函数
        # 注意：这里会返回带有 alpha (归一化系数) 的 6 元组指令
        self.irreps_mid, instructions = get_feasible_tp(
            irreps_in1, irreps_in2, irreps_out, tp_mode="uvw"
        )

        # 2. OEQ 的 TPProblem
        # 注意：OEQ 的 TPProblem 接受的指令格式可以是 5 元组或 6 元组
        # 如果是 6 元组，最后一个值会被当作 path_weight (alpha)
        self.problem = oeq.TPProblem(
            irreps_in1, irreps_in2, self.irreps_mid, instructions,
            shared_weights=True,  # 模拟 internal weights
            internal_weights=False
        )

        self.tp = oeq.TensorProduct(self.problem, torch_op=True)

        # 3. 权重初始化
        self.weights = nn.Parameter(torch.randn(self.problem.weight_numel))
        # 因为 get_feasible_tp_v1 已经计算了 alpha 归一化，
        # 这里的权重通常初始化为标准正态分布即可，或者轻微缩放
        with torch.no_grad():
            self.weights.normal_(0, 1.0)

            # 4. 后处理线性层
        self.post_linear = o3.Linear(self.irreps_mid, irreps_out)

    def forward(self, x1, x2):
        out = self.tp(x1, x2, self.weights)
        return self.post_linear(out)

# --- UpdateNode ---
class UpdateNode(torch.nn.Module):
    def __init__(self, edge_irreps_in, irreps_in, irreps_out, latent_dim, avg_num_neighbors, tp_class, res_update,
                 sh_irreps, **kwargs):
        super().__init__()
        self.irreps_out = irreps_out
        self.register_buffer("env_sum_norm", torch.as_tensor(avg_num_neighbors).rsqrt())

        self.sln_n = SeperableLayerNorm(irreps_in)
        self.sln_e = SeperableLayerNorm(edge_irreps_in)

        # Gate Activation
        irreps_scalar = o3.Irreps([(mul, ir) for mul, ir in self.irreps_out if ir.l == 0]).simplify()
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_out if ir.l > 0]).simplify()
        irreps_gates = o3.Irreps([(mul, "0e") for mul, _ in irreps_gated]).simplify()
        self.activation = Gate(irreps_scalar, [torch.nn.functional.silu for _ in irreps_scalar],
                               irreps_gates, [torch.sigmoid for _ in irreps_gates], irreps_gated)

        # Full TP: (Node + Edge) x SH -> Activation Input
        tp_in = irreps_in + edge_irreps_in
        if tp_class == FullyConnectedTensorProduct:
            self.tp = tp_class(tp_in, sh_irreps, self.activation.irreps_in, internal_weights=True)
        else:
            self.tp = tp_class(tp_in, sh_irreps, self.activation.irreps_in)

        self.tp.irreps_in = tp_in
        self.tp.irreps_out = self.activation.irreps_in
        self.lin_post = Linear(self.activation.irreps_out, self.irreps_out, internal_weights=True)
        self._env_weighter = E3ElementLinear(irreps_in=self.irreps_out)
        self.env_embed_mlps = ScalarMLPFunction(mlp_input_dimension=latent_dim, mlp_latent_dimensions=[],
                                                mlp_output_dimension=self._env_weighter.weight_numel)

        self.res_update = res_update
        if res_update:
            self.linear_res = Linear(irreps_in, irreps_out, internal_weights=True)

    def forward(self, latents, node_features, edge_features, edge_sh, edge_index, atom_type, active_edges):
        edge_center = edge_index[0][active_edges]
        node_input = torch.cat([self.sln_n(node_features)[edge_center], self.sln_e(edge_features)], dim=-1)

        message = self.tp(node_input, edge_sh)
        message = self.activation(message)
        message = self.lin_post(message)

        # Latent weighting
        weights = self.env_embed_mlps(latents[active_edges])
        message = self._env_weighter(message, weights)

        new_node_features = scatter(message, edge_center, dim=0, dim_size=node_features.shape[0]) * self.env_sum_norm

        if self.res_update:
            return self.linear_res(node_features) + new_node_features
        return new_node_features


# --- UpdateEdge ---
class UpdateEdge(torch.nn.Module):
    def __init__(self, num_types, node_irreps_in, irreps_in, irreps_out, latent_dim, latent_channels, tp_class,
                 res_update, sh_irreps, **kwargs):
        super().__init__()
        self.irreps_out = irreps_out
        self.sln_n = SeperableLayerNorm(node_irreps_in)
        self.sln_e = SeperableLayerNorm(irreps_in)
        self.ln_lat = torch.nn.LayerNorm(latent_dim)

        irreps_scalar = o3.Irreps([(mul, ir) for mul, ir in self.irreps_out if ir.l == 0]).simplify()
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_out if ir.l > 0]).simplify()
        irreps_gates = o3.Irreps([(mul, "0e") for mul, _ in irreps_gated]).simplify()
        self.activation = Gate(irreps_scalar, [torch.nn.functional.silu for _ in irreps_scalar],
                               irreps_gates, [torch.sigmoid for _ in irreps_gates], irreps_gated)

        # Full TP: (CenterNode + Edge + NeighborNode) x SH -> Activation Input
        tp_in = node_irreps_in + irreps_in + node_irreps_in
        if tp_class == FullyConnectedTensorProduct:
            self.tp = tp_class(tp_in, sh_irreps, self.activation.irreps_in, internal_weights=True)
        else:
            self.tp = tp_class(tp_in, sh_irreps, self.activation.irreps_in)

        self.tp.irreps_in = tp_in
        self.tp.irreps_out = self.activation.irreps_in
        self.lin_post = Linear(self.activation.irreps_out, self.irreps_out, internal_weights=True)
        self._edge_weighter = E3ElementLinear(irreps_in=self.irreps_out)
        self.edge_embed_mlps = ScalarMLPFunction(mlp_input_dimension=latent_dim, mlp_latent_dimensions=[],
                                                 mlp_output_dimension=self._edge_weighter.weight_numel)

        # Latents Update Logic (Same as original)
        self.latents_mlp = ScalarMLPFunction(
            mlp_input_dimension=latent_dim + self.activation.irreps_in[0].dim + 2 * num_types,
            mlp_output_dimension=latent_dim, mlp_latent_dimensions=latent_channels
        )

        self.res_update = res_update
        if res_update:
            self.linear_res = Linear(irreps_in, irreps_out, internal_weights=True)

    def forward(self, latents, node_features, node_onehot, edge_features, edge_sh, edge_index, cutoff_coeffs,
                active_edges):
        idx_i, idx_j = edge_index[0][active_edges], edge_index[1][active_edges]
        node_i, node_j = self.sln_n(node_features)[idx_i], self.sln_n(node_features)[idx_j]

        combined = torch.cat([node_i, self.sln_e(edge_features), node_j], dim=-1)

        # TP and Activation
        raw_msg = self.tp(combined, edge_sh)
        scalars_for_lat = raw_msg[:, :self.tp.irreps_out[0].dim]

        message = self.activation(raw_msg)
        message = self.lin_post(message)

        # Latent weighting for edge features
        weights = self.edge_embed_mlps(latents[active_edges])
        edge_out = self._edge_weighter(message, weights)

        # Update Latents
        lat_in = torch.cat(
            [node_onehot[idx_i], self.ln_lat(latents[active_edges]), scalars_for_lat, node_onehot[idx_j]], dim=-1)
        new_latents_active = self.latents_mlp(lat_in) * cutoff_coeffs[active_edges].unsqueeze(-1)

        latents_out = torch.index_copy(latents, 0, active_edges, new_latents_active)

        if self.res_update:
            return self.linear_res(edge_features) + edge_out, latents_out
        return edge_out, latents_out


# --- Layer Container ---
class Layer(torch.nn.Module):
    def __init__(self, num_types, avg_num_neighbors, irreps_in, irreps_out, latent_dim, latent_channels, tp_class,
                 res_update, sh_irreps, **kwargs):
        super().__init__()
        self.edge_update = UpdateEdge(num_types, irreps_in, irreps_in, irreps_out, latent_dim, latent_channels,
                                      tp_class, res_update, sh_irreps)
        self.node_update = UpdateNode(irreps_out, irreps_in, irreps_out, latent_dim, avg_num_neighbors, tp_class,
                                      res_update, sh_irreps)

    def forward(self, latents, node_features, edge_features, node_onehot, edge_index, edge_sh, atom_type, cutoff_coeffs,
                active_edges):
        edge_features, latents = self.edge_update(latents, node_features, node_onehot, edge_features, edge_sh,
                                                  edge_index, cutoff_coeffs, active_edges)
        node_features = self.node_update(latents, node_features, edge_features, edge_sh, edge_index, atom_type,
                                         active_edges)
        return latents, node_features, edge_features


# --- Main Main Main: Lem ---
@Embedding.register("lem_full_tp")
class LemFullTP(torch.nn.Module):
    def __init__(
            self, basis: Dict = None, idp: OrbitalMapper = None, n_layers: int = 3, n_radial_basis: int = 10,
            r_max: float = 5.0, irreps_hidden: o3.Irreps = None, avg_num_neighbors: Optional[float] = None,
            latent_dim: int = 128, latent_channels: List[int] = [128, 128], res_update: bool = True,
            universal: bool = False, device="cpu", dtype=torch.float32, **kwargs
    ):
        super().__init__()
        self.idp = idp if idp else OrbitalMapper(basis, method="e3tb")
        self.idp.get_irreps(no_parity=False)
        self.device, self.dtype = device, dtype

        irreps_hidden = o3.Irreps(irreps_hidden)
        irreps_sh = o3.Irreps.spherical_harmonics(irreps_hidden.lmax)
        orbpair_irreps = self.idp.orbpair_irreps.sort()[0].simplify()

        self.sh = SphericalHarmonics(irreps_sh, True, "component")
        self.onehot = OneHotAtomEncoding(num_types=95 if universal else len(self.idp.basis), idp=self.idp,
                                         universal=universal)
        self.n_atom = int(self.onehot.num_types)

        # InitLayer Style preserved
        self.init_layer = InitLayer(
            idp=self.idp, num_types=self.n_atom, n_radial_basis=n_radial_basis, r_max=r_max,
            avg_num_neighbors=avg_num_neighbors, irreps_sh=irreps_sh, env_embed_multiplicity=32,
            latent_dim=latent_dim, two_body_latent_channels=latent_channels, device=device, dtype=dtype
        )

        self.layers = torch.nn.ModuleList()
        print(f"Building {self.__class__.__name__} with {n_layers} layers...")
        for i in tqdm(range(n_layers)):
            # Irreps流转控制
            in_layer = self.init_layer.irreps_out if i == 0 else irreps_hidden
            out_layer = orbpair_irreps if i == n_layers - 1 else irreps_hidden

            self.layers.append(Layer(
                num_types=self.n_atom, avg_num_neighbors=avg_num_neighbors,
                irreps_in=in_layer, irreps_out=out_layer, latent_dim=latent_dim,
                latent_channels=latent_channels, tp_class=self.get_tp_class(),
                res_update=res_update, sh_irreps=irreps_sh
            ))

    @property
    def out_edge_irreps(self):
        return self.idp.orbpair_irreps

    @property
    def out_node_irreps(self):
        return self.idp.orbpair_irreps

    def get_tp_class(self):
        return FullyConnectedTensorProduct

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = with_edge_vectors(data, with_lengths=True)
        data = with_batch(data)
        edge_index, edge_length = data[_keys.EDGE_INDEX_KEY], data[_keys.EDGE_LENGTH_KEY]
        edge_sh = self.sh(data[_keys.EDGE_VECTORS_KEY][:, [1, 2, 0]])

        data = self.onehot(data)
        node_oh, atom_ty, bond_ty = data[_keys.NODE_ATTRS_KEY], data[_keys.ATOM_TYPE_KEY].flatten(), data[
            _keys.EDGE_TYPE_KEY].flatten()

        latents, node_features, edge_features, cutoff, active_edges = self.init_layer(
            edge_index, atom_ty, bond_ty, edge_sh, edge_length, node_oh
        )

        for layer in self.layers:
            latents, node_features, edge_features = layer(
                latents, node_features, edge_features, node_oh, edge_index,
                edge_sh[active_edges], atom_ty, cutoff, active_edges
            )

        data[_keys.NODE_FEATURES_KEY] = node_features
        full_edge_feat = torch.zeros(edge_index.shape[1], edge_features.shape[-1], device=edge_features.device,
                                     dtype=edge_features.dtype)
        full_edge_feat[active_edges] = edge_features
        data[_keys.EDGE_FEATURES_KEY] = full_edge_feat
        data[_keys.EDGE_OVERLAP_KEY] = latents
        return data


# --- OpenEqui Version Class ---
@Embedding.register("lem_full_tp_oeq")
class LemFullTPOpenEqui(LemFullTP):
    def get_tp_class(self):
        return OEQFullyConnectedTP