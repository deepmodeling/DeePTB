from typing import Optional, List, Union, Dict
import math
import os

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


class InitLayer(torch.nn.Module):
    """
    与之前的 InitLayer 完全一致，只是加了 ln_flag，用于控制是否使用 LayerNorm。
    """

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
        # ---- new flag ----
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

        # ---- LayerNorm 开关 ----
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


class UpdateNodeInFrame(torch.nn.Module):
    """
    Keep-in-Frame 版本的 UpdateNode。
    - in_frame_flag=True（默认）: 使用 keep-in-frame 策略，只在必要时旋转。
    - in_frame_flag=False : 每个 SO2_Linear 都 rotate_in & rotate_out，等价于每层都在全局坐标系中工作。
    - ln_flag: 控制是否使用 SeperableLayerNorm。
    """

    def __init__(
        self,
        node_irreps_in: o3.Irreps,
        edge_irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        latent_dim: int,
        norm_eps: float = 1e-8,
        radial_emb: bool = False,
        radial_channels: list = [128, 128],
        res_update: bool = True,
        use_interpolation_tp: bool = False,
        res_update_ratios: Optional[List[float]] = None,
        res_update_ratios_learnable: bool = False,
        avg_num_neighbors: Optional[float] = None,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
        # === SO2 flags ===
        tp_rotate_in: bool = True,
        tp_rotate_out: bool = True,
        # === new global flags ===
        ln_flag: bool = True,
        in_frame_flag: bool = True,
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

        self.register_buffer(
            "env_sum_normalizations",
            torch.as_tensor(avg_num_neighbors).rsqrt(),
        )

        # ---- LayerNorm with flag ----
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

        # SO2_Linear 的具体 rotate_in/out 行为由上层 LemInFrame 根据 in_frame_flag 决定
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
        edge_index: torch.Tensor,
        edge_vector: torch.Tensor,
        active_edges: torch.Tensor,
        wigner_D_all: Optional[torch.Tensor],
    ):
        edge_center = edge_index[0]
        edge_neighbor = edge_index[1]

        norm_node_features = self.sln_n(node_features)
        norm_edge_features = self.sln_e(edge_features)

        # 1. 计算 / 复用 Wigner-D
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

        # 2. 准备 TP 输入
        center_node = norm_node_features[edge_center[active_edges]]
        neighbor_node = norm_node_features[edge_neighbor[active_edges]]

        # keep-in-frame: 中间层 node 需手动转 local；全局模式下（in_frame_flag=False 时）
        # LemInFrame 会把 tp_rotate_in 设为 True，这里不会触发该分支。
        if self.in_frame_flag and (not self.tp_rotate_in):
            center_node = rotate_vector(center_node, self.node_irreps_in, wigner_D_all, back=False)
            neighbor_node = rotate_vector(neighbor_node, self.node_irreps_in, wigner_D_all, back=False)
            # edge_features 已经在 local frame，无需处理

        edge_input = torch.cat([center_node, norm_edge_features, neighbor_node], dim=-1)

        # 3. SO2 Tensor Product
        edge_messages, wigner_D_all = self.tp(
            edge_input,
            edge_vector[active_edges],
            latents[active_edges],
            wigner_D_all,
        )

        edge_messages = self.activation(edge_messages)
        edge_messages = self.lin_post(edge_messages)

        # 4. 节点更新（需要全局帧）
        msg_for_node = edge_messages
        if self.in_frame_flag and (not self.tp_rotate_out):
            # keep-in-frame: 如果 TP 输出保持在 local，这里手动转回 global
            msg_for_node = rotate_vector(edge_messages, self.irreps_out, wigner_D_all, back=True)

        weights = self.env_embed_mlps(latents[active_edges])
        edge_messages_weighted = self._env_weighter(msg_for_node, weights)

        aggregated_node_messages = scatter(edge_messages_weighted, edge_center[active_edges], dim=0)

        if self.env_sum_normalizations.ndim < 1:
            norm_const = self.env_sum_normalizations
        else:
            norm_const = self.env_sum_normalizations[atom_type.flatten()].unsqueeze(-1)

        new_node_features = aggregated_node_messages * norm_const

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

        # 返回：
        # - node_features: 始终在 global frame
        # - edge_messages: 如果 in_frame_flag=True，则中间层为 local，首尾层为 global；
        #                  如果 in_frame_flag=False，则各层总是 global。
        return node_features, edge_messages, wigner_D_all


class PoolingLayer(torch.nn.Module):
    """
    Pooling layer that performs self full tensor product on node and edge features,
    then adds a residual connection with the original features.
    """

    def __init__(
        self,
        node_irreps_in: o3.Irreps,
        edge_irreps_in: o3.Irreps,
        out_irreps: o3.Irreps,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(PoolingLayer, self).__init__()

        self.node_irreps_in = o3.Irreps(node_irreps_in)
        self.edge_irreps_in = o3.Irreps(edge_irreps_in)
        self.out_irreps = o3.Irreps(out_irreps)
        self.dtype = dtype
        self.device = device

        # Self full tensor product for node features
        self.node_self_tp = FullyConnectedTensorProduct(
            irreps_in1=self.node_irreps_in,
            irreps_in2=self.node_irreps_in,
            irreps_out=self.out_irreps,
        )

        # Self full tensor product for edge features
        self.edge_self_tp = FullyConnectedTensorProduct(
            irreps_in1=self.edge_irreps_in,
            irreps_in2=self.edge_irreps_in,
            irreps_out=self.out_irreps,
        )

        # Linear projection for residual connection
        self.node_res_linear = Linear(
            self.node_irreps_in,
            self.out_irreps,
            shared_weights=True,
            internal_weights=True,
            biases=True,
        )

        self.edge_res_linear = Linear(
            self.edge_irreps_in,
            self.out_irreps,
            shared_weights=True,
            internal_weights=True,
            biases=True,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
    ):
        # Self full tensor product
        node_tp_out = self.node_self_tp(node_features, node_features)
        edge_tp_out = self.edge_self_tp(edge_features, edge_features)

        # Residual connection
        node_res = self.node_res_linear(node_features)
        edge_res = self.edge_res_linear(edge_features)

        # Add residual (short connection)
        out_node_features = node_tp_out + node_res
        out_edge_features = edge_tp_out + edge_res

        return out_node_features, out_edge_features


@Embedding.register("lem_in_frame")
class LemInFrame(torch.nn.Module):
    """
    带有几个全局 flag 的 LemInFrame：
    - ln_flag: 控制是否启用所有 SeperableLayerNorm（默认 True）。
    - in_frame_flag:
        * True（默认）: keep-in-frame 模式（当前实现的默认行为）
        * False       : 每个 SO2_Linear 都 rotate_in & rotate_out，等价于旧版 Lem 的全局坐标行为
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
        res_update: bool = True,
        res_update_ratios: Optional[List[float]] = None,
        res_update_ratios_learnable: bool = False,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
        universal: Optional[bool] = False,
        use_interpolation_out: Optional[bool] = True,
        prune_edges_by_cutoff: bool = True,
        prune_log_path: Optional[str] = None,
        # ---- new flags ----
        ln_flag: bool = True,
        in_frame_flag: bool = True,
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
            # ---- 决定本层的 node/edge 输入 irreps ----
            if i == 0:
                irreps_in_layer = self.init_layer.irreps_out
            else:
                irreps_in_layer = irreps_hidden

            # ---- 决定 keep-in-frame 下的 rotate_in/out; 若关闭 in_frame_flag，则所有层都 rotate_in & rotate_out ----
            if self.in_frame_flag:
                if i == 0:
                    # 第 0 层：Global In -> TP 内部 Rotate In -> Local Out（不 Rotate Out）
                    rotate_in = True
                    rotate_out = False
                else:
                    # 中间层默认输入为 Local
                    rotate_in = False
                    if i == n_layers - 1:
                        # 最后一层：Local In -> TP 内部 Rotate Out -> Global Out
                        rotate_out = True
                    else:
                        # 中间层：Local In & Local Out
                        rotate_out = False
            else:
                # 关闭 keep-in-frame：所有层都在 TP 内部完成 rotate_in & rotate_out
                rotate_in = True
                rotate_out = True

            # ---- 决定本层输出 irreps 及是否使用 interpolation TP ----
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
                    norm_eps=norm_eps,
                    radial_emb=tp_radial_emb,
                    radial_channels=tp_radial_channels,
                    res_update=res_update,
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
                )
            )

            current_irreps = irreps_out_layer
            if use_interpolation_tp:
                print(f"Use interpolation SO2 layer in layer {i}")

        self.node_irreps_out = current_irreps
        self.edge_irreps_out = current_irreps

        # ---- Pooling Layer ----
        self.pooling_layer = PoolingLayer(
            node_irreps_in=self.node_irreps_out,
            edge_irreps_in=self.edge_irreps_out,
            out_irreps=self.idp.orbpair_irreps,
            dtype=dtype,
            device=device,
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

        data[_keys.EDGE_OVERLAP_KEY] = latents

        wigner_D_all = None
        for layer in self.layers:
            node_features, edge_features, wigner_D_all = layer(
                latents,
                node_features,
                edge_features,
                atom_type,
                edge_index,
                edge_vector,
                active_edges,
                wigner_D_all,
            )

        # ---- Apply Pooling Layer with self full tp and residual connection ----
        out_node_features, out_edge_features = self.pooling_layer(node_features, edge_features)

        # pad zero for inactive nodes and edges
        if out_node_features.shape[0] < num_nodes_total:
            pad = torch.zeros(
                num_nodes_total - out_node_features.shape[0],
                out_node_features.shape[1],
                device=out_node_features.device,
                dtype=out_node_features.dtype,
            )
            out_node_features = torch.cat([out_node_features, pad], dim=0)

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