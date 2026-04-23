from typing import Optional, List, Union, Dict, Tuple
import math
import functools
import torch
from torch_runstats.scatter import scatter
from torch import fx
from e3nn import o3
from torch_scatter import scatter_mean
from e3nn.o3 import Linear, SphericalHarmonics, FullyConnectedTensorProduct, TensorProduct
from dptb.data import AtomicDataDict
from dptb.nn.embedding.emb import Embedding
from ..radial_basis import BesselBasis
from ..base import ScalarMLPFunction
from dptb.nn.embedding.from_deephe3.deephe3 import tp_path_exists
from dptb.data import _keys
from dptb.nn.cutoff import cosine_cutoff, polynomial_cutoff
from dptb.nn.rescale import E3ElementLinear
from .lem_moe_v3_plugins import (
    EqV3StyleNodeFFN,
    FlatSwiGLUS2Merge,
    build_equivariant_norm,
    build_gate_activation,
    can_use_flat_s2_patch,
)
# Note: Modified SO2_Linear and MOLE classes imported here
from dptb.nn.tensor_product_moe_v3 import SO2_Linear, MOLEGlobals, MOLERouterV3
import math
from dptb.data.transforms import OrbitalMapper
from ..type_encode.one_hot import OneHotAtomEncoding, OneHotEdgeEmbedding
from dptb.data.AtomicDataDict import with_edge_vectors, with_batch

from math import ceil

import logging

log = logging.getLogger(__name__)


@Embedding.register("lem_moe_v3")
class LemMoEV3(torch.nn.Module):
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
            equivariant_norm_type: str = "none",
            hidden_edge_activation_type: str = "gate",
            hidden_node_activation_type: str = "gate",
            swiglu_s2_grid_resolution: Tuple[int, int] = (14, 14),
            swiglu_s2_compat_mode: str = "modern",
            ffn_hidden_factor: float = 0.0,
            ffn_apply_to_last: bool = False,
            so2_wigner_apply_mode: str = "compact_blocks",
            so2_fusion_mode: str = "staged",
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
            universal: Optional[bool] = False,
            use_interpolation_out: Optional[bool] = True,
            # MOE parameters
            num_experts: int = 8,
            num_shared_experts: int = 1,
            top_k: Optional[int] = 1,
            **kwargs,
    ):

        super(LemMoEV3, self).__init__()

        irreps_hidden = o3.Irreps(irreps_hidden)
        lmax = irreps_hidden.lmax
        self.num_experts = num_experts

        # 使用 log.info 打印参数
        log.info(f'[LemMoEV3] Initialized DeepSeek-V3 Style MoE.')
        log.info(f'  - Num Shared Experts: {num_shared_experts}')
        log.info(f'  - Num Routed Experts: {self.num_experts}')
        log.info(f'  - Top-K Actived Routed Experts: {top_k}')
        log.info(f'  - Strategy: Shared Expert + Aux-Loss-Free Balancing (Sigmoid Routing)')
        if ffn_hidden_factor > 1.0:
            log.info(
                f"  - EqV3 SO3 Grid FFN: enabled (hidden_factor={ffn_hidden_factor}, "
                f"apply_to_last={ffn_apply_to_last})"
            )
        else:
            log.info("  - EqV3 SO3 Grid FFN: disabled")
        mean_max_prob_lower_bound = 1.0 / self.num_experts
        # 上界：One-Hot 分布，max为 1.0，平方为 1.0
        mean_max_prob_upper_bound = 1.0

        log.info(f"[LemMoEV3] Theoretical mean_max_prob Bounds -> "
                 f"Min (Uniform): {mean_max_prob_lower_bound:.6f} | Max (One-Hot): {mean_max_prob_upper_bound:.6f}")
        cv_lower_bound = 0.0
        cv_upper_bound = math.sqrt((self.num_experts - top_k) / top_k) if top_k else 0.0
        log.info(f"[LemMoEV3] Theoretical expert_load_cv Bounds -> "
                 f"Min (Balanced): {cv_lower_bound:.6f} | Max (Collapsed): {cv_upper_bound:.6f}")

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

        latent_kwargs = {
            "mlp_latent_dimensions": latent_channels + [latent_dim],
            "mlp_nonlinearity": "silu",
            "mlp_initialization": "uniform"
        },
        self.latent_dim = latent_dim

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

        assert all(ir in irreps_out for _, ir in
                   orbpair_irreps), "hidden irreps should at least cover all the reqired irreps in the hamiltonian data {}".format(
            orbpair_irreps)

        self.sh = SphericalHarmonics(
            irreps_sh, sh_normalized, sh_normalization
        )
        self.onehot = OneHotAtomEncoding(num_types=self.n_atom, set_features=False, idp=self.idp, universal=universal)
        self.edge_one_hot = OneHotEdgeEmbedding(num_types=self.n_atom, idp=self.idp, universal=universal,
                                                d_emb=edge_one_hot_dim)

        # --- MOE Router V3 (DeepSeek Style) ---
        self.router = MOLERouterV3(
            in_features=self.n_atom,
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_free=True,  # 开启 DeepSeek 负载均衡
            bias_update_speed=0.005
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
            norm_eps=norm_eps
        )

        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                irreps_in = self.init_layer.irreps_out
            else:
                irreps_in = irreps_hidden

            if i == n_layers - 1:
                irreps_out = orbpair_irreps.sort()[0].simplify()
                use_interpolation_tp = bool(use_interpolation_out)
            else:
                irreps_out = irreps_hidden
                use_interpolation_tp = False

            if i == n_layers - 1:
                edge_activation_type = "gate"
                node_activation_type = "gate"
            else:
                edge_activation_type = hidden_edge_activation_type
                node_activation_type = hidden_node_activation_type

            use_node_ffn = ffn_hidden_factor > 1.0 and ((i < n_layers - 1) or ffn_apply_to_last)

            self.layers.append(Layer(
                num_types=self.n_atom,
                avg_num_neighbors=avg_num_neighbors,
                irreps_in=irreps_in,
                irreps_out=irreps_out,
                tp_radial_emb=tp_radial_emb,
                tp_radial_channels=tp_radial_channels,
                use_layer_onehot_tp=use_layer_onehot_tp,
                edge_one_hot_dim=edge_one_hot_dim,
                latent_channels=latent_channels,
                latent_dim=latent_dim,
                res_update=res_update,
                res_update_ratios=res_update_ratios,
                res_update_ratios_learnable=res_update_ratios_learnable,
                equivariant_norm_type=equivariant_norm_type,
                edge_activation_type=edge_activation_type,
                node_activation_type=node_activation_type,
                swiglu_s2_grid_resolution=swiglu_s2_grid_resolution,
                swiglu_s2_compat_mode=swiglu_s2_compat_mode,
                ffn_hidden_factor=ffn_hidden_factor,
                use_node_ffn=use_node_ffn,
                so2_wigner_apply_mode=so2_wigner_apply_mode,
                so2_fusion_mode=so2_fusion_mode,
                dtype=dtype,
                device=device,
                use_interpolation_tp=use_interpolation_tp,
                num_experts=num_experts,
                num_shared_experts=num_shared_experts,  # Pass down to Layer -> SO2_Linear
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
        preserved_split_sizes = data.get(_keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY, None)
        if preserved_split_sizes is not None:
            data = data.copy()
            data.pop(_keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY, None)
        data = with_edge_vectors(data, with_lengths=True)
        data = with_batch(data)
        if preserved_split_sizes is not None:
            data[_keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY] = preserved_split_sizes

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

        # --- MOLE Routing Logic ---
        # 1. Global Feature per system: Mean of node one-hot
        global_feat = scatter_mean(node_one_hot, batch, dim=0)  # [Batch, n_atom]

        # 2. Compute Routing Coefficients
        # 返回: coeffs [Batch, Num_Experts], monitor_val (mean max prob)
        coeffs, monitor_val, expert_load_cv = self.router(global_feat)

        # 不再记录 z_loss，改为记录监控指标 mean_max_prob
        # 这个值越接近 1.0 表示路由越自信，接近 0.5 (TopK=1时) 表示犹豫
        data["mean_max_prob"] = monitor_val
        data["expert_load_cv"] = expert_load_cv
        # 3. Prepare MOLEGlobals
        num_nodes_total = node_one_hot.shape[0]
        precomputed_active_edges = data.get(_keys.LEM_ACTIVE_EDGES_KEY, None)
        precomputed_cutoff_coeffs = data.get(_keys.LEM_CUTOFF_COEFFS_KEY, None)
        precomputed_split_sizes = data.get(_keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY, None)
        if precomputed_cutoff_coeffs is not None and edge_length.requires_grad:
            raise RuntimeError(
                "Precomputed LEM cutoff coefficients cannot be used when edge_length requires gradients. "
                "Set train_options.precompute_lem_cutoff_coeffs=false for force/stress/virial training."
            )
        latents, node_features, edge_features, cutoff_coeffs, active_edges = self.init_layer(edge_index, atom_type,
                                                                                             bond_type, edge_sh,
                                                                                             edge_length, edge_one_hot,
                                                                                             precomputed_active_edges,
                                                                                             precomputed_cutoff_coeffs)

        n_active_nodes = node_features.shape[0]
        if n_active_nodes < num_nodes_total:
            safe_node_one_hot = node_one_hot[:n_active_nodes]
        else:
            safe_node_one_hot = node_one_hot

        edge_one_hot = edge_one_hot[active_edges]

        # Determine sizes for active edges for Weight Merging in MOLELinear
        if precomputed_split_sizes is not None:
            mole_globals = MOLEGlobals(coefficients=coeffs, split_sizes=precomputed_split_sizes)
        else:
            edge_batch = batch[edge_index[0][active_edges]]  # Map edge to graph index
            num_systems = coeffs.shape[0]
            edge_sizes = torch.bincount(edge_batch, minlength=num_systems)
            mole_globals = MOLEGlobals(coefficients=coeffs, sizes=edge_sizes)
        # --------------------------

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
                    mole_globals  # Pass globals to layers
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

        data.pop(_keys.LEM_ACTIVE_EDGES_KEY, None)
        data.pop(_keys.LEM_ACTIVE_EDGE_SPLIT_SIZES_KEY, None)
        data.pop(_keys.LEM_CUTOFF_COEFFS_KEY, None)
        return data

@torch.jit.script
def ShiftedSoftPlus(x: torch.Tensor):
    return torch.nn.functional.softplus(x) - math.log(2.0)


def _cosine_cutoff_per_edge(
    x: torch.Tensor, r_max: torch.Tensor, r_start_cos_ratio: float = 0.8
) -> torch.Tensor:
    r_decay = r_start_cos_ratio * r_max
    x = torch.minimum(torch.maximum(x, r_decay), r_max)
    return 0.5 * (torch.cos((math.pi / (r_max - r_decay)) * (x - r_decay)) + 1.0)


def _polynomial_cutoff_per_edge(
    x: torch.Tensor, r_max: torch.Tensor, p: float = 6.0
) -> torch.Tensor:
    assert p >= 2.0
    x = x / r_max
    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p))
    out = out + (p * (p + 2.0) * torch.pow(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * torch.pow(x, p + 2.0))
    return out * (x < 1.0)


class InitLayer(torch.nn.Module):
    def __init__(
            self,
            # required params
            idp,
            num_types: int,
            n_radial_basis: int,
            r_max: float,
            avg_num_neighbors: Optional[float] = None,
            irreps_sh: o3.Irreps = None,
            env_embed_multiplicity: int = 32,
            # MLP parameters:
            two_body_latent_channels: list = [128, 128],
            latent_dim: int = 128,
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
            max_r_max_value = float(r_max)
            r_max_tensor = torch.tensor(r_max, device=device, dtype=dtype)
            self.r_max_dict = None
        elif isinstance(r_max, dict):
            c_set = set(list(r_max.values()))
            max_r_max_value = max(list(r_max.values()))
            r_max_tensor = torch.tensor(max_r_max_value, device=device, dtype=dtype)
            if len(r_max) == 1 or len(c_set) == 1:
                self.r_max_dict = None
            else:
                self.r_max_dict = {}
                for k, v in r_max.items():
                    self.r_max_dict[k] = torch.tensor(v, device=device, dtype=dtype)
        else:
            raise TypeError("r_max should be either float, int or dict")

        self.idp = idp
        self.register_buffer("r_max", r_max_tensor)
        self._r_max_cpu = r_max_tensor.detach().cpu()
        r_max_by_edge_type = None
        r_max_edge_type_valid = None
        if self.r_max_dict is not None:
            max_edge_type = max(int(v) for v in self.idp.bond_to_type.values())
            edge_type_count = max(max_edge_type + 1, int(num_types) * int(num_types))
            r_max_by_edge_type = torch.zeros(edge_type_count, device=device, dtype=dtype)
            r_max_edge_type_valid = torch.zeros(edge_type_count, device=device, dtype=torch.bool)
            for bond, ty in self.idp.bond_to_type.items():
                iatom, jatom = bond.split("-")
                if iatom not in self.r_max_dict or jatom not in self.r_max_dict:
                    continue
                r_max_by_edge_type[int(ty)] = 0.5 * (self.r_max_dict[iatom] + self.r_max_dict[jatom])
                r_max_edge_type_valid[int(ty)] = True
        self.register_buffer("r_max_by_edge_type", r_max_by_edge_type)
        self.register_buffer("r_max_edge_type_valid", r_max_edge_type_valid)
        self._r_max_by_edge_type_cpu = (
            None if r_max_by_edge_type is None else r_max_by_edge_type.detach().cpu()
        )
        self._r_max_edge_type_valid_cpu = (
            None if r_max_edge_type_valid is None else r_max_edge_type_valid.detach().cpu()
        )
        self.r_start_cos_ratio = r_start_cos_ratio
        self.polynomial_cutoff_p = PolynomialCutoff_p
        self.cutoff_type = cutoff_type
        self.device = device
        self.dtype = dtype
        self.irreps_out = o3.Irreps([(env_embed_multiplicity, ir) for _, ir in irreps_sh])

        assert all(mul == 1 for mul, _ in irreps_sh)
        assert (
                irreps_sh[0].ir == SCALAR
        ), "env_embed_irreps must start with scalars"

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

        self.bessel = BesselBasis(r_max=float(max_r_max_value), num_basis=n_radial_basis, trainable=True)

    def _r_max_for(self, edge_length: torch.Tensor) -> torch.Tensor:
        if edge_length.device.type == "cpu":
            return self._r_max_cpu.to(dtype=edge_length.dtype)
        return self.r_max.to(device=edge_length.device, dtype=edge_length.dtype)

    def _r_max_tables_for(self, edge_length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if edge_length.device.type == "cpu":
            return (
                self._r_max_by_edge_type_cpu.to(dtype=edge_length.dtype),
                self._r_max_edge_type_valid_cpu,
            )
        return (
            self.r_max_by_edge_type.to(device=edge_length.device, dtype=edge_length.dtype),
            self.r_max_edge_type_valid.to(device=edge_length.device),
        )

    def cutoff_coefficients(self, edge_length: torch.Tensor, bond_type: torch.Tensor) -> torch.Tensor:
        if self.r_max_dict is None:
            r_max = self._r_max_for(edge_length)
            if self.cutoff_type == "cosine":
                cutoff_coeffs = cosine_cutoff(
                    edge_length,
                    r_max.reshape(-1),
                    r_start_cos_ratio=self.r_start_cos_ratio,
                ).flatten()

            elif self.cutoff_type == "polynomial":
                cutoff_coeffs = polynomial_cutoff(
                    edge_length, r_max.reshape(-1), p=self.polynomial_cutoff_p
                ).flatten()

            else:
                assert False, "Invalid cutoff type"
        else:
            r_max_by_edge_type, r_max_edge_type_valid = self._r_max_tables_for(edge_length)
            bond_type_flat = bond_type.reshape(-1).to(device=edge_length.device, dtype=torch.long)
            edge_length_flat = edge_length.reshape(-1)
            table_size = r_max_by_edge_type.shape[0]
            in_range = (bond_type_flat >= 0) & (bond_type_flat < table_size)
            safe_bond_type = torch.where(in_range, bond_type_flat, torch.zeros_like(bond_type_flat))
            bond_r_max = r_max_by_edge_type.index_select(0, safe_bond_type)
            valid_bond_type = r_max_edge_type_valid.index_select(0, safe_bond_type) & in_range
            safe_bond_r_max = torch.where(
                valid_bond_type,
                bond_r_max.clamp_min(torch.finfo(edge_length.dtype).eps),
                torch.ones_like(bond_r_max),
            )
            safe_edge_length = torch.where(
                valid_bond_type,
                edge_length_flat,
                torch.zeros_like(edge_length_flat),
            )
            if self.cutoff_type == "cosine":
                cutoff_coeffs = _cosine_cutoff_per_edge(
                    safe_edge_length,
                    safe_bond_r_max,
                    r_start_cos_ratio=self.r_start_cos_ratio,
                )
            elif self.cutoff_type == "polynomial":
                cutoff_coeffs = _polynomial_cutoff_per_edge(
                    safe_edge_length,
                    safe_bond_r_max,
                    p=self.polynomial_cutoff_p,
                )
            else:
                assert False, "Invalid cutoff type"
            cutoff_coeffs = cutoff_coeffs * valid_bond_type.to(dtype=cutoff_coeffs.dtype)

        return cutoff_coeffs

    def precompute_cutoff_metadata(
        self,
        edge_length: torch.Tensor,
        bond_type: torch.Tensor,
        compute_cutoff: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        with torch.no_grad():
            cutoff_coeffs = self.cutoff_coefficients(edge_length, bond_type)
            active_edges = (cutoff_coeffs > 0).nonzero().squeeze(-1).to(dtype=torch.long)
            if compute_cutoff:
                return active_edges, cutoff_coeffs
            return active_edges, None

    def cutoff_config_signature(self):
        table = None
        valid = None
        if self._r_max_by_edge_type_cpu is not None:
            table = tuple(float(v) for v in self._r_max_by_edge_type_cpu.reshape(-1).tolist())
            valid = tuple(bool(v) for v in self._r_max_edge_type_valid_cpu.reshape(-1).tolist())
        return (
            self.cutoff_type,
            float(self.r_start_cos_ratio),
            float(self.polynomial_cutoff_p),
            tuple(float(v) for v in self._r_max_cpu.reshape(-1).tolist()),
            table,
            valid,
        )

    def forward(
        self,
        edge_index,
        atom_type,
        bond_type,
        edge_sh,
        edge_length,
        edge_one_hot,
        active_edges: Optional[torch.Tensor] = None,
        cutoff_coeffs: Optional[torch.Tensor] = None,
    ):
        edge_center = edge_index[0]

        edge_invariants = self.bessel(edge_length)

        if cutoff_coeffs is None:
            cutoff_coeffs = self.cutoff_coefficients(edge_length, bond_type)
        else:
            cutoff_coeffs = cutoff_coeffs.to(device=edge_length.device, dtype=edge_length.dtype).reshape(-1)

        if active_edges is None:
            active_edges = (cutoff_coeffs > 0).nonzero().squeeze(-1)
        else:
            active_edges = active_edges.to(device=edge_length.device, dtype=torch.long).reshape(-1)

        latents = torch.zeros(
            (edge_sh.shape[0], self.two_body_latent.out_features),
            dtype=edge_sh.dtype,
            device=edge_sh.device,
        )

        new_latents = self.two_body_latent(torch.cat([
            edge_one_hot[active_edges],
            edge_invariants[active_edges],
        ], dim=-1))

        latents = torch.index_copy(
            latents, 0, active_edges,
            cutoff_coeffs[active_edges].unsqueeze(-1) * new_latents
        )

        weights_e = self.env_embed_mlp(latents[active_edges])

        edge_features = self._env_weighter(
            edge_sh[active_edges], weights_e
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

        return latents, node_features, edge_features, cutoff_coeffs, active_edges


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
            equivariant_norm_type: str = "none",
            activation_type: str = "gate",
            swiglu_s2_grid_resolution: Tuple[int, int] = (14, 14),
            swiglu_s2_compat_mode: str = "modern",
            avg_num_neighbors: Optional[float] = None,
            so2_wigner_apply_mode: str = "compact_blocks",
            so2_fusion_mode: str = "staged",
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
            num_experts: int = 8,
            num_shared_experts: int = 1,
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
            torch.as_tensor(avg_num_neighbors).rsqrt(),
        )

        self._env_weighter = E3ElementLinear(
            irreps_in=irreps_out,
            dtype=dtype,
            device=device,
        )

        assert irreps_out[0].ir.l == 0

        self.env_embed_mlps = ScalarMLPFunction(
            mlp_input_dimension=latent_dim,
            mlp_latent_dimensions=[],
            mlp_output_dimension=self._env_weighter.weight_numel,
        )

        self.node_norm = build_equivariant_norm(
            equivariant_norm_type,
            self.irreps_in,
            norm_eps,
            dtype,
            device,
        )
        self.edge_norm = build_equivariant_norm(
            equivariant_norm_type,
            self.edge_irreps_in,
            norm_eps,
            dtype,
            device,
        )

        if activation_type not in {"gate", "swiglu_s2"}:
            raise ValueError(f"Unsupported activation_type={activation_type!r}")
        use_s2 = activation_type == "swiglu_s2" and can_use_flat_s2_patch(
            self.irreps_out,
            mode=swiglu_s2_compat_mode,
        )
        if use_s2:
            self.activation = FlatSwiGLUS2Merge(
                self.irreps_out,
                grid_resolution=swiglu_s2_grid_resolution,
            )
            tp_out_irreps = self.activation.tp_main_irreps
            extra_m0_outsize = self.activation.extra_m0_outsize
        else:
            self.activation = build_gate_activation(self.irreps_out)
            tp_out_irreps = self.activation.irreps_in
            extra_m0_outsize = 0

        self.tp = SO2_Linear(
            irreps_in=self.irreps_in + self.edge_irreps_in,
            irreps_out=tp_out_irreps,
            latent_dim=latent_dim,
            radial_emb=radial_emb,
            radial_channels=radial_channels,
            extra_m0_outsize=extra_m0_outsize,
            use_interpolation=use_interpolation_tp,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            wigner_apply_mode=so2_wigner_apply_mode,
            so2_fusion_mode=so2_fusion_mode,
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

        if res_update_ratios is None:
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
        self.use_identity_res = (self.irreps_in == self.irreps_out) and res_update
        if not self.use_identity_res:
            if res_update:
                self.linear_res = Linear(
                    self.irreps_in,
                    self.irreps_out,
                    shared_weights=True,
                    internal_weights=True,
                    biases=True,
                )

    def forward(self, latents, node_features, edge_features, atom_type, node_onehot, edge_index, edge_vector,
                active_edges, wigner_D_all, mole_globals):  # Accept globals
        edge_center = edge_index[0]
        edge_neighbor = edge_index[1]

        new_node_features = node_features
        node_in = self.node_norm(new_node_features) if self.node_norm is not None else new_node_features
        edge_in = self.edge_norm(edge_features) if self.edge_norm is not None else edge_features
        message, _ = self.tp(
            torch.cat(
                [node_in[edge_center[active_edges]], edge_in],
                dim=-1,
            ),
            edge_vector[active_edges],
            mole_globals,
            latents[active_edges],
            wigner_D_all,
        )  # Pass globals

        message = self.activation(message)
        message = self.lin_post(message)
        scalars = message[:, :self.irreps_out[0].dim]

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

            if self.use_identity_res:
                node_features = coefficient_old * node_features + coefficient_new * new_node_features
            else:
                # 维度不同，必须经过 linear_res
                node_features = coefficient_old * self.linear_res(node_features) + coefficient_new * new_node_features

        else:
            node_features = new_node_features

        if self.use_layer_onehot_tp:
            onehot_tune_node_feat = self.node_onehot_tp(node_features, node_onehot)
            node_features = node_features + onehot_tune_node_feat

        return node_features


class UpdateEdge(torch.nn.Module):
    def __init__(
            self,
            num_types,
            node_irreps_in: o3.Irreps,
            irreps_in: o3.Irreps,
            irreps_out: o3.Irreps,
            latent_dim: int,
            norm_eps: float = 1e-8,
            latent_channels: list = [128, 128],
            radial_emb: bool = False,
            radial_channels: list = [128, 128],
            res_update: bool = True,
            use_layer_onehot_tp: bool = True,
            use_interpolation_tp: bool = False,
            edge_one_hot_dim: int = 128,
            res_update_ratios: Optional[List[float]] = None,
            res_update_ratios_learnable: bool = False,
            equivariant_norm_type: str = "none",
            activation_type: str = "gate",
            swiglu_s2_grid_resolution: Tuple[int, int] = (14, 14),
            swiglu_s2_compat_mode: str = "modern",
            so2_wigner_apply_mode: str = "compact_blocks",
            so2_fusion_mode: str = "staged",
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
            num_experts: int = 8,
            num_shared_experts: int = 1,
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

        self.node_norm = build_equivariant_norm(
            equivariant_norm_type,
            self.node_irreps_in,
            norm_eps,
            dtype,
            device,
        )
        self.edge_norm = build_equivariant_norm(
            equivariant_norm_type,
            self.irreps_in,
            norm_eps,
            dtype,
            device,
        )

        if activation_type not in {"gate", "swiglu_s2"}:
            raise ValueError(f"Unsupported activation_type={activation_type!r}")
        use_s2 = activation_type == "swiglu_s2" and can_use_flat_s2_patch(
            self.irreps_out,
            mode=swiglu_s2_compat_mode,
        )
        if use_s2:
            self.activation = FlatSwiGLUS2Merge(
                self.irreps_out,
                grid_resolution=swiglu_s2_grid_resolution,
            )
            tp_out_irreps = self.activation.tp_main_irreps
            extra_m0_outsize = self.activation.extra_m0_outsize
        else:
            self.activation = build_gate_activation(self.irreps_out)
            tp_out_irreps = self.activation.irreps_in
            extra_m0_outsize = 0

        self.tp = SO2_Linear(
            irreps_in=self.node_irreps_in + self.irreps_in + self.node_irreps_in,
            irreps_out=tp_out_irreps,
            latent_dim=latent_dim,
            radial_emb=radial_emb,
            radial_channels=radial_channels,
            extra_m0_outsize=extra_m0_outsize,
            use_interpolation=use_interpolation_tp,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            wigner_apply_mode=so2_wigner_apply_mode,
            so2_fusion_mode=so2_fusion_mode,
        )

        self.latents_mlp_1 = ScalarMLPFunction(
            mlp_input_dimension=latent_dim + self.irreps_out[0].dim,
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

        if res_update_ratios is None:
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

        self.use_identity_res = (self.irreps_in == self.irreps_out) and res_update
        if not self.use_identity_res:
            if res_update:
                self.linear_res = Linear(
                    self.irreps_in,
                    self.irreps_out,
                    shared_weights=True,
                    internal_weights=True,
                    biases=True,
                )

    def forward(self, latents, node_features, node_onehot, edge_features, edge_index, edge_vector, cutoff_coeffs,
                active_edges, edge_one_hot, wigner_D_all, mole_globals):  # Accept globals
        edge_center = edge_index[0]
        edge_neighbor = edge_index[1]

        new_node_features = node_features
        node_in = self.node_norm(new_node_features) if self.node_norm is not None else new_node_features
        edge_in = self.edge_norm(edge_features) if self.edge_norm is not None else edge_features

        new_edge_features, wigner_D_all = self.tp(
            torch.cat(
                [
                    node_in[edge_center[active_edges]],
                    edge_in,
                    node_in[edge_neighbor[active_edges]],
                ],
                dim=-1,
            ),
            edge_vector[active_edges],
            mole_globals,
            latents[active_edges],
            wigner_D_all,
        )  # Pass globals

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

            if self.use_identity_res:
                edge_features = coefficient_old * edge_features + coefficient_new * new_edge_features
            else:
                # 维度不同，必须经过 linear_res
                edge_features = coefficient_old * self.linear_res(edge_features) + coefficient_new * new_edge_features

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


class Layer(torch.nn.Module):
    def __init__(
            self,
            num_types: int,
            # required params
            avg_num_neighbors: Optional[float] = None,
            irreps_in: o3.Irreps = None,
            irreps_out: o3.Irreps = None,
            tp_radial_emb: bool = False,
            tp_radial_channels: list = [128, 128],
            # MLP parameters:
            norm_eps: float = 1e-8,
            latent_channels: list = [128, 128],
            latent_dim: int = 128,
            res_update: bool = True,
            use_layer_onehot_tp: bool = True,
            use_interpolation_tp: bool = False,
            edge_one_hot_dim: int = 128,
            res_update_ratios: Optional[List[float]] = None,
            res_update_ratios_learnable: bool = False,
            equivariant_norm_type: str = "none",
            edge_activation_type: str = "gate",
            node_activation_type: str = "gate",
            swiglu_s2_grid_resolution: Tuple[int, int] = (14, 14),
            swiglu_s2_compat_mode: str = "modern",
            ffn_hidden_factor: float = 0.0,
            use_node_ffn: bool = False,
            so2_wigner_apply_mode: str = "compact_blocks",
            so2_fusion_mode: str = "staged",
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
            num_experts: int = 8,
            num_shared_experts: int = 1,
    ):
        super(Layer, self).__init__()

        self.res_update = res_update
        self.avg_num_neighbors = avg_num_neighbors
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.dtype = dtype
        self.device = device
        self.num_types = num_types

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
            equivariant_norm_type=equivariant_norm_type,
            activation_type=edge_activation_type,
            swiglu_s2_grid_resolution=swiglu_s2_grid_resolution,
            swiglu_s2_compat_mode=swiglu_s2_compat_mode,
            dtype=dtype,
            device=device,
            use_interpolation_tp=use_interpolation_tp,
            norm_eps=norm_eps,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            so2_wigner_apply_mode=so2_wigner_apply_mode,
            so2_fusion_mode=so2_fusion_mode,
        )

        self.node_update = UpdateNode(
            edge_irreps_in=self.edge_update.irreps_out,
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            latent_dim=latent_dim,
            radial_emb=tp_radial_emb,
            use_layer_onehot_tp=use_layer_onehot_tp,
            radial_channels=tp_radial_channels,
            res_update=res_update,
            res_update_ratios=res_update_ratios,
            res_update_ratios_learnable=res_update_ratios_learnable,
            avg_num_neighbors=avg_num_neighbors,
            equivariant_norm_type=equivariant_norm_type,
            activation_type=node_activation_type,
            swiglu_s2_grid_resolution=swiglu_s2_grid_resolution,
            swiglu_s2_compat_mode=swiglu_s2_compat_mode,
            dtype=dtype,
            device=device,
            use_interpolation_tp=use_interpolation_tp,
            norm_eps=norm_eps,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            so2_wigner_apply_mode=so2_wigner_apply_mode,
            so2_fusion_mode=so2_fusion_mode,
        )

        self.node_ffn = None
        if use_node_ffn:
            self.node_ffn = EqV3StyleNodeFFN(
                self.irreps_out,
                hidden_factor=ffn_hidden_factor,
                norm_type=equivariant_norm_type,
                norm_eps=norm_eps,
                grid_resolution=swiglu_s2_grid_resolution,
                dtype=dtype,
                device=device,
            )

    def forward(self, latents, node_features, edge_features, node_onehot, edge_index, edge_vector, atom_type,
                cutoff_coeffs, active_edges, edge_one_hot, wigner_D_all, mole_globals):
        edge_features, latents, wigner_D_all = self.edge_update(latents, node_features, node_onehot, edge_features,
                                                                edge_index, edge_vector, cutoff_coeffs, active_edges,
                                                                edge_one_hot, wigner_D_all, mole_globals)
        node_features = self.node_update(latents, node_features, edge_features, atom_type, node_onehot, edge_index,
                                         edge_vector, active_edges, wigner_D_all, mole_globals)
        if self.node_ffn is not None:
            node_features = self.node_ffn(node_features)

        return latents, node_features, edge_features, wigner_D_all
