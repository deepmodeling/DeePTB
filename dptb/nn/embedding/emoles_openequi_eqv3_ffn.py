from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Union

import torch
from e3nn import o3
from e3nn.o3 import FromS2Grid, Linear, ToS2Grid
from torch import nn

from dptb.nn.embedding.emb import Embedding
from dptb.nn.tensor_product import SO2_Linear

from .emoles import EMolES, EAMPOpenequi, OEQTensorProduct, create_gate, oeq


_GRID_MAT_CACHE: dict[tuple[int, int, str, tuple[int, int]], tuple[torch.Tensor, torch.Tensor]] = {}


def _as_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(dtype, str):
        return getattr(torch, dtype)
    return dtype


def _as_device(device: Union[str, torch.device]) -> torch.device:
    if isinstance(device, str):
        return torch.device(device)
    return device


def _build_uniform_pack_indices(
    irreps: Union[str, o3.Irreps],
) -> tuple[o3.Irreps, int, int, int, torch.Tensor, torch.Tensor]:
    irreps = o3.Irreps(irreps).simplify()
    lmax = irreps.lmax

    blocks: dict[int, tuple[int, int, int]] = {}
    num_channels = None
    offset = 0

    for mul, ir in irreps:
        if ir.l in blocks:
            raise ValueError(
                f"EqV3 flat SO(3) layout needs at most one block per degree, got duplicated l={ir.l}."
            )
        if num_channels is None:
            num_channels = mul
        elif num_channels != mul:
            raise ValueError(
                f"EqV3 flat SO(3) layout needs uniform multiplicity across degrees, got {num_channels} and {mul}."
            )
        blocks[ir.l] = (offset, ir.dim, mul)
        offset += mul * ir.dim

    expected = set(range(lmax + 1))
    if set(blocks.keys()) != expected:
        raise ValueError(f"EqV3 flat SO(3) layout needs contiguous degrees 0..{lmax}, got {sorted(blocks.keys())}.")

    pack_index = []
    for degree in range(lmax + 1):
        start, dim, mul = blocks[degree]
        for m in range(dim):
            for channel in range(mul):
                pack_index.append(start + channel * dim + m)

    unpack_index = [0] * len(pack_index)
    for packed_pos, flat_pos in enumerate(pack_index):
        unpack_index[flat_pos] = packed_pos

    return (
        irreps,
        num_channels,
        (lmax + 1) ** 2,
        lmax,
        torch.tensor(pack_index, dtype=torch.long),
        torch.tensor(unpack_index, dtype=torch.long),
    )


def can_use_eqv3_flat_layout(irreps: Union[str, o3.Irreps]) -> bool:
    try:
        _build_uniform_pack_indices(irreps)
    except ValueError:
        return False
    return True


class UniformDegreePacker(nn.Module):
    def __init__(self, irreps: Union[str, o3.Irreps]):
        super().__init__()
        irreps, num_channels, num_coeffs, lmax, pack_index, unpack_index = _build_uniform_pack_indices(irreps)
        self.irreps = irreps
        self.num_channels = num_channels
        self.num_coeffs = num_coeffs
        self.lmax = lmax
        self.dim = irreps.dim
        self.register_buffer("pack_index", pack_index)
        self.register_buffer("unpack_index", unpack_index)

    def pack(self, x_flat: torch.Tensor) -> torch.Tensor:
        if x_flat.ndim != 2 or x_flat.shape[-1] != self.dim:
            raise ValueError(f"Expected [N, {self.dim}], got {tuple(x_flat.shape)}")
        packed = x_flat.index_select(1, self.pack_index)
        return packed.view(x_flat.shape[0], self.num_coeffs, self.num_channels)

    def unpack(self, x_arr: torch.Tensor) -> torch.Tensor:
        expected = (self.num_coeffs, self.num_channels)
        if x_arr.ndim != 3 or tuple(x_arr.shape[1:]) != expected:
            raise ValueError(f"Expected [N, {expected[0]}, {expected[1]}], got {tuple(x_arr.shape)}")
        flat = x_arr.reshape(x_arr.shape[0], -1)
        return flat.index_select(1, self.unpack_index)


def doubled_irreps(irreps: Union[str, o3.Irreps]) -> o3.Irreps:
    irreps = o3.Irreps(irreps).simplify()
    return o3.Irreps([(2 * mul, ir) for mul, ir in irreps]).simplify()


def _get_grid_mats(
    lmax: int,
    mmax: int,
    normalization: str,
    resolution: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (lmax, mmax, normalization, resolution)
    mats = _GRID_MAT_CACHE.get(key)
    if mats is not None:
        return mats

    to_grid = ToS2Grid(
        lmax,
        resolution,
        normalization=normalization,
        device="cpu",
    )
    to_grid_mat = torch.einsum("mbi, am -> bai", to_grid.shb, to_grid.sha).detach()
    to_grid_mat = to_grid_mat.flatten(0, 1).contiguous()

    from_grid = FromS2Grid(
        resolution,
        lmax,
        normalization=normalization,
        device="cpu",
    )
    from_grid_mat = torch.einsum("am, mbi -> bai", from_grid.sha, from_grid.shb).detach()
    from_grid_mat = from_grid_mat.flatten(0, 1).permute(1, 0).contiguous()

    mats = (to_grid_mat, from_grid_mat)
    _GRID_MAT_CACHE[key] = mats
    return mats


class SO3Grid(nn.Module):
    def __init__(
        self,
        lmax: int,
        mmax: int,
        normalization: str = "component",
        resolution: tuple[int, int] | None = None,
        use_m_primary: bool = False,
    ):
        super().__init__()

        if mmax != lmax:
            raise NotImplementedError("This EqV3 FFN helper only supports mmax == lmax.")
        if use_m_primary:
            raise NotImplementedError("This EqV3 FFN helper only supports use_m_primary=False.")

        if resolution is None:
            resolution = (2 * (lmax + 1), 2 * (mmax + 1) + 1)
        if len(resolution) != 2:
            raise ValueError(f"Expected 2D S2 resolution, got {resolution!r}")

        self.lmax = lmax
        self.mmax = mmax
        self.lat_resolution = int(resolution[0])
        self.long_resolution = int(resolution[1])
        self.use_m_primary = use_m_primary

        to_grid_mat, from_grid_mat = _get_grid_mats(
            self.lmax,
            self.mmax,
            normalization,
            (self.lat_resolution, self.long_resolution),
        )
        self.register_buffer("to_grid_mat", to_grid_mat)
        self.register_buffer("from_grid_mat", from_grid_mat)

    def to_grid(self, embedding: torch.Tensor) -> torch.Tensor:
        return torch.einsum("aj, njc -> nac", self.to_grid_mat, embedding)

    def from_grid(self, grid: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ja, nac -> njc", self.from_grid_mat, grid)


class EquivariantMergedRMSNormFlat(nn.Module):
    def __init__(
        self,
        irreps: Union[str, o3.Irreps],
        eps: float = 1e-6,
        affine: bool = True,
        normalization: str = "component",
        std_balance_degrees: bool = True,
        center_0e: bool = True,
        treat_0o_as_scalar: bool = False,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super().__init__()

        dtype = _as_dtype(dtype)
        device = _as_device(device)

        self.irreps = o3.Irreps(irreps).simplify()
        self.eps = eps
        self.affine = affine
        self.normalization = normalization
        self.std_balance_degrees = std_balance_degrees
        self.center_0e = center_0e
        self.treat_0o_as_scalar = treat_0o_as_scalar

        if normalization not in ("component", "norm"):
            raise ValueError(f"Unsupported normalization={normalization!r}")

        dim_to_group = []
        group_to_degree = []
        group_inv_dims = []
        scalar_dim_idx = []
        degree_counts = {}

        offset = 0
        group_id = 0
        for mul, ir in self.irreps:
            for _ in range(mul):
                dim_to_group.extend([group_id] * ir.dim)
                group_to_degree.append(ir.l)
                group_inv_dims.append(1.0 / ir.dim)
                if ir.l == 0 and (ir.p == 1 or self.treat_0o_as_scalar):
                    scalar_dim_idx.append(offset)
                degree_counts[ir.l] = degree_counts.get(ir.l, 0) + 1
                offset += ir.dim
                group_id += 1

        self.dim = offset
        self.num_groups = group_id
        self.num_scalar = len(scalar_dim_idx)
        self.num_degrees = len(degree_counts)

        degree_ids = sorted(degree_counts.keys())
        degree_map = {degree: idx for idx, degree in enumerate(degree_ids)}
        group_to_degree = [degree_map[degree] for degree in group_to_degree]
        degree_inv_group_counts = [1.0 / degree_counts[degree] for degree in degree_ids]

        self.register_buffer("dim_to_group", torch.tensor(dim_to_group, dtype=torch.long, device=device))
        self.register_buffer("group_to_degree", torch.tensor(group_to_degree, dtype=torch.long, device=device))
        self.register_buffer(
            "group_inv_dims",
            torch.tensor(group_inv_dims, dtype=dtype, device=device).unsqueeze(0),
        )
        self.register_buffer(
            "degree_inv_group_counts",
            torch.tensor(degree_inv_group_counts, dtype=dtype, device=device).unsqueeze(0),
        )
        self.register_buffer("scalar_dim_idx", torch.tensor(scalar_dim_idx, dtype=torch.long, device=device))

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(1, self.num_groups, dtype=dtype, device=device))
            if self.center_0e and self.num_scalar > 0:
                self.affine_bias = nn.Parameter(torch.zeros(1, self.num_scalar, dtype=dtype, device=device))
            else:
                self.register_parameter("affine_bias", None)
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

    @torch.amp.autocast(device_type="cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected [N, dim], got shape={tuple(x.shape)}")
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {x.shape[-1]}")

        orig_dtype = x.dtype
        y = x.to(torch.float32)
        if y.data_ptr() == x.data_ptr():
            y = y.clone()

        if self.center_0e and self.scalar_dim_idx.numel() > 0:
            scalars = y.index_select(1, self.scalar_dim_idx)
            scalars = scalars - scalars.mean(dim=1, keepdim=True)
            y[:, self.scalar_dim_idx] = scalars

        group_sums = y.new_zeros((y.shape[0], self.num_groups))
        group_sums.scatter_add_(1, self.dim_to_group.expand(y.shape[0], -1), y.square())

        if self.normalization == "component":
            group_ms = group_sums * self.group_inv_dims
        else:
            group_ms = group_sums

        if self.std_balance_degrees:
            degree_sums = group_ms.new_zeros((y.shape[0], self.num_degrees))
            degree_sums.scatter_add_(1, self.group_to_degree.expand(y.shape[0], -1), group_ms)
            degree_ms = degree_sums * self.degree_inv_group_counts
            merged_ms = degree_ms.mean(dim=1, keepdim=True)
        else:
            merged_ms = group_ms.mean(dim=1, keepdim=True)

        group_scale = torch.rsqrt(merged_ms + self.eps)
        if self.affine:
            group_scale = group_scale * self.affine_weight
        group_scale = group_scale.index_select(1, self.dim_to_group)
        y = y * group_scale

        if self.affine and self.affine_bias is not None and self.scalar_dim_idx.numel() > 0:
            y[:, self.scalar_dim_idx] = y[:, self.scalar_dim_idx] + self.affine_bias

        return y.to(orig_dtype)


class SimpleSwiGLU(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(inputs, chunks=2, dim=-1)
        return torch.nn.functional.silu(x1) * x2


class LinearSwiGLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super().__init__()
        dtype = _as_dtype(dtype)
        device = _as_device(device)
        self.linear = nn.Linear(in_channels, 2 * out_channels, bias=bias, dtype=dtype, device=device)
        self.act = SimpleSwiGLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(inputs))


class SO3LinearFlat(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        lmax: int,
        bias: bool = True,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super().__init__()

        dtype = _as_dtype(dtype)
        device = _as_device(device)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lmax = lmax

        self.weight = nn.Parameter(torch.empty((self.lmax + 1), out_channels, in_channels, dtype=dtype, device=device))
        bound = 1.0 / math.sqrt(self.in_channels)
        nn.init.uniform_(self.weight, -bound, bound)

        if bias:
            self.bias = nn.Parameter(torch.zeros(1, 1, out_channels, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

        expand_index = torch.zeros([(lmax + 1) ** 2], dtype=torch.long, device=device)
        for degree in range(lmax + 1):
            start = degree ** 2
            expand_index[start : start + 2 * degree + 1] = degree
        self.register_buffer("expand_index", expand_index)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = torch.index_select(self.weight, dim=0, index=self.expand_index)
        outputs = torch.einsum("bmi, moi -> bmo", inputs, weight)
        if self.bias is not None:
            outputs[:, 0:1, :] = outputs.narrow(1, 0, 1) + self.bias
        return outputs


class FlatSwiGLUS2Merge(nn.Module):
    def __init__(self, irreps_out: Union[str, o3.Irreps], grid_resolution: Tuple[int, int] = (14, 14)):
        super().__init__()

        self.irreps_out = o3.Irreps(irreps_out).simplify()
        self.tp_main_irreps = doubled_irreps(self.irreps_out)

        self.out_packer = UniformDegreePacker(self.irreps_out)
        self.in_packer = UniformDegreePacker(self.tp_main_irreps)

        self.num_channels = self.out_packer.num_channels
        self.extra_m0_outsize = self.num_channels
        self.irreps_in = (o3.Irreps(f"{self.extra_m0_outsize}x0e") + self.tp_main_irreps).simplify()

        self.scalar_act = SimpleSwiGLU()
        self.gate_act = nn.Sigmoid()
        self.so3_grid = SO3Grid(
            self.out_packer.lmax,
            self.out_packer.lmax,
            resolution=grid_resolution,
            use_m_primary=False,
        )

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        gate_scalars = x_flat[:, :self.extra_m0_outsize]
        main_flat = x_flat[:, self.extra_m0_outsize :]

        x_arr = self.in_packer.pack(main_flat)
        out_scalars = self.scalar_act(x_arr[:, 0, :])

        x_grid = self.so3_grid.to_grid(x_arr)
        x1, x2 = torch.chunk(x_grid, chunks=2, dim=-1)
        out_arr = self.so3_grid.from_grid(x1 * x2)

        out_arr.mul_(self.gate_act(gate_scalars).unsqueeze(1))
        out_arr[:, 0, :] = out_arr[:, 0, :] + out_scalars
        return self.out_packer.unpack(out_arr)


class EqV3StyleNodeFFN(nn.Module):
    def __init__(
        self,
        irreps: Union[str, o3.Irreps],
        hidden_factor: float = 4.0,
        norm_type: str = "merged_rms",
        norm_eps: float = 1e-6,
        grid_resolution: Tuple[int, int] = (14, 14),
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super().__init__()

        if hidden_factor <= 1.0:
            raise ValueError(f"EqV3StyleNodeFFN needs hidden_factor > 1.0, got {hidden_factor}.")

        dtype = _as_dtype(dtype)
        device = _as_device(device)

        self.irreps = o3.Irreps(irreps).simplify()
        self.packer = UniformDegreePacker(self.irreps)
        self.hidden_channels = max(1, int(round(self.packer.num_channels * hidden_factor)))
        self.norm = build_equivariant_norm(norm_type, self.irreps, norm_eps, dtype, device)

        self.scalar_mlp = LinearSwiGLU(
            self.packer.num_channels,
            self.hidden_channels,
            bias=True,
            dtype=dtype,
            device=device,
        )
        self.so3_linear_1 = SO3LinearFlat(
            self.packer.num_channels,
            self.hidden_channels,
            self.packer.lmax,
            bias=True,
            dtype=dtype,
            device=device,
        )
        self.so3_grid = SO3Grid(
            self.packer.lmax,
            self.packer.lmax,
            resolution=grid_resolution,
            use_m_primary=False,
        )
        self.grid_mlp = nn.Sequential(
            LinearSwiGLU(
                self.hidden_channels,
                self.hidden_channels,
                bias=False,
                dtype=dtype,
                device=device,
            ),
            nn.Linear(
                self.hidden_channels,
                self.hidden_channels,
                bias=False,
                dtype=dtype,
                device=device,
            ),
        )
        self.so3_linear_2 = SO3LinearFlat(
            self.hidden_channels,
            self.packer.num_channels,
            self.packer.lmax,
            bias=True,
            dtype=dtype,
            device=device,
        )

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        residual = x_flat
        x_flat = self.norm(x_flat)
        x_arr = self.packer.pack(x_flat)

        scalar_out = self.scalar_mlp(x_arr[:, 0, :])
        x_arr = self.so3_linear_1(x_arr)
        x_grid = self.so3_grid.to_grid(x_arr)
        x_grid = self.grid_mlp(x_grid)
        x_arr = self.so3_grid.from_grid(x_grid)
        x_arr[:, 0:1, :] = x_arr[:, 0:1, :] + scalar_out.unsqueeze(1)
        x_arr = self.so3_linear_2(x_arr)

        return residual + self.packer.unpack(x_arr)


def build_equivariant_norm(
    norm_type: str,
    irreps: o3.Irreps,
    norm_eps: float,
    dtype: Union[str, torch.dtype],
    device: Union[str, torch.device],
) -> nn.Module:
    if norm_type == "none":
        return nn.Identity()
    if norm_type == "merged_rms":
        return EquivariantMergedRMSNormFlat(
            irreps,
            eps=norm_eps,
            dtype=dtype,
            device=device,
        )
    raise ValueError(f"Unsupported equivariant_norm_type={norm_type!r}")


def _create_eqv3_ffn_layer_worker(args):
    idx, layer_kwargs = args
    t_start = time.time()
    layer = EAMPOpenequiEqV3FFN(**layer_kwargs)
    duration = time.time() - t_start
    return idx, layer, duration


def _create_tp_worker(args):
    name, tp_kwargs = args
    t_start = time.time()
    tp = OEQTensorProduct(**tp_kwargs)
    duration = time.time() - t_start
    return name, tp, duration


class EAMPOpenequiEqV3FFN(EAMPOpenequi):
    def __init__(self, **kwargs):
        layer_kwargs = dict(kwargs)
        self.ln_flag = layer_kwargs.get("ln_flag", True)
        self.equivariant_norm_type = layer_kwargs.pop("equivariant_norm_type", "merged_rms")
        self.hidden_edge_activation_type = layer_kwargs.pop("hidden_edge_activation_type", "swiglu_s2")
        self.swiglu_s2_grid_resolution = tuple(layer_kwargs.pop("swiglu_s2_grid_resolution", [14, 14]))
        self.ffn_hidden_factor = float(layer_kwargs.pop("ffn_hidden_factor", 0.0))
        self.use_node_ffn = bool(layer_kwargs.pop("use_node_ffn", False))
        super().__init__(**layer_kwargs)

        self.sln_n = self._build_eq_norm(self.node_irreps_in)
        self.sln_e = self._build_eq_norm(self.edge_irreps_in)
        self.activation = self._build_main_activation(self.hidden_edge_activation_type)

        real_tp_rotate_out = self.tp_rotate_out
        if self.in_frame_flag and self.optimized_in_frame:
            real_tp_rotate_out = False

        tp_irreps_out = getattr(self.activation, "tp_main_irreps", self.activation.irreps_in)
        extra_m0_outsize = getattr(self.activation, "extra_m0_outsize", 0)
        self.tp = SO2_Linear(
            irreps_in=self.node_irreps_in + self.edge_irreps_in + self.node_irreps_in,
            irreps_out=tp_irreps_out,
            latent_dim=kwargs.get("latent_dim", 128),
            radial_emb=kwargs.get("radial_emb", False),
            radial_channels=kwargs.get("radial_channels", [128, 128]),
            extra_m0_outsize=extra_m0_outsize,
            use_interpolation=kwargs.get("use_interpolation_tp", False),
            rotate_in=self.tp_rotate_in,
            rotate_out=real_tp_rotate_out,
        )
        self.lin_post = Linear(
            self.activation.irreps_out,
            self.irreps_out,
            shared_weights=True,
            internal_weights=True,
            biases=True,
        )

        self.node_ffn = None
        if self.use_node_ffn and self.ffn_hidden_factor > 1.0:
            self.node_ffn = EqV3StyleNodeFFN(
                self.irreps_out,
                hidden_factor=self.ffn_hidden_factor,
                norm_type=self.equivariant_norm_type,
                norm_eps=self.norm_eps,
                grid_resolution=self.swiglu_s2_grid_resolution,
                dtype=self.dtype,
                device=self.device,
            )

    def _build_eq_norm(self, irreps: o3.Irreps) -> nn.Module:
        if not self.ln_flag:
            return nn.Identity()
        return build_equivariant_norm(
            self.equivariant_norm_type,
            irreps,
            self.norm_eps,
            self.dtype,
            self.device,
        )

    def _build_main_activation(self, activation_type: str) -> nn.Module:
        if activation_type == "gate":
            return create_gate(self.irreps_out)
        if activation_type == "swiglu_s2":
            if not can_use_eqv3_flat_layout(self.irreps_out):
                raise ValueError(f"SwiGLU-S2 requires uniform EqV3-compatible hidden irreps, got {self.irreps_out}.")
            return FlatSwiGLUS2Merge(
                self.irreps_out,
                grid_resolution=self.swiglu_s2_grid_resolution,
            )
        raise ValueError(f"Unsupported hidden_edge_activation_type={activation_type!r}")

    def _build_mixer_module(self):
        mixer = torch.nn.ModuleDict()
        mixer["norm"] = self._build_eq_norm(self.irreps_out)

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
                        tp_mode="uvw",
                    )
                elif "channelwise" in self.self_mix_mode:
                    tp_layer = OEQTensorProduct(
                        irreps_in1=self.irreps_out,
                        irreps_in2=irreps_in2,
                        irreps_out=self.irreps_out,
                        tp_mode="uvu",
                    )
                    if pre_gate_linear is None:
                        pre_gate_linear = Linear(
                            self.irreps_out,
                            gate.irreps_in,
                            internal_weights=True,
                            shared_weights=True,
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
                        tp_mode="uvu",
                    )
                    if pre_gate_linear is None:
                        pre_gate_linear = Linear(
                            self.irreps_out,
                            gate.irreps_in,
                            internal_weights=True,
                            shared_weights=True,
                        )
                elif "uuw" in self.self_mix_mode:
                    tp_layer = OEQTensorProduct(
                        irreps_in1=self.irreps_out,
                        irreps_in2=irreps_in2,
                        irreps_out=self.irreps_out,
                        tp_mode="uuw",
                    )
                    if pre_gate_linear is None:
                        pre_gate_linear = Linear(
                            self.irreps_out,
                            gate.irreps_in,
                            internal_weights=True,
                            shared_weights=True,
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
            internal_weights=True,
            shared_weights=True,
        )
        return mixer

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
        wigner_D_all: torch.Tensor | None,
    ):
        node_features, edge_features, wigner_D_all = super().forward(
            latents,
            node_features,
            edge_features,
            atom_type,
            node_onehot,
            edge_index,
            edge_vector,
            active_edges,
            wigner_D_all,
        )
        if self.node_ffn is not None:
            node_features = self.node_ffn(node_features)
        return node_features, edge_features, wigner_D_all


@Embedding.register("emoles_openequi_eqv3_ffn")
class EMolESOpenequiEqV3FFN(EMolES):
    def __init__(self, **kwargs):
        n_layers = kwargs.get("n_layers", 3)
        irreps_hidden = kwargs.get("irreps_hidden")
        use_interpolation_out = kwargs.get("use_interpolation_out", True)
        edge_one_hot_dim = kwargs.get("edge_one_hot_dim", 128)
        ln_flag = kwargs.get("ln_flag", True)
        equivariant_norm_type = kwargs.get("equivariant_norm_type", "merged_rms")
        hidden_edge_activation_type = kwargs.get("hidden_edge_activation_type", "swiglu_s2")
        swiglu_s2_grid_resolution = kwargs.get("swiglu_s2_grid_resolution", [14, 14])
        ffn_hidden_factor = float(kwargs.get("ffn_hidden_factor", 0.0))
        ffn_apply_to_last = bool(kwargs.get("ffn_apply_to_last", False))

        super().__init__(**kwargs)

        if oeq is None:
            raise ImportError("OpenEquivariance is not installed.")

        if ln_flag:
            self.init_layer.sln_n = build_equivariant_norm(
                equivariant_norm_type,
                self.init_layer.irreps_out,
                kwargs.get("norm_eps", 1e-8),
                self.dtype,
                self.device,
            )
        else:
            self.init_layer.sln_n = nn.Identity()

        self.layers = torch.nn.ModuleList([None] * n_layers)
        irreps_hidden_obj = o3.Irreps(irreps_hidden)
        orbpair_irreps = self.idp.orbpair_irreps.sort()[0].simplify()

        base_layer_kwargs = {
            "latent_dim": kwargs.get("latent_dim", 128),
            "norm_eps": kwargs.get("norm_eps", 1e-8),
            "radial_emb": kwargs.get("tp_radial_emb", False),
            "radial_channels": kwargs.get("tp_radial_channels", [128, 128]),
            "res_update": kwargs.get("res_update", True),
            "use_layer_onehot_tp": kwargs.get("use_layer_onehot_tp", True),
            "res_update_ratios": kwargs.get("res_update_ratios", None),
            "res_update_ratios_learnable": kwargs.get("res_update_ratios_learnable", False),
            "avg_num_neighbors": kwargs.get("avg_num_neighbors", None),
            "dtype": self.dtype,
            "device": self.device,
            "ln_flag": ln_flag,
            "in_frame_flag": kwargs.get("in_frame_flag", True),
            "optimized_in_frame": kwargs.get("optimized_in_frame", True),
            "onehot_mode": kwargs.get("onehot_mode", "FullTP"),
            "self_mix_flag": kwargs.get("self_mix_flag", False),
            "self_mix_mode": kwargs.get("self_mix_mode", "scalar_channelwise"),
            "self_mix_iter": kwargs.get("self_mix_iter", 1),
            "self_mix_type": kwargs.get("self_mix_type", "node"),
            "equivariant_norm_type": equivariant_norm_type,
            "swiglu_s2_grid_resolution": swiglu_s2_grid_resolution,
            "ffn_hidden_factor": ffn_hidden_factor,
        }

        tasks = []
        for i in range(n_layers):
            if i == 0:
                irreps_in_layer = self.init_layer.irreps_out
            else:
                irreps_in_layer = irreps_hidden_obj

            if self.in_frame_flag:
                rotate_in = i == 0
                rotate_out = i == n_layers - 1
            else:
                rotate_in, rotate_out = True, True

            if i == n_layers - 1:
                irreps_out_layer = orbpair_irreps
                use_interpolation_tp = bool(use_interpolation_out)
                activation_type = "gate"
            else:
                irreps_out_layer = irreps_hidden_obj
                use_interpolation_tp = False
                activation_type = hidden_edge_activation_type

            use_node_ffn = ffn_hidden_factor > 1.0 and ((i < n_layers - 1) or ffn_apply_to_last)

            current_kwargs = base_layer_kwargs.copy()
            current_kwargs.update(
                {
                    "node_irreps_in": irreps_in_layer,
                    "edge_irreps_in": irreps_in_layer,
                    "irreps_out": irreps_out_layer,
                    "tp_rotate_in": rotate_in,
                    "tp_rotate_out": rotate_out,
                    "use_interpolation_tp": use_interpolation_tp,
                    "node_one_hot_dim": self.n_atom,
                    "hidden_edge_activation_type": activation_type,
                    "use_node_ffn": use_node_ffn,
                }
            )
            tasks.append((i, current_kwargs))

        print(f"Starting parallel compilation for {n_layers} EqV3-FFN layers...")
        t_start_all = time.time()

        with ThreadPoolExecutor(max_workers=8) as executor:
            layer_futures = [executor.submit(_create_eqv3_ffn_layer_worker, task) for task in tasks]

            tp_futures = []
            if self.use_out_onehot_tp:
                tp1_kwargs = {
                    "irreps_in1": self.node_irreps_out,
                    "irreps_in2": o3.Irreps(f"{self.n_atom}x0e"),
                    "irreps_out": self.idp.orbpair_irreps,
                    "tp_mode": "uvw",
                }
                tp2_kwargs = {
                    "irreps_in1": self.edge_irreps_out,
                    "irreps_in2": o3.Irreps(f"{edge_one_hot_dim}x0e"),
                    "irreps_out": self.idp.orbpair_irreps,
                    "tp_mode": "uvw",
                }
                tp_futures.append(executor.submit(_create_tp_worker, ("out_node_ele_tp", tp1_kwargs)))
                tp_futures.append(executor.submit(_create_tp_worker, ("out_edge_ele_tp", tp2_kwargs)))

            for future in layer_futures:
                idx, layer, _ = future.result()
                self.layers[idx] = layer

            for future in tp_futures:
                name, tp_module, _ = future.result()
                setattr(self, name, tp_module)

        print(f"EqV3-FFN compilation finished in {time.time() - t_start_all:.2f}s")
