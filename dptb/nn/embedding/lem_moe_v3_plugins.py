from __future__ import annotations

from typing import Tuple, Union

import torch
from torch import nn
from e3nn import o3
from e3nn.nn import Gate
from e3nn.o3 import FromS2Grid, ToS2Grid


_GRID_MAT_CACHE: dict[tuple[int, int, str, tuple[int, int]], tuple[torch.Tensor, torch.Tensor]] = {}


def _canonical_irreps(irreps: Union[str, o3.Irreps]) -> o3.Irreps:
    return o3.Irreps(irreps).sort()[0].simplify()


def _build_uniform_pack_indices(
    irreps: Union[str, o3.Irreps],
) -> tuple[o3.Irreps, int, int, torch.Tensor, torch.Tensor]:
    irreps = _canonical_irreps(irreps)
    lmax = irreps.lmax

    blocks = {}
    num_channels = None
    offset = 0

    for mul, ir in irreps:
        if ir.l in blocks:
            raise ValueError(
                f"Need at most one block per degree for flat SwiGLU-S2, got duplicated l={ir.l}."
            )
        if num_channels is None:
            num_channels = mul
        elif num_channels != mul:
            raise ValueError(
                f"Need uniform multiplicity across degrees, got {num_channels} and {mul}."
            )

        blocks[ir.l] = (offset, ir.dim, mul)
        offset += mul * ir.dim

    expected = set(range(lmax + 1))
    if set(blocks.keys()) != expected:
        raise ValueError(f"Need contiguous degrees 0..{lmax}, got {sorted(blocks.keys())}.")

    pack_index = []
    for l in range(lmax + 1):
        start, dim, mul = blocks[l]
        for m in range(dim):
            for c in range(mul):
                pack_index.append(start + c * dim + m)

    unpack_index = [0] * len(pack_index)
    for packed_pos, flat_pos in enumerate(pack_index):
        unpack_index[flat_pos] = packed_pos

    return (
        irreps,
        num_channels,
        (lmax + 1) ** 2,
        torch.tensor(pack_index, dtype=torch.long),
        torch.tensor(unpack_index, dtype=torch.long),
    )


def can_use_flat_s2_patch(irreps) -> bool:
    try:
        _build_uniform_pack_indices(irreps)
    except ValueError:
        return False
    return True


class UniformDegreePacker(nn.Module):
    def __init__(self, irreps: Union[str, o3.Irreps]):
        super().__init__()
        irreps, num_channels, num_coeffs, pack_index, unpack_index = _build_uniform_pack_indices(irreps)
        self.irreps = irreps
        self.num_channels = num_channels
        self.num_coeffs = num_coeffs
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
            raise ValueError(
                f"Expected [N, {expected[0]}, {expected[1]}], got {tuple(x_arr.shape)}"
            )

        flat = x_arr.reshape(x_arr.shape[0], -1)
        return flat.index_select(1, self.unpack_index)


def doubled_irreps(irreps: Union[str, o3.Irreps]) -> o3.Irreps:
    irreps = _canonical_irreps(irreps)
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
    """
    Minimal grid projection helper for full l-primary coefficients.

    This only supports the layout used by the flat SwiGLU-S2 adapter:
    full degrees `0..lmax`, `mmax == lmax`, and `use_m_primary == False`.
    """

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
            raise NotImplementedError("Minimal SO3Grid only supports mmax == lmax.")
        if use_m_primary:
            raise NotImplementedError("Minimal SO3Grid only supports use_m_primary=False.")

        self.lmax = lmax
        self.mmax = mmax
        self.use_m_primary = use_m_primary

        if resolution is None:
            resolution = (2 * (self.lmax + 1), 2 * (self.mmax + 1) + 1)
        elif len(resolution) != 2:
            raise ValueError(f"Expected 2D resolution, got {resolution!r}")

        self.lat_resolution = int(resolution[0])
        self.long_resolution = int(resolution[1])

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

    def extra_repr(self):
        return (
            f"lmax={self.lmax}, mmax={self.mmax}, "
            f"lat_resolution={self.lat_resolution}, "
            f"long_resolution={self.long_resolution}, "
            f"use_m_primary={self.use_m_primary}"
        )


class EquivariantMergedRMSNormFlat(nn.Module):
    """
    Flat e3nn analogue of EquiformerV3 merged RMS norm.

    Behavior:
    1. Center only 0e scalars.
    2. Compute one shared RMS over all degrees.
    3. Affine weight is per irrep-copy, expanded over all m.
    4. Bias is added only to centered 0e scalars.
    """

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

        self.irreps = o3.Irreps(irreps).simplify()
        self.eps = eps
        self.affine = affine
        self.normalization = normalization
        self.std_balance_degrees = std_balance_degrees
        self.center_0e = center_0e
        self.treat_0o_as_scalar = treat_0o_as_scalar

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if isinstance(device, str):
            device = torch.device(device)

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
        self.register_buffer(
            "group_to_degree",
            torch.tensor(group_to_degree, dtype=torch.long, device=device),
        )
        self.register_buffer(
            "group_inv_dims",
            torch.tensor(group_inv_dims, dtype=dtype, device=device).unsqueeze(0),
        )
        self.register_buffer(
            "degree_inv_group_counts",
            torch.tensor(degree_inv_group_counts, dtype=dtype, device=device).unsqueeze(0),
        )
        self.register_buffer(
            "scalar_dim_idx",
            torch.tensor(scalar_dim_idx, dtype=torch.long, device=device),
        )

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(1, self.num_groups, dtype=dtype, device=device))
            if self.center_0e and self.num_scalar > 0:
                self.affine_bias = nn.Parameter(torch.zeros(1, self.num_scalar, dtype=dtype, device=device))
            else:
                self.register_parameter("affine_bias", None)
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"irreps={self.irreps}, eps={self.eps}, "
            f"normalization={self.normalization}, "
            f"std_balance_degrees={self.std_balance_degrees}, "
            f"center_0e={self.center_0e})"
        )

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


class FlatSwiGLUS2Merge(nn.Module):
    """
    Single-input flat adapter for DeePTB's existing activation call sites.

    Expected TP output layout:
        [ extra_gate_scalars (C x 0e) ] + [ doubled_main_irreps ]
    """

    def __init__(self, irreps_out, grid_resolution: Tuple[int, int] = (14, 14)):
        super().__init__()

        self.irreps_out = _canonical_irreps(irreps_out)
        self.tp_main_irreps = doubled_irreps(self.irreps_out)

        self.out_packer = UniformDegreePacker(self.irreps_out)
        self.in_packer = UniformDegreePacker(self.tp_main_irreps)

        self.num_channels = self.out_packer.num_channels
        self.extra_m0_outsize = self.num_channels
        self.irreps_in = (o3.Irreps(f"{self.extra_m0_outsize}x0e") + self.tp_main_irreps).simplify()

        self.scalar_act = SimpleSwiGLU()
        self.gate_act = nn.Sigmoid()
        self.so3_grid = SO3Grid(
            self.irreps_out.lmax,
            self.irreps_out.lmax,
            resolution=grid_resolution,
            use_m_primary=False,
        )

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        gate_scalars = x_flat[:, :self.extra_m0_outsize]
        main_flat = x_flat[:, self.extra_m0_outsize:]

        x_arr = self.in_packer.pack(main_flat)
        out_scalars = self.scalar_act(x_arr[:, 0, :])

        x_grid = self.so3_grid.to_grid(x_arr)
        x1, x2 = torch.chunk(x_grid, chunks=2, dim=-1)
        out_arr = self.so3_grid.from_grid(x1 * x2)

        out_arr.mul_(self.gate_act(gate_scalars).unsqueeze(1))
        out_arr[:, 0, :] = out_arr[:, 0, :] + out_scalars
        return self.out_packer.unpack(out_arr)


def build_gate_activation(irreps_out: o3.Irreps) -> Gate:
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


def build_equivariant_norm(
    norm_type: str,
    irreps: o3.Irreps,
    norm_eps: float,
    dtype: Union[str, torch.dtype],
    device: Union[str, torch.device],
):
    if norm_type == "none":
        return None
    if norm_type == "merged_rms":
        return EquivariantMergedRMSNormFlat(
            irreps,
            eps=norm_eps,
            dtype=dtype,
            device=device,
        )
    raise ValueError(f"Unsupported equivariant_norm_type={norm_type!r}")
