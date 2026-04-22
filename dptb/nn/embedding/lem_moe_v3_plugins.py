from __future__ import annotations

import torch
from e3nn import o3
from e3nn.nn import Gate

from .eqv3_grid_helpers import (
    EqV3StyleNodeFFN,
    FlatSwiGLUS2Merge,
    build_equivariant_norm,
    can_use_flat_s2_patch,
)


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
