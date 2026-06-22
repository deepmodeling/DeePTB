"""Compatibility wrapper for k-point sampling utilities.

New code should prefer importing from :mod:`dptb.kpoints`.
"""

from dptb.kpoints.geometry import (
    calculate_reciprocal_vectors,
    get_symm_ops,
    is_integer,
)
from dptb.kpoints.mesh import (
    build_kmeshgrid,
    monkhorst_pack_sampling,
    mp,
)
from dptb.kpoints.reduction import (
    HAS_NUMBA,
    _build_hash_table,
    _build_hash_table_numba,
    _compute_hash_key,
    _norm_3d,
    _reduce_by_symmetry,
    _reduce_by_symmetry_direct,
    _reduce_by_symmetry_hash,
    _reduce_by_symmetry_numba,
    _reduce_by_symmetry_numba_direct_core,
    _reduce_by_symmetry_numba_hash_core,
    _wrap_to_bz,
    _wrap_to_bz_numba,
    reduce,
    reduce_rotation,
    reduce_time_inversion,
)
from dptb.kpoints.sampling import sample

__all__ = [
    "HAS_NUMBA",
    "_build_hash_table",
    "_build_hash_table_numba",
    "_compute_hash_key",
    "_norm_3d",
    "_reduce_by_symmetry",
    "_reduce_by_symmetry_direct",
    "_reduce_by_symmetry_hash",
    "_reduce_by_symmetry_numba",
    "_reduce_by_symmetry_numba_direct_core",
    "_reduce_by_symmetry_numba_hash_core",
    "_wrap_to_bz",
    "_wrap_to_bz_numba",
    "build_kmeshgrid",
    "calculate_reciprocal_vectors",
    "get_symm_ops",
    "is_integer",
    "monkhorst_pack_sampling",
    "mp",
    "reduce",
    "reduce_rotation",
    "reduce_time_inversion",
    "sample",
]
