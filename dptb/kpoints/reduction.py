"""K-point symmetry reduction routines."""

from typing import Optional, Tuple

import numpy as np

from dptb.kpoints.geometry import is_integer

# Try to import numba for JIT compilation
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Define a no-op decorator if numba is not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

def _wrap_to_bz(kpt: np.ndarray) -> np.ndarray:
    '''Wrap k-point to the first Brillouin zone [-0.5, 0.5).'''
    return (kpt + 0.5) % 1 - 0.5


def _validate_symm_prec(symm_prec: float) -> float:
    if not np.isfinite(symm_prec) or symm_prec <= 0:
        raise ValueError("symm_prec must be positive and finite.")
    return float(symm_prec)


def _validate_kvec_d(kvec_d: np.ndarray) -> np.ndarray:
    kvec_d = np.asarray(kvec_d)
    if kvec_d.ndim == 1:
        if kvec_d.shape[0] != 3:
            raise ValueError("kvec_d must have shape (3,) or (n, 3).")
        kvec_d = kvec_d.reshape(1, 3)
    elif kvec_d.ndim == 2:
        if kvec_d.shape[1] != 3:
            raise ValueError("kvec_d must have shape (3,) or (n, 3).")
    else:
        raise ValueError("kvec_d must have shape (3,) or (n, 3).")
    if kvec_d.shape[0] == 0:
        raise ValueError("kvec_d must contain at least one k-point.")
    if not np.all(np.isfinite(kvec_d)):
        raise ValueError("kvec_d must contain only finite values.")
    return kvec_d.astype(float, copy=False)


def _validate_symm_ops(symm_op) -> np.ndarray:
    symm_ops = np.asarray(symm_op)
    if symm_ops.ndim == 2 and symm_ops.shape == (3, 3):
        symm_ops = symm_ops.reshape(1, 3, 3)
    if symm_ops.ndim != 3 or symm_ops.shape[1:] != (3, 3) or symm_ops.shape[0] == 0:
        raise ValueError("symm_op must contain one or more 3x3 symmetry matrices.")
    if not np.all(np.isfinite(symm_ops)):
        raise ValueError("symm_op must contain only finite values.")
    return symm_ops.astype(float, copy=False)


def _can_use_packed_hash(symm_prec: float) -> bool:
    scale = 1.0 / symm_prec
    return np.isfinite(scale) and scale < 2**20


# =============================================================================
# Numba-optimized functions for k-point reduction
# =============================================================================

@njit(cache=True)
def _wrap_to_bz_numba(kpt):
    '''Wrap k-point to the first Brillouin zone [-0.5, 0.5). Numba-compatible.'''
    return (kpt + 0.5) % 1 - 0.5


@njit(cache=True)
def _norm_3d(v):
    '''Compute norm of a 3D vector. Numba-compatible.'''
    return np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])


@njit(cache=True)
def _reduce_by_symmetry_numba_direct_core(k, wk, symm_ops, symm_prec):
    '''
    Core reduction loop optimized with Numba.

    Parameters
    ----------
    k : np.ndarray
        k-points in direct coordinates, shape (nk, 3), dtype float64
    wk : np.ndarray
        weights/degeneracies for each k-point, dtype float64
    symm_ops : np.ndarray
        symmetry operation matrices, shape (m, 3, 3), dtype float64
    symm_prec : float
        precision for considering two k-points identical

    Returns
    -------
    wk : np.ndarray
        Updated weights (modified in place, but also returned)
    '''
    nk = k.shape[0]
    n_ops = symm_ops.shape[0]

    for i in range(nk - 1):
        if wk[i] == 0:
            continue

        ki = k[i]
        found = False

        for op_idx in range(n_ops):
            if found:
                break

            # Apply symmetry operation: transformed = ki @ op
            op = symm_ops[op_idx]
            transformed = np.zeros(3)
            for a in range(3):
                for b in range(3):
                    transformed[a] += ki[b] * op[b, a]

            # Search for equivalent k-point in k[i+1:]
            for j in range(i + 1, nk):
                if wk[j] == 0:
                    continue

                # Compute wrapped difference
                dk = np.zeros(3)
                for a in range(3):
                    diff = transformed[a] - k[j, a]
                    dk[a] = (diff + 0.5) % 1 - 0.5

                # Compute norm
                dist = _norm_3d(dk)

                if dist < symm_prec:
                    wk[j] += wk[i]
                    wk[i] = 0
                    found = True
                    break

    return wk


@njit(cache=True)
def _reduce_by_symmetry_numba_hash_core(k, wk, symm_ops, symm_prec,
                                         hash_keys, sorted_indices,
                                         group_starts, group_ends, scale):
    '''
    Hash-based reduction loop optimized with Numba.

    Uses pre-computed hash groups for O(n*m) complexity instead of O(n^2*m).

    Parameters
    ----------
    k : np.ndarray
        k-points in direct coordinates, shape (nk, 3)
    wk : np.ndarray
        weights/degeneracies for each k-point
    symm_ops : np.ndarray
        symmetry operation matrices, shape (m, 3, 3)
    symm_prec : float
        precision for considering two k-points identical
    hash_keys : np.ndarray
        unique hash keys, shape (n_groups,), dtype int64
    sorted_indices : np.ndarray
        indices sorted by hash group, shape (nk,)
    group_starts : np.ndarray
        start index for each group in sorted_indices, shape (n_groups,)
    group_ends : np.ndarray
        end index for each group in sorted_indices, shape (n_groups,)
    scale : float
        scale factor for hashing (1/symm_prec)
    '''
    nk = k.shape[0]
    n_ops = symm_ops.shape[0]
    n_groups = len(hash_keys)

    for i in range(nk - 1):
        if wk[i] == 0:
            continue

        ki = k[i]
        found = False

        for op_idx in range(n_ops):
            if found:
                break

            # Apply symmetry operation: transformed = ki @ op
            op = symm_ops[op_idx]
            transformed = np.zeros(3)
            for a in range(3):
                for b in range(3):
                    transformed[a] += ki[b] * op[b, a]

            # Compute hash for transformed k-point
            wrapped = _wrap_to_bz_numba(transformed)
            h0 = int(np.round(wrapped[0] * scale))
            h1 = int(np.round(wrapped[1] * scale))
            h2 = int(np.round(wrapped[2] * scale))
            # Encode as single int64 (assuming coordinates in reasonable range)
            h = h0 + h1 * 2097152 + h2 * 4398046511104  # 2^21 and 2^42

            # Binary search for hash key
            group_idx = -1
            lo, hi = 0, n_groups
            while lo < hi:
                mid = (lo + hi) // 2
                if hash_keys[mid] < h:
                    lo = mid + 1
                elif hash_keys[mid] > h:
                    hi = mid
                else:
                    group_idx = mid
                    break

            if group_idx < 0:
                continue

            # Search within the hash group
            start = group_starts[group_idx]
            end = group_ends[group_idx]

            for idx in range(start, end):
                j = sorted_indices[idx]
                if j <= i or wk[j] == 0:
                    continue

                # Compute wrapped difference
                dk = np.zeros(3)
                for a in range(3):
                    diff = transformed[a] - k[j, a]
                    dk[a] = (diff + 0.5) % 1 - 0.5

                dist = _norm_3d(dk)

                if dist < symm_prec:
                    wk[j] += wk[i]
                    wk[i] = 0
                    found = True
                    break

    return wk


def _build_hash_table_numba(k: np.ndarray, scale: float):
    '''
    Build hash table data structures compatible with Numba.

    Returns arrays that can be passed to the Numba-optimized reduction function.
    '''
    # Vectorized: wrap all k-points and compute integer hashes at once
    wrapped = _wrap_to_bz(k)
    rounded = np.round(wrapped * scale).astype(np.int64)

    # Encode as single int64 for sorting
    hash_vals = rounded[:, 0] + rounded[:, 1] * 2097152 + rounded[:, 2] * 4398046511104

    # Sort by hash value
    sorted_order = np.argsort(hash_vals)
    sorted_hashes = hash_vals[sorted_order]

    # Find unique hash keys and group boundaries
    unique_mask = np.concatenate([[True], sorted_hashes[1:] != sorted_hashes[:-1]])
    unique_indices = np.where(unique_mask)[0]

    hash_keys = sorted_hashes[unique_indices]
    group_starts = unique_indices.astype(np.int64)
    group_ends = np.concatenate([unique_indices[1:], [len(k)]]).astype(np.int64)

    return hash_keys, sorted_order.astype(np.int64), group_starts, group_ends


def _build_hash_table(k: np.ndarray, scale: float) -> Tuple[dict, np.ndarray]:
    '''
    Build hash table mapping k-point hashes to indices using vectorized operations.

    Parameters
    ----------
    k : np.ndarray
        k-points array of shape (nk, 3)
    scale : float
        scale factor for hashing (typically 1/symm_prec)

    Returns
    -------
    Tuple[dict, np.ndarray]
        - hash_to_indices: mapping from hash tuple to list of indices
        - rounded: the rounded integer coordinates (nk, 3)
    '''
    from collections import defaultdict

    # Vectorized: wrap all k-points and compute integer hashes at once
    wrapped = _wrap_to_bz(k)
    rounded = np.round(wrapped * scale).astype(np.int32)

    # Convert to tuple keys using a view-based approach for speed
    # Using structured array view for fast row-wise hashing
    hash_to_indices = defaultdict(list)

    # Direct iteration with tuple conversion - simpler and often faster for pure Python dict
    for i in range(len(rounded)):
        key = (rounded[i, 0], rounded[i, 1], rounded[i, 2])
        hash_to_indices[key].append(i)

    return dict(hash_to_indices), rounded


def _compute_hash_key(kpt_rounded: np.ndarray) -> tuple:
    '''Compute hash key from rounded k-point coordinates.'''
    return (int(kpt_rounded[0]), int(kpt_rounded[1]), int(kpt_rounded[2]))


def _reduce_by_symmetry_direct(k: np.ndarray,
                               wk: np.ndarray,
                               symm_ops: np.ndarray,
                               symm_prec: float) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Direct O(n^2*m) method for reducing k-points by symmetry operations.
    Faster for small meshes due to low overhead.

    Parameters
    ----------
    k : np.ndarray
        k-points in direct coordinates, shape (nk, 3)
    wk : np.ndarray
        weights/degeneracies for each k-point
    symm_ops : List[np.ndarray]
        list of symmetry operation matrices (3x3)
    symm_prec : float
        precision for considering two k-points identical
    '''
    nk = len(k)
    # For each kpoint, apply each symmetry operation and check against all subsequent kpoints
    # If found equivalent, merge weights and mark the original as zero
    for i in range(nk - 1):
        if wk[i] == 0:
            continue
        ki = k[i]
        for op in symm_ops:
            alive = wk[i + 1:] > 0
            if not np.any(alive):
                break
            # Compute periodic distance: wrap to [-0.5, 0.5) for minimum distance
            dk = np.dot(ki, op) - k[i + 1:][alive]
            dk = _wrap_to_bz(dk)
            dk = np.linalg.norm(dk, axis=1)
            if np.any(dk < symm_prec):
                alive_idx = np.flatnonzero(alive) + i + 1
                j = alive_idx[np.argmin(dk)] # if exits multiple minima, take the first one
                wk[j] += wk[i]
                wk[i] = 0
                break  # once found one, stop searching
    return k[wk > 0], wk[wk > 0]


def _reduce_by_symmetry_hash(k: np.ndarray,
                             wk: np.ndarray,
                             symm_ops: np.ndarray,
                             symm_prec: float) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Hash-based O(n*m) method for reducing k-points by symmetry operations.
    Faster for large meshes.

    Parameters
    ----------
    k : np.ndarray
        k-points in direct coordinates, shape (nk, 3)
    wk : np.ndarray
        weights/degeneracies for each k-point
    symm_ops : List[np.ndarray]
        list of symmetry operation matrices (3x3)
    symm_prec : float
        precision for considering two k-points identical
    '''
    nk = len(k)
    scale = 1.0 / symm_prec

    # Build hash table using vectorized operations
    hash_to_indices, _ = _build_hash_table(k, scale)

    # Stack all symmetry operations for batch transformation
    symm_ops_arr = np.array(symm_ops)  # (m, 3, 3)

    # For each k-point, apply symmetry operations and look up equivalent points
    for i in range(nk - 1):
        if wk[i] == 0:
            continue
        ki = k[i]

        # Apply all symmetry operations: ki @ op for each op
        # Here we use einsum for batch matrix multiplication to transform all ops at once
        # For row vector ki: ki @ op[m] for each m
        transformed_all = np.einsum('j,mjk->mk', ki, symm_ops_arr)  # ki @ op for each op

        # Find the first matching j > i (consistent with direct method which stops
        # at the first symmetry operation that finds any match)
        found = False # flag to stop symm ops loop when found a match
        for transformed in transformed_all:
            if found:
                break

            # Compute hash for transformed k-point
            wrapped = _wrap_to_bz(transformed)
            rounded = np.round(wrapped * scale).astype(np.int32)
            h = _compute_hash_key(rounded)

            if h not in hash_to_indices:
                continue

            for j in hash_to_indices[h]:
                if j <= i or wk[j] == 0:
                    continue
                dk = _wrap_to_bz(transformed - k[j])
                if np.linalg.norm(dk) < symm_prec:
                    wk[j] += wk[i]
                    wk[i] = 0
                    found = True
                    break  # Found match for this op, stop

    return k[wk > 0], wk[wk > 0]


def _reduce_by_symmetry_numba(k: np.ndarray,
                              wk: np.ndarray,
                              symm_ops: np.ndarray,
                              symm_prec: float,
                              use_hash: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Numba-optimized k-point reduction by symmetry operations.

    Parameters
    ----------
    k : np.ndarray
        k-points in direct coordinates, shape (nk, 3)
    wk : np.ndarray
        weights/degeneracies for each k-point
    symm_ops : List[np.ndarray]
        list of symmetry operation matrices (3x3)
    symm_prec : float
        precision for considering two k-points identical
    use_hash : bool
        if True, use hash-based O(n*m) method; otherwise use direct O(n^2*m)
    '''
    # Ensure correct dtypes for Numba
    k_arr = np.ascontiguousarray(k, dtype=np.float64)
    wk_arr = np.ascontiguousarray(wk, dtype=np.float64)
    symm_ops_arr = np.ascontiguousarray(np.array(symm_ops), dtype=np.float64)

    if use_hash:
        scale = 1.0 / symm_prec
        hash_keys, sorted_indices, group_starts, group_ends = _build_hash_table_numba(k_arr, scale)
        wk_result = _reduce_by_symmetry_numba_hash_core(
            k_arr, wk_arr, symm_ops_arr, symm_prec,
            hash_keys, sorted_indices, group_starts, group_ends, scale
        )
    else:
        wk_result = _reduce_by_symmetry_numba_direct_core(k_arr, wk_arr, symm_ops_arr, symm_prec)

    mask = wk_result > 0
    return k_arr[mask], wk_result[mask]


def _reduce_by_symmetry(k: np.ndarray,
                        wk: np.ndarray,
                        symm_ops: np.ndarray,
                        symm_prec: float,
                        threshold: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Reduce k-points using symmetry operations (rotation or time-inversion).
    Automatically selects the best method based on mesh size and Numba availability.

    Parameters
    ----------
    k : np.ndarray
        k-points in direct coordinates, shape (nk, 3)
    wk : np.ndarray
        weights/degeneracies for each k-point
    symm_ops : List[np.ndarray]
        list of symmetry operation matrices (3x3)
    symm_prec : float
        precision for considering two k-points identical
    threshold : int
        crossover point for switching between direct and hash methods
    '''
    nk = len(k)
    use_packed_hash = _can_use_packed_hash(symm_prec)

    # Use Numba-optimized version if available and mesh is large enough
    if HAS_NUMBA and nk > 100:
        use_hash = nk > threshold and use_packed_hash
        return _reduce_by_symmetry_numba(k, wk, symm_ops, symm_prec, use_hash=use_hash)

    # Fallback to pure Python implementation
    if nk > threshold and use_packed_hash:
        return _reduce_by_symmetry_hash(k, wk, symm_ops, symm_prec)
    else:
        return _reduce_by_symmetry_direct(k, wk, symm_ops, symm_prec)


def reduce_time_inversion(kvec_d: np.ndarray,
                          symm_prec: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Reduce k-points using time-inversion symmetry (k -> -k).

    Uses a hybrid approach: hash-based O(n) for large meshes,
    direct O(n^2) for small meshes where it has less overhead.
    '''
    symm_prec = _validate_symm_prec(symm_prec)
    k = _validate_kvec_d(kvec_d)
    nk = len(k)
    wk = np.ones(shape=(nk,), dtype=float)

    if nk == 1:
        return k, wk

    # Time inversion: k -> -k, represented as -I matrix
    time_inversion_op = [-np.eye(3)]
    return _reduce_by_symmetry(k, wk, time_inversion_op, symm_prec, threshold=2000)


def reduce_rotation(kvec_d: np.ndarray,
                    symm_op: np.ndarray,
                    symm_prec: float = 1e-8,
                    degeneracies_: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Reduce the Brillouin zone by iteratively imposing the symmetry operations on k-points.
    If the transformed k-point is identical with any other, they are considered equivalent
    and their weights are merged.

    Parameters
    ----------
    kvec_d : np.ndarray
        the collection of k points in direct coordinates
    symm_op : List[np.ndarray]
        the list of symmetry operations of the crystal
    symm_prec : float, optional
        The precision for considering two k points to be identical, default is 1e-8
    degeneracies_ : List[float], optional
        The initial weights for the k points, default is None

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The reduced k points and their degeneracies
    '''
    k = _validate_kvec_d(kvec_d)
    symm_op = _validate_symm_ops(symm_op)
    symm_prec = _validate_symm_prec(symm_prec)
    nk = len(k)

    degen = np.ones(shape=(nk,), dtype=float)
    if degeneracies_ is not None:
        degeneracies_ = np.asarray(degeneracies_, dtype=float)
        if degeneracies_.shape != (nk,):
            raise ValueError("degeneracies_ must have one value per k-point.")
        if not np.all(np.isfinite(degeneracies_)):
            raise ValueError("degeneracies_ must contain only finite values.")
        if not all(is_integer(x) for x in degeneracies_):
            degeneracies_ = degeneracies_ * nk
            if not all(is_integer(x) for x in degeneracies_):
                raise ValueError("degeneracies_ must be integer counts or normalized integer counts.")
        degen = degeneracies_.astype(float)

    if nk == 1:
        return k, degen.astype(int)

    k_reduced, degen_reduced = _reduce_by_symmetry(k, degen, symm_op, symm_prec, threshold=500)
    return k_reduced, degen_reduced.astype(int)

def reduce(kvec_d: np.ndarray,
           symm_op: np.ndarray,
           time_inversion_symmetry: bool = True,
           symm_prec: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    '''
    reduce the primitive k vectors in direct coordinates, optionally (but default) also taking
    the time-inversion symmetry into consideration. All kpoints that are identical (indistinguishable
    under the threshold `symm_prec`) after any symmetry operation will be considered equivalent
    and their weights will be merged.

    Parameters
    ----------
    kvec_d : np.ndarray
        the collection of k points in direct coordinates
    symm_op : List[np.ndarray]
        the list of symmetry operations of the crystal
    time_inversion_symmetry : bool, optional
        Whether to include time-inversion symmetry, default is True
    symm_prec : float, optional
        The precision for considering two k points to be identical, default is 1e-8

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The reduced k points and their degeneracies
    '''
    symm_op = _validate_symm_ops(symm_op)
    symm_prec = _validate_symm_prec(symm_prec)
    kvec_d = _validate_kvec_d(kvec_d)
    wk = None
    if time_inversion_symmetry:
        kvec_d, wk = reduce_time_inversion(kvec_d, symm_prec)
    return reduce_rotation(kvec_d, symm_op, symm_prec=symm_prec, degeneracies_=wk)
