"""AtomicData: neighbor graphs in (periodic) real space.

Authors: Albert Musaelian
"""

import warnings
from copy import deepcopy
from typing import Union, Tuple, Dict, Optional, List, Set, Sequence
from collections.abc import Mapping
import os

import numpy as np
import ase.neighborlist
import ase
from ase.calculators.singlepoint import SinglePointCalculator, SinglePointDFTCalculator
from ase.calculators.calculator import all_properties as ase_all_properties
from ase.stress import voigt_6_to_full_3x3_stress, full_3x3_to_voigt_6_stress

import torch
import e3nn.o3

from . import AtomicDataDict
from .util import _TORCH_INTEGER_DTYPES
from dptb.utils.torch_geometric.data import Data
from dptb.utils.constants import atomic_num_dict
import logging

log = logging.getLogger(__name__)

# A type representing ASE-style periodic boundary condtions, which can be partial (the tuple case)
PBC = Union[bool, Tuple[bool, bool, bool]]


_DEFAULT_NESTED_FIELDS : Set[str] = {
    # AtomicDataDict.HAMILTONIAN_KEY,
    # AtomicDataDict.OVERLAP_KEY, not support nested type in this two since nested format does not support complex dtype
    AtomicDataDict.ENERGY_EIGENVALUE_KEY,
    AtomicDataDict.KPOINT_KEY,
}


_DEFAULT_LONG_FIELDS: Set[str] = {
    AtomicDataDict.EDGE_INDEX_KEY,
    AtomicDataDict.ENV_INDEX_KEY, # new
    AtomicDataDict.ONSITENV_INDEX_KEY, # new
    AtomicDataDict.ATOMIC_NUMBERS_KEY,
    AtomicDataDict.ATOM_TYPE_KEY,
    AtomicDataDict.BATCH_KEY,
    AtomicDataDict.ATOM_INDEX_KEY,
    AtomicDataDict.GHOST_ATOM_INDEX_KEY,
    AtomicDataDict.GHOST_ATOM_OWNER_KEY,
    AtomicDataDict.GHOST_ATOM_TYPE_KEY,
    AtomicDataDict.GHOST_ATOMIC_NUMBERS_KEY,
}

_DEFAULT_NODE_FIELDS: Set[str] = {
    AtomicDataDict.POSITIONS_KEY,
    AtomicDataDict.NODE_FEATURES_KEY,
    AtomicDataDict.NODE_ATTRS_KEY,
    AtomicDataDict.ATOMIC_NUMBERS_KEY,
    AtomicDataDict.ATOM_TYPE_KEY,
    AtomicDataDict.FORCE_KEY,
    AtomicDataDict.PER_ATOM_ENERGY_KEY,
    AtomicDataDict.NODE_HAMILTONIAN_KEY,
    AtomicDataDict.NODE_OVERLAP_KEY,
    AtomicDataDict.BATCH_KEY,
    AtomicDataDict.ATOM_INDEX_KEY,
    AtomicDataDict.NODE_SOC_KEY, # new
    AtomicDataDict.NODE_THREECENTER_KEY # new
}

_DEFAULT_EDGE_FIELDS: Set[str] = {
    AtomicDataDict.EDGE_CELL_SHIFT_KEY,
    AtomicDataDict.EDGE_VECTORS_KEY,
    AtomicDataDict.EDGE_LENGTH_KEY,
    AtomicDataDict.EDGE_ATTRS_KEY,
    AtomicDataDict.EDGE_EMBEDDING_KEY,
    AtomicDataDict.EDGE_FEATURES_KEY,
    AtomicDataDict.EDGE_CUTOFF_KEY,
    AtomicDataDict.EDGE_ENERGY_KEY,
    AtomicDataDict.EDGE_OVERLAP_KEY,
    AtomicDataDict.EDGE_HAMILTONIAN_KEY,
    AtomicDataDict.EDGE_TYPE_KEY,
    AtomicDataDict.EDGE_THREECENTER_KEY # new
}

_DEFAULT_ENV_FIELDS: Set[str] = {
    AtomicDataDict.ENV_CELL_SHIFT_KEY,
    AtomicDataDict.ENV_VECTORS_KEY,
    AtomicDataDict.ENV_LENGTH_KEY,
    AtomicDataDict.ENV_ATTRS_KEY,
    AtomicDataDict.ENV_EMBEDDING_KEY,
    AtomicDataDict.ENV_FEATURES_KEY,
    AtomicDataDict.ENV_CUTOFF_KEY,
}

_DEFAULT_ONSITENV_FIELDS: Set[str] = {
    AtomicDataDict.ONSITENV_CELL_SHIFT_KEY,
    AtomicDataDict.ONSITENV_VECTORS_KEY,
    AtomicDataDict.ONSITENV_LENGTH_KEY,
    AtomicDataDict.ONSITENV_ATTRS_KEY,
    AtomicDataDict.ONSITENV_EMBEDDING_KEY,
    AtomicDataDict.ONSITENV_FEATURES_KEY,
    AtomicDataDict.ONSITENV_CUTOFF_KEY,
}

_DEFAULT_GRAPH_FIELDS: Set[str] = {
    AtomicDataDict.TOTAL_ENERGY_KEY,
    AtomicDataDict.STRESS_KEY,
    AtomicDataDict.VIRIAL_KEY,
    AtomicDataDict.PBC_KEY,
    AtomicDataDict.CELL_KEY,
    AtomicDataDict.BATCH_PTR_KEY,
    AtomicDataDict.KPOINT_KEY, # new # should be nested
    AtomicDataDict.HAMILTONIAN_KEY, # new # should be nested
    AtomicDataDict.OVERLAP_KEY, # new # should be nested
    AtomicDataDict.ENERGY_EIGENVALUE_KEY, # new # should be nested
    AtomicDataDict.EIGENVECTOR_KEY, # new # should be nested
    AtomicDataDict.ENERGY_WINDOWS_KEY, # new,
    AtomicDataDict.BAND_WINDOW_KEY, # new,
    AtomicDataDict.NODE_SOC_SWITCH_KEY # new
}

_NODE_FIELDS: Set[str] = set(_DEFAULT_NODE_FIELDS)
_EDGE_FIELDS: Set[str] = set(_DEFAULT_EDGE_FIELDS)
_ENV_FIELDS: Set[str] = set(_DEFAULT_ENV_FIELDS)
_ONSITENV_FIELDS: Set[str] = set(_DEFAULT_ONSITENV_FIELDS)
_GRAPH_FIELDS: Set[str] = set(_DEFAULT_GRAPH_FIELDS)
_LONG_FIELDS: Set[str] = set(_DEFAULT_LONG_FIELDS)
_NESTED_FIELDS: Set[str] = set(_DEFAULT_NESTED_FIELDS)
_GHOST_FIELDS: Set[str] = {
    AtomicDataDict.GHOST_ATOM_INDEX_KEY,
    AtomicDataDict.GHOST_ATOM_OWNER_KEY,
    AtomicDataDict.GHOST_POSITIONS_KEY,
    AtomicDataDict.GHOST_ATOM_TYPE_KEY,
    AtomicDataDict.GHOST_ATOMIC_NUMBERS_KEY,
}
class _RIBNode:
    """Internal finite geometry for one recursive inertial partition node."""

    __slots__ = ("rank", "axis", "threshold", "left", "right", "bbox_min", "bbox_max")

    def __init__(self, bbox_min, bbox_max, rank=None, axis=None, threshold=0.0,
                 left=None, right=None):
        self.rank = rank
        self.axis = axis
        self.threshold = float(threshold)
        self.left = left
        self.right = right
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max

    @property
    def is_leaf(self):
        return self.rank is not None


def _partition_pbc(pbc) -> tuple[bool, bool, bool]:
    """Normalize the PBC spellings accepted by ``from_points``."""
    if pbc is None:
        return (False, False, False)
    if isinstance(pbc, (bool, np.bool_)):
        value = bool(pbc)
        return (value, value, value)
    values = torch.as_tensor(pbc, dtype=torch.bool).flatten().tolist()
    if len(values) != 3:
        raise ValueError("pbc must be a bool or contain exactly three booleans")
    return tuple(bool(value) for value in values)


def _max_cutoff(*cutoffs) -> float:
    """Return the largest numeric value in scalar/dict cutoff specifications."""
    values = []
    for cutoff in cutoffs:
        if cutoff is None:
            continue
        if isinstance(cutoff, Mapping):
            values.extend(float(v) for v in cutoff.values())
        elif isinstance(cutoff, torch.Tensor):
            values.append(float(cutoff.detach().max().cpu()))
        else:
            values.append(float(cutoff))
    if not values:
        raise ValueError("At least one cutoff must be provided")
    if not all(np.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("Neighbor cutoffs must be finite and positive")
    return max(values)


def _construction_cell(cell, pbc):
    """Return a full-rank internal cell while preserving periodic vectors."""
    pbc = _partition_pbc(pbc)
    if cell is None:
        if any(pbc):
            raise ValueError("Periodic graph construction requires a cell")
        return None
    raw = torch.as_tensor(
        np.array(cell, dtype=np.float64, copy=True), dtype=torch.float64
    ).view(3, 3).cpu()
    if not bool(torch.isfinite(raw).all()):
        raise ValueError("Cell contains non-finite values")
    periodic = raw[torch.as_tensor(pbc, dtype=torch.bool)]
    if len(periodic):
        singular_values = torch.linalg.svdvals(periodic)
        scale = max(float(singular_values.max()), 1.0)
        rank = int((singular_values > 64.0 * torch.finfo(torch.float64).eps * scale).sum())
        if rank != len(periodic):
            raise ValueError("Periodic cell vectors must be nonzero and linearly independent")
    completed = torch.as_tensor(
        ase.geometry.complete_cell(raw.numpy()), dtype=torch.float64
    )
    if abs(float(torch.linalg.det(completed))) < 1e-14:
        raise ValueError("Cell could not be completed to a nonsingular geometry cell")
    return completed


def _geometry_tolerance(cell, cutoff, dtype=torch.float64):
    """Roundoff allowance used only by the temporary candidate search."""
    scale = max(1.0, float(cutoff))
    if cell is not None:
        scale = max(scale, float(torch.as_tensor(cell, dtype=torch.float64).abs().max()))
    if not torch.empty((), dtype=dtype).is_floating_point():
        dtype = torch.float64
    return 64.0 * torch.finfo(dtype).eps * scale


def _partition_coordinates(pos: torch.Tensor, cell, pbc, return_wrap=False):
    """Return a deterministic, continuous coordinate chart for periodic RIB.

    The chart differs from the caller's coordinates only by per-atom lattice
    translations and one common origin translation.  The former are returned
    so neighbour-list shifts can always be expressed against the untouched
    public positions.
    """
    pbc = _partition_pbc(pbc)
    xyz = pos.to(torch.float64)
    if cell is None or not any(pbc):
        wrap = torch.zeros((len(xyz), 3), dtype=torch.long)
        return (xyz, wrap) if return_wrap else xyz
    cell_t = _construction_cell(cell, pbc)
    frac = xyz @ torch.linalg.inv(cell_t)
    wrap = torch.zeros((len(xyz), 3), dtype=torch.long)
    seam_fractional = torch.zeros(3, dtype=torch.float64)
    for dim, periodic in enumerate(pbc):
        if not periodic or len(xyz) == 0:
            continue
        floor = torch.floor(frac[:, dim])
        wrapped = frac[:, dim] - floor
        # Put the periodic seam in the largest empty gap.  Equal gaps are
        # resolved by the atom id at the right side of the gap.
        order = sorted(range(len(xyz)), key=lambda i: (float(wrapped[i]), i))
        values = wrapped[torch.as_tensor(order, dtype=torch.long)]
        gaps = torch.empty_like(values)
        if len(values) == 1:
            gaps[0] = 1.0
        else:
            gaps[:-1] = values[1:] - values[:-1]
            gaps[-1] = values[0] + 1.0 - values[-1]
        max_gap = float(gaps.max())
        choices = [i for i, gap in enumerate(gaps) if abs(float(gap) - max_gap) <= 1e-14]
        gap_index = min(choices, key=lambda i: order[(i + 1) % len(order)])
        seam = float(values[(gap_index + 1) % len(values)])
        below = wrapped < seam
        wrap[:, dim] = (-floor).to(torch.long) + below.to(torch.long)
        seam_fractional[dim] = seam
    # Written directly against the caller's coordinates so incomplete cells
    # and atoms outside the primary image retain exact (i,j,R) semantics.
    result = xyz + wrap.to(torch.float64) @ cell_t - seam_fractional @ cell_t
    return (result, wrap) if return_wrap else result


def recursive_inertial_bisection(
    pos,
    world_size: int,
    cell=None,
    pbc=(False, False, False),
    return_tree: bool = False,
):
    """Return deterministic equal-atom RIB ownership and optionally its cut tree."""
    pos = torch.as_tensor(pos, dtype=torch.get_default_dtype(), device="cpu")
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("RIB positions must have shape [N, 3]")
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if pos.shape[0] < world_size:
        raise ValueError(
            f"RIB needs at least one atom per rank, got {pos.shape[0]} atoms and {world_size} ranks"
        )

    owners = torch.empty(pos.shape[0], dtype=torch.long)
    partition_pos = _partition_coordinates(pos, cell, pbc)

    def split(indices: torch.Tensor, ranks: list[int]):
        xyz = partition_pos[indices]
        bbox_min = xyz.amin(dim=0)
        bbox_max = xyz.amax(dim=0)
        if len(ranks) == 1:
            owners[indices] = ranks[0]
            return _RIBNode(bbox_min, bbox_max, rank=ranks[0])
        centered = xyz - xyz.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered
        axis = torch.linalg.eigh(covariance).eigenvectors[:, -1]
        # Eigenvectors are sign-ambiguous. Canonicalize the sign so rank
        # assignments do not flip between LAPACK implementations.
        pivot = int(torch.argmax(axis.abs()))
        if axis[pivot] < 0:
            axis = -axis
        projection = centered @ axis
        order = sorted(
            range(len(indices)),
            key=lambda i: (float(projection[i]), int(indices[i])),
        )
        ordered = indices[torch.as_tensor(order, dtype=torch.long)]
        left_nranks = len(ranks) // 2
        left_natoms = (len(indices) * left_nranks) // len(ranks)
        left_natoms = max(
            left_nranks,
            min(left_natoms, len(indices) - (len(ranks) - left_nranks)),
        )
        absolute_projection = xyz @ axis
        ordered_projection = absolute_projection[torch.as_tensor(order, dtype=torch.long)]
        threshold = 0.5 * (
            float(ordered_projection[left_natoms - 1])
            + float(ordered_projection[left_natoms])
        )
        left = split(ordered[:left_natoms], ranks[:left_nranks])
        right = split(ordered[left_natoms:], ranks[left_nranks:])
        return _RIBNode(
            bbox_min, bbox_max, axis=axis, threshold=threshold,
            left=left, right=right,
        )

    tree = split(torch.arange(pos.shape[0], dtype=torch.long), list(range(world_size)))
    return (owners, tree) if return_tree else owners


def _mark_ranks_within_radius(tree, point: torch.Tensor, radius: float, marked: set):
    """Mark finite RIB subtrees whose atom bounds intersect a cutoff sphere."""
    outside = torch.maximum(tree.bbox_min - point, point - tree.bbox_max).clamp_min(0.0)
    if float(torch.dot(outside, outside)) > float(radius) * float(radius):
        return
    if tree.is_leaf:
        marked.add(int(tree.rank))
        return
    distance = float(torch.dot(point, tree.axis)) - tree.threshold
    if distance <= radius:
        _mark_ranks_within_radius(tree.left, point, radius, marked)
    if distance >= -radius:
        _mark_ranks_within_radius(tree.right, point, radius, marked)


def _partition_image_shifts(cell, pbc, cutoff: float):
    """Periodic images whose cutoff spheres can reach the primary-cell RIB tree."""
    pbc = _partition_pbc(pbc)
    if cell is None or not any(pbc):
        return torch.zeros((1, 3), dtype=torch.long)
    cell_t = _construction_cell(cell, pbc)
    reciprocal_norm = torch.linalg.vector_norm(torch.linalg.inv(cell_t), dim=0)
    shells = [
        int(np.ceil(cutoff * float(reciprocal_norm[dim]))) if pbc[dim] else 0
        for dim in range(3)
    ]
    return torch.tensor(
        [
            (i, j, k)
            for i in range(-shells[0], shells[0] + 1)
            for j in range(-shells[1], shells[1] + 1)
            for k in range(-shells[2], shells[2] + 1)
        ],
        dtype=torch.long,
    )


def _exchange_halo_records_neighbor(outgoing, rank, comm):
    """Exchange variable halo records over a sparse MPI distributed graph."""
    from mpi4py import MPI

    local_records = list(outgoing[rank])
    remote_peers = [peer for peer, records in enumerate(outgoing)
                    if peer != rank and records]
    graph = comm.Create_dist_graph(
        [rank], [len(remote_peers)], remote_peers, reorder=False
    )
    try:
        sources, destinations, _weights = graph.Get_dist_neighbors()
        send_counts = np.asarray(
            [len(outgoing[int(peer)]) for peer in destinations], dtype=np.int32
        )
        recv_counts = np.empty(len(sources), dtype=np.int32)
        graph.Neighbor_alltoall(send_counts, recv_counts)
        send_displs = np.zeros(len(send_counts), dtype=np.int64)
        recv_displs = np.zeros(len(recv_counts), dtype=np.int64)
        if len(send_counts) > 1:
            send_displs[1:] = np.cumsum(send_counts[:-1], dtype=np.int64)
        if len(recv_counts) > 1:
            recv_displs[1:] = np.cumsum(recv_counts[:-1], dtype=np.int64)
        packed = [record for peer in destinations for record in outgoing[int(peer)]]
        send_meta = np.asarray(
            [[r[0], r[2], r[3], r[4], r[5], r[6]] for r in packed],
            dtype=np.int64,
        ).reshape(-1, 6)
        send_pos = np.asarray(
            [[r[7], r[8], r[9]] for r in packed], dtype=np.float64
        ).reshape(-1, 3)
        recv_total = int(recv_counts.sum())
        recv_meta = np.empty((recv_total, 6), dtype=np.int64)
        recv_pos = np.empty((recv_total, 3), dtype=np.float64)
        graph.Neighbor_alltoallv(
            [send_meta.reshape(-1), send_counts * 6, send_displs * 6, MPI.INT64_T],
            [recv_meta.reshape(-1), recv_counts * 6, recv_displs * 6, MPI.INT64_T],
        )
        graph.Neighbor_alltoallv(
            [send_pos.reshape(-1), send_counts * 3, send_displs * 3, MPI.DOUBLE],
            [recv_pos.reshape(-1), recv_counts * 3, recv_displs * 3, MPI.DOUBLE],
        )
        records = local_records
        cursor = 0
        for source, count in zip(sources, recv_counts.tolist()):
            for row in range(cursor, cursor + count):
                meta = recv_meta[row]
                xyz = recv_pos[row]
                records.append((
                    int(meta[0]), int(source), int(meta[1]), int(meta[2]),
                    int(meta[3]), int(meta[4]), int(meta[5]),
                    float(xyz[0]), float(xyz[1]), float(xyz[2]),
                ))
            cursor += count
        return records
    finally:
        graph.Free()


def _collect_halo_images(pos, atomic_numbers, atom_types, owners, tree, rank,
                         world_size, cutoff, cell=None, pbc=False):
    """Offer explicit periodic atom images to every intersected RIB region."""
    pos = torch.as_tensor(pos, dtype=torch.float64, device="cpu")
    numbers = torch.as_tensor(atomic_numbers, dtype=torch.long).flatten().cpu()
    types = torch.as_tensor(atom_types, dtype=torch.long).flatten().cpu()
    owners = torch.as_tensor(owners, dtype=torch.long, device="cpu")
    chart_pos, chart_wrap = _partition_coordinates(pos, cell, pbc, return_wrap=True)
    shifts = _partition_image_shifts(cell, pbc, cutoff)
    cell_t = (
        torch.as_tensor(cell, dtype=torch.float64).view(3, 3)
        if cell is not None
        else torch.zeros((3, 3), dtype=torch.float64)
    )
    owned = torch.nonzero(owners == rank, as_tuple=False).flatten()
    outgoing = [[] for _ in range(world_size)]

    for atom in owned.tolist():
        for image_shift in shifts:
            image = chart_pos[atom] + image_shift.to(torch.float64) @ cell_t
            destinations = set()
            _mark_ranks_within_radius(tree, image, cutoff, destinations)
            identity = not bool(torch.any(image_shift))
            total_shift = chart_wrap[atom] + image_shift
            record = (
                atom, rank, int(numbers[atom]), int(types[atom]),
                int(total_shift[0]), int(total_shift[1]), int(total_shift[2]),
                float(image[0]), float(image[1]), float(image[2]),
            )
            for destination in destinations:
                if destination == rank and identity:
                    continue
                outgoing[destination].append(record)

    comm = None
    try:
        from dptb.utils.distributed import get_mpi_comm
        candidate = get_mpi_comm()
        if (candidate is not None and candidate.Get_size() == world_size
                and candidate.Get_rank() == rank):
            comm = candidate
    except (ImportError, AttributeError):
        pass

    if comm is not None and world_size > 1:
        if hasattr(comm, "Create_dist_graph"):
            records = _exchange_halo_records_neighbor(outgoing, rank, comm)
        else:
            # Lightweight communicator stand-ins used by unit tests.
            incoming = comm.alltoall(outgoing)
            records = [record for source in incoming for record in source]
    else:
        # Deterministic single-process construction path used by unit tests and
        # preprocessing tools that materialize partitions one rank at a time.
        records = []
        for atom in range(len(pos)):
            owner = int(owners[atom])
            for image_shift in shifts:
                image = chart_pos[atom] + image_shift.to(torch.float64) @ cell_t
                destinations = set()
                _mark_ranks_within_radius(tree, image, cutoff, destinations)
                identity = not bool(torch.any(image_shift))
                if rank not in destinations or (owner == rank and identity):
                    continue
                total_shift = chart_wrap[atom] + image_shift
                records.append((
                    atom, owner, int(numbers[atom]), int(types[atom]),
                    int(total_shift[0]), int(total_shift[1]), int(total_shift[2]),
                    float(image[0]), float(image[1]), float(image[2]),
                ))
    records = list({
        (item[1], item[0], item[4], item[5], item[6]): item
        for item in records
    }.values())
    records.sort(key=lambda item: (item[1], item[0], item[4], item[5], item[6]))
    return owned, chart_pos, chart_wrap, records


def _deduplicate_relation(index, shifts):
    """Sort and deduplicate a directed relation by compact endpoints and R."""
    entries = sorted(set(
        (int(index[0, k]), int(index[1, k]),
         int(shifts[k, 0]), int(shifts[k, 1]), int(shifts[k, 2]))
        for k in range(index.shape[1])
    ))
    if not entries:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 3), dtype=shifts.dtype)
    return (
        torch.tensor([[e[0] for e in entries], [e[1] for e in entries]], dtype=torch.long),
        torch.tensor([e[2:] for e in entries], dtype=shifts.dtype),
    )


def _partitioned_neighbor_candidates(pos, cell, pbc, search_cutoff,
                                     self_interaction, partition_options, kwargs):
    """Build this rank's owner-row candidate relations for ``from_points``.

    Partitions the structure with RIB, collects the halo images that can
    reach the owned region, and runs one neighbour search over owned plus
    halo coordinates.  ``kwargs`` is updated in place: ghost-column fields
    are added and public node fields are sliced to owner rows.  Returns
    ``(pos, cell, candidate_index, candidate_cell_shift, column_pos,
    column_numbers)`` in the same layout as the serial neighbour search.
    """
    forbidden = {"rank", "world_size"}.intersection(partition_options)
    if forbidden:
        raise ValueError(
            f"{', '.join(sorted(forbidden))} must not be supplied in "
            "partition_options; process identity comes from the initialized "
            "distributed context"
        )
    method = str(partition_options.get("method", "rib")).lower()
    if method != "rib":
        raise ValueError(f"Unsupported graph partition method {method!r}; expected 'rib'")
    # Import lazily to avoid a module cycle: the communication helpers
    # themselves operate on AtomicDataDict.
    from dptb.utils.distributed import (
        get_rank,
        get_world_size,
        require_distributed_initialized,
    )
    require_distributed_initialized()
    rank = get_rank()
    world_size = get_world_size()

    global_natoms = pos.shape[0]
    global_pos = pos
    if kwargs.get(AtomicDataDict.ATOMIC_NUMBERS_KEY) is None:
        raise ValueError(
            "partitioned AtomicData requires atomic_numbers for compact halo metadata"
        )
    global_numbers = torch.as_tensor(
        kwargs.get(AtomicDataDict.ATOMIC_NUMBERS_KEY), dtype=torch.long
    ).flatten()
    if global_numbers.numel() != global_natoms:
        raise ValueError("partitioned AtomicData requires one atomic number per atom")
    global_types_value = kwargs.get(AtomicDataDict.ATOM_TYPE_KEY)
    global_types = (
        torch.as_tensor(global_types_value, dtype=torch.long).flatten()
        if global_types_value is not None
        else torch.full((global_natoms,), -1, dtype=torch.long)
    )
    owners, rib_tree = recursive_inertial_bisection(
        pos=global_pos,
        world_size=world_size,
        cell=cell,
        pbc=pbc,
        return_tree=True,
    )
    owned_ids, chart_pos, chart_wrap, halo_records = _collect_halo_images(
        pos=global_pos,
        atomic_numbers=global_numbers,
        atom_types=global_types,
        owners=owners,
        tree=rib_tree,
        rank=rank,
        world_size=world_size,
        cutoff=search_cutoff,
        cell=cell,
        pbc=pbc,
    )
    owned_ids = owned_ids.sort().values
    n_owned = int(owned_ids.numel())

    owned_public_pos = global_pos[owned_ids].clone()
    owned_chart_pos = chart_pos[owned_ids].clone()
    owned_chart_wrap = chart_wrap[owned_ids].clone()
    halo_public_pos = {
        int(record[0]): global_pos[int(record[0])].clone()
        for record in halo_records
    }
    # From this point graph construction sees only owned coordinates
    # plus explicitly offered images, never the replicated full chart.
    del global_pos, chart_pos, chart_wrap
    pos = None

    construction_pos = owned_chart_pos.tolist()
    construction_numbers = global_numbers[owned_ids].tolist()
    construction_gid = owned_ids.tolist()
    construction_shift = owned_chart_wrap.tolist()
    for record in halo_records:
        construction_gid.append(int(record[0]))
        construction_numbers.append(int(record[2]))
        construction_shift.append([int(record[4]), int(record[5]), int(record[6])])
        construction_pos.append([float(record[7]), float(record[8]), float(record[9])])

    construction_pos = torch.as_tensor(
        construction_pos, dtype=owned_public_pos.dtype
    )
    construction_numbers = torch.as_tensor(construction_numbers, dtype=torch.long)
    raw_index, _, cell_tensor = neighbor_list_and_relative_vec(
        pos=construction_pos,
        r_max=search_cutoff,
        self_interaction=False,
        cell=cell,
        reduce=False,
        atomic_numbers=construction_numbers,
        pbc=False,
    )
    raw_index = raw_index[:, raw_index[0] < n_owned]
    if self_interaction:
        owned_self = torch.arange(n_owned, dtype=torch.long)
        raw_index = torch.cat(
            (raw_index, torch.stack((owned_self, owned_self), dim=0)),
            dim=1,
        )
    owned_lookup = {int(gid): lid for lid, gid in enumerate(owned_ids.tolist())}
    used_ghosts = {
        int(construction_gid[int(dst)])
        for dst in raw_index[1].tolist()
        if int(construction_gid[int(dst)]) not in owned_lookup
    }
    ghost_ids = sorted(used_ghosts, key=lambda gid: (int(owners[gid]), gid))
    ghost_lookup = {gid: n_owned + k for k, gid in enumerate(ghost_ids)}

    compact_src, compact_dst, relation_shift = [], [], []
    for src, dst in raw_index.T.tolist():
        src_gid = int(construction_gid[src])
        dst_gid = int(construction_gid[dst])
        compact_src.append(owned_lookup[src_gid])
        compact_dst.append(owned_lookup.get(dst_gid, ghost_lookup.get(dst_gid)))
        relation_shift.append([
            int(construction_shift[dst][d]) - int(owned_chart_wrap[src, d])
            for d in range(3)
        ])
    candidate_index = torch.tensor(
        [compact_src, compact_dst], dtype=torch.long
    ) if compact_src else torch.empty((2, 0), dtype=torch.long)
    candidate_cell_shift = torch.tensor(
        relation_shift, dtype=owned_public_pos.dtype
    ) if relation_shift else torch.empty((0, 3), dtype=owned_public_pos.dtype)
    candidate_index, candidate_cell_shift = _deduplicate_relation(
        candidate_index, candidate_cell_shift
    )

    ghost_ids_t = torch.as_tensor(ghost_ids, dtype=torch.long)
    kwargs[AtomicDataDict.ATOM_INDEX_KEY] = owned_ids
    kwargs[AtomicDataDict.GHOST_ATOM_INDEX_KEY] = ghost_ids_t
    kwargs[AtomicDataDict.GHOST_ATOM_OWNER_KEY] = owners[ghost_ids_t]
    kwargs[AtomicDataDict.GHOST_POSITIONS_KEY] = (
        torch.stack([halo_public_pos[gid] for gid in ghost_ids], dim=0)
        if ghost_ids
        else owned_public_pos.new_empty((0, 3))
    )
    kwargs[AtomicDataDict.GHOST_ATOMIC_NUMBERS_KEY] = global_numbers[ghost_ids_t]
    kwargs[AtomicDataDict.GHOST_ATOM_TYPE_KEY] = global_types[ghost_ids_t]

    # Slice public node fields to owner rows only.
    for key in list(kwargs):
        value = kwargs[key]
        if key not in _NODE_FIELDS or value is None:
            continue
        value_t = torch.as_tensor(value)
        if value_t.ndim > 0 and value_t.shape[0] == global_natoms:
            kwargs[key] = value_t[owned_ids]
    pos = owned_public_pos
    column_pos = torch.cat((pos, kwargs[AtomicDataDict.GHOST_POSITIONS_KEY]), dim=0)
    column_numbers = torch.cat((
        torch.as_tensor(kwargs[AtomicDataDict.ATOMIC_NUMBERS_KEY]).flatten(),
        kwargs[AtomicDataDict.GHOST_ATOMIC_NUMBERS_KEY].flatten(),
    ))
    cell = cell_tensor
    return pos, cell, candidate_index, candidate_cell_shift, column_pos, column_numbers


def _relation_cutoff_mask(pos, index, shifts, cell, cutoff, atomic_numbers,
                          tolerance=0.0):
    """Filter one maximum-cutoff relation using DeePTB's existing cutoff rules."""
    vectors = pos[index[1]] - pos[index[0]]
    if cell is not None:
        vectors = vectors + shifts.to(pos.dtype) @ cell.reshape(3, 3).to(pos.dtype)
    lengths = torch.linalg.vector_norm(vectors, dim=1)
    if not isinstance(cutoff, Mapping):
        return lengths <= float(cutoff) + float(tolerance)

    numbers = torch.as_tensor(atomic_numbers, dtype=torch.long).flatten()
    first_key = next(iter(cutoff))
    if len(first_key.split("-")) == 1:
        r_map = get_r_map(cutoff, numbers)
        limits = 0.5 * (r_map[numbers[index[0]] - 1] + r_map[numbers[index[1]] - 1])
    elif len(first_key.split("-")) == 2:
        r_map = get_r_map_bondwise(cutoff, numbers)
        limits = r_map[numbers[index[0]] - 1, numbers[index[1]] - 1]
    else:
        raise ValueError("Cutoff keys must be atomic symbols or symbol pairs")
    return lengths <= limits.to(lengths.device) + float(tolerance)


def register_fields(
    node_fields: Sequence[str] = [],
    edge_fields: Sequence[str] = [],
    env_fields: Sequence[str] = [],
    onsitenv_fields: Sequence[str] = [],
    graph_fields: Sequence[str] = [],
    long_fields: Sequence[str] = [],
) -> None:
    
    r"""Register fields as being per-atom, per-edge, or per-frame.

    Args:
        node_permute_fields: fields that are equivariant to node permutations.
        edge_permute_fields: fields that are equivariant to edge permutations.
    """
    
    node_fields: set = set(node_fields)
    edge_fields: set = set(edge_fields)
    env_fields: set = set(env_fields)
    onsitenv_fields: set = set(onsitenv_fields)
    graph_fields: set = set(graph_fields)
    long_fields: set = set(long_fields)
    allfields = node_fields.union(edge_fields, graph_fields, env_fields, onsitenv_fields)
    assert len(allfields) == len(node_fields) + len(edge_fields) + len(graph_fields)
    _NODE_FIELDS.update(node_fields)
    _EDGE_FIELDS.update(edge_fields)
    _ENV_FIELDS.update(env_fields)
    _ONSITENV_FIELDS.update(onsitenv_fields)
    _GRAPH_FIELDS.update(graph_fields)
    _LONG_FIELDS.update(long_fields)
    if len(set.union(_NODE_FIELDS, _EDGE_FIELDS, _GRAPH_FIELDS)) < (
        len(_NODE_FIELDS) + len(_EDGE_FIELDS) + len(_GRAPH_FIELDS)
    ):
        raise ValueError(
            "At least one key was registered as more than one of node, edge, or graph!"
        )


def deregister_fields(*fields: Sequence[str]) -> None:
    r"""Deregister a field registered with ``register_fields``.

    Silently ignores fields that were never registered to begin with.

    Args:
        *fields: fields to deregister.
    """
    for f in fields:
        assert f not in _DEFAULT_NODE_FIELDS, "Cannot deregister built-in field"
        assert f not in _DEFAULT_EDGE_FIELDS, "Cannot deregister built-in field"
        assert f not in _DEFAULT_GRAPH_FIELDS, "Cannot deregister built-in field"
        assert f not in _DEFAULT_ENV_FIELDS, "Cannot deregister built-in field"
        assert f not in _DEFAULT_ONSITENV_FIELDS, "Cannot deregister built-in field"
        _NODE_FIELDS.discard(f)
        _EDGE_FIELDS.discard(f)
        _ENV_FIELDS.discard(f)
        _ONSITENV_FIELDS.discard(f)
        _GRAPH_FIELDS.discard(f)


def _register_field_prefix(prefix: str) -> None:
    """Re-register all registered fields as the same type, but with `prefix` added on."""
    assert prefix.endswith("_")
    register_fields(
        node_fields=[prefix + e for e in _NODE_FIELDS],
        edge_fields=[prefix + e for e in _EDGE_FIELDS],
        env_fields=[prefix + e for e in _ENV_FIELDS],
        onsitenv_fields=[prefix + e for e in _ONSITENV_FIELDS],
        graph_fields=[prefix + e for e in _GRAPH_FIELDS],
        long_fields=[prefix + e for e in _LONG_FIELDS],
    )


def _process_dict(kwargs, ignore_fields=[]):
    """Convert a dict of data into correct dtypes/shapes according to key"""
    # Deal with _some_ dtype issues

    # assert all nested_field is also graph_field
    for field in _NESTED_FIELDS:
        assert field in _GRAPH_FIELDS

    if AtomicDataDict.BATCH_KEY in kwargs:
        num_frames = kwargs[AtomicDataDict.BATCH_KEY].max() + 1
    else:
        num_frames = 1

    for k, v in kwargs.items():
        if k in ignore_fields:
            continue

        if k in _LONG_FIELDS:
            # Any property used as an index must be long (or byte or bool, but those are not relevant for atomic scale systems)
            # int32 would pass later checks, but is actually disallowed by torch
            kwargs[k] = torch.as_tensor(v, dtype=torch.long)
        elif k in _NESTED_FIELDS: # we need to transform the input array or tensors into nested tensors
            if num_frames > 1:
                if isinstance(v, np.ndarray): # this suggest that the dimension for each features are the same
                    v = torch.nested.as_nested_tensor(list(torch.as_tensor(v)), dtype=torch.get_default_dtype())
                elif isinstance(v, torch.Tensor) and not getattr(v, "is_nested"): # this also suggest the dimenion are the same
                    v = torch.nested.as_nested_tensor(list(v), dtype=torch.get_default_dtype())
                elif isinstance(v, list):
                    v = torch.nested.as_nested_tensor(v, dtype=torch.get_default_dtype())
                else:
                    assert v.is_nested # this assert that v is a tensor and is nested
            else:
                if isinstance(v, np.ndarray):
                    v = torch.as_tensor(v, dtype=torch.get_default_dtype())
                    v = torch.nested.as_nested_tensor([v], dtype=torch.get_default_dtype())
                elif isinstance(v, torch.Tensor) and not getattr(v, "is_nested"):
                    v = torch.nested.as_nested_tensor([v], dtype=torch.get_default_dtype())
                else:
                    assert v.is_nested

            kwargs[k] = v
        elif isinstance(v, bool):
            kwargs[k] = torch.as_tensor(v)
        elif isinstance(v, np.ndarray):
            if np.issubdtype(v.dtype, np.floating):
                kwargs[k] = torch.as_tensor(v, dtype=torch.get_default_dtype())
            else:
                kwargs[k] = torch.as_tensor(v)
        elif isinstance(v, list):
            ele_dtype = np.array(v).dtype
            if np.issubdtype(ele_dtype, np.floating):
                kwargs[k] = torch.as_tensor(v, dtype=torch.get_default_dtype())
            else:
                kwargs[k] = torch.as_tensor(v)
        elif np.issubdtype(type(v), np.floating):
            # Force scalars to be tensors with a data dimension
            # This makes them play well with irreps
            kwargs[k] = torch.as_tensor(v, dtype=torch.get_default_dtype())
        elif isinstance(v, torch.Tensor) and len(v.shape) == 0:
            # ^ this tensor is a scalar; we need to give it
            # a data dimension to play nice with irreps
            kwargs[k] = v

    for k, v in kwargs.items():
        if k in ignore_fields:
            continue

        if k not in _NESTED_FIELDS:
            if len(v.shape) == 0:
                kwargs[k] = v.unsqueeze(-1)
                v = kwargs[k]

            if k in set.union(_NODE_FIELDS, _EDGE_FIELDS) and len(v.shape) == 1:
                kwargs[k] = v.unsqueeze(-1)
                v = kwargs[k]

            if (
                k in _NODE_FIELDS
                and AtomicDataDict.POSITIONS_KEY in kwargs
                and v.shape[0] != kwargs[AtomicDataDict.POSITIONS_KEY].shape[0]
            ):
                raise ValueError(
                    f"{k} is a node field but has the wrong dimension {v.shape}"
                )
            elif (
                k in _EDGE_FIELDS
                and AtomicDataDict.EDGE_INDEX_KEY in kwargs
                and v.shape[0] != kwargs[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
            ):
                raise ValueError(
                    f"{k} is a edge field but has the wrong dimension {v.shape}"
                )
            elif (
                k in _ENV_FIELDS
                and AtomicDataDict.ENV_INDEX_KEY in kwargs
                and v.shape[0] != kwargs[AtomicDataDict.ENV_INDEX_KEY].shape[1]
            ):
                raise ValueError(
                    f"{k} is a env field but has the wrong dimension {v.shape}"
                )
            elif (
                k in _ONSITENV_FIELDS
                and AtomicDataDict.ONSITENV_INDEX_KEY in kwargs
                and v.shape[0] != kwargs[AtomicDataDict.ONSITENV_INDEX_KEY].shape[1]
            ):
                raise ValueError(
                    f"{k} is a env field but has the wrong dimension {v.shape}"
                )
            elif k in _GHOST_FIELDS:
                ghost_ids = kwargs.get(AtomicDataDict.GHOST_ATOM_INDEX_KEY)
                if ghost_ids is None or v.shape[0] != ghost_ids.shape[0]:
                    raise ValueError(
                        f"{k} is a ghost-column field but has the wrong dimension {v.shape}"
                    )
            elif k in _GRAPH_FIELDS:
                if num_frames > 1 and v.shape[0] != num_frames:
                    raise ValueError(f"Wrong shape for graph property {k}")
        else: # when k in NESTED_FIELD, and also k must be in GRAPH_FIELDS
            if num_frames > 1 and v.size(0) != num_frames:
                raise ValueError(f"Wrong shape for NESTED property {k}")


class AtomicData(Data):
    """A neighbor graph for points in (periodic triclinic) real space.

    For typical cases either ``from_points`` or ``from_ase`` should be used to
    construct a AtomicData; they also standardize and check their input much more
    thoroughly.

    In general, ``node_features`` are features or input information on the nodes that will be fed through and transformed by the network, while ``node_attrs`` are _encodings_ fixed, inherant attributes of the atoms themselves that remain constant through the network.
    For example, a one-hot _encoding_ of atomic species is a node attribute, while some observed instantaneous property of that atom (current partial charge, for example), would be a feature.

    In general, ``torch.Tensor`` arguments should be of consistant dtype. Numpy arrays will be converted to ``torch.Tensor``s; those of floating point dtype will be converted to ``torch.get_current_dtype()`` regardless of their original precision. Scalar values (Python scalars or ``torch.Tensor``s of shape ``()``) a resized to tensors of shape ``[1]``. Per-atom scalar values should be given with shape ``[N_at, 1]``.

    ``AtomicData`` should be used for all data creation and manipulation outside of the model; inside of the model ``AtomicDataDict.Type`` is used.

    Args:
        pos (Tensor [n_nodes, 3]): Positions of the nodes.
        edge_index (LongTensor [2, n_edges]): ``edge_index[0]`` is the per-edge
            index of the source node and ``edge_index[1]`` is the target node.
        edge_cell_shift (Tensor [n_edges, 3], optional): which periodic image
            of the target point each edge goes to, relative to the source point.
        cell (Tensor [1, 3, 3], optional): the periodic cell for
            ``edge_cell_shift`` as the three triclinic cell vectors.
        node_features (Tensor [n_atom, ...]): the input features of the nodes, optional
        node_attrs (Tensor [n_atom, ...]): the attributes of the nodes, for instance the atom type, optional
        batch (Tensor [n_atom]): the graph to which the node belongs, optional
        atomic_numbers (Tensor [n_atom]): optional.
        atom_type (Tensor [n_atom]): optional.
        **kwargs: other data, optional.
    """

    def __init__(
        self, irreps: Dict[str, e3nn.o3.Irreps] = {}, _validate: bool = True, **kwargs
    ):

        # empty init needed by get_example
        if len(kwargs) == 0 and len(irreps) == 0:
            super().__init__()
            return

        # Check the keys
        if _validate:
            AtomicDataDict.validate_keys(kwargs)
            _process_dict(kwargs)

        super().__init__(num_nodes=len(kwargs["pos"]), **kwargs)

        if _validate:
            # Validate shapes
            assert self.pos.dim() == 2 and self.pos.shape[1] == 3
            assert self.edge_index.dim() == 2 and self.edge_index.shape[0] == 2
            if "edge_cell_shift" in self and self.edge_cell_shift is not None:
                assert self.edge_cell_shift.shape == (self.num_edges, 3)
                assert self.edge_cell_shift.dtype == self.pos.dtype
            # TODO: should we add checks for env too ?
            if "cell" in self and self.cell is not None:
                assert (self.cell.shape == (3, 3)) or (
                    self.cell.dim() == 3 and self.cell.shape[1:] == (3, 3)
                )
                assert self.cell.dtype == self.pos.dtype
            if "node_features" in self and self.node_features is not None:
                assert self.node_features.shape[0] == self.num_nodes
                assert self.node_features.dtype == self.pos.dtype
            if "node_attrs" in self and self.node_attrs is not None:
                assert self.node_attrs.shape[0] == self.num_nodes
                assert self.node_attrs.dtype == self.pos.dtype

            if (
                AtomicDataDict.ATOMIC_NUMBERS_KEY in self
                and self.atomic_numbers is not None
            ):
                assert self.atomic_numbers.dtype in _TORCH_INTEGER_DTYPES
            if "batch" in self and self.batch is not None:
                assert self.batch.dim() == 2 and self.batch.shape[0] == self.num_nodes
                # Check that there are the right number of cells
                if "cell" in self and self.cell is not None:
                    cell = self.cell.view(-1, 3, 3)
                    assert cell.shape[0] == self.batch.max() + 1

            # Validate irreps
            # __*__ is the only way to hide from torch_geometric
            self.__irreps__ = AtomicDataDict._fix_irreps_dict(irreps)
            for field, irreps in self.__irreps__:
                if irreps is not None:
                    assert self[field].shape[-1] == irreps.dim

    @classmethod
    def from_points(
        cls,
        pos=None,
        r_max: Union[float, int, dict] = None,
        self_interaction: bool = False,
        cell=None,
        pbc: Optional[PBC] = None,
        er_max: Optional[float] = None,
        oer_max: Optional[float] = None,
        partition_options: Optional[Dict[str, object]] = None,
        **kwargs,
    ):
        """Build neighbor graph from points, optionally with PBC.

        Args:
            pos (np.ndarray/torch.Tensor shape [N, 3]): node positions. If Tensor, must be on the CPU.
            r_max (float): neighbor cutoff radius.
            cell (ase.Cell/ndarray [3,3], optional): periodic cell for the points. Defaults to ``None``.
            pbc (bool or 3-tuple of bool, optional): whether to apply periodic boundary conditions to all or each of
            the three cell vector directions. Defaults to ``False``.
            self_interaction (bool, optional): whether to include self edges for points. Defaults to ``False``. Note
            that edges between the same atom in different periodic images are still included. (See
            ``strict_self_interaction`` to control this behaviour.)
            strict_self_interaction (bool): Whether to include *any* self interaction edges in the graph, even if the
                two instances of the atom are in different periodic images. Defaults to True, should be True for most
                applications.
            partition_options (dict, optional): Partition algorithm settings, for example
                ``{"method": "rib"}``. The rank and world size are taken from DeePTB's
                initialized distributed context and must not be supplied here.
            **kwargs (optional): other fields to add. Keys listed in ``AtomicDataDict.*_KEY` will be treated specially.
        """
        if pos is None or r_max is None:
            raise ValueError("pos and r_max must be given.")

        if pbc is None:
            if cell is not None:
                raise ValueError(
                    "A cell was provided, but pbc weren't. Please explicitly probide PBC."
                )
            # there are no PBC if cell and pbc are not provided
            pbc = False

        if isinstance(pbc, bool):
            pbc = (pbc,) * 3
        else:
            assert len(pbc) == 3

        pos = torch.as_tensor(pos, dtype=torch.get_default_dtype())
        if not bool(torch.isfinite(pos).all()):
            raise ValueError("Positions contain non-finite values")
        maximum_cutoff = _max_cutoff(r_max, er_max, oer_max)
        construction_cell = _construction_cell(cell, pbc) if cell is not None else None
        geometry_tolerance = _geometry_tolerance(
            construction_cell, maximum_cutoff, pos.dtype
        )
        search_cutoff = maximum_cutoff + geometry_tolerance

        if partition_options is not None:
            (pos, cell, candidate_index, candidate_cell_shift,
             column_pos, column_numbers) = _partitioned_neighbor_candidates(
                pos=pos,
                cell=cell,
                pbc=pbc,
                search_cutoff=search_cutoff,
                self_interaction=self_interaction,
                partition_options=partition_options,
                kwargs=kwargs,
            )
        else:
            candidate_index, candidate_cell_shift, cell = neighbor_list_and_relative_vec(
                pos=pos,
                r_max=search_cutoff,
                self_interaction=self_interaction,
                cell=cell,
                reduce=False,
                atomic_numbers=kwargs.get("atomic_numbers", None),
                pbc=pbc,
            )
            column_pos = pos
            column_numbers = kwargs.get(AtomicDataDict.ATOMIC_NUMBERS_KEY)

        edge_keep = _relation_cutoff_mask(
            column_pos, candidate_index, candidate_cell_shift, cell, r_max,
            column_numbers,
        )
        edge_index = candidate_index[:, edge_keep]
        edge_cell_shift = candidate_cell_shift[edge_keep]

        # Make torch tensors for data:
        if cell is not None:
            kwargs[AtomicDataDict.CELL_KEY] = cell.view(3, 3)
            kwargs[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = edge_cell_shift
        if pbc is not None:
            kwargs[AtomicDataDict.PBC_KEY] = torch.as_tensor(
                pbc, dtype=torch.bool
            ).view(3)

        # add env index
        if er_max is not None:
            env_keep = _relation_cutoff_mask(
                column_pos, candidate_index, candidate_cell_shift, cell, er_max,
                column_numbers,
            )
            env_index = candidate_index[:, env_keep]
            env_cell_shift = candidate_cell_shift[env_keep]

            if cell is not None:
                kwargs[AtomicDataDict.ENV_CELL_SHIFT_KEY] = env_cell_shift
            kwargs[AtomicDataDict.ENV_INDEX_KEY] = env_index
        
        # add onsitenv index
        if oer_max is not None:
            onsitenv_keep = _relation_cutoff_mask(
                column_pos, candidate_index, candidate_cell_shift, cell, oer_max,
                column_numbers,
            )
            onsitenv_index = candidate_index[:, onsitenv_keep]
            onsitenv_cell_shift = candidate_cell_shift[onsitenv_keep]

            if cell is not None:
                kwargs[AtomicDataDict.ONSITENV_CELL_SHIFT_KEY] = onsitenv_cell_shift
            kwargs[AtomicDataDict.ONSITENV_INDEX_KEY] = onsitenv_index
            
        return cls(edge_index=edge_index, pos=torch.as_tensor(pos), **kwargs)

    @classmethod
    def from_ase(
        cls,
        atoms,
        r_max: Union[float, int, dict],
        er_max: Optional[float] = None,
        oer_max: Optional[float] = None,
        key_mapping: Optional[Dict[str, str]] = {},
        include_keys: Optional[list] = [],
        partition_options: Optional[Dict[str, object]] = None,
        **kwargs,
    ):
        """Build a ``AtomicData`` from an ``ase.Atoms`` object.

        Respects ``atoms``'s ``pbc`` and ``cell``.

        First tries to extract energies and forces from a single-point calculator associated with the ``Atoms`` if one is present and has those fields.
        If either is not found, the method will look for ``energy``/``energies`` and ``force``/``forces`` in ``atoms.arrays``.

        `get_atomic_numbers()` will be stored as the atomic_numbers attribute.

        Args:
            atoms (ase.Atoms): the input.
            r_max (float): neighbor cutoff radius.
            features (torch.Tensor shape [N, M], optional): per-atom M-dimensional feature vectors. If ``None`` (the
             default), uses a one-hot encoding of the species present in ``atoms``.
            include_keys (list): list of additional keys to include in AtomicData aside from the ones defined in
                 ase.calculators.calculator.all_properties. Optional
            key_mapping (dict): rename ase property name to a new string name. Optional
            **kwargs (optional): other arguments for the ``AtomicData`` constructor.

        Returns:
            A ``AtomicData``.
        """
        # from nequip.ase import NequIPCalculator

        assert "pos" not in kwargs

        default_args = set(
            [
                "numbers",
                "positions",
            ]  # ase internal names for position and atomic_numbers
            + ["pbc", "cell", "pos", "r_max", "er_max", "oer_max", "partition_options"]  # arguments for from_points method
            + list(kwargs.keys())
        )
        # the keys that are duplicated in kwargs are removed from the include_keys
        include_keys = list(
            set(include_keys + ase_all_properties + list(key_mapping.keys()))
            - default_args
        )

        km = {
            "forces": AtomicDataDict.FORCE_KEY,
            "energy": AtomicDataDict.TOTAL_ENERGY_KEY,
        }
        km.update(key_mapping)
        key_mapping = km

        add_fields = {}

        # Get info from atoms.arrays; lowest priority. copy first
        add_fields = {
            key_mapping.get(k, k): v
            for k, v in atoms.arrays.items()
            if k in include_keys
        }

        # Get info from atoms.info; second lowest priority.
        add_fields.update(
            {
                key_mapping.get(k, k): v
                for k, v in atoms.info.items()
                if k in include_keys
            }
        )

        # if atoms.calc is not None:

        #     if isinstance(
        #         atoms.calc, (SinglePointCalculator, SinglePointDFTCalculator)
        #     ):
        #         add_fields.update(
        #             {
        #                 key_mapping.get(k, k): deepcopy(v)
        #                 for k, v in atoms.calc.results.items()
        #                 if k in include_keys
        #             }
        #         )
        #     elif isinstance(atoms.calc, NequIPCalculator):
        #         pass  # otherwise the calculator breaks
        #     else:
        #         raise NotImplementedError(
        #             f"`from_ase` does not support calculator {atoms.calc}"
        #         )

        add_fields[AtomicDataDict.ATOMIC_NUMBERS_KEY] = atoms.get_atomic_numbers()

        # cell and pbc in kwargs can override the ones stored in atoms
        cell = kwargs.pop("cell", atoms.get_cell())
        pbc = kwargs.pop("pbc", atoms.pbc)

        # handle ASE-style 6 element Voigt order stress
        for key in (AtomicDataDict.STRESS_KEY, AtomicDataDict.VIRIAL_KEY):
            if key in add_fields:
                if add_fields[key].shape == (3, 3):
                    # it's already 3x3, do nothing else
                    pass
                elif add_fields[key].shape == (6,):
                    # it's Voigt order
                    add_fields[key] = voigt_6_to_full_3x3_stress(add_fields[key])
                else:
                    raise RuntimeError(f"bad shape for {key}")

        return cls.from_points(
            pos=atoms.positions,
            r_max=r_max,
            er_max=er_max,
            oer_max=oer_max,
            cell=cell,
            pbc=pbc,
            partition_options=partition_options,
            **kwargs,
            **add_fields,
        )

    def to_ase(
        self,
        type_mapper=None,
        extra_fields: List[str] = [],
    ) -> Union[List[ase.Atoms], ase.Atoms]:
        """Build a (list of) ``ase.Atoms`` object(s) from an ``AtomicData`` object.

        For each unique batch number provided in ``AtomicDataDict.BATCH_KEY``,
        an ``ase.Atoms`` object is created. If ``AtomicDataDict.BATCH_KEY`` does not
        exist in self, a single ``ase.Atoms`` object is created.

        Args:
            type_mapper: if provided, will be used to map ``ATOM_TYPES`` back into
                elements, if the configuration of the ``type_mapper`` allows.
            extra_fields: fields other than those handled explicitly (currently
                those defining the structure as well as energy, per-atom energy,
                and forces) to include in the output object. Per-atom (per-node)
                quantities will be included in ``arrays``; per-graph and per-edge
                quantities will be included in ``info``.

        Returns:
            A list of ``ase.Atoms`` objects if ``AtomicDataDict.BATCH_KEY`` is in self
            and is not None. Otherwise, a single ``ase.Atoms`` object is returned.
        """
        positions = self.pos
        edge_index = self[AtomicDataDict.EDGE_INDEX_KEY]
        if positions.device != torch.device("cpu"):
            raise TypeError(
                "Explicitly move this `AtomicData` to CPU using `.to()` before calling `to_ase()`."
            )
        if AtomicDataDict.ATOMIC_NUMBERS_KEY in self:
            atomic_nums = self.atomic_numbers
        elif type_mapper is not None and type_mapper.has_chemical_symbols:
            atomic_nums = type_mapper.untransform(self[AtomicDataDict.ATOM_TYPE_KEY])
        else:
            warnings.warn(
                "AtomicData.to_ase(): self didn't contain atomic numbers... using atom_type as atomic numbers instead, but this means the chemical symbols in ASE (outputs) will be wrong"
            )
            atomic_nums = self[AtomicDataDict.ATOM_TYPE_KEY]
        pbc = getattr(self, AtomicDataDict.PBC_KEY, None)
        cell = getattr(self, AtomicDataDict.CELL_KEY, None)
        batch = getattr(self, AtomicDataDict.BATCH_KEY, None)
        energy = getattr(self, AtomicDataDict.TOTAL_ENERGY_KEY, None)
        energies = getattr(self, AtomicDataDict.PER_ATOM_ENERGY_KEY, None)
        force = getattr(self, AtomicDataDict.FORCE_KEY, None)
        do_calc = any(
            k in self
            for k in [
                AtomicDataDict.TOTAL_ENERGY_KEY,
                AtomicDataDict.FORCE_KEY,
                AtomicDataDict.PER_ATOM_ENERGY_KEY,
                AtomicDataDict.STRESS_KEY,
            ]
        )

        # exclude those that are special for ASE and that we process seperately
        special_handling_keys = [
            AtomicDataDict.POSITIONS_KEY,
            AtomicDataDict.CELL_KEY,
            AtomicDataDict.PBC_KEY,
            AtomicDataDict.ATOMIC_NUMBERS_KEY,
            AtomicDataDict.TOTAL_ENERGY_KEY,
            AtomicDataDict.FORCE_KEY,
            AtomicDataDict.PER_ATOM_ENERGY_KEY,
            AtomicDataDict.STRESS_KEY,
        ]
        assert (
            len(set(extra_fields).intersection(special_handling_keys)) == 0
        ), f"Cannot specify keys handled in special ways ({special_handling_keys}) as `extra_fields` for atoms output--- they are output by default"

        if cell is not None:
            cell = cell.view(-1, 3, 3)
        if pbc is not None:
            pbc = pbc.view(-1, 3)

        if batch is not None:
            n_batches = batch.max() + 1
            cell = cell.expand(n_batches, 3, 3) if cell is not None else None
            pbc = pbc.expand(n_batches, 3) if pbc is not None else None
        else:
            n_batches = 1

        batch_atoms = []
        for batch_idx in range(n_batches):
            if batch is not None:
                mask = batch == batch_idx
                mask = mask.view(-1)
                if hasattr(self, AtomicDataDict.GHOST_ATOM_INDEX_KEY):
                    # Owner-row relations belong to the graph of their source;
                    # destination indices use a separate compact column space.
                    edge_mask = mask[edge_index[0]]
                else:
                    edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
            else:
                mask = slice(None)
                edge_mask = slice(None)

            mol = ase.Atoms(
                numbers=atomic_nums[mask].view(-1),  # must be flat for ASE
                positions=positions[mask],
                cell=cell[batch_idx] if cell is not None else None,
                pbc=pbc[batch_idx] if pbc is not None else None,
            )

            if do_calc:
                fields = {}
                if energies is not None:
                    fields["energies"] = energies[mask].cpu().numpy()
                if energy is not None:
                    fields["energy"] = energy[batch_idx].cpu().numpy()
                if force is not None:
                    fields["forces"] = force[mask].cpu().numpy()
                if AtomicDataDict.STRESS_KEY in self:
                    fields["stress"] = full_3x3_to_voigt_6_stress(
                        self["stress"].view(-1, 3, 3)[batch_idx].cpu().numpy()
                    )
                mol.calc = SinglePointCalculator(mol, **fields)

            # add other information
            for key in extra_fields:
                if key in _NODE_FIELDS:
                    # mask it
                    mol.arrays[key] = (
                        self[key][mask].cpu().numpy().reshape(mask.sum(), -1)
                    )
                elif key in _EDGE_FIELDS:
                    mol.info[key] = (
                        self[key][edge_mask].cpu().numpy().reshape(edge_mask.sum(), -1)
                    )
                elif key == AtomicDataDict.EDGE_INDEX_KEY:
                    mol.info[key] = self[key][:, edge_mask].cpu().numpy()
                elif key in _GRAPH_FIELDS:
                    mol.info[key] = self[key][batch_idx].cpu().numpy().reshape(-1)
                else:
                    raise RuntimeError(
                        f"Extra field `{key}` isn't registered as node/edge/graph"
                    )

            batch_atoms.append(mol)

        if batch is not None:
            return batch_atoms
        else:
            assert len(batch_atoms) == 1
            return batch_atoms[0]

    def get_edge_vectors(data: Data) -> torch.Tensor:
        data = AtomicDataDict.with_edge_vectors(AtomicData.to_AtomicDataDict(data))
        return data[AtomicDataDict.EDGE_VECTORS_KEY]
    
    def get_env_vectors(data: Data) -> torch.Tensor:
        data = AtomicDataDict.with_env_vectors(AtomicData.to_AtomicDataDict(data))
        return data[AtomicDataDict.ENV_VECTORS_KEY]

    @staticmethod
    def to_AtomicDataDict(
        data: Union[Data, Mapping], exclude_keys=tuple()
    ) -> AtomicDataDict.Type:
        if isinstance(data, Data):
            keys = data.keys
        elif isinstance(data, Mapping):
            keys = data.keys()
        else:
            raise ValueError(f"Invalid data `{repr(data)}`")

        return {
            k: data[k]
            for k in keys
            if (
                k not in exclude_keys
                and data[k] is not None
                and isinstance(data[k], torch.Tensor)
            )
        }

    @classmethod
    def from_AtomicDataDict(cls, data: AtomicDataDict.Type):
        # it's an AtomicDataDict, so don't validate-- assume valid:
        return cls(_validate=False, **data)

    @property
    def irreps(self):
        return self.__irreps__

    def __cat_dim__(self, key, value):
        if key == AtomicDataDict.EDGE_INDEX_KEY or key == AtomicDataDict.ENV_INDEX_KEY or key == AtomicDataDict.ONSITENV_INDEX_KEY:
            return 1  # always cat in the edge dimension
        elif key in _GRAPH_FIELDS:
            # graph-level properties and so need a new batch dimension
            if key in _NESTED_FIELDS:
                return 0
            else:
                return None
        else:
            return 0  # cat along node/edge dimension

    def __inc__(self, key, value):
        if key in (
            AtomicDataDict.EDGE_INDEX_KEY,
            AtomicDataDict.ENV_INDEX_KEY,
            AtomicDataDict.ONSITENV_INDEX_KEY,
        ):
            n_ghost = 0
            if hasattr(self, AtomicDataDict.GHOST_ATOM_INDEX_KEY):
                n_ghost = int(self[AtomicDataDict.GHOST_ATOM_INDEX_KEY].numel())
            return torch.tensor([[self.num_nodes], [self.num_nodes + n_ghost]])
        if key in (
            AtomicDataDict.ATOM_INDEX_KEY,
            AtomicDataDict.GHOST_ATOM_INDEX_KEY,
        ):
            # These are original structure identities, not local indices.
            return 0
        return super().__inc__(key, value)

    def without_nodes(self, which_nodes):
        """Return a copy of ``self`` with ``which_nodes`` removed.
        The returned object may share references to some underlying data tensors with ``self``.
        Args:
            which_nodes (index tensor or boolean mask)
        Returns:
            A new data object.
        """
        if hasattr(self, AtomicDataDict.GHOST_ATOM_INDEX_KEY):
            raise NotImplementedError(
                "without_nodes is not defined for an owner-row partition; rebuild the partition from points"
            )
        which_nodes = torch.as_tensor(which_nodes)
        if which_nodes.dtype == torch.bool:
            mask = ~which_nodes
        else:
            mask = torch.ones(self.num_nodes, dtype=torch.bool)
            mask[which_nodes] = False
        assert mask.shape == (self.num_nodes,)
        n_keeping = mask.sum()

        # Only keep edges where both from and to are kept
        edge_mask = mask[self.edge_index[0]] & mask[self.edge_index[1]]
        if hasattr(self, AtomicDataDict.ENV_INDEX_KEY):
            env_mask = mask[self.env_index[0]] & mask[self.env_index[1]]
        if hasattr(self, AtomicDataDict.ONSITENV_INDEX_KEY):
            onsitenv_mask = mask[self.onsitenv_index[0]] & mask[self.onsitenv_index[1]]
        # Create an index mapping:
        new_index = torch.full((self.num_nodes,), -1, dtype=torch.long)
        new_index[mask] = torch.arange(n_keeping, dtype=torch.long)

        new_dict = {}
        for k in self.keys:
            if k == AtomicDataDict.EDGE_INDEX_KEY:
                new_dict[AtomicDataDict.EDGE_INDEX_KEY] = new_index[
                    self.edge_index[:, edge_mask]
                ]
            elif k == AtomicDataDict.EDGE_CELL_SHIFT_KEY:
                new_dict[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = self.edge_cell_shift[
                    edge_mask
                ]
            elif k == AtomicDataDict.CELL_KEY:
                new_dict[k] = self[k]
            elif k == AtomicDataDict.ENV_INDEX_KEY:
                new_dict[AtomicDataDict.ENV_INDEX_KEY] = new_index[
                    self.env_index[:, env_mask]
                ]
            elif k == AtomicDataDict.ENV_CELL_SHIFT_KEY:
                new_dict[AtomicDataDict.ENV_CELL_SHIFT_KEY] = self.env_cell_shift[
                    env_mask
                ]
            elif k == AtomicDataDict.ONSITENV_INDEX_KEY:
                new_dict[AtomicDataDict.ONSITENV_INDEX_KEY] = new_index[
                    self.onsitenv_index[:, onsitenv_mask]
                ]
            elif k == AtomicDataDict.ONSITENV_CELL_SHIFT_KEY:
                new_dict[AtomicDataDict.ONSITENV_CELL_SHIFT_KEY] = self.onsitenv_cell_shift[
                    onsitenv_mask
                ]
            else:
                if isinstance(self[k], torch.Tensor) and len(self[k]) == self.num_nodes:
                    new_dict[k] = self[k][mask]
                else:
                    new_dict[k] = self[k]

        new_dict["irreps"] = self.__irreps__

        return type(self)(**new_dict)


_ERROR_ON_NO_EDGES = os.environ.get("NEQUIP_ERROR_ON_NO_EDGES", "true").lower()
assert _ERROR_ON_NO_EDGES in ("true", "false"), "NEQUIP_ERROR_ON_NO_EDGES must be 'true' or 'false'"
_ERROR_ON_NO_EDGES = _ERROR_ON_NO_EDGES == "true"

def neighbor_list_and_relative_vec(
    pos,
    r_max,
    self_interaction=False,
    reduce=True,
    atomic_numbers=None,
    cell=None,
    pbc=False,
):
    """Create neighbor list and neighbor vectors based on radial cutoff.

    Create neighbor list (``edge_index``) and relative vectors
    (``edge_attr``) based on radial cutoff.

    Edges are given by the following convention:
    - ``edge_index[0]`` is the *source* (convolution center).
    - ``edge_index[1]`` is the *target* (neighbor).

    Thus, ``edge_index`` has the same convention as the relative vectors:
    :math:`\\vec{r}_{source, target}`

    If the input positions are a tensor with ``requires_grad == True``,
    the output displacement vectors will be correctly attached to the inputs
    for autograd.

    All outputs are Tensors on the same device as ``pos``; this allows future
    optimization of the neighbor list on the GPU.

    Args:
        pos (shape [N, 3]): Positional coordinate; Tensor or numpy array. If Tensor, must be on CPU.
        r_max (float): Radial cutoff distance for neighbor finding.
        cell (numpy shape [3, 3]): Cell for periodic boundary conditions. Ignored if ``pbc == False``.
        pbc (bool or 3-tuple of bool): Whether the system is periodic in each of the three cell dimensions.
        self_interaction (bool): Whether or not to include same periodic image self-edges in the neighbor list.
        strict_self_interaction (bool): Whether to include *any* self interaction edges in the graph, even if the two
            instances of the atom are in different periodic images. Defaults to True, should be True for most applications.

    Returns:
        edge_index (torch.tensor shape [2, num_edges]): List of edges.
        edge_cell_shift (torch.tensor shape [num_edges, 3]): Relative cell shift
            vectors. Returned only if cell is not None.
        cell (torch.Tensor [3, 3]): the cell as a tensor on the correct device.
            Returned only if cell is not None.
    """
    if isinstance(pbc, bool):
        pbc = (pbc,) * 3

    mask_r = False
    if isinstance(r_max, dict):
        _r_max = max(r_max.values())
        if _r_max - min(r_max.values()) > 1e-5:
            mask_r = True
        
        if len(r_max) < len(set(atomic_numbers)):
            raise ValueError("The number of r_max is less than the number of required atom species.")
    else:
        _r_max = r_max
        assert isinstance(r_max, (float, int))

    # Either the position or the cell may be on the GPU as tensors
    if isinstance(pos, torch.Tensor):
        temp_pos = pos.detach().cpu().numpy()
        out_device = pos.device
        out_dtype = pos.dtype
    else:
        temp_pos = np.asarray(pos)
        out_device = torch.device("cpu")
        out_dtype = torch.get_default_dtype()

    # Right now, GPU tensors require a round trip
    if out_device.type != "cpu":
        warnings.warn(
            "Currently, neighborlists require a round trip to the CPU. Please pass CPU tensors if possible."
        )

    # Get a cell on the CPU no matter what
    if isinstance(cell, torch.Tensor):
        temp_cell = cell.detach().cpu().numpy()
        cell_tensor = cell.to(device=out_device, dtype=out_dtype)
    elif cell is not None:
        temp_cell = np.asarray(cell)
        cell_tensor = torch.as_tensor(temp_cell, device=out_device, dtype=out_dtype)
    else:
        # ASE will "complete" this correctly.
        temp_cell = np.zeros((3, 3), dtype=temp_pos.dtype)
        cell_tensor = torch.as_tensor(temp_cell, device=out_device, dtype=out_dtype)

    # ASE dependent part
    temp_cell = ase.geometry.complete_cell(temp_cell)

    first_idex, second_idex, shifts = ase.neighborlist.primitive_neighbor_list(
        "ijS",
        pbc,
        temp_cell,
        temp_pos,
        cutoff=float(_r_max),
        self_interaction=self_interaction,  # we want edges from atom to itself in different periodic images!
        use_scaled_positions=False,
    )

    # Eliminate true self-edges that don't cross periodic boundaries
    # if not self_interaction:
    #     bad_edge = first_idex == second_idex
    #     bad_edge &= np.all(shifts == 0, axis=1)
    #     keep_edge = ~bad_edge
    #     if _ERROR_ON_NO_EDGES and (not np.any(keep_edge)):
    #         raise ValueError(
    #             f"Every single atom has no neighbors within the cutoff r_max={r_max} (after eliminating self edges, no edges remain in this system)"
    #         )
    #     first_idex = first_idex[keep_edge]
    #     second_idex = second_idex[keep_edge]
    #     shifts = shifts[keep_edge]

    """
    bond list is: i, j, shift; but i j shift and j i -shift are the same bond. so we need to remove the duplicate bonds.s
    first for i != j; we only keep i < j; then the j i -shift will be removed.
    then, for i == j; we only keep i i shift and remove i i -shift.
    """
    # 1. for i != j, keep i < j
    assert atomic_numbers is not None
    atomic_numbers = torch.as_tensor(atomic_numbers, dtype=torch.long)
    mask = first_idex <= second_idex
    first_idex = first_idex[mask]
    second_idex = second_idex[mask]
    shifts = shifts[mask]

    # 2. for i == j

    mask = torch.ones(len(first_idex), dtype=torch.bool)
    mask[first_idex == second_idex] = False
    # get index bool type ~mask  for i == j.
    # Convert mask to numpy for consistent indexing behavior
    mask_np = mask.cpu().numpy()
    o_first_idex = first_idex[~mask_np]
    o_second_idex = second_idex[~mask_np]
    o_shift = shifts[~mask_np]
    o_mask = mask[~mask]  # this is all False, with length being the number all  the bonds with i == j.

    # Ensure arrays are proper numpy arrays (not scalars) for isolated systems
    o_first_idex = np.atleast_1d(o_first_idex)
    o_second_idex = np.atleast_1d(o_second_idex)
    o_shift = np.atleast_2d(o_shift)

    # using the dict key to remove the duplicate bonds, because it is O(1) to check if a key is in the dict.
    rev_dict = {}
    for i in range(len(o_first_idex)):
        key = str(o_first_idex[i])+str(o_shift[i])
        key_rev = str(o_first_idex[i])+str(-o_shift[i])
        rev_dict[key] = True
        # key_rev is the reverse key of key, if key_rev is in the dict, then the bond is duplicate.
        # so， only when key_rev is not in the dict, we keep the bond. that is when rev_dict.get(key_rev, False) is False, we set o_mast = True.
        if not (rev_dict.get(key_rev, False) and rev_dict.get(key, False)):
            o_mask[i] = True
        
        if self_interaction:
            log.warning("self_interaction is True, but usually we do not want the self-interaction, please check if it is correct.")
            # for self-interaction, the above will remove the self-interaction, i.e. i == j, shift == [0, 0, 0]. since -0 = 0.
            if (o_shift[i] == np.array([0, 0, 0])).all():
                o_mask[i] = True

    del rev_dict
    del o_first_idex
    del o_second_idex
    del o_shift
    mask[~mask] = o_mask
    del o_mask

    # Convert mask to numpy for indexing numpy arrays (avoids torch/numpy compatibility issues)
    mask_np = mask.cpu().numpy()
    first_idex = torch.as_tensor(first_idex[mask_np], dtype=torch.long, device=out_device)
    second_idex = torch.as_tensor(second_idex[mask_np], dtype=torch.long, device=out_device)
    shifts = torch.as_tensor(shifts[mask_np], dtype=out_dtype, device=out_device)

    if not reduce:
        assert self_interaction == False, "for self_interaction = True,  i i 0 0 0 will be duplicated."
        first_idex, second_idex = torch.cat((first_idex, second_idex), dim=0), torch.cat((second_idex, first_idex), dim=0)
        shifts = torch.cat((shifts, -shifts), dim=0)
    
    # Build output:
    edge_index = torch.vstack(
        (torch.LongTensor(first_idex), torch.LongTensor(second_idex))
    )

    # TODO: mask the edges that is larger than r_max
    if mask_r:
        edge_vec = pos[edge_index[1]] - pos[edge_index[0]]
        if cell is not None :
            edge_vec = edge_vec + torch.einsum(
                "ni,ij->nj",
                shifts,
                cell_tensor.reshape(3,3),  # remove batch dimension
            )

        edge_length = torch.linalg.norm(edge_vec, dim=-1)

        # atom_species_num = [atomic_num_dict[k] for k in r_max.keys()]
        # for i in set(atomic_numbers):
        #     assert i in atom_species_num
        # r_map = torch.zeros(max(atom_species_num))
        # for k, v in r_max.items():
        #     r_map[atomic_num_dict[k]-1] = v

        first_key = next(iter(r_max.keys()))
        key_parts = first_key.split("-")
        
        if len(key_parts)==1:
            r_map = get_r_map(r_max, atomic_numbers)
            edge_length_max = 0.5 * (r_map[atomic_numbers[edge_index[0]]-1] + r_map[atomic_numbers[edge_index[1]]-1])
        
        elif len(key_parts)==2:
            r_map = get_r_map_bondwise(r_max, atomic_numbers)
            edge_length_max = r_map[atomic_numbers[edge_index[0]]-1,atomic_numbers[edge_index[1]]-1]            
        else:
            raise ValueError("The r_max keys should be either atomic number or atomic number pair.")
        
        r_mask = edge_length <= edge_length_max
        if any(~r_mask):
            edge_index = edge_index[:, r_mask]
            shifts = shifts[r_mask]
        # 收集不同类型的边及其对应的最大截断半径
        #edge_types = {}
        #for i in range(edge_index.shape[1]):
        #    atom_type_pair = (atomic_numbers[edge_index[0, i]], atomic_numbers[edge_index[1, i]])
        #    if atom_type_pair not in edge_types:
        #        edge_types[atom_type_pair] = edge_length_max[i].item()
        
        del edge_length
        del edge_vec
        del r_map
        del edge_length_max
        del r_mask

    return edge_index, shifts, cell_tensor

def get_r_map(r_max: dict, atomic_numbers=None):
    """
    Returns a torch tensor representing the mapping of atomic species to their maximum distances.

    Args:
        r_max (dict): A dictionary mapping atomic species to their maximum distances.
        atomic_numbers (list, optional): A list of atomic numbers to validate against the atomic species. Defaults to None.

    Returns:
        torch.Tensor: A torch tensor representing the mapping of atomic species to their maximum distances.
    """
    atom_species_num = [atomic_num_dict[k] for k in r_max.keys()]
    if atomic_numbers is not None:
        for i in atomic_numbers:
            assert i in atom_species_num
    r_map = torch.zeros(max(atom_species_num))
    for k, v in r_max.items():
        r_map[atomic_num_dict[k]-1] = v
    return r_map

def get_r_map_bondwise(r_max:dict, atomic_numbers=None):
    """
    Calculate the bondwise distance map based on the maximum bond length dictionary.

    Args:
        r_max (dict): A dictionary containing the maximum bond lengths for different atom pairs.
        atomic_numbers (list, optional): A list of atomic numbers. Defaults to None.

    Returns:
        torch.Tensor: A torch tensor representing the bondwise distance map.
    """
    atom_species_num = []
    for k in r_max.keys():
        assert len(k.split('-')) == 2
        atom_a, atom_b = k.split('-')
        if atomic_num_dict[atom_a] not in atom_species_num:
            atom_species_num.append(atomic_num_dict[atom_a])    
        if atomic_num_dict[atom_b] not in atom_species_num:
            atom_species_num.append(atomic_num_dict[atom_b])    
    
    if atomic_numbers is not None:
        for i in atomic_numbers:
            assert i in atom_species_num

    r_map = torch.zeros(max(atom_species_num), max(atom_species_num))
    for k, v in r_max.items():
        atom_a, atom_b = k.split('-')
        
        inv_value = r_map[atomic_num_dict[atom_b]-1, atomic_num_dict[atom_a]-1]
        if inv_value == 0:
            r_map[atomic_num_dict[atom_a]-1, atomic_num_dict[atom_b]-1] = v
            r_map[atomic_num_dict[atom_b]-1, atomic_num_dict[atom_a]-1] = v
        else:
            mean_val = (v + inv_value) / 2
            r_map[atomic_num_dict[atom_a]-1, atomic_num_dict[atom_b]-1] = mean_val
            r_map[atomic_num_dict[atom_b]-1, atomic_num_dict[atom_a]-1] = mean_val

    return r_map
