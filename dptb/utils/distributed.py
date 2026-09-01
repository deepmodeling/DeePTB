"""Owner-row graph-parallel helpers for ordinary ``AtomicDataDict`` objects.

Only process state and immutable communication plans live here.  Every graph
operation receives its data explicitly; public node fields always contain
owned rows only.
"""

import os
import socket
import hashlib
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from dptb.data import AtomicDataDict


_MPI_COMM = None
_RANK = 0
_WORLD_SIZE = 1
_LOCAL_RANK = 0
_INITIALIZED = False
_PLAN_CACHE: OrderedDict = OrderedDict()
_MAX_CACHED_PLANS = 16


def initialize_distributed(device: Optional[str] = None, backend: str = "auto") -> Tuple[int, int, int]:
    """Initialize torch.distributed from an MPI launch, if MPI has multiple ranks."""
    global _MPI_COMM, _RANK, _WORLD_SIZE, _LOCAL_RANK, _INITIALIZED
    if _INITIALIZED:
        return _RANK, _WORLD_SIZE, _LOCAL_RANK
    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise RuntimeError(
            "Distributed execution requires mpi4py (install DeePTB's distributed extra)"
        ) from exc

    _MPI_COMM = MPI.COMM_WORLD
    _RANK = _MPI_COMM.Get_rank()
    _WORLD_SIZE = _MPI_COMM.Get_size()
    shared = _MPI_COMM.Split_type(MPI.COMM_TYPE_SHARED)
    _LOCAL_RANK = shared.Get_rank()
    shared.Free()
    if _WORLD_SIZE == 1:
        _INITIALIZED = True
        return _RANK, _WORLD_SIZE, _LOCAL_RANK

    import torch.distributed as dist

    use_cuda = str(device or "").startswith("cuda")
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed CUDA was requested, but CUDA is unavailable")
        if _LOCAL_RANK >= torch.cuda.device_count():
            raise RuntimeError(
                f"MPI local rank {_LOCAL_RANK} has no matching GPU; only "
                f"{torch.cuda.device_count()} devices are visible"
            )
        torch.cuda.set_device(_LOCAL_RANK) # so we assume single card per MPI rank?

    if not dist.is_initialized():
        master_addr = os.environ.get("MASTER_ADDR")
        master_port = os.environ.get("MASTER_PORT")
        if _RANK == 0:
            master_addr = master_addr or socket.gethostname()
            if master_port is None:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind(("", 0))
                    master_port = str(sock.getsockname()[1])
        master_addr, master_port = _MPI_COMM.bcast((master_addr, master_port), root=0)
        selected_backend = backend
        if selected_backend == "auto":
            selected_backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(
            backend=selected_backend,
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=_RANK,
            world_size=_WORLD_SIZE,
        )
    elif dist.get_rank() != _RANK or dist.get_world_size() != _WORLD_SIZE:
        raise RuntimeError(
            "The initialized torch.distributed process group does not match "
            "MPI.COMM_WORLD"
        )
    # PyTorch requires every rank to participate when batch_isend_irecv is the
    # first operation on a process group.  Establish the group collectively so
    # later sparse exchanges may involve only the graph-neighbour ranks.
    dist.barrier()
    _INITIALIZED = True
    return _RANK, _WORLD_SIZE, _LOCAL_RANK


def is_distributed() -> bool:
    return _WORLD_SIZE > 1


def distributed_is_initialized() -> bool:
    """Whether DeePTB's MPI/PyTorch distributed context was initialized."""
    return _INITIALIZED


def require_distributed_initialized() -> None:
    """Fail if graph-parallel state has not been initialized by DeePTB."""
    if not _INITIALIZED:
        raise RuntimeError(
            "Partitioned graph construction requires initialize_distributed() "
            "to be called first; rank and world size are inferred from that context"
        )


def get_rank() -> int:
    return _RANK


def get_world_size() -> int:
    return _WORLD_SIZE


def get_local_rank() -> int:
    return _LOCAL_RANK


def get_mpi_comm():
    """Return the initialized MPI communicator, or ``None`` in serial setup."""
    return _MPI_COMM


def _remember_plan(key, value):
    _PLAN_CACHE[key] = value
    _PLAN_CACHE.move_to_end(key)
    while len(_PLAN_CACHE) > _MAX_CACHED_PLANS:
        _PLAN_CACHE.popitem(last=False)
    return value


def _request_plan(owned_ids: torch.Tensor, request_ids: torch.Tensor,
                  request_owners: torch.Tensor, cache_tag: str):
    if _MPI_COMM is None:
        raise RuntimeError("MPI graph exchange was used before initialize_distributed()")
    owned_cpu = owned_ids.detach().to("cpu", dtype=torch.long)
    request_cpu = request_ids.detach().to("cpu", dtype=torch.long)
    owners_cpu = request_owners.detach().to("cpu", dtype=torch.long).flatten()
    fingerprint = hashlib.blake2b(digest_size=16)
    fingerprint.update(owned_cpu.contiguous().numpy().tobytes())
    fingerprint.update(request_cpu.contiguous().numpy().tobytes())
    fingerprint.update(owners_cpu.contiguous().numpy().tobytes())
    key = (cache_tag, fingerprint.digest(), tuple(request_cpu.shape), _RANK)
    cached = _PLAN_CACHE.get(key)
    if cached is not None:
        _PLAN_CACHE.move_to_end(key)
        return cached

    request_keys = {}
    recv_indices = {}
    for local_index, (identity, owner) in enumerate(zip(request_cpu.tolist(), owners_cpu.tolist())):
        ident = tuple(identity) if isinstance(identity, list) else (identity,)
        if owner != _RANK:
            request_keys.setdefault(owner, []).append(ident)
            recv_indices.setdefault(owner, []).append(local_index)

    owned_lookup = {}
    for local_index, identity in enumerate(owned_cpu.tolist()):
        ident = tuple(identity) if isinstance(identity, list) else (identity,)
        owned_lookup[ident] = local_index

    if hasattr(_MPI_COMM, "Create_dist_graph"):
        from mpi4py import MPI

        requested_owners = sorted(request_keys)
        graph = _MPI_COMM.Create_dist_graph(
            [_RANK], [len(requested_owners)], requested_owners, reorder=False
        )
        try:
            send_peers, recv_peers, _weights = graph.Get_dist_neighbors()
            request_counts = np.asarray(
                [len(request_keys.get(int(peer), ())) for peer in recv_peers],
                dtype=np.int32,
            )
            incoming_counts = np.empty(len(send_peers), dtype=np.int32)
            graph.Neighbor_alltoall(request_counts, incoming_counts)
            width = int(request_cpu.shape[1])
            request_displs = np.zeros(len(request_counts), dtype=np.int64)
            incoming_displs = np.zeros(len(incoming_counts), dtype=np.int64)
            if len(request_counts) > 1:
                request_displs[1:] = np.cumsum(
                    request_counts[:-1], dtype=np.int64
                )
            if len(incoming_counts) > 1:
                incoming_displs[1:] = np.cumsum(
                    incoming_counts[:-1], dtype=np.int64
                )
            packed = np.asarray(
                [identity for peer in recv_peers
                 for identity in request_keys.get(int(peer), ())],
                dtype=np.int64,
            ).reshape(-1, width)
            incoming = np.empty((int(incoming_counts.sum()), width), dtype=np.int64)
            graph.Neighbor_alltoallv(
                [packed.reshape(-1), request_counts * width,
                 request_displs * width, MPI.INT64_T],
                [incoming.reshape(-1), incoming_counts * width,
                 incoming_displs * width, MPI.INT64_T],
            )
            requests_from = []
            cursor = 0
            for count in incoming_counts.tolist():
                requests_from.append([
                    tuple(row.tolist()) for row in incoming[cursor:cursor + count]
                ])
                cursor += count
            send_peers = [int(peer) for peer in send_peers]
            recv_peers = [int(peer) for peer in recv_peers]
            send_splits = incoming_counts.tolist()
            recv_splits = request_counts.tolist()
        finally:
            graph.Free()
    else:
        # Unit-test communicator fallback. Runtime MPI uses the sparse path.
        dense_requests = [request_keys.get(peer, []) for peer in range(_WORLD_SIZE)]
        dense_incoming = _MPI_COMM.alltoall(dense_requests)
        send_peers = [peer for peer, items in enumerate(dense_incoming) if items]
        recv_peers = sorted(request_keys)
        requests_from = [dense_incoming[peer] for peer in send_peers]
        send_splits = [len(items) for items in requests_from]
        recv_splits = [len(request_keys[peer]) for peer in recv_peers]

    send_indices = []
    recv_flat = []
    for peer, requested in zip(send_peers, requests_from):
        missing = [ident for ident in requested if tuple(ident) not in owned_lookup]
        if missing:
            raise RuntimeError(
                f"Rank {_RANK} is missing {len(missing)} requested halo samples; first={missing[0]}"
            )
        send_indices.extend(owned_lookup[tuple(ident)] for ident in requested)
    for peer in recv_peers:
        recv_flat.extend(recv_indices[peer])

    plan = (
        torch.as_tensor(send_indices, dtype=torch.long),
        torch.as_tensor(recv_flat, dtype=torch.long),
        tuple(send_peers),
        tuple(recv_peers),
        send_splits,
        recv_splits,
    )
    return _remember_plan(key, plan)


def _batch_peer_exchange(send, send_peers, send_splits, recv_peers, recv_splits):
    """Packed point-to-point tensor exchange in peer-list order."""
    import torch.distributed as dist

    send_chunks = list(send.split(list(send_splits), dim=0)) if send_splits else []
    recv_chunks = [
        send.new_empty((count,) + tuple(send.shape[1:])) for count in recv_splits
    ]
    operations = []
    for tensor, peer in zip(recv_chunks, recv_peers):
        operations.append(dist.P2POp(dist.irecv, tensor, int(peer)))
    for tensor, peer in zip(send_chunks, send_peers):
        operations.append(dist.P2POp(dist.isend, tensor.contiguous(), int(peer)))
    if operations:
        for request in dist.batch_isend_irecv(operations):
            request.wait()
    if recv_chunks:
        return torch.cat(recv_chunks, dim=0)
    return send.new_empty((0,) + tuple(send.shape[1:]))


class _NeighborExchange(torch.autograd.Function):
    """Owner-to-ghost exchange whose backward reduces gradients to owners."""

    @staticmethod
    def forward(ctx, send, send_peers, send_splits, recv_peers, recv_splits):
        ctx.send_peers = send_peers
        ctx.send_splits = tuple(send_splits)
        ctx.recv_peers = recv_peers
        ctx.recv_splits = tuple(recv_splits)
        return _batch_peer_exchange(
            send, send_peers, send_splits, recv_peers, recv_splits
        )

    @staticmethod
    def backward(ctx, grad_received):
        grad_send = _batch_peer_exchange(
            grad_received.contiguous(),
            ctx.recv_peers, ctx.recv_splits,
            ctx.send_peers, ctx.send_splits,
        )
        return grad_send, None, None, None, None


def _request_values(values: torch.Tensor, owned_ids: torch.Tensor,
                    request_ids: torch.Tensor, request_owners: torch.Tensor,
                    cache_tag: str) -> torch.Tensor:
    """Return values for requested identities in request order."""
    local_lookup = {
        tuple(v) if isinstance(v, list) else (v,): i
        for i, v in enumerate(owned_ids.detach().cpu().tolist())
    }
    result = values.new_empty((request_ids.shape[0],) + tuple(values.shape[1:]))
    remote_mask = request_owners.flatten().ne(_RANK)
    for request_index in torch.nonzero(~remote_mask, as_tuple=False).flatten().tolist():
        identity = request_ids[request_index].detach().cpu().tolist()
        identity = tuple(identity) if isinstance(identity, list) else (identity,)
        if identity not in local_lookup:
            raise RuntimeError(f"Missing local requested sample {identity}")
        result[request_index] = values[local_lookup[identity]]
    if not is_distributed():
        if bool(remote_mask.any()):
            raise RuntimeError("Remote graph columns require initialized distributed execution")
        return result
    if _MPI_COMM is None:
        raise RuntimeError("Remote graph columns require initialized distributed execution")
    send_index, recv_index, send_peers, recv_peers, send_splits, recv_splits = _request_plan(
        owned_ids, request_ids, request_owners, cache_tag
    )
    send_index = send_index.to(values.device)
    recv_index = recv_index.to(values.device)
    send = values.index_select(0, send_index)
    received = _NeighborExchange.apply(
        send, send_peers, send_splits, recv_peers, recv_splits
    )
    return torch.index_copy(result, 0, recv_index, received)


def column_atom_index(data: AtomicDataDict.Type) -> torch.Tensor:
    owned = data.get(AtomicDataDict.ATOM_INDEX_KEY)
    if owned is None:
        owned = torch.arange(
            data[AtomicDataDict.POSITIONS_KEY].shape[0],
            device=data[AtomicDataDict.POSITIONS_KEY].device,
        )
    ghost = data.get(AtomicDataDict.GHOST_ATOM_INDEX_KEY)
    return owned.flatten() if ghost is None else torch.cat((owned.flatten(), ghost.flatten()))


def column_atom_owner(data: AtomicDataDict.Type) -> torch.Tensor:
    n_owned = data[AtomicDataDict.POSITIONS_KEY].shape[0]
    device = data[AtomicDataDict.POSITIONS_KEY].device
    owned = torch.full((n_owned,), _RANK, dtype=torch.long, device=device)
    ghost = data.get(AtomicDataDict.GHOST_ATOM_OWNER_KEY)
    return owned if ghost is None else torch.cat((owned, ghost.flatten().to(device)))


def node_column_values(data: AtomicDataDict.Type, owned_values: torch.Tensor,
                       ghost_values: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Expand owner rows into a temporary owned-plus-ghost column tensor."""
    ghost_ids = data.get(AtomicDataDict.GHOST_ATOM_INDEX_KEY)
    if ghost_ids is None:
        return owned_values
    if not is_distributed() and ghost_ids.numel() == 0:
        return owned_values
    if is_distributed():
        ghost_owners = data.get(AtomicDataDict.GHOST_ATOM_OWNER_KEY)
        if ghost_owners is None:
            ghost_owners = torch.empty(
                0, dtype=torch.long, device=owned_values.device
            )
        received = _request_values(
            owned_values,
            data[AtomicDataDict.ATOM_INDEX_KEY].reshape(-1, 1),
            ghost_ids.reshape(-1, 1),
            ghost_owners,
            "node-columns",
        )
    elif ghost_values is not None:
        received = ghost_values.to(device=owned_values.device, dtype=owned_values.dtype)
    else:
        raise RuntimeError("ghost_values are required outside initialized distributed execution")
    return torch.cat((owned_values, received), dim=0)


def reverse_edge_values(data: AtomicDataDict.Type, values: torch.Tensor) -> torch.Tensor:
    """Return the reversed directed edge value aligned with each local edge."""
    edge = data[AtomicDataDict.EDGE_INDEX_KEY]
    shifts = data.get(AtomicDataDict.EDGE_CELL_SHIFT_KEY)
    if shifts is None:
        shifts = torch.zeros((edge.shape[1], 3), dtype=torch.long, device=edge.device)
    ids = column_atom_index(data).to(edge.device)
    owners = column_atom_owner(data).to(edge.device)
    local_ids = torch.cat((
        ids[edge[0]].reshape(-1, 1), ids[edge[1]].reshape(-1, 1),
        shifts.round().long(),
    ), dim=1)
    requested = torch.cat((
        ids[edge[1]].reshape(-1, 1), ids[edge[0]].reshape(-1, 1),
        -shifts.round().long(),
    ), dim=1)
    return _request_values(
        values, local_ids, requested, owners[edge[1]], "reverse-edges"
    )


def _route_plan(keys, target_owners):
    keys_cpu = keys.detach().to("cpu", dtype=torch.long).contiguous()
    owners_cpu = target_owners.detach().to("cpu", dtype=torch.long).flatten().contiguous()
    fingerprint = hashlib.blake2b(digest_size=16)
    fingerprint.update(keys_cpu.numpy().tobytes())
    fingerprint.update(owners_cpu.numpy().tobytes())
    cache_key = ("relation-route", fingerprint.digest(), tuple(keys_cpu.shape), _RANK)
    cached = _PLAN_CACHE.get(cache_key)
    if cached is not None:
        _PLAN_CACHE.move_to_end(cache_key)
        return cached

    by_peer = {}
    for peer in range(_WORLD_SIZE):
        if peer == _RANK:
            continue
        selected = torch.nonzero(owners_cpu.eq(peer), as_tuple=False).flatten()
        if selected.numel():
            by_peer[peer] = selected

    if hasattr(_MPI_COMM, "Create_dist_graph"):
        from mpi4py import MPI

        destinations_requested = sorted(by_peer)
        graph = _MPI_COMM.Create_dist_graph(
            [_RANK], [len(destinations_requested)], destinations_requested,
            reorder=False,
        )
        try:
            recv_peers, send_peers, _weights = graph.Get_dist_neighbors()
            send_counts = np.asarray(
                [int(by_peer[int(peer)].numel()) for peer in send_peers],
                dtype=np.int32,
            )
            recv_counts = np.empty(len(recv_peers), dtype=np.int32)
            graph.Neighbor_alltoall(send_counts, recv_counts)
            send_order = torch.cat([by_peer[int(peer)] for peer in send_peers]) \
                if len(send_peers) else torch.empty(0, dtype=torch.long)
            packed_keys = keys_cpu.index_select(0, send_order).numpy()
            width = int(keys_cpu.shape[1])
            sdisp = np.zeros(len(send_counts), dtype=np.int64)
            rdisp = np.zeros(len(recv_counts), dtype=np.int64)
            if len(send_counts) > 1:
                sdisp[1:] = np.cumsum(send_counts[:-1], dtype=np.int64)
            if len(recv_counts) > 1:
                rdisp[1:] = np.cumsum(recv_counts[:-1], dtype=np.int64)
            recv_keys = np.empty((int(recv_counts.sum()), width), dtype=np.int64)
            graph.Neighbor_alltoallv(
                [packed_keys.reshape(-1), send_counts * width, sdisp * width, MPI.INT64_T],
                [recv_keys.reshape(-1), recv_counts * width, rdisp * width, MPI.INT64_T],
            )
            plan = (
                send_order, torch.as_tensor(recv_keys, dtype=torch.long),
                tuple(int(p) for p in send_peers),
                tuple(int(p) for p in recv_peers),
                send_counts.tolist(), recv_counts.tolist(),
            )
        finally:
            graph.Free()
    else:
        dense = [keys_cpu.index_select(0, by_peer[p]).tolist() if p in by_peer else []
                 for p in range(_WORLD_SIZE)]
        incoming = _MPI_COMM.alltoall(dense)
        send_peers = sorted(by_peer)
        recv_peers = [p for p, items in enumerate(incoming) if items]
        send_order = torch.cat([by_peer[p] for p in send_peers]) \
            if send_peers else torch.empty(0, dtype=torch.long)
        recv_rows = [row for p in recv_peers for row in incoming[p]]
        plan = (
            send_order,
            torch.as_tensor(recv_rows, dtype=torch.long).reshape(-1, keys_cpu.shape[1]),
            tuple(send_peers), tuple(recv_peers),
            [len(by_peer[p]) for p in send_peers],
            [len(incoming[p]) for p in recv_peers],
        )
    return _remember_plan(cache_key, plan)


def route_relation_contributions(data: AtomicDataDict.Type, keys: torch.Tensor,
                                 values: torch.Tensor,
                                 target_owners: Optional[torch.Tensor] = None):
    """Route keyed sparse contributions to their row owners.

    ``keys[:, 0]`` is the original atom id of the target row.  Returned local
    and received contributions remain differentiable with respect to values.
    """
    if target_owners is None:
        ids = column_atom_index(data).detach().cpu().tolist()
        owners = column_atom_owner(data).detach().cpu().tolist()
        owner_of = {int(identity): int(owner) for identity, owner in zip(ids, owners)}
        target_owners = torch.tensor(
            [owner_of[int(identity)] for identity in keys[:, 0].detach().cpu().tolist()],
            dtype=torch.long,
            device=keys.device,
        )
    local_mask = target_owners.flatten().eq(_RANK)
    local_keys = keys[local_mask]
    local_values = values[local_mask]
    if not is_distributed():
        if bool((~local_mask).any()):
            raise RuntimeError("Remote sparse routing requires initialized distributed execution")
        return local_keys, local_values

    send_order, recv_keys, send_peers, recv_peers, send_counts, recv_counts = \
        _route_plan(keys, target_owners)
    send_order = send_order.to(values.device)
    send_values = values.index_select(0, send_order)
    recv_values = _NeighborExchange.apply(
        send_values, send_peers, send_counts, recv_peers, recv_counts
    )
    recv_keys = recv_keys.to(keys.device)
    return (
        torch.cat((local_keys, recv_keys), dim=0),
        torch.cat((local_values, recv_values), dim=0),
    )


def barrier():
    if is_distributed():
        torch.distributed.barrier()
