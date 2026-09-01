import pytest
import torch
import torch.distributed as torch_dist
import torch.multiprocessing as torch_mp

from dptb.data import AtomicData, AtomicDataDict
from dptb.data.AtomicData import recursive_inertial_bisection
from dptb.data.interfaces.ham_to_feature import block_to_feature, feature_to_block
from dptb.data.transforms import OrbitalMapper
from dptb.utils.distributed import column_atom_index


@pytest.fixture
def distributed_context(monkeypatch):
    """Install a synthetic DeePTB context for non-collective partition tests."""
    import dptb.utils.distributed as distributed

    def configure(rank, world_size, comm=None):
        monkeypatch.setattr(distributed, "_RANK", rank)
        monkeypatch.setattr(distributed, "_WORLD_SIZE", world_size)
        monkeypatch.setattr(distributed, "_LOCAL_RANK", rank)
        monkeypatch.setattr(distributed, "_MPI_COMM", comm)
        monkeypatch.setattr(distributed, "_INITIALIZED", True)

    return configure


def _relation_set(data, index_key, shift_key):
    columns = column_atom_index(data.to_dict()).cpu()
    index = data[index_key].cpu()
    shift = getattr(data, shift_key, None)
    if shift is None:
        shift = torch.zeros((index.shape[1], 3), dtype=torch.long)
    return {
        (
            int(columns[index[0, edge]]),
            int(columns[index[1, edge]]),
            *map(int, shift[edge]),
        )
        for edge in range(index.shape[1])
    }


def _serial_relation_set(data, owners, rank, index_key, shift_key):
    index = data[index_key].cpu()
    shift = getattr(data, shift_key, None)
    if shift is None:
        shift = torch.zeros((index.shape[1], 3), dtype=torch.long)
    return {
        (int(index[0, edge]), int(index[1, edge]), *map(int, shift[edge]))
        for edge in range(index.shape[1])
        if int(owners[index[0, edge]]) == rank
    }


@pytest.mark.parametrize("world_size", [2, 3, 4, 5, 7])
def test_rib_balances_arbitrary_world_sizes(world_size):
    gen = torch.Generator().manual_seed(123)
    pos = torch.rand((31, 3), generator=gen)
    owners = recursive_inertial_bisection(pos, world_size)
    counts = torch.bincount(owners, minlength=world_size)
    assert int(counts.max() - counts.min()) <= 1
    assert torch.equal(owners, recursive_inertial_bisection(pos, world_size))


def test_rib_projection_ties_are_deterministic_and_too_many_ranks_fail():
    pos = torch.zeros((7, 3))
    first = recursive_inertial_bisection(pos, 5)
    assert torch.equal(first, recursive_inertial_bisection(pos, 5))
    assert int(torch.bincount(first, minlength=5).max()) == 2
    with pytest.raises(ValueError, match="at least one atom per rank"):
        recursive_inertial_bisection(pos[:2], 3)


def test_partition_process_identity_is_inferred(distributed_context):
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [1.4, 0.0, 0.0], [2.1, 0.0, 0.0]]
    )
    distributed_context(1, 2)
    graph = AtomicData.from_points(
        pos=pos,
        atomic_numbers=torch.ones(4, dtype=torch.long),
        r_max=0.9,
        pbc=False,
        partition_options={"method": "rib"},
    )
    owners = recursive_inertial_bisection(pos, 2)
    assert torch.equal(
        graph.atom_index.flatten(),
        torch.nonzero(owners == 1, as_tuple=False).flatten(),
    )

    with pytest.raises(ValueError, match="must not be supplied"):
        AtomicData.from_points(
            pos=pos,
            atomic_numbers=torch.ones(4, dtype=torch.long),
            r_max=0.9,
            pbc=False,
            partition_options={"method": "rib", "rank": 0},
        )


def test_partition_requires_initialized_distributed_context(monkeypatch):
    import dptb.utils.distributed as distributed

    monkeypatch.setattr(distributed, "_INITIALIZED", False)
    with pytest.raises(RuntimeError, match="initialize_distributed"):
        AtomicData.from_points(
            pos=torch.zeros((2, 3)),
            atomic_numbers=torch.ones(2, dtype=torch.long),
            r_max=1.0,
            pbc=False,
            partition_options={"method": "rib"},
        )


def test_incomplete_partial_pbc_matches_serial(distributed_context):
    pos = torch.tensor([
        [0.1, 0.2, 0.0], [1.9, 0.2, 0.0],
        [1.0, 1.0, 0.0], [1.0, 1.5, 0.0],
    ])
    cell = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    kwargs = dict(
        pos=pos, cell=cell, pbc=(True, True, False),
        atomic_numbers=torch.ones(4, dtype=torch.long), r_max=0.31,
    )
    serial = AtomicData.from_points(**kwargs)
    owners = recursive_inertial_bisection(pos, 2, cell, (True, True, False))
    for rank in range(2):
        distributed_context(rank, 2)
        local = AtomicData.from_points(**kwargs, partition_options={"method": "rib"})
        assert _relation_set(
            local, AtomicDataDict.EDGE_INDEX_KEY, AtomicDataDict.EDGE_CELL_SHIFT_KEY
        ) == _serial_relation_set(
            serial, owners, rank,
            AtomicDataDict.EDGE_INDEX_KEY, AtomicDataDict.EDGE_CELL_SHIFT_KEY,
        )


@pytest.mark.parametrize("case", ["skew_translated", "multiple_shells"])
def test_adversarial_periodic_partitions_match_serial(case, distributed_context):
    generator = torch.Generator().manual_seed(91)
    if case == "skew_translated":
        # A left-handed triclinic cell with atoms deliberately represented in
        # different, noncentral periodic images.
        cell = torch.tensor([
            [2.4, 0.0, 0.0], [0.8, 2.0, 0.0], [0.3, 0.4, -1.8],
        ])
        fractional = torch.rand((17, 3), generator=generator)
        translations = torch.randint(-4, 5, (17, 3), generator=generator)
        pos = (fractional + translations) @ cell
        cutoffs = (0.86, 0.67, 0.51)
    else:
        # The cutoff spans more than one periodic shell in every direction.
        cell = torch.tensor([
            [1.0, 0.0, 0.0], [0.2, 0.9, 0.0], [0.1, 0.15, 0.8],
        ])
        pos = torch.rand((9, 3), generator=generator) @ cell
        cutoffs = (1.65, 1.13, 0.74)

    kwargs = dict(
        pos=pos, cell=cell, pbc=True,
        atomic_numbers=torch.ones(len(pos), dtype=torch.long),
        r_max=cutoffs[0], er_max=cutoffs[1], oer_max=cutoffs[2],
    )
    serial = AtomicData.from_points(**kwargs)
    owners = recursive_inertial_bisection(pos, 3, cell, True)
    relations = (
        (AtomicDataDict.EDGE_INDEX_KEY, AtomicDataDict.EDGE_CELL_SHIFT_KEY),
        (AtomicDataDict.ENV_INDEX_KEY, AtomicDataDict.ENV_CELL_SHIFT_KEY),
        (AtomicDataDict.ONSITENV_INDEX_KEY, AtomicDataDict.ONSITENV_CELL_SHIFT_KEY),
    )
    for rank in range(3):
        distributed_context(rank, 3)
        local = AtomicData.from_points(
            **kwargs, partition_options={"method": "rib"}
        )
        for index_key, shift_key in relations:
            assert _relation_set(local, index_key, shift_key) == _serial_relation_set(
                serial, owners, rank, index_key, shift_key
            )


def test_partition_geometry_validation(distributed_context):
    distributed_context(0, 2)
    common = dict(
        atomic_numbers=torch.ones(2, dtype=torch.long), r_max=1.0,
        partition_options={"method": "rib"},
    )
    with pytest.raises(ValueError, match="non-finite"):
        AtomicData.from_points(
            pos=torch.tensor([[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]]),
            pbc=False, **common,
        )
    with pytest.raises(ValueError, match="linearly independent"):
        AtomicData.from_points(
            pos=torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            cell=torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            pbc=(True, True, False), **common,
        )


def test_finite_rib_bounds_avoid_volume_image_replication():
    from dptb.data.AtomicData import _collect_halo_images

    generator = torch.Generator().manual_seed(5)
    n_atoms, world_size = 1000, 8
    cell = torch.eye(3) * 10.0
    pos = torch.rand((n_atoms, 3), generator=generator) @ cell
    numbers = torch.ones(n_atoms, dtype=torch.long)
    owners, tree = recursive_inertial_bisection(
        pos, world_size, cell, True, return_tree=True
    )
    received = 0
    for rank in range(world_size):
        _, _, _, records = _collect_halo_images(
            pos, numbers, torch.zeros_like(numbers), owners, tree,
            rank, world_size, 1.0, cell, True,
        )
        received += len(records)
    # The old unbounded half-space routing generated 32,189 records here.
    assert received < 4 * n_atoms


@pytest.mark.parametrize("periodic", [False, True])
def test_owner_row_partition_matches_serial_relations(periodic, distributed_context):
    gen = torch.Generator().manual_seed(7)
    pos = 4.0 * torch.rand((24, 3), generator=gen)
    kwargs = dict(
        pos=pos,
        r_max=1.35,
        er_max=1.05,
        oer_max=0.75,
        atomic_numbers=torch.ones(24, dtype=torch.long),
    )
    if periodic:
        kwargs.update(
            cell=torch.tensor(
                [[4.0, 0.2, 0.0], [0.0, 4.1, 0.3], [0.1, 0.0, 4.2]]
            ),
            pbc=True,
        )
    else:
        kwargs.update(pbc=False)

    serial = AtomicData.from_points(**kwargs)
    owners = recursive_inertial_bisection(
        pos, 3, kwargs.get("cell"), (periodic,) * 3
    )
    relations = (
        (AtomicDataDict.EDGE_INDEX_KEY, AtomicDataDict.EDGE_CELL_SHIFT_KEY),
        (AtomicDataDict.ENV_INDEX_KEY, AtomicDataDict.ENV_CELL_SHIFT_KEY),
        (AtomicDataDict.ONSITENV_INDEX_KEY, AtomicDataDict.ONSITENV_CELL_SHIFT_KEY),
    )
    for rank in range(3):
        distributed_context(rank, 3)
        local = AtomicData.from_points(
            **kwargs,
            partition_options={"method": "rib"},
        )
        n_owned = local.pos.shape[0]
        n_ghost = local.ghost_atom_index.numel()
        atom_index = local.atom_index.flatten()
        assert torch.equal(atom_index, atom_index.sort().values)
        assert torch.equal(local.pos, pos[atom_index])
        assert set(atom_index.tolist()) == set(
            torch.nonzero(owners == rank, as_tuple=False).flatten().tolist()
        )
        assert not hasattr(local, "atom_owner")
        for field in ("pos", "atomic_numbers", "atom_index"):
            assert local[field].shape[0] == n_owned
        for index_key, shift_key in relations:
            index = local[index_key]
            assert bool((index[0] < n_owned).all())
            assert bool((index[1] < n_owned + n_ghost).all())
            assert _relation_set(local, index_key, shift_key) == _serial_relation_set(
                serial, owners, rank, index_key, shift_key
            )


def test_partition_build_invokes_neighbour_list_once(monkeypatch, distributed_context):
    import importlib

    atomic_module = importlib.import_module("dptb.data.AtomicData")
    original = atomic_module.neighbor_list_and_relative_vec
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs.get("r_max"))
        return original(*args, **kwargs)

    monkeypatch.setattr(atomic_module, "neighbor_list_and_relative_vec", counted)
    distributed_context(0, 2)
    AtomicData.from_points(
        pos=torch.tensor(
            [[0.1, 0.2, 0.3], [1.0, 0.2, 0.3], [2.1, 0.4, 0.2], [3.0, 0.2, 0.3]]
        ),
        cell=torch.tensor([[4.0, 0.2, 0.0], [0.0, 3.8, 0.1], [0.1, 0.0, 4.1]]),
        pbc=True,
        atomic_numbers=torch.ones(4, dtype=torch.long),
        r_max=1.3,
        er_max=1.8,
        oer_max=0.9,
        partition_options={"method": "rib"},
    )
    assert len(calls) == 1
    assert calls[0] > 1.8
    assert calls[0] == pytest.approx(1.8, abs=1e-3)


def _two_row_partition(rank):
    other = 1 - rank
    global_id = 4 if rank == 0 else 9
    ghost_id = 9 if rank == 0 else 4
    return AtomicData(
        pos=torch.tensor([[float(rank), 0.0, 0.0]]),
        cell=torch.eye(3),
        edge_index=torch.tensor([[0], [1]]),
        edge_cell_shift=torch.zeros((1, 3)),
        atomic_numbers=torch.ones((1, 1), dtype=torch.long),
        atom_types=torch.zeros((1, 1), dtype=torch.long),
        atom_index=torch.tensor([global_id]),
        ghost_atom_index=torch.tensor([ghost_id]),
        ghost_atom_owner=torch.tensor([other]),
        ghost_pos=torch.tensor([[float(other), 0.0, 0.0]]),
        ghost_atomic_numbers=torch.ones(1, dtype=torch.long),
        ghost_atom_types=torch.zeros(1, dtype=torch.long),
        node_features=torch.tensor([[2.0 + rank]]),
        edge_features=torch.tensor([[5.0]]),
    )


def test_block_conversion_uses_owner_rows_and_global_column_ids():
    idp = OrbitalMapper({"H": ["1s"]})
    combined = {}
    for rank in range(2):
        data = _two_row_partition(rank)
        for key, value in feature_to_block(data.to_dict(), idp).items():
            combined[key] = combined.get(key, torch.zeros_like(value)) + value
    assert set(combined) == {"4_4_0_0_0", "9_9_0_0_0", "4_9_0_0_0"}
    assert torch.equal(combined["4_4_0_0_0"], torch.tensor([[2.0]]))
    assert torch.equal(combined["9_9_0_0_0"], torch.tensor([[3.0]]))

    loaded = _two_row_partition(0)
    del loaded.node_features
    del loaded.edge_features
    block_to_feature(
        loaded,
        idp,
        blocks={
            "0_0_0_0_0": torch.tensor([[0.0]]),
            "4_4_0_0_0": torch.tensor([[7.0]]),
            "4_9_0_0_0": torch.tensor([[13.0]]),
        },
    )
    assert torch.equal(loaded.node_features, torch.tensor([[7.0]]))
    assert torch.equal(loaded.edge_features, torch.tensor([[13.0]]))


class _TorchObjectComm:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

    def alltoall(self, outgoing):
        gathered = [None] * self.world_size
        torch_dist.all_gather_object(gathered, outgoing)
        return [gathered[source][self.rank] for source in range(self.world_size)]

    def allgather(self, value):
        gathered = [None] * self.world_size
        torch_dist.all_gather_object(gathered, value)
        return gathered

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.world_size


def _communication_worker(rank, rendezvous):
    import dptb.utils.distributed as distributed

    torch_dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        distributed._RANK = rank
        distributed._WORLD_SIZE = 2
        distributed._MPI_COMM = _TorchObjectComm(rank, 2)
        distributed._INITIALIZED = True
        distributed._PLAN_CACHE.clear()

        graph = AtomicData.from_points(
            pos=torch.tensor(
                [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [1.6, 0.0, 0.0], [2.4, 0.0, 0.0]]
            ),
            atomic_numbers=torch.ones(4, dtype=torch.long),
            r_max=1.0,
            pbc=False,
            partition_options={"method": "rib"},
        ).to_dict()
        assert graph[AtomicDataDict.POSITIONS_KEY].shape[0] == 2
        ids = graph[AtomicDataDict.ATOM_INDEX_KEY].flatten()
        values = ids.to(torch.float32).unsqueeze(-1).requires_grad_()
        columns = distributed.node_column_values(graph, values)
        assert torch.equal(columns.flatten(), distributed.column_atom_index(graph).float())
        columns[2:].sum().mul(10.0 * (rank + 1)).backward()
        expected = torch.zeros_like(values)
        expected[ids.eq(1 if rank == 0 else 2)] = 20.0 if rank == 0 else 10.0
        assert torch.equal(values.grad, expected)

        edge = graph[AtomicDataDict.EDGE_INDEX_KEY]
        column_ids = distributed.column_atom_index(graph)
        edge_values = (
            100 * column_ids[edge[0]] + column_ids[edge[1]]
        ).float().unsqueeze(-1).requires_grad_()
        reversed_values = distributed.reverse_edge_values(graph, edge_values)
        expected_reverse = (
            100 * column_ids[edge[1]] + column_ids[edge[0]]
        ).float().unsqueeze(-1)
        assert torch.equal(reversed_values, expected_reverse)
        reversed_values.sum().backward()
        assert torch.equal(edge_values.grad, torch.ones_like(edge_values))

        route_keys = torch.tensor(
            [[9 if rank == 0 else 4, rank]], dtype=torch.long
        )
        route_values = torch.tensor(
            [[2.0 + rank]], requires_grad=True
        )
        received_keys, received_values = distributed.route_relation_contributions(
            graph, route_keys, route_values,
            target_owners=torch.tensor([1 - rank]),
        )
        assert int(received_keys[0, 0]) == (4 if rank == 0 else 9)
        received_values.sum().mul(float(rank + 1)).backward()
        assert torch.equal(
            route_values.grad,
            torch.tensor([[2.0 if rank == 0 else 1.0]]),
        )

        graph[AtomicDataDict.POSITIONS_KEY] = graph[
            AtomicDataDict.POSITIONS_KEY
        ].detach().requires_grad_()
        graph.pop(AtomicDataDict.EDGE_VECTORS_KEY, None)
        graph.pop(AtomicDataDict.EDGE_LENGTH_KEY, None)
        graph = AtomicDataDict.with_edge_vectors(graph, with_lengths=True)
        graph[AtomicDataDict.EDGE_LENGTH_KEY].square().sum().backward()
        local_position_grad = graph[AtomicDataDict.POSITIONS_KEY].grad.clone()

        serial_pos = torch.tensor(
            [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [1.6, 0.0, 0.0], [2.4, 0.0, 0.0]],
            requires_grad=True,
        )
        serial = AtomicData.from_points(
            pos=serial_pos,
            atomic_numbers=torch.ones(4, dtype=torch.long),
            r_max=1.0,
            pbc=False,
        ).to_dict()
        serial = AtomicDataDict.with_edge_vectors(serial, with_lengths=True)
        serial[AtomicDataDict.EDGE_LENGTH_KEY].square().sum().backward()
        assert torch.allclose(local_position_grad, serial_pos.grad[ids])
    finally:
        torch_dist.destroy_process_group()


def test_node_columns_and_reverse_edges_forward_backward(tmp_path):
    torch_mp.start_processes(
        _communication_worker,
        args=(str(tmp_path / "communication-rendezvous"),),
        nprocs=2,
        join=True,
        start_method="fork",
    )
