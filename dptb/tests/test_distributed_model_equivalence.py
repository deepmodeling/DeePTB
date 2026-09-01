"""Serial vs parallel model-inference equivalence for an owner-row partitioned graph.

Follows the communication emulation of ``test_distributed_atomic_data``: gloo
process groups carry the tensor exchanges while an object-transport mock stands
in for the MPI communicator used to build communication plans and route blocks.
Workers are started with the ``spawn`` method: forking after the parent has run
a model forward deadlocks OpenMP.
"""
import pickle

import pytest
import torch
import torch.distributed as torch_dist
import torch.multiprocessing as torch_mp

from dptb.data import AtomicData
from dptb.data.interfaces.ham_to_feature import feature_to_block
from dptb.nn.build import build_model

COMMON_OPTIONS = {
    "basis": {"C": ["2s", "2p"]},
    "device": "cpu",
    "dtype": "float64",
    "overlap": False,
    "seed": 42,
}

MODEL_OPTIONS = {
    "embedding": {"method": "slem", "r_max": 6.0, "irreps_hidden": "4x0e+4x1o+2x2e",
                  "n_layers": 1, "avg_num_neighbors": 10},
    "prediction": {"method": "e3tb", "neurons": [8, 8]},
}

LEM_OPTIONS = {
    "embedding": {"method": "lem", "r_max": 6.0, "irreps_hidden": "4x0e+4x1o+2x2e",
                  "n_layers": 1, "avg_num_neighbors": 10},
    "prediction": {"method": "e3tb", "neurons": [8, 8]},
}

TRINITY_OPTIONS = {
    "embedding": {"method": "trinity", "r_max": 6.0, "irreps_hidden": "4x0e+4x1o+2x2e",
                  "n_layers": 1, "avg_num_neighbors": 10, "mode": "full",
                  "three_center": {"projectors": {"C": ["1s", "1p"]}, "er_max": 6.0}},
    "prediction": {"method": "e3tb", "neurons": [8, 8]},
}


def _structure_kwargs(structure: str = "periodic"):
    generator = torch.Generator().manual_seed(2024)
    if structure == "clusters":
        # two bonded clusters + four isolated atoms in between (spacing larger
        # than every cutoff): with three RIB ranks the middle rank owns only
        # isolated atoms, i.e. zero edges and zero ghosts, while the other
        # ranks have work — the empty rank must still join every collective.
        cluster = torch.tensor([
            [0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.7, 0.7, 1.2],
        ], dtype=torch.float64)
        isolated = torch.tensor([
            [50.0, 0.3, 0.1], [60.0, 0.0, 0.5], [70.0, 0.4, 0.0], [80.0, 0.1, 0.2],
        ], dtype=torch.float64)
        pos = torch.cat((cluster, isolated, cluster + torch.tensor([130.0, 0.2, 0.1])))
        pos = pos + 0.05 * torch.rand(pos.shape, generator=generator, dtype=torch.float64)
        return dict(
            pos=pos, pbc=False,
            atomic_numbers=torch.full((len(pos),), 6, dtype=torch.long), r_max=6.0,
        )
    cell = torch.tensor(
        [[7.0, 0.0, 0.0], [0.6, 7.2, 0.0], [0.1, 0.4, 6.8]], dtype=torch.float64
    )
    pos = torch.rand((10, 3), generator=generator, dtype=torch.float64) @ cell
    return dict(
        pos=pos, cell=cell, pbc=True,
        atomic_numbers=torch.full((10,), 6, dtype=torch.long), r_max=6.0,
    )


def _infer_blocks(model, data):
    dd = AtomicData.to_AtomicDataDict(data)
    with torch.no_grad():
        dd = model.idp(dd)
        dd = model(dd)
        return {k: v.detach().cpu() for k, v in feature_to_block(data=dd, idp=model.idp).items()}


def _save_checkpoint(path, model_options):
    torch.manual_seed(42)
    model = build_model(model_options=model_options, common_options=dict(COMMON_OPTIONS))
    torch.save(
        {
            "config": {
                "model_options": model_options,
                "common_options": dict(COMMON_OPTIONS),
            },
            "model_state_dict": model.state_dict(),
            "task": "hamiltonian",
        },
        path,
    )


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


def _partition_worker(rank, world_size, structure, rendezvous, ckpt_path, out_path):
    torch.set_default_dtype(torch.float64)
    torch_dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size
    )
    try:
        import dptb.utils.distributed as distributed

        distributed._RANK = rank
        distributed._WORLD_SIZE = world_size
        distributed._LOCAL_RANK = rank
        distributed._MPI_COMM = _TorchObjectComm(rank, world_size)
        distributed._INITIALIZED = True
        distributed._PLAN_CACHE.clear()

        model = build_model(checkpoint=ckpt_path)
        model.eval()
        data = AtomicData.from_points(
            **_structure_kwargs(structure), partition_options={"method": "rib"}
        )
        blocks = _infer_blocks(model, data)

        gathered = [None] * world_size
        torch_dist.all_gather_object(
            gathered,
            {
                "n_edge": int(data.edge_index.shape[1]),
                "blocks": {k: v.numpy() for k, v in blocks.items()},
            },
        )
        if rank == 0:
            with open(out_path, "wb") as fid:
                pickle.dump(gathered, fid)
    finally:
        torch_dist.destroy_process_group()


def _run_equivalence(tmp_path, model_options, structure, world_size):
    ckpt_path = str(tmp_path / "model_ckpt.pth")
    _save_checkpoint(ckpt_path, model_options)

    serial_model = build_model(checkpoint=ckpt_path)
    serial_model.eval()
    serial_blocks = _infer_blocks(
        serial_model, AtomicData.from_points(**_structure_kwargs(structure))
    )

    out_path = str(tmp_path / "parallel_blocks.pkl")
    torch_mp.start_processes(
        _partition_worker,
        args=(world_size, structure, str(tmp_path / "rendezvous"),
              ckpt_path, out_path),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )
    with open(out_path, "rb") as fid:
        gathered = pickle.load(fid)

    rank_blocks = [item["blocks"] for item in gathered]
    for a in range(world_size):
        for b in range(a + 1, world_size):
            overlap = set(rank_blocks[a]) & set(rank_blocks[b])
            assert not overlap, f"blocks assembled on two ranks: {sorted(overlap)[:3]}"
    combined = {
        key: torch.as_tensor(value)
        for blocks in rank_blocks
        for key, value in blocks.items()
    }
    assert set(combined) == set(serial_blocks)
    for key, serial_value in serial_blocks.items():
        assert torch.allclose(serial_value, combined[key], atol=1e-10), key
    return gathered


@pytest.mark.order(1)
def test_partitioned_inference_matches_serial(tmp_path):
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        _run_equivalence(tmp_path, MODEL_OPTIONS, "periodic", world_size=2)
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.order(2)
def test_partitioned_inference_with_empty_rank_matches_serial(tmp_path):
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        gathered = _run_equivalence(tmp_path, MODEL_OPTIONS, "clusters", world_size=3)
        assert any(item["n_edge"] == 0 for item in gathered), \
            "the clusters structure should give one rank zero edges"
    finally:
        torch.set_default_dtype(previous_dtype)


@pytest.mark.order(3)
def test_zero_edge_serial_forward(tmp_path):
    # two atoms far beyond every cutoff: zero edges, env and onsitenv — the
    # forward must still produce the two onsite blocks (covering the
    # SO2/E3Hamiltonian/three-center empty paths of all three e3 models).
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        pos = torch.tensor([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        numbers = torch.full((2,), 6, dtype=torch.long)
        for name, model_options in (
            ("slem", MODEL_OPTIONS), ("lem", LEM_OPTIONS), ("trinity", TRINITY_OPTIONS),
        ):
            ckpt_path = str(tmp_path / f"{name}_ckpt.pth")
            _save_checkpoint(ckpt_path, model_options)
            model = build_model(checkpoint=ckpt_path)
            model.eval()
            kwargs = dict(pos=pos, pbc=False, atomic_numbers=numbers, r_max=6.0)
            if name == "trinity":
                kwargs["er_max"] = 6.0
            blocks = _infer_blocks(model, AtomicData.from_points(**kwargs))
            assert set(blocks) == {"0_0_0_0_0", "1_1_0_0_0"}, name
    finally:
        torch.set_default_dtype(previous_dtype)
