import numpy as np
from dptb.utils.tools import j_must_have
from ase.io import read
import ase
from typing import Union
import matplotlib.pyplot as plt
import torch
from typing import Optional
import matplotlib
import logging
import os
from dptb.data import AtomicData, AtomicDataDict
from dptb.data.interfaces.ham_to_feature import feature_to_block

log = logging.getLogger(__name__)

def write_block(
        data: Union[AtomicData, ase.Atoms, str],
        model: torch.nn.Module,
        AtomicData_options: dict={},
        device: Union[str, torch.device]=None
        ):

    model.eval()
    if isinstance(device, str):
        device = torch.device(device)
    # get the AtomicData structure and the ase structure
    if isinstance(data, str):
        structase = read(data)
        data = AtomicData.from_ase(structase, **AtomicData_options)
    elif isinstance(data, ase.Atoms):
        structase = data
        data = AtomicData.from_ase(structase, **AtomicData_options)
    elif isinstance(data, AtomicData):
        structase = data.to("cpu").to_ase()
        data = data

    data = AtomicData.to_AtomicDataDict(data.to(device))
    with torch.no_grad():
        data = model.idp(data)

        # set the kpoint of the AtomicData
        data = model(data)
        block = feature_to_block(data=data, idp=model.idp)

    return block


def write_block_parallel_hdf5(path: str, blocks: dict, group_name: str = "0", chunk_size: int = 4096):
    """Write rank-owned blocks to one legacy-layout HDF5 file.

    Uses collective parallel HDF5 when h5py was built with MPI support;
    otherwise falls back to gathering all blocks on rank 0 and writing with
    ordinary serial h5py (correct at any scale that fits rank 0's memory).
    """
    import h5py

    if h5py.get_config().mpi:
        _write_block_collective(path, blocks, group_name, chunk_size)
    else:
        log.warning(
            "h5py was not built with MPI support; gathering all Hamiltonian "
            "blocks to rank 0 for a serial write. Install parallel HDF5 and "
            "rebuild h5py with HDF5_MPI=ON for a scalable collective write."
        )
        _write_block_gathered(path, blocks, group_name)


def _write_block_collective(path: str, blocks: dict, group_name: str, chunk_size: int):
    """Collective parallel-HDF5 write; requires mpi4py and MPI-enabled h5py."""
    from mpi4py import MPI
    import h5py

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    partial = path + ".partial"
    if rank == 0 and os.path.exists(partial):
        os.remove(partial)
    comm.Barrier()

    local_items = sorted(blocks.items())
    local_chunks = (len(local_items) + chunk_size - 1) // chunk_size
    nchunks = comm.allreduce(local_chunks, op=MPI.MAX)
    with h5py.File(partial, "w", driver="mpio", comm=comm) as fid:
        group = fid.create_group(group_name)
        for chunk_id in range(nchunks):
            begin = chunk_id * chunk_size
            selected = local_items[begin:begin + chunk_size]
            metadata = [
                (key, tuple(value.shape), value.detach().cpu().numpy().dtype.str, rank)
                for key, value in selected
            ]
            gathered = comm.allgather(metadata)
            merged = sorted(item for rank_items in gathered for item in rank_items)
            for key, shape, dtype, owner in merged:
                if key in group:
                    raise RuntimeError(f"Block {key!r} has more than one canonical owner")
                group.create_dataset(key, shape=shape, dtype=dtype)
            comm.Barrier()
            for key, value in selected:
                group[key][...] = value.detach().cpu().numpy()
            comm.Barrier()

    comm.Barrier()
    if rank == 0:
        os.replace(partial, path)
    comm.Barrier()


def _write_block_gathered(path: str, blocks: dict, group_name: str):
    """Gather every rank's blocks to rank 0 and write with serial h5py.

    Communicates through DeePTB's communicator abstraction (``alltoall`` of
    plain objects), so it works with any communicator the distributed context
    provides, and degenerates to a plain local write in serial runs.
    """
    import h5py

    from dptb.utils.distributed import get_mpi_comm, get_rank, get_world_size

    world_size = get_world_size()
    rank = get_rank()
    local_items = [(key, value.detach().cpu().numpy()) for key, value in blocks.items()]
    if world_size > 1:
        comm = get_mpi_comm()
        if comm is None:
            raise RuntimeError("Distributed block writing requires an initialized communicator")
        outgoing = [local_items if peer == 0 else [] for peer in range(world_size)]
        incoming = comm.alltoall(outgoing)
    else:
        incoming = [local_items]

    if rank == 0:
        partial = path + ".partial"
        if os.path.exists(partial):
            os.remove(partial)
        with h5py.File(partial, "w") as fid:
            group = fid.create_group(group_name)
            for source_items in incoming:
                for key, value in source_items:
                    if key in group:
                        raise RuntimeError(f"Block {key!r} has more than one canonical owner")
                    group[key] = value
        os.replace(partial, path)
