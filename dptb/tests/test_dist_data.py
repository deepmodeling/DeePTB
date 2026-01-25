import pytest
from dptb.data import AtomicData, _keys
import ase.io as io
try:
    import mpi4py
    _MPI = True
except:
    ImportError("The test is bypassed since the lack of MPI")
    _MPI = False
import os
from pathlib import Path

rootdir = os.path.join(Path(os.path.abspath(__file__)).parent, "data")



def test_hilbert_part():
    if _MPI:
        data = AtomicData.from_ase(io.read(os.path.join(rootdir, "hBN", "hBN.vasp")), r_max=4.0)
        data.partition_graph_hilbert(2, split=False)
        print(data[_keys.CLUSTER_CONNECTIVITY])
        print(data[_keys.CLUSTER_NODE_RANGE])
        print(data[_keys.CLUSTER_GHOST_LIST])

def test_metis_part():
    if _MPI:
        data = AtomicData.from_ase(io.read(os.path.join(rootdir, "hBN", "hBN.vasp")), r_max=4.0)
        data.partition_graph(2, split=False)
        print(data[_keys.CLUSTER_CONNECTIVITY])
        print(data[_keys.CLUSTER_NODE_RANGE])
        print(data[_keys.CLUSTER_GHOST_LIST])


# if __name__ == "__main__":
#     test_hilbert_part()
#     test_metis_part()

    