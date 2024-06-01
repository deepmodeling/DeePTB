import pytest
from dptb.data.interfaces.abacus import _abacus_parse
import lmdb
import os
import pickle
import h5py

@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    return str(request.config.rootdir)



def test_abacus_parse(root_directory):
    print(root_directory)
    _abacus_parse(
        input_path=root_directory+"/dptb/tests/data/mos2/abacus",
        data_name="OUT.ABACUS",
        output_path=root_directory+"/dptb/tests/data/mos2/abacus/conv.0",
        idx=0,
        get_Ham=True,
    )

    lmdb_env = lmdb.open(os.path.join(root_directory+"/dptb/tests/data/mos2/abacus/", 'data.lmdb'), map_size=1048576000000)

    _abacus_parse(
        input_path=root_directory+"/dptb/tests/data/mos2/abacus",
        data_name="OUT.ABACUS",
        output_path=root_directory+"/dptb/tests/data/mos2/abacus/conv.0",
        idx=0,
        lmdb_env=lmdb_env,
        output_mode="lmdb",
        get_Ham=True,
    )

    lmdb_env.close()

    lmdb_env = lmdb.open(os.path.join(root_directory+"/dptb/tests/data/mos2/abacus/", 'data.lmdb'), readonly=True, lock=False)
    with lmdb_env.begin() as txn:
        data_dict = txn.get(int(0).to_bytes(length=4, byteorder='big'))
        data_dict = pickle.loads(data_dict)
        ham_lmdb = data_dict["hamiltonian"]
    lmdb_env.close()


    file = h5py.File(root_directory+"/dptb/tests/data/mos2/abacus/conv.0/hamiltonians.h5", "r")
    ham_h5 = file['0']

    for k in ham_lmdb.keys():
        assert (ham_h5[k][:] - ham_lmdb[k]).sum() < 1e-7
    
    file.close()
