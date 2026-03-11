import os
import glob
import lmdb
import pickle
from ase.db.core import connect
import ase.io
import numpy as np

lmdb_path = r'data.3457651.lmdb'
db_env = lmdb.open(lmdb_path, readonly=True, lock=False)
with db_env.begin() as txn:
    stat = txn.stat()
    entries = stat['entries']
    print(entries)
    for valid_idx in range(50):
        data_dict = txn.get(valid_idx.to_bytes(length=4, byteorder='big'))
        data_dict = pickle.loads(data_dict)
        print(data_dict.keys())
        for a_key in ['pos', 'atomic_numbers']:
            print(data_dict[a_key])
            print(type(data_dict[a_key]))
        break
    for a_key in data_dict['hamiltonian_0'].keys():
        break
    print(a_key)
    print(data_dict['hamiltonian_0'][a_key])
db_env.close()
first_mat = data_dict['hamiltonian_0'][a_key]
max_abs = np.abs(first_mat).max() < 1e-10
print(max_abs)
print(np.abs(first_mat).max())