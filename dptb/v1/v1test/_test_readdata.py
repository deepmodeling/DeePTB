from dptb.dataprocess.datareader import read_data
import numpy as np
import pytest
from ase.io import read
from ase.io.trajectory import Trajectory
import os

@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
        return str(request.config.rootdir)

kpo = np.array([[ 0.        ,  0.        ,  0.        ],
       [ 0.11111111,  0.        ,  0.        ],
       [ 0.22222222, -0.11111111,  0.        ],
       [ 0.22222222,  0.        ,  0.        ],
       [ 0.33333333, -0.11111111,  0.        ],
       [ 0.33333333,  0.        ,  0.        ],
       [ 0.33333333,  0.33333333,  0.        ],
       [ 0.44444444, -0.22222222,  0.        ],
       [ 0.44444444, -0.11111111,  0.        ],
       [ 0.44444444,  0.        ,  0.        ],
       [ 0.44444444,  0.33333333,  0.        ],
       [ 0.44444444,  0.44444444,  0.        ]])

eig000=np.array([-22.51867585, -10.28791391,  -6.41103909,  -6.40648338,
         3.13601486,   3.7619632 ,   6.47297807,   6.47308359,
         9.06268509,  12.59532112,  23.07222237,  23.56770135,
        27.43609078,  27.44191137,  36.92642387,  36.94885367,
        37.01588031,  40.39335872,  40.40952965,  42.16299308,
        42.19160683,  54.37409228,  54.39658144,  70.81407143])

def test_read_data(root_directory):
    read2bin(root_directory)
    filepath = root_directory + '/dptb/tests/data/hBN/data'
    prefix = 'set'
    cutoff=4
    proj_atom_anglr_m = {"N":["s","p"],"B":["s","p"]}
    proj_atom_neles =  {"N":5,"B":3}

    struct_list_sets, kpoints_sets, eigens_sets, bandinfo_sets, wannier_sets = read_data(filepath, prefix, cutoff, proj_atom_anglr_m, proj_atom_neles)

    assert len(struct_list_sets) == len(kpoints_sets) == len(eigens_sets) == 1
    assert len(struct_list_sets[0]) == 1
    assert kpoints_sets[0].shape ==(12,3)
    assert (np.abs(kpoints_sets[0]-kpo) < 1e-6).all()
    assert eigens_sets[0].shape == (1, 12, 24)
    assert (np.abs(eigens_sets[0][0,0] - eig000) < 1e-6).all()

def read2bin(root_directory):
    in_dir = root_directory + '/dptb/tests/data/hBN/data/set.0'
    eig_file ='eigenvalues.dat'
    kpoints_file = 'kpoints.dat'
    struct_file='struct.vasp'

    fp = open(in_dir + '/' + eig_file,'r')
    for i in range(2):
        line = fp.readline()
    fp.close()
    nsp,nkp,nbnd = int(line.split()[1]), int(line.split()[2]),int(line.split()[3])
    data = np.loadtxt(in_dir + '/' + eig_file)
    eigvaules = np.reshape(data,[nsp,nkp,nbnd])

    kpoints = np.loadtxt(in_dir + '/' + kpoints_file)

    trajstrs = read(in_dir + '/' + struct_file, format='vasp',index=':')
    traj = Trajectory(in_dir + '/' + 'xdat2.traj',mode='w')
    for i  in range(1):
        traj.write(atoms=trajstrs[i])  
    traj.close()

    #np.save(in_dir + '/' + 'eigs.npy',eigvaules)
    #np.save(in_dir + '/' + 'kpoints.npy',kpoints)
    #os.remove(in_dir + '/' + 'xdat.traj')
