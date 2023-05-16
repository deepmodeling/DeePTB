import pytest
from dptb.utils.make_kpoints import abacus_kpath, ase_kpath
from ase.io import read
import numpy as np

@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
        return str(request.config.rootdir)

true_klist = np.array([[0.        , 0.        , 0.        ],
                       [0.125     , 0.        , 0.        ],
                       [0.25      , 0.        , 0.        ],
                       [0.375     , 0.        , 0.        ],
                       [0.5       , 0.        , 0.        ],
                       [0.45833333, 0.08333333, 0.        ],
                       [0.41666666, 0.16666666, 0.        ],
                       [0.375     , 0.25      , 0.        ],
                       [0.33333333, 0.33333333, 0.        ],
                       [0.25      , 0.25      , 0.        ],
                       [0.16666666, 0.16666666, 0.        ],
                       [0.08333333, 0.08333333, 0.        ],
                       [0.        , 0.        , 0.        ]])

true_xlist = np.array([0.        , 0.0576428 , 0.1152856 , 0.1729284 , 0.2305712 ,
       0.26385128, 0.29713137, 0.33041146, 0.36369154, 0.43025171,
       0.49681188, 0.56337205, 0.62993223])
true_hskp = np.array([0.        , 0.2305712 , 0.36369154, 0.62993223])

def test_intpkpath(root_directory):
    hbn_file=f'{root_directory}/dptb/tests/data/hBN/hBN.vasp'
    kpath=[[0.00000000, 0.00000000, 0.00000000, 4],
           [0.50000000, 0.00000000, 0.00000000, 4],
           [0.33333333, 0.33333333, 0.00000000, 4],
           [0.00000000, 0.00000000, 0.00000000, 1]]
    klabels = ['G','M','K','G']
    struct = read(hbn_file)
    klist, xlist, high_sym_kpoints = abacus_kpath(structase=struct, kpath=kpath)
    assert (np.abs(klist -  true_klist) < 1e-6).all()
    assert (np.abs(xlist -  true_xlist) < 1e-6).all()
    assert (np.abs(high_sym_kpoints - true_hskp)< 1e-6).all()


true_klist2= np.array([[0.        , 0.        , 0.        ],
                       [0.125     , 0.        , 0.        ],
                       [0.25      , 0.        , 0.        ],
                       [0.375     , 0.        , 0.        ],
                       [0.5       , 0.        , 0.        ],
                       [0.33333333, 0.33333333, 0.        ],
                       [0.25      , 0.25      , 0.        ],
                       [0.16666666, 0.16666666, 0.        ],
                       [0.08333333, 0.08333333, 0.        ],
                       [0.        , 0.        , 0.        ]])

true_xlist2 = np.array([0.        , 0.0576428 , 0.1152856 , 0.1729284 , 0.2305712 ,
       0.2305712 , 0.29713137, 0.36369154, 0.43025171, 0.49681188])
true_hskp2 = np.array([0.        , 0.2305712 , 0.2305712 , 0.49681188])

def test_intpkpath2(root_directory):
    hbn_file=f'{root_directory}/dptb/tests/data/hBN/hBN.vasp'
    kpath=[[0.00000000, 0.00000000, 0.00000000, 4],
           [0.50000000, 0.00000000, 0.00000000, 1],
           [0.33333333, 0.33333333, 0.00000000, 4],
           [0.00000000, 0.00000000, 0.00000000, 1]]
    klabels = ['G','M','K','G']
    struct = read(hbn_file)
    klist, xlist, high_sym_kpoints = abacus_kpath(structase=struct, kpath=kpath)
    assert (np.abs(klist -  true_klist2) < 1e-6).all()
    assert (np.abs(xlist -  true_xlist2) < 1e-6).all()
    assert (np.abs(high_sym_kpoints - true_hskp2)< 1e-6).all()