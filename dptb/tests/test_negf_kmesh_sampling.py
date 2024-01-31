from dptb.entrypoints import run
import pytest
import torch
import numpy as np
from dptb.utils.make_kpoints import kmesh_sampling_negf


@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    """
    :return:
    """
    return str(request.config.rootdir)


def test_negf_ksampling(root_directory):

    #-------- 1D ksampling-----------

    ## even meshgrid
    meshgrid = [1,4,1]
    ### Gamma center
    is_gamma_center = True
    is_time_reversal = True
    kp , wk = kmesh_sampling_negf(meshgrid, is_gamma_center, is_time_reversal)
    assert abs(kp - np.array([[0.  , 0.  , 0.  ],[0.  , 0.25, 0.  ],[0.  , 0.5 , 0.  ]])).max()<1e-5
    assert abs(wk - np.array([0.25, 0.5 , 0.25])).max()<1e-5

    is_time_reversal = False
    kp , wk = kmesh_sampling_negf(meshgrid, is_gamma_center, is_time_reversal)
    assert abs(kp - np.array([[0., 0., 0.],[0., 0.25, 0.],[0., 0.5, 0.],[0., 0.75, 0.]])).max()<1e-5
    assert abs(wk - np.array([0.25, 0.25 ,0.25, 0.25])).max()<1e-5

    ### Monkhorst-Packing sampling
    is_gamma_center = False
    is_time_reversal = True
    kp , wk = kmesh_sampling_negf(meshgrid, is_gamma_center, is_time_reversal)
    assert abs(kp-np.array([[0.  , 0.125  , 0.  ],[0.  , 0.375, 0.  ]])).max()<1e-5
    assert abs(wk-np.array([0.5, 0.5])).max()<1e-5

    is_time_reversal = False
    kp , wk = kmesh_sampling_negf(meshgrid, is_gamma_center, is_time_reversal)
    assert abs(kp-np.array([[ 0., -0.375,  0.],[ 0., -0.125, 0.],[ 0.,0.125,0.],[ 0.,0.375, 0.]])).max()<1e-5
    assert abs(wk - np.array([0.25, 0.25, 0.25, 0.25])).max()<1e-5

    ## odd meshgrid
    meshgrid = [1,5,1]
    ### Gamma center
    is_gamma_center = True
    is_time_reversal = True
    kp , wk = kmesh_sampling_negf(meshgrid, is_gamma_center, is_time_reversal)
    assert abs(kp - np.array([[0., 0., 0.],[0., 0.2, 0. ],[0., 0.4 , 0.]])).max()<1e-5
    assert abs(wk - np.array([0.2, 0.4, 0.4])).max()<1e-5

    is_time_reversal = False
    kp , wk = kmesh_sampling_negf(meshgrid, is_gamma_center, is_time_reversal)
    assert abs(kp - np.array([[0., 0., 0.],[0., 0.2, 0. ],[0., 0.4 , 0.],[0., 0.6 , 0.],[0., 0.8 , 0.]])).max()<1e-5
    assert abs(wk - np.array([0.2, 0.2 ,0.2, 0.2, 0.2])).max()<1e-5

    ### Monkhorst-Packing sampling
    is_gamma_center = False
    is_time_reversal = True
    kp , wk = kmesh_sampling_negf(meshgrid, is_gamma_center, is_time_reversal)
    assert abs(kp-np.array([[0.  , 0.0  ,0 ],[0.  , 0.2  ,0 ],[0., 0.4, 0 ]])).max()<1e-5
    assert abs(wk-np.array([0.2, 0.4, 0.4])).max()<1e-5

    is_time_reversal = False
    kp , wk = kmesh_sampling_negf(meshgrid, is_gamma_center, is_time_reversal)
    assert abs(kp-np.array([[ 0., -0.4,  0.],[ 0., -0.2, 0.],[ 0.,0.0,0.],[ 0.,0.2, 0.],[ 0.,0.4,0.]])).max()<1e-5
    assert abs(wk - np.array([0.2, 0.2, 0.2, 0.2, 0.2])).max()<1e-5

    #-------- 1D ksampling-----------




    #-------- 2D ksampling-----------
    ## even meshgrid
    meshgrid = [2,2,1]
    ### Gamma center
    is_gamma_center = True
    is_time_reversal = True
    kp , wk = kmesh_sampling_negf(meshgrid, is_gamma_center, is_time_reversal)
    assert abs(kp - np.array([[0.  , 0.  , 0.  ],[0.  , 0.5, 0.  ],[0.5  , 0. , 0.  ],[0.5  , 0.5 , 0.  ]])).max()<1e-5
    assert abs(wk - np.array([0.25, 0.25 , 0.25, 0.25])).max()<1e-5

    is_time_reversal = False
    kp , wk = kmesh_sampling_negf(meshgrid, is_gamma_center, is_time_reversal)
    assert abs(kp - np.array([[0.  , 0.  , 0.  ],[0.  , 0.5, 0.  ],[0.5  , 0. , 0.  ],[0.5  , 0.5 , 0.  ]])).max()<1e-5
    assert abs(wk - np.array([0.25, 0.25 , 0.25, 0.25])).max()<1e-5

    ### Monkhorst-Packing sampling
    is_gamma_center = False
    is_time_reversal = True
    kp , wk = kmesh_sampling_negf(meshgrid, is_gamma_center, is_time_reversal)
    assert abs(kp - np.array([[ 0.25, -0.25,  0.  ],[ 0.25,  0.25,  0.  ]])).max()<1e-5
    assert abs(wk - np.array([0.5,0.5])).max()<1e-5

    is_time_reversal = False
    kp , wk = kmesh_sampling_negf(meshgrid, is_gamma_center, is_time_reversal)
    assert abs(kp - np.array([[ -0.25, -0.25,  0.  ],[ -0.25,  0.25,  0.  ],[ 0.25, -0.25,  0.  ],[ 0.25,0.25, 0.  ]])).max()<1e-5
    assert abs(wk - np.array([0.25, 0.25, 0.25, 0.25])).max()<1e-5

    ## odd meshgrid
    meshgrid = [3,3,1]
    ### Gamma center
    is_gamma_center = True
    is_time_reversal = True
    kp , wk = kmesh_sampling_negf(meshgrid, is_gamma_center, is_time_reversal)
    assert abs(kp - np.array([[0.        , 0.        , 0.        ],
       [0.        , 0.33333333, 0.        ],
       [0.33333333, 0.        , 0.        ],
       [0.33333333, 0.33333333, 0.        ],
       [0.33333333, 0.66666667, 0.        ]])).max()<1e-5
    assert abs(wk - np.array([0.11111111, 0.22222222, 0.22222222, 0.22222222, 0.22222222])).max()<1e-5

    is_time_reversal = False
    kp , wk = kmesh_sampling_negf(meshgrid, is_gamma_center, is_time_reversal)
    assert abs(kp - np.array([[0.        , 0.        , 0.        ],
       [0.        , 0.33333333, 0.        ],
       [0.        , 0.66666667, 0.        ],
       [0.33333333, 0.        , 0.        ],
       [0.33333333, 0.33333333, 0.        ],
       [0.33333333, 0.66666667, 0.        ],
       [0.66666667, 0.        , 0.        ],
       [0.66666667, 0.33333333, 0.        ],
       [0.66666667, 0.66666667, 0.        ]])).max()<1e-5
    assert abs(wk - np.array([0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111,
       0.11111111, 0.11111111, 0.11111111, 0.11111111])).max()<1e-5

    ### Monkhorst-Packing sampling
    is_gamma_center = False
    is_time_reversal = True
    kp , wk = kmesh_sampling_negf(meshgrid, is_gamma_center, is_time_reversal)
    assert abs(kp - np.array([[ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.33333333,  0.        ],
       [ 0.33333333, -0.33333333,  0.        ],
       [ 0.33333333,  0.        ,  0.        ],
       [ 0.33333333,  0.33333333,  0.        ]])).max()<1e-5
    assert abs(wk - np.array([0.11111111, 0.22222222, 0.22222222, 0.22222222, 0.22222222])).max()<1e-5

    is_time_reversal = False
    kp , wk = kmesh_sampling_negf(meshgrid, is_gamma_center, is_time_reversal)
    assert abs(kp - np.array([[-0.33333333, -0.33333333,  0.        ],
       [-0.33333333,  0.        ,  0.        ],
       [-0.33333333,  0.33333333,  0.        ],
       [ 0.        , -0.33333333,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.33333333,  0.        ],
       [ 0.33333333, -0.33333333,  0.        ],
       [ 0.33333333,  0.        ,  0.        ],
       [ 0.33333333,  0.33333333,  0.        ]])).max()<1e-5
    assert abs(wk - np.array([0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111,
       0.11111111, 0.11111111, 0.11111111, 0.11111111])).max()<1e-5