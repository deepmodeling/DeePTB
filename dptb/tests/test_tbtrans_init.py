import pytest
from dptb.nnops.apihost import NNSKHost
from dptb.plugins.init_nnsk import InitSKModel
from dptb.nnops.NN2HRK import NN2HRK
from dptb.postprocess.tbtrans_init import TBTransInputSet
from dptb.utils.tools import j_loader
import numpy as np

@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    return str(request.config.rootdir)


def test_tbtrans_init(root_directory):

    # check whether sisl is installed: if not, skip this test
    try:
        import sisl
    except:
        pytest.skip('sisl is not installed which is necessary for TBtrans Input Generation. Therefore, skipping test_tbtrans_init.')

    model_ckpt = f'{root_directory}/dptb/tests/data/test_tbtrans/best_nnsk_b3.600_c3.600_w0.300.json'
    config = f'{root_directory}/dptb/tests/data/test_tbtrans/negf_tbtrans.json'
    apihost = NNSKHost(checkpoint=model_ckpt, config=config)
    apihost.register_plugin(InitSKModel())
    apihost.build()
    apiHrk = NN2HRK(apihost=apihost, mode='nnsk')

    run_opt = {
            "run_sk": True,
            "init_model":model_ckpt,
            "results_path":f'{root_directory}/dptb/tests/data/test_tbtrans/',
            "structure":f'{root_directory}/dptb/tests/data/test_tbtrans/test_hBN_zigzag_struct.xyz',
            "log_path": '/data/DeepTB/dptb_Zjj/DeePTB/dptb/tests/data/test_tbtrans/output',
            "log_level": 5,
            "use_correction":False
        }

    jdata = j_loader(config)
    jdata = jdata['task_options']
    tbtrans_hBN = TBTransInputSet(apiHrk=apiHrk,run_opt=run_opt, jdata=jdata)

    # check _orbitals_name_get
    element_orbital_name = tbtrans_hBN._orbitals_name_get(['2s', '2p']) 
    assert element_orbital_name == ['2s', '2py', '2pz', '2px']
    element_orbital_name = tbtrans_hBN._orbitals_name_get(['3s', '3p', 'd*']) 
    assert element_orbital_name == ['3s', '3py', '3pz', '3px', 'dxy*', 'dyz*', 'dz2*', 'dxz*', 'dx2-y2*']



    tbtrans_hBN.hamil_get_write(write_nc=False)
    assert (tbtrans_hBN.allbonds_all[0].detach().numpy()-np.array([5, 0, 5, 0, 0, 0, 0])).max()<1e-5
    assert (tbtrans_hBN.allbonds_all[50].detach().numpy()-np.array([ 5,  2,  5, 18,  0,  0, -1])).max()<1e-5
    assert (tbtrans_hBN.hamil_block_all[0].detach().numpy()-   np.array([[-0.73303634,  0.        ,  0.        ,  0.        ],
                                                                        [ 0.        ,  0.04233637,  0.        ,  0.        ],
                                                                        [ 0.        ,  0.        ,  0.04233636,  0.        ],
                                                                        [ 0.        ,  0.        ,  0.        , -0.27156556]])).max()<1e-5
    assert (tbtrans_hBN.hamil_block_all[50].detach().numpy()-np.array([[-0.03164609, -0.        ,  0.02028139, -0.        ],
                                                                        [ 0.        ,  0.00330366,  0.        ,  0.        ],
                                                                        [-0.02028139,  0.        , -0.05393751,  0.        ],
                                                                        [ 0.        ,  0.        ,  0.        ,  0.00330366]])).max()<1e-5
    assert (tbtrans_hBN.allbonds_lead_L[0].detach().numpy()-np.array([5, 0, 5, 0, 0, 0, 0])).max()<1e-5
    assert (tbtrans_hBN.allbonds_lead_L[50].detach().numpy()-np.array([5, 4, 7, 7, 0, 0, 0])).max()<1e-5
    assert (tbtrans_hBN.hamil_block_lead_L[0].detach().numpy()-np.array([[-0.73303634,  0.        ,  0.        ,  0.        ],
                                                                        [ 0.        ,  0.04233637,  0.        ,  0.        ],
                                                                        [ 0.        ,  0.        ,  0.04233636,  0.        ],
                                                                        [ 0.        ,  0.        ,  0.        , -0.27156556]])).max()<1e-5
    assert (tbtrans_hBN.hamil_block_lead_L[50].detach().numpy()-np.array([[ 0.1145315 , -0.06116847,  0.10594689,  0.        ],
                                                                            [ 0.15539739, -0.04634972,  0.22751862,  0.        ],
                                                                            [-0.26915616,  0.22751862, -0.30906558,  0.        ],
                                                                            [-0.        ,  0.        ,  0.        ,  0.08500822]])).max()<1e-5

    
    # tbtrans_hBN.hamil_write()
    # check the hamiltonian through Hk at Gamma and M 
    H_lead = tbtrans_hBN.H_lead_L
    G_eigs = sisl.BandStructure(H_lead, [[0., 0.,0.]], 1, ["G"])
    M_eigs = sisl.BandStructure(H_lead, [[0, 0.5 ,0 ]], 1, ["M"])
    Ef = -9.874358177185059
    G_eigs = G_eigs.apply.array.eigh() -Ef
    M_eigs = M_eigs.apply.array.eigh() -Ef
   
    G_eigs_right = np.array([[-19.95431228, -17.10836511, -17.10836511, -16.91249093, -11.20658215,
 -10.01096331, -10.01096331,  -8.11484357,  -8.11484357,  -7.60482165,
  -6.65543971,  -3.79243033,  -3.79243032,  -3.22893608,  -3.22893608,
  -3.17565711,   3.18687914,   3.2401581,    3.2401581,    6.4529848,
   7.70578579,   7.91102607 , 10.91061078 , 10.91061078 , 23.6481713,
  23.6481713,   28.30797724 , 28.30797738 , 28.65762375 , 30.78500847,
  33.2545355 ,  33.2545355 ]])

    M_eigs_right = np.array([[-18.69719147, -18.69719147, -16.91249065, -16.91249065, -11.20658214,
 -11.20658214,  -7.4476675,   -7.4476675 ,  -6.65543987  ,-6.65543987,
  -5.79288269,  -5.79288269,  -5.21972335 , -5.21972335 , -3.17565711,
  -3.17565711,   3.18687914,   3.18687914  , 5.84897577 ,  5.84897577,
   7.74721734,   7.74721734,   7.91102615 ,  7.91102615 , 28.12953979,
  28.12953979,  28.65762339,  28.65762339,  29.54228118,  29.54228118,
  30.78500872,  30.78500872]])

    print(M_eigs[0])
    assert (G_eigs[0]-G_eigs_right[0]).max()<1e-5
    assert (M_eigs[0]-M_eigs_right[0]).max()<1e-5




