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

    tbtrans_hBN.load_model()
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

    tbtrans_hBN.hamil_get()
    # tbtrans_hBN.hamil_write()
    # check the hamiltonian through Hk at Gamma and M 
    H_lead = tbtrans_hBN.H_lead_L
    G_eigs = sisl.BandStructure(H_lead, [[0., 0.,0.]], 1, ["G"])
    M_eigs = sisl.BandStructure(H_lead, [[0, 0.5 ,0 ]], 1, ["M"])
    Ef = -9.874358177185059
    G_eigs = G_eigs.apply.array.eigh() -Ef
    M_eigs = M_eigs.apply.array.eigh() -Ef
   
    G_eigs_right = np.array([[-19.95362763, -17.10774579, -17.10774579, -16.9118761 ,
            -11.20609829, -10.01050689, -10.01050689,  -8.11443067,
            -8.11443067,  -7.60442045,  -6.6550603 ,  -3.79211664,
            -3.79211663,  -3.22863532,  -3.22863532,  -3.17535758,
            3.18703263,   3.24031037,   3.24031037,   6.45306332,
            7.70583557,   7.91107113,  10.91058699,  10.91058699,
            23.64785516,  23.64785516,  28.30755414,  28.30755428,
            28.65719263,  30.78452851,  33.25399887,  33.25399887]])

    M_eigs_right = np.array([[-18.69653568, -18.69653568, -16.91187582, -16.91187582,
            -11.20609828, -11.20609828,  -7.44726991,  -7.44726991,
            -6.65506047,  -6.65506047,  -5.79252308,  -5.79252308,
            -5.2193769 ,  -5.2193769 ,  -3.17535758,  -3.17535758,
            3.18703263,   3.18703263,   5.84906816,   5.84906816,
            7.74726616,   7.74726616,   7.91107121,   7.91107121,
            28.12912079,  28.12912079,  28.65719227,  28.65719227,
            29.54182975,  29.54182975,  30.78452877,  30.78452877]])

    assert (G_eigs[0]-G_eigs_right[0]).max()<1e-5
    assert (M_eigs[0]-M_eigs_right[0]).max()<1e-5




