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
            "structure":f'{root_directory}/dptb/tests/data/test_tbtrans/test_hBN_struct.xyz',
            "log_path": '/data/DeepTB/dptb_Zjj/DeePTB/dptb/tests/data/test_tbtrans/output',
            "log_level": 5,
            "use_correction":False
        }

    jdata = j_loader(config)
    jdata = jdata['task_options']
    tbtrans_hBN = TBTransInputSet(apiHrk=apiHrk,run_opt=run_opt, jdata=jdata)

    tbtrans_hBN.load_model()
    assert (tbtrans_hBN.allbonds_all[0].detach().numpy()-np.array([5, 0, 5, 0, 0, 0, 0])).max()<1e-5
    assert (tbtrans_hBN.allbonds_all[50].detach().numpy()-np.array([ 7,  2,  7, 18,  0,  0, -1])).max()<1e-5
    assert (tbtrans_hBN.hamil_block_all[0].detach().numpy()-np.array([[-7.3303628e-01, -4.8828125e-03, -2.7465820e-03,  0.0000000e+00],
                                                        [-4.8828125e-03,  4.2336360e-02, -2.9802322e-08,  0.0000000e+00],
                                                        [-2.7465820e-03, -2.9802322e-08,  4.2336382e-02,  0.0000000e+00],
                                                        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00, -2.7156556e-01]])).max()<1e-5
    assert (tbtrans_hBN.hamil_block_all[50].detach().numpy()-np.array([[ 0.03893254, -0.01220665,  0.02114254, -0],
                                                        [ 0.01220665,  0.01277969, -0.02714315,  0.        ],
                                                        [-0.02114254, -0.02714315,  0.0441219 ,  0.        ],
                                                        [ 0.        ,  0.        ,  0.        , -0.00289141]])).max()<1e-5
    assert (tbtrans_hBN.allbonds_lead_L[0].detach().numpy()-np.array([5, 0, 5, 0, 0, 0, 0])).max()<1e-5
    assert (tbtrans_hBN.allbonds_lead_L[50].detach().numpy()-np.array([5, 4, 7, 6, 0, 0, 0])).max()<1e-5
    assert (tbtrans_hBN.hamil_block_lead_L[0].detach().numpy()-np.array([[-7.3303634e-01, -4.8828125e-03, -2.7465820e-03,  0.0000000e+00],
                                                        [-4.8828125e-03,  4.2336360e-02, -2.9802322e-08,  0.0000000e+00],
                                                        [-2.7465820e-03, -2.9802322e-08,  4.2336386e-02,  0.0000000e+00],
                                                        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00, -2.7156556e-01]])).max()<1e-5
    assert (tbtrans_hBN.hamil_block_lead_L[50].detach().numpy()-np.array([[ 0.1145315 , -0.10594689,  0.06116847,  0.        ],
                                                        [ 0.26915616, -0.30906558,  0.22751863,  0.        ],
                                                        [-0.1553974 ,  0.22751863, -0.04634975,  0.        ],
                                                        [-0.        ,  0.        ,  0.        ,  0.08500822]])).max()<1e-5

    tbtrans_hBN.hamil_get()
    tbtrans_hBN.hamil_write()
    # check the hamiltonian through Hk at Gamma and M 
    H_lead = tbtrans_hBN.H_lead_L
    G_eigs = sisl.BandStructure(H_lead, [[0., 0.,0.]], 1, ["G"])
    M_eigs = sisl.BandStructure(H_lead, [[0, 0.5 ,0 ]], 1, ["M"])
    Ef = -9.874358177185059
    G_eigs = G_eigs.apply.array.eigh() -Ef
    M_eigs = M_eigs.apply.array.eigh() -Ef
   
    G_eigs_right = np.array([[-19.95531891, -16.93914904, -16.93841612, -16.86674503,
            -11.22071648, -11.22027601, -11.17725089,  -7.60441858,
            -6.65505841,  -6.65185484,  -6.65177689,  -3.79211871,
            -3.79077201,  -3.17535786,  -3.1753559 ,  -3.1753559 ,
            3.18703176,   3.18703176,   3.18703372,   6.4516442 ,
            7.70583451,   7.75399143,   7.98816478,   7.99058251,
            28.30755911,  28.30931748,  28.61445576,  28.61568511,
            28.74029286,  30.78453022,  30.78638238,  30.78643695]])

    M_eigs_right = np.array([[-18.67664519, -18.67634743, -17.14453635, -17.143923,
            -10.08802197, -10.08757032,  -8.02545742,  -8.02536069,
            -7.41175673,  -7.41082759,  -5.79252192,  -5.79252192,
            -5.21937684,  -5.21937662,  -3.22863363,  -3.22863363,
            3.24030949,   3.24030949,   5.84906781,   5.84906781,
            7.63450212,   7.6363856 ,  10.98470396,  10.98561984,
            23.62405886,  23.62457958,  28.18439875,  28.18544892,
            29.54183293,  29.54183343,  33.22755142,  33.22787417]])

    assert (G_eigs[0]-G_eigs_right[0]).max()<1e-5
    assert (M_eigs[0]-M_eigs_right[0]).max()<1e-5




