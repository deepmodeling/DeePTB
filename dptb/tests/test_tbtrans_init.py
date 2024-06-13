import pytest
import json
from dptb.negf.device_property import DeviceProperty
from dptb.nn.build import build_model
from dptb.negf.negf_hamiltonian_init import NEGFHamiltonianInit
import torch
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
    results_path=f'{root_directory}/dptb/tests/data/test_tbtrans/test_output'
    input_path = root_directory +"/dptb/tests/data/test_tbtrans/negf_tbtrans.json"
    structure=root_directory +"/dptb/tests/data/test_tbtrans/test_hBN_zigzag_struct.xyz"

   
    negf_json = json.load(open(input_path))
    model_json = json.load(open(model_ckpt))
    model = build_model(model_ckpt,model_options=negf_json['model_options'],common_options=negf_json['common_options'])


      
    # apihost = NNSKHost(checkpoint=model_ckpt, config=config)
    # apihost.register_plugin(InitSKModel())
    # apihost.build()
    # apiHrk = NN2HRK(apihost=apihost, mode='nnsk')

    # run_opt = {
    #         "run_sk": True,
    #         "init_model":model_ckpt,
    #         "results_path":f'{root_directory}/dptb/tests/data/test_tbtrans/',
    #         "structure":f'{root_directory}/dptb/tests/data/test_tbtrans/test_hBN_zigzag_struct.xyz',
    #         "log_path": '/data/DeepTB/dptb_Zjj/DeePTB/dptb/tests/data/test_tbtrans/output',
    #         "log_level": 5,
    #         "use_correction":False
    #     }

    # jdata = j_loader(config)
    # jdata = jdata['task_options']
    tbtrans_hBN = TBTransInputSet(model = model,
                                 AtomicData_options = negf_json['AtomicData_options'], 
                                 structure = structure,
                                 results_path= results_path,
                                 basis_dict=negf_json['common_options']['basis'],
                                 **negf_json['task_options'])

    # # check _orbitals_name_get
    element_orbital_name = tbtrans_hBN._orbitals_name_get(['2s', '2p']) 
    assert element_orbital_name == ['2s', '2py', '2pz', '2px']
    element_orbital_name = tbtrans_hBN._orbitals_name_get(['3s', '3p', 'd*']) 
    assert element_orbital_name == ['3s', '3py', '3pz', '3px', 'dxy*', 'dyz*', 'dz2*', 'dxz*', 'dx2-y2*']



    tbtrans_hBN.hamil_get_write(write_nc=False)
    assert (tbtrans_hBN.allbonds_all[0].detach().numpy()-np.array([5, 0, 5, 0, 0, 0, 0])).max()<1e-5
    assert (tbtrans_hBN.allbonds_all[50].detach().numpy()-np.array([ 5,  2,  5, 18,  0,  0, -1])).max()<1e-5
    assert (tbtrans_hBN.hamil_block_all[0].detach().numpy()*1.0000/13.605662285137/2-   np.array([[-0.73303634,  0.        ,  0.        ,  0.        ],
                                                                        [ 0.        ,  0.04233637,  0.        ,  0.        ],
                                                                        [ 0.        ,  0.        ,  0.04233636,  0.        ],
                                                                        [ 0.        ,  0.        ,  0.        , -0.27156556]])).max()<1e-5
    assert (tbtrans_hBN.hamil_block_all[50].detach().numpy()*1.0000/13.605662285137/2-np.array([[-0.03164609, -0.        ,  0.02028139, -0.        ],
                                                                        [ 0.        ,  0.00330366,  0.        ,  0.        ],
                                                                        [-0.02028139,  0.        , -0.05393751,  0.        ],
                                                                        [ 0.        ,  0.        ,  0.        ,  0.00330366]])).max()<1e-5
    assert (tbtrans_hBN.allbonds_lead_L[0].detach().numpy()-np.array([5, 0, 5, 0, 0, 0, 0])).max()<1e-5
    assert (tbtrans_hBN.allbonds_lead_L[50].detach().numpy()-np.array([5, 4, 7, 7, 0, 0, 0])).max()<1e-5
    assert (tbtrans_hBN.hamil_block_lead_L[0].detach().numpy()*1.0000/13.605662285137/2-np.array([[-0.73303634,  0.        ,  0.        ,  0.        ],
                                                                        [ 0.        ,  0.04233637,  0.        ,  0.        ],
                                                                        [ 0.        ,  0.        ,  0.04233636,  0.        ],
                                                                        [ 0.        ,  0.        ,  0.        , -0.27156556]])).max()<1e-5
    assert (tbtrans_hBN.hamil_block_lead_L[50].detach().numpy()*1.0000/13.605662285137/2-np.array([[ 0.1145315 , -0.06116847,  0.10594689,  0.        ],
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
   
    G_eigs_correct = np.array([[-19.9606056213, -17.1669273376, -17.0796546936, -16.8952560425,
        -11.2022151947, -10.1263637543,  -9.9411945343,  -8.1748561859,
         -8.0049209595,  -7.6048202515,  -6.6505813599,  -3.7902746201,
         -3.7882986069,  -3.2289390564,  -3.2289323807,  -3.1756639481,
          3.1868720055,   3.2401518822,   3.2401537895,   6.4502696991,
          7.7058019638,   7.8654155731,  10.8850431442,  10.9806480408,
         23.6024589539,  23.6807479858,  28.3076782227,  28.3091220856,
         28.6845626831,  30.7864990234,  33.2417755127,  33.2541275024]])

    M_eigs_correct = np.array([[-18.7672195435, -18.6261577606, -16.9293708801, -16.9104366302,
        -11.2307147980, -11.1986064911,  -7.5301399231,  -7.3464851379,
         -6.6541309357,  -6.6479454041,  -5.7928838730,  -5.7928771973,
         -5.2177238464,  -5.2155799866,  -3.1756563187,  -3.1756496429,
          3.1868739128,   3.1868796349,   5.8489637375,   5.8489723206,
          7.6596388817,   7.7445230484,   7.9219026566,   7.9709992409,
         28.1335067749,  28.1563091278,  28.6396503448,  28.6454830170,
         29.5444660187,  29.5520248413,  30.7856063843,  30.7873668671]])

    
    assert (G_eigs[0]-G_eigs_correct[0]).max()<1e-4
    assert (M_eigs[0]-M_eigs_correct[0]).max()<1e-4



