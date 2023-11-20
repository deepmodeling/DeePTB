import pytest
from dptb.nnops.apihost import NNSKHost
from dptb.plugins.init_nnsk import InitSKModel
from dptb.nnops.NN2HRK import NN2HRK
from dptb.postprocess.tbtrans_init import TBTransInputSet


@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    return str(request.config.rootdir)


def test_tbtrans_init(root_directory):
    # checkfile = f'{root_directory}/dptb/tests/data/hBN/checkpoint/best_nnsk.pth'
    # nnskapi = NNSKHost(checkpoint=checkfile)
    # nnskapi.register_plugin(InitSKModel())
    # nnskapi.build()

    model_ckpt = f'{root_directory}/dptb/tests/data/test_tbtrans/best_nnsk_b3.600_c3.600_w0.300.json'
    jdata = f'{root_directory}/dptb/tests/data/test_tbtrans/negf_tbtrans.json'
    apihost = NNSKHost(checkpoint=model_ckpt, config=jdata)
    apihost.register_plugin(InitSKModel())
    apihost.build()
    apiHrk = NN2HRK(apihost=apihost, mode='nnsk')

    run_opt = {
            "run_sk": True,
            "init_model":model_ckpt,
            "results_path":f'{root_directory}/dptb/tests/data/test_negf/',
            "structure":f'{root_directory}/dptb/tests/data/test_tbtrans/test_hBN_struct.xyz',
            "log_path": '/data/DeepTB/dptb_Zjj/DeePTB/dptb/tests/data/test_tbtrans/output',
            "log_level": 5,
            "use_correction":False
        }

    tbtrans_init = TBTransInputSet(apiHrk=apiHrk,run_opt=run_opt, jdata=jdata)


