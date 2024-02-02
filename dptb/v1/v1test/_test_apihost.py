import pytest
from dptb.v1.init_nnsk import InitSKModel
from dptb.v1.init_dptb import InitDPTBModel
from dptb.nnops.v1.NN2HRK import NN2HRK
from dptb.nnops.v1.apihost import NNSKHost,DPTBHost
from dptb.entrypoints.run import run

@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
        return str(request.config.rootdir)


def test_dptbhost(root_directory):
    checkfile = f'{root_directory}/dptb/tests/data/hBN/checkpoint/best_dptb.pth'
    use_correction = f'{root_directory}/dptb/tests/data/hBN/checkpoint/best_nnsk.pth'
    dptbapi = DPTBHost(dptbmodel=checkfile,use_correction=use_correction)
    dptbapi.register_plugin(InitDPTBModel())
    dptbapi.build()


def test_nnskhost(root_directory):
    checkfile = f'{root_directory}/dptb/tests/data/hBN/checkpoint/best_nnsk.pth'
    nnskapi = NNSKHost(checkpoint=checkfile)
    nnskapi.register_plugin(InitSKModel())
    nnskapi.build()

def test_nnsk_nrl_json(root_directory):
    checkfile = f'{root_directory}/examples/NRL-TB/silicon/ckpt/nrl_ckpt.json'
    config=f'{root_directory}/examples/NRL-TB/silicon/input_nrl.json'

    nnskapi = NNSKHost(checkpoint=checkfile, config=config)
    nnskapi.register_plugin(InitSKModel())
    nnskapi.build()

def test_nnsk_nrl_pth(root_directory):
    checkfile = f'{root_directory}/examples/NRL-TB/silicon/ckpt/nrl_ckpt.pth'
    nnskapi = NNSKHost(checkpoint=checkfile)
    nnskapi.register_plugin(InitSKModel())
    nnskapi.build()


def test_nnsk2HRK(root_directory):
    checkfile = f'{root_directory}/dptb/tests/data/hBN/checkpoint/best_nnsk.pth'
    nnskapi = NNSKHost(checkpoint=checkfile)
    nnskapi.register_plugin(InitSKModel())
    nnskapi.build()
    nnHrk = NN2HRK(apihost=nnskapi, mode='nnsk')

def test_nnsk2HRK_pth(root_directory):
    checkfile = f'{root_directory}/examples/NRL-TB/silicon/ckpt/nrl_ckpt.pth'
    nnskapi = NNSKHost(checkpoint=checkfile)
    nnskapi.register_plugin(InitSKModel())
    nnskapi.build()
    nnHrk = NN2HRK(apihost=nnskapi, mode='nnsk')


def test_dptb2HRK(root_directory):
    checkfile = f'{root_directory}/dptb/tests/data/hBN/checkpoint/best_dptb.pth'
    use_correction = f'{root_directory}/dptb/tests/data/hBN/checkpoint/best_nnsk.pth'
    dptbapi = DPTBHost(dptbmodel=checkfile,use_correction=use_correction)
    dptbapi.register_plugin(InitDPTBModel())
    dptbapi.build()
    nnHrk = NN2HRK(apihost=dptbapi, mode='dptb')

    
def test_run_nnsk(root_directory):
    run(
        INPUT=f'{root_directory}/dptb/tests/data/post_nnsk.json',
        model_ckpt=None,
        output=f"{root_directory}/dptb/tests/data/postrun",
        init_model=f"{root_directory}/dptb/tests/data/hBN/checkpoint/best_nnsk.pth",
        run_sk=True,
        structure=None,
        log_level=2,
        log_path=None,
        use_correction=None
    )

def test_run_band_nnsk_nrl(root_directory):
    run(
        INPUT=f'{root_directory}/dptb/tests/data/nrl/band_pthckpt.json',
        model_ckpt=None,
        output=f"{root_directory}/dptb/tests/data/postrun",
        init_model=f"{root_directory}/examples/NRL-TB/silicon/ckpt/nrl_ckpt.pth",
        run_sk=True,
        structure=None,
        log_level=2,
        log_path=None,
        use_correction=None
    )

def test_run_band_nnsk_nrl_json(root_directory):
    run(
        INPUT=f'{root_directory}/dptb/tests/data/nrl/band_jsonckpt.json',
        model_ckpt=None,
        output=f"{root_directory}/dptb/tests/data/postrun",
        init_model=f"{root_directory}/examples/NRL-TB/silicon/ckpt/nrl_ckpt.json",
        run_sk=True,
        structure=None,
        log_level=2,
        log_path=None,
        use_correction=None
    )

def test_run_dptb(root_directory):
    run(
        INPUT=f'{root_directory}/dptb/tests/data/post_dptb.json',
        model_ckpt=None,
        output=f"{root_directory}/dptb/tests/data/postrun",
        init_model=f"{root_directory}/dptb/tests/data/hBN/checkpoint/best_dptb.pth",
        run_sk=False,
        structure=None,
        log_level=2,
        log_path=None,
        use_correction=None
    )

def test_run_dptbnnsk(root_directory):
    run(
        INPUT=f'{root_directory}/dptb/tests/data/post_dptb.json',
        model_ckpt=None,
        output=f"{root_directory}/dptb/tests/data/postrun",
        init_model=f"{root_directory}/dptb/tests/data/hBN/checkpoint/best_dptb.pth",
        run_sk=False,
        structure=None,
        log_level=2,
        log_path=None,
        use_correction=f"{root_directory}/dptb/tests/data/hBN/checkpoint/best_nnsk.pth"
    )