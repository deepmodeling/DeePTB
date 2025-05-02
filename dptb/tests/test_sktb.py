from dptb.entrypoints import train
from dptb.utils.config_check import check_config_train
import pytest
import torch
import numpy as np
from dptb.utils.tools import j_loader
import json

@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    return str(request.config.rootdir)

# test nnsk model run, with s p orbital and non onsite.
@pytest.mark.order(1)
def test_nnsk_valence(root_directory):
    INPUT_file = root_directory+"/dptb/tests/data/test_sktb/input/input_valence.json"
    output = root_directory+"/dptb/tests/data/test_sktb/output"
    check_config_train(INPUT=INPUT_file, init_model=None, restart=None)
    train(INPUT=INPUT_file, init_model=None, restart=None,\
          output=output+"/test_valence", log_level=5, log_path=output+"/test_valence.log")

# test nnsk model run, with s p d* orbital and strain mode for onsite.
# using init_model of the model trained in the previous test.
@pytest.mark.order(2)
def test_nnsk_strain_polar(root_directory):
    INPUT_file = root_directory+"/dptb/tests/data/test_sktb/input/input_strain_polar.json"
    output = root_directory+"/dptb/tests/data/test_sktb/output"
    init_model = root_directory+"/dptb/tests/data/test_sktb/output/test_valence/checkpoint/nnsk.latest.pth"

    check_config_train(INPUT=INPUT_file, init_model=init_model, restart=None)
    train(INPUT=INPUT_file, init_model=init_model, restart=None,\
          output=output+"/test_strain_polar", log_level=5, log_path=output+"/test_strain_polar.log")

# test push  rs and w in nnsk model run, with s p d* orbital and strain mode for onsite.
# using init_model of the model trained in the previous test. 
@pytest.mark.order(3)
def test_nnsk_push(root_directory):
    INPUT_file_rs = root_directory + "/dptb/tests/data/test_sktb/input/input_push_rs.json"
    INPUT_file_w = root_directory + "/dptb/tests/data/test_sktb/input/input_push_w.json"
    output = root_directory + "/dptb/tests/data/test_sktb/output"
    init_model = root_directory + "/dptb/tests/data/test_sktb/output/test_strain_polar/checkpoint/nnsk.best.pth"
    
    check_config_train(INPUT=INPUT_file_rs, init_model=init_model, restart=None)
    train(INPUT=INPUT_file_rs, init_model=init_model, restart=None,\
          output=output+"/test_push_rs", log_level=5, log_path=output+"/test_push_rs.log")
    
    check_config_train(INPUT=INPUT_file_w, init_model=init_model, restart=None)
    train(INPUT=INPUT_file_w, init_model=init_model, restart=None,\
          output=output+"/test_push_w", log_level=5, log_path=output+"/test_push_w.log")
    
    model_rs = torch.load(f"{root_directory}/dptb/tests/data/test_sktb/output/test_push_rs/checkpoint/nnsk.iter_rs2.700_w0.300.pth", weights_only=False)
    model_w = torch.load(f"{root_directory}/dptb/tests/data/test_sktb/output/test_push_w/checkpoint/nnsk.iter_rs5.000_w0.400.pth", weights_only=False)
    # test push limits
    # 10 epoch, 0.01 step, 1 period -> 0.05 added.
    assert np.isclose(model_rs["config"]["model_options"]["nnsk"]["hopping"]["rs"], 2.700)
    assert np.isclose(model_w["config"]["model_options"]["nnsk"]["hopping"]["w"], 0.40)

# train on md structures.
@pytest.mark.order(4)
def test_md(root_directory):
    INPUT_file =root_directory + "/dptb/tests/data/test_sktb/input/input_md.json"
    output = root_directory + "/dptb/tests/data/test_sktb/output"
    init_model = root_directory + "/dptb/tests/data/test_sktb/output/test_push_w/checkpoint/nnsk.iter_rs5.000_w0.400.pth"

    check_config_train(INPUT=INPUT_file, init_model=init_model, restart=None)
    train(INPUT=INPUT_file, init_model=init_model, restart=None,\
          output=output+"/test_md", log_level=5, log_path=output+"/test_md.log")
    
# train  dptb with env.
@pytest.mark.order(5)
def test_dptb(root_directory):
    INPUT_file =root_directory + "/dptb/tests/data/test_sktb/input/input_dptb.json"
    output = root_directory + "/dptb/tests/data/test_sktb/output"
    init_model = root_directory + "/dptb/tests/data/test_sktb/output/test_md/checkpoint/nnsk.latest.pth"

    check_config_train(INPUT=INPUT_file, init_model=init_model, restart=None)
    train(INPUT=INPUT_file, init_model=init_model, restart=None,\
          output=output+"/test_dptb", log_level=5, log_path=output+"/test_dptb.log")
    

@pytest.mark.order(2)
def test_init_V1_json(root_directory):
    INPUT_file =root_directory + "/dptb/tests/data/test_sktb/input/input_initv1json.json"
    output = root_directory + "/dptb/tests/data/test_sktb/output"
    init_model = root_directory + "/dptb/tests/data/json_model/Si_v1_nnsk_b2.600_c2.600_w0.300.json"
    
    check_config_train(INPUT=INPUT_file, init_model=init_model, restart=None)
    train(INPUT=INPUT_file, init_model=init_model, restart=None,\
          output=output+"/test_v1json", log_level=5, log_path=output+"/test_v1json.log")

@pytest.mark.order(2)
def test_init_V2_json(root_directory):
    INPUT_file =root_directory + "/dptb/tests/data/test_sktb/input/input_initv1json.json"
    output = root_directory + "/dptb/tests/data/test_sktb/output"
    init_model = root_directory + "/dptb/tests/data/json_model/Si_v2ckpt.json"
    
    check_config_train(INPUT=INPUT_file, init_model=init_model, restart=None)
    train(INPUT=INPUT_file, init_model=init_model, restart=None,\
          output=output+"/test_v2json", log_level=5, log_path=output+"/test_v2json.log")


@pytest.mark.order(2)
def test_init_nrl_json(root_directory):
    INPUT_file =root_directory + "/dptb/tests/data/test_sktb/input/input_nrl.json"
    output = root_directory + "/dptb/tests/data/test_sktb/output"
    init_model = root_directory + "/dptb/tests/data/json_model/Si_nrl.json"
    
    check_config_train(INPUT=INPUT_file, init_model=init_model, restart=None)
    train(INPUT=INPUT_file, init_model=init_model, restart=None,\
          output=output+"/test_nrl", log_level=5, log_path=output+"/test_nrl.log")


@pytest.mark.order(2)
def test_init_nrl_jsonfz(root_directory):
    INPUT_file =root_directory + "/dptb/tests/data/test_sktb/input/input_nrl_fz.json"
    output = root_directory + "/dptb/tests/data/test_sktb/output"
    init_model = root_directory + "/dptb/tests/data/json_model/Si_nrl.json"
    
    check_config_train(INPUT=INPUT_file, init_model=init_model, restart=None)
    train(INPUT=INPUT_file, init_model=init_model, restart=None,\
          output=output+"/test_nrlfz", log_level=5, log_path=output+"/test_nrlfz.log")


@pytest.mark.order(1)
def test_soc_from_nonsoc_v1json(root_directory):
    INPUT_file =root_directory + "/dptb/tests/data/Sn/soc/input/input_soc.json"
    output = root_directory + "/dptb/tests/data/Sn/soc/output"
    init_model = root_directory + "/dptb/tests/data/Sn/soc/ckpt_nsoc/v1ckpt_c6.0w0.1.json"
    
    check_config_train(INPUT=INPUT_file, init_model=init_model, restart=None)
    with pytest.raises(KeyError):
        train(INPUT=INPUT_file, init_model=init_model, restart=None,\
            output=output+"/test_nsoc_v1json", log_level=5, log_path=output+"/test_nsoc_v1json.log")

@pytest.mark.order(1)
def test_soc_from_nonsoc_v2json(root_directory):
    INPUT_file =root_directory + "/dptb/tests/data/Sn/soc/input/input_soc.json"
    output = root_directory + "/dptb/tests/data/Sn/soc/output"
    init_model = root_directory + "/dptb/tests/data/Sn/soc/ckpt_nsoc/v2ckpt.json"
    
    check_config_train(INPUT=INPUT_file, init_model=init_model, restart=None)
    train(INPUT=INPUT_file, init_model=init_model, restart=None,\
            output=output+"/test_nsoc_v2json", log_level=5, log_path=output+"/test_nsoc_v2json.log")


@pytest.mark.order(1)
def test_soc_from_nonsoc_v2pth(root_directory):
    INPUT_file =root_directory + "/dptb/tests/data/Sn/soc/input/input_soc.json"
    output = root_directory + "/dptb/tests/data/Sn/soc/output"
    # init_model = root_directory + "/dptb/tests/data/Sn/soc/output/test_nsoc_v2json/checkpoint/nnsk.best.pth"
    init_model = root_directory + "/dptb/tests/data/Sn/soc/ckpt_nsoc/nnsk.ep100.pth"

    check_config_train(INPUT=INPUT_file, init_model=init_model, restart=None)
    train(INPUT=INPUT_file, init_model=init_model, restart=None,\
            output=output+"/test_nsoc_v2pth", log_level=5, log_path=output+"/test_nsoc_v2pth.log")


@pytest.mark.order(1)
def test_soc_from_soc_v1json(root_directory):
    INPUT_file =root_directory + "/dptb/tests/data/Sn/soc/input/input_soc.json"
    output = root_directory + "/dptb/tests/data/Sn/soc/output"
    init_model = root_directory + "/dptb/tests/data/Sn/soc/ckpt_soc/v1ckpt_c6.0w0.1.json"
    
    check_config_train(INPUT=INPUT_file, init_model=init_model, restart=None)
    train(INPUT=INPUT_file, init_model=init_model, restart=None,\
            output=output+"/test_soc_v1json", log_level=5, log_path=output+"/test_soc_v1json.log")


@pytest.mark.order(1)
def test_soc_from_soc_v2json(root_directory):
    INPUT_file =root_directory + "/dptb/tests/data/Sn/soc/input/input_soc.json"
    output = root_directory + "/dptb/tests/data/Sn/soc/output"
    init_model = root_directory + "/dptb/tests/data/Sn/soc/ckpt_soc/v2ckpt.json"
    
    check_config_train(INPUT=INPUT_file, init_model=init_model, restart=None)
    train(INPUT=INPUT_file, init_model=init_model, restart=None,\
            output=output+"/test_soc_v2json", log_level=5, log_path=output+"/test_soc_v2json.log")


@pytest.mark.order(1)
def test_soc_from_soc_v2pth(root_directory):
    INPUT_file =root_directory + "/dptb/tests/data/Sn/soc/input/input_soc.json"
    output = root_directory + "/dptb/tests/data/Sn/soc/output"
    init_model = root_directory + "/dptb/tests/data/Sn/soc/ckpt_soc/nnsk.ep100.pth"
    
    check_config_train(INPUT=INPUT_file, init_model=init_model, restart=None)
    train(INPUT=INPUT_file, init_model=init_model, restart=None,\
            output=output+"/test_soc_v2json", log_level=5, log_path=output+"/test_soc_v2json.log")


@pytest.mark.order(1)
def test_soc_from_nonsoc_v2json_fz(root_directory):
    INPUT_file =root_directory + "/dptb/tests/data/Sn/soc/input/input_soc_fz.json"
    output = root_directory + "/dptb/tests/data/Sn/soc/output"
    init_model = root_directory + "/dptb/tests/data/Sn/soc/ckpt_nsoc/v2ckpt.json"
    
    check_config_train(INPUT=INPUT_file, init_model=init_model, restart=None)
    train(INPUT=INPUT_file, init_model=init_model, restart=None,\
            output=output+"/test_nsoc_v2jsonfz", log_level=5, log_path=output+"/test_nsoc_v2jsonfz.log")


@pytest.mark.order(1)
def test_soc_from_nonsoc_v2pth_fz(root_directory):
    INPUT_file =root_directory + "/dptb/tests/data/Sn/soc/input/input_soc_fz.json"
    output = root_directory + "/dptb/tests/data/Sn/soc/output"
    # init_model = root_directory + "/dptb/tests/data/Sn/soc/output/test_nsoc_v2json/checkpoint/nnsk.best.pth"
    init_model = root_directory + "/dptb/tests/data/Sn/soc/ckpt_nsoc/nnsk.ep100.pth"

    check_config_train(INPUT=INPUT_file, init_model=init_model, restart=None)
    train(INPUT=INPUT_file, init_model=init_model, restart=None,\
            output=output+"/test_nsoc_v2pthfz", log_level=5, log_path=output+"/test_nsoc_v2pthfz.log")

@pytest.mark.order(1)
def test_soc_from_nonsoc_v2json_fz_all(root_directory):
    INPUT_file =root_directory + "/dptb/tests/data/Sn/soc/input/input_soc_fz.json"
    output = root_directory + "/dptb/tests/data/Sn/soc/output"
    init_model = root_directory + "/dptb/tests/data/Sn/soc/ckpt_nsoc/v2ckpt.json"

    jdata = j_loader(INPUT_file)
    jdata['model_options']['nnsk']['freeze'] = True

    inputfz = output + "/inputfz.json"
    with open(inputfz, 'w') as f:
        json.dump(jdata, f, indent=4)

    check_config_train(INPUT=inputfz, init_model=init_model, restart=None)
    with pytest.raises(RuntimeError):
        train(INPUT=inputfz, init_model=init_model, restart=None,\
            output=output+"/test_nsoc_v2jsonfz_all", log_level=5, log_path=output+"/test_nsoc_v2jsonfz_all.log")

@pytest.mark.order(1)
def test_soc_from_nonsoc_v2json_fz_fail(root_directory):
    INPUT_file =root_directory + "/dptb/tests/data/Sn/soc/input/input_soc_fz.json"
    output = root_directory + "/dptb/tests/data/Sn/soc/output"
    init_model = root_directory + "/dptb/tests/data/Sn/soc/ckpt_nsoc/v2ckpt.json"

    jdata = j_loader(INPUT_file)
    jdata['model_options']['nnsk']['freeze'] = ["wrongtag", "hopping"]

    inputfz = output + "/inputfz.json"
    with open(inputfz, 'w') as f:
        json.dump(jdata, f, indent=4)

    check_config_train(INPUT=inputfz, init_model=init_model, restart=None)
    with pytest.raises(ValueError):
        train(INPUT=inputfz, init_model=init_model, restart=None,\
            output=output+"/test_nsoc_v2jsonfz_all", log_level=5, log_path=output+"/test_nsoc_v2jsonfz_all.log")


@pytest.mark.order(1)
def test_dftbsk_mixed(root_directory):
    INPUT_file =root_directory + "/dptb/tests/data/hBN/input/input_mix_dftbsk.json"
    output = root_directory + "/dptb/tests/data/hBN/output"
    init_model = None
    
    check_config_train(INPUT=INPUT_file, init_model=init_model, restart=None)
    train(INPUT=INPUT_file, init_model=init_model, restart=None,\
            output=output+"/test_dftbsk_mixed", log_level=5, log_path=output+"/test_dftbsk_mixed.log")

@pytest.mark.order(2)
def test_dftbsk_mixed_init(root_directory):
    INPUT_file =root_directory + "/dptb/tests/data/hBN/input/input_mix_dftbsk.json"
    output = root_directory + "/dptb/tests/data/hBN/output"
    init_model = root_directory + "/dptb/tests/data/hBN/output/test_dftbsk_mixed/checkpoint/mix.best.pth"
    
    check_config_train(INPUT=INPUT_file, init_model=init_model, restart=None)
    train(INPUT=INPUT_file, init_model=init_model, restart=None,\
            output=output+"/test_dftbsk_mixed", log_level=5, log_path=output+"/test_dftbsk_mixed.log")

@pytest.mark.order(1)
def test_nnsk_from_dftbsk(root_directory):
    INPUT_file =root_directory + "/dptb/tests/data/hBN/input/input_nnsk_from_dftbsk.json"
    output = root_directory + "/dptb/tests/data/hBN/output"
    init_model = None
    
    check_config_train(INPUT=INPUT_file, init_model=init_model, restart=None)
    train(INPUT=INPUT_file, init_model=init_model, restart=None,\
            output=output+"/test_nnsk_from_dftbsk", log_level=5, log_path=output+"/test_nnsk_from_dftbsk.log")

@pytest.mark.order(2)
def test_nnsk_i(root_directory):
    INPUT_file =root_directory + "/dptb/tests/data/hBN/input/input_nnsk_i.json"
    output = root_directory + "/dptb/tests/data/hBN/output"
    init_model = root_directory + "/dptb/tests/data/hBN/output/test_nnsk_from_dftbsk/checkpoint/nnsk.best.pth"
    
    check_config_train(INPUT=INPUT_file, init_model=init_model, restart=None)
    train(INPUT=INPUT_file, init_model=init_model, restart=None,\
            output=output+"/test_nnsk_i", log_level=5, log_path=output+"/test_nnsk_i.log")
