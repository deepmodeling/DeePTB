from dptb.entrypoints import train
import pytest
import torch
import numpy as np

@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    return str(request.config.rootdir)

# test nnsk model run, with s p orbital and non onsite.
def test_nnsk_valence(root_directory):
    INPUT_file = root_directory+"/dptb/tests/data/test_sktb/input/input_valence.json"
    output = root_directory+"/dptb/tests/data/test_sktb/output"

    train(INPUT=INPUT_file, init_model=None, restart=None, train_soc=False,\
          output=output+"/test_valence", log_level=5, log_path=output+"/test_valence.log")

# test nnsk model run, with s p d* orbital and strain mode for onsite.
# using init_model of the model trained in the previous test.
def test_nnsk_strain_polar(root_directory):
    INPUT_file = root_directory+"/dptb/tests/data/test_sktb/input/input_strain_polar.json"
    output = root_directory+"/dptb/tests/data/test_sktb/output"
    init_model = root_directory+"/dptb/tests/data/test_sktb/output/test_valence/checkpoint/nnsk.iter2.pth"

    train(INPUT=INPUT_file, init_model=init_model, restart=None, train_soc=False,\
          output=output+"/test_strain_polar", log_level=5, log_path=output+"/test_strain_polar.log")

# test push  rs and w in nnsk model run, with s p d* orbital and strain mode for onsite.
# using init_model of the model trained in the previous test. 
def test_nnsk_push(root_directory):
    INPUT_file_rs = root_directory + "/dptb/tests/data/test_sktb/input/input_push_rs.json"
    INPUT_file_w = root_directory + "/dptb/tests/data/test_sktb/input/input_push_w.json"
    output = root_directory + "/dptb/tests/data/test_sktb/output"
    init_model = root_directory + "/dptb/tests/data/test_sktb/output/test_strain_polar/checkpoint/nnsk.ep2.pth"

    train(INPUT=INPUT_file_rs, init_model=init_model, restart=None, train_soc=False,\
          output=output+"/test_push_rs", log_level=5, log_path=output+"/test_push_rs.log")
    train(INPUT=INPUT_file_w, init_model=init_model, restart=None, train_soc=False,\
          output=output+"/test_push_w", log_level=5, log_path=output+"/test_push_w.log")
    
    model_rs = torch.load(f"{root_directory}/dptb/tests/data/test_sktb/output/test_push_rs/checkpoint/nnsk.iter10.pth")
    model_w = torch.load(f"{root_directory}/dptb/tests/data/test_sktb/output/test_push_w/checkpoint/nnsk.iter10.pth")
    # test push limits
    # 10 epoch, 0.01 step, 1 period -> 0.05 added.
    assert np.isclose(model_rs["config"]["model_options"]["nnsk"]["hopping"]["rs"], 2.65)
    assert np.isclose(model_w["config"]["model_options"]["nnsk"]["hopping"]["w"], 0.35)

# train on md structures.
def test_md(root_directory):
    INPUT_file =root_directory + "/dptb/tests/data/test_sktb/input/input_md.json"
    output = root_directory + "/dptb/tests/data/test_sktb/output"
    init_model = root_directory + "/dptb/tests/data/test_sktb/output/test_push_w/checkpoint/nnsk.iter10.pth"

    train(INPUT=INPUT_file, init_model=init_model, restart=None, train_soc=False,\
          output=output+"/test_md", log_level=5, log_path=output+"/test_md.log")
    
# train  dptb with env.
def test_dptb(root_directory):
    INPUT_file =root_directory + "/dptb/tests/data/test_sktb/input/input_dptb.json"
    output = root_directory + "/dptb/tests/data/test_sktb/output"
    init_model = root_directory + "/dptb/tests/data/test_sktb/output/test_md/checkpoint/nnsk.ep2.pth"

    train(INPUT=INPUT_file, init_model=init_model, restart=None, train_soc=False,\
          output=output+"/test_dptb", log_level=5, log_path=output+"/test_dptb.log")