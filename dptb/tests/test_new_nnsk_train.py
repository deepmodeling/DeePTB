from dptb.entrypoints import train
import pytest
import torch
import numpy as np

@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    return str(request.config.rootdir)

def test_nnsk_valence():
    INPUT_file = "./dptb/tests/data/test_data_nequip/input/input_valence.json"
    output = "./dptb/tests/data/test_data_nequip/output"

    train(INPUT=INPUT_file, init_model=None, restart=None, train_soc=False,\
          output=output+"/test_valence", log_level=5, log_path=output+"/test_valence.log")
    
def test_nnsk_strain_polar():
    INPUT_file = "./dptb/tests/data/test_data_nequip/input/input_strain_polar.json"
    output = "./dptb/tests/data/test_data_nequip/output"
    init_model = "./dptb/tests/data/test_data_nequip/output/test_valence/checkpoint/nnsk.iter6.pth"

    train(INPUT=INPUT_file, init_model=init_model, restart=None, train_soc=False,\
          output=output+"/test_strain_polar", log_level=5, log_path=output+"/test_strain_polar.log")

def test_nnsk_push():
    INPUT_file_rs = "./dptb/tests/data/test_data_nequip/input/input_push_rs.json"
    INPUT_file_w = "./dptb/tests/data/test_data_nequip/input/input_push_w.json"
    output = "./dptb/tests/data/test_data_nequip/output"
    init_model = "./dptb/tests/data/test_data_nequip/output/test_strain_polar/checkpoint/nnsk.iter6.pth"

    train(INPUT=INPUT_file_rs, init_model=init_model, restart=None, train_soc=False,\
          output=output+"/test_push_rs", log_level=5, log_path=output+"/test_push_rs.log")
    train(INPUT=INPUT_file_w, init_model=init_model, restart=None, train_soc=False,\
          output=output+"/test_push_w", log_level=5, log_path=output+"/test_push_w.log")
    
    model_rs = torch.load("./dptb/tests/data/test_data_nequip/output/test_push_rs/checkpoint/nnsk.iter11.pth")
    model_w = torch.load("./dptb/tests/data/test_data_nequip/output/test_push_w/checkpoint/nnsk.iter11.pth")
    # test push limits
    # 10 epoch, 0.01 step, 1 period -> 0.05 added.
    assert np.isclose(model_rs["config"]["model_options"]["nnsk"]["hopping"]["rs"], 2.65)
    assert np.isclose(model_w["config"]["model_options"]["nnsk"]["hopping"]["w"], 0.35)
    
def test_md():
    INPUT_file = "./dptb/tests/data/test_data_nequip/input/input_md.json"
    output = "./dptb/tests/data/test_data_nequip/output"
    init_model = "./dptb/tests/data/test_data_nequip/output/test_push_w/checkpoint/nnsk.iter11.pth"

    train(INPUT=INPUT_file, init_model=init_model, restart=None, train_soc=False,\
          output=output+"/test_md", log_level=5, log_path=output+"/test_md.log")