import pytest
import torch
from dptb.nn.nnsk import NNSK
from dptb.nn.deeptb import NNENV, MIX
from dptb.utils.tools import j_must_have
from dptb.nn.build import build_model
import os
from pathlib import Path
from dptb.data.transforms import OrbitalMapper


rootdir = os.path.join(Path(os.path.abspath(__file__)).parent, "data/out")


def test_build_nnsk_from_scratch():
    run_options = {
        "init_model": None,
        "restart": None,
        "train_soc": False,
        "log_path": f"{rootdir}/log.txt",
        "log_level": "INFO"
    }
    model_options = {
        "nnsk": {
            "onsite": {
                "method": "strain",
                "rs": 2.5,
                "w": 0.3
            },
            "hopping": {
                "method": "powerlaw",
                "rs": 2.6,
                "w": 0.35
            },
            "freeze": False,
            "std": 0.01,
            "push": None,
        }
    }
    common_options = {
        "basis": {
            "Si": ["3s","3p"]
        },
        "device": "cpu",
        "dtype": "float32",
        "overlap": False,
        "seed": 3982377700
    }
    model = build_model(None, model_options, common_options)

    assert isinstance(model, NNSK)
    assert model.device == "cpu"
    assert model.dtype ==  getattr(torch, common_options["dtype"])
    assert model.name == "nnsk"
    assert model.transform == True
    # OrbitalMapper(basis=common_options["basis"], method="sktb", device=common_options["device"])
    assert hasattr(model, "idp_sk")
    assert isinstance(model.idp_sk, OrbitalMapper)
    assert model.idp_sk.basis == common_options["basis"]
    assert model.idp_sk.method == "sktb"
    
def test_build_model_MIX_from_scratch():
    run_options = {
        "init_model": None,
        "restart": None,
        "train_soc": False,
        "log_path": f"{rootdir}/log.txt",
        "log_level": "INFO"
    }
    model_options = {
        "nnsk": {
            "onsite": {
                "method": "strain",
                "rs": 2.5,
                "w": 0.3
            },
            "hopping": {
                "method": "powerlaw",
                "rs": 2.6,
                "w": 0.35
            },
            "freeze": False,
            "std": 0.01,
            "push": None,
        },
        "embedding": {
            "method": "se2",
            "rs": 2.5,
            "rc": 5.0,
            "radial_net": {
                "neurons": [10,20,30],
                "activation": "tanh",
                "if_batch_normalized": False
            },
            "n_axis": None
        },
        "prediction": {
            "method": "sktb",
            "neurons": [16,16,16],
            "activation": "tanh",
            "if_batch_normalized": False
        }
    }
    common_options = {
        "basis": {
            "Si": ["3s","3p"]
        },
        "device": "cpu",
        "dtype": "float32",
        "overlap": False,
        "seed": 3982377700
    }

    model = build_model(None, model_options, common_options)

    assert isinstance(model, MIX)
    assert model.name == "mix"
    assert model.nnenv.method == 'sktb'
    assert model.nnenv.name == 'nnenv'
    assert model.nnsk.name == 'nnsk'

    assert model.nnsk.transform == False
    assert model.nnenv.transform == False

def test_build_dftbsk_from_scratch():
    skdatapath = f"{rootdir}/../../../../examples/hBN_dftb/slakos"
    common_options = {
        "basis": {
            "B": ["2s", "2p"],
            "N": ["2s", "2p"]
        },
        "device": "cpu",
        "dtype": "float32",
        "overlap": True,
        "seed": 3982377700
    }
    model_options = {
    "dftbsk": {
            "skdata": skdatapath
        }
    }
    model = build_model(None, model_options, common_options)
    assert model.name == 'dftbsk'

def test_build_model_MIX_dftbsk_from_scratch():
    skdatapath = f"{rootdir}/../../../../examples/hBN_dftb/slakos"
    common_options = {
        "basis": {
            "B": ["2s", "2p"],
            "N": ["2s", "2p"]
        },
        "device": "cpu",
        "dtype": "float32",
        "overlap": True,
        "seed": 3982377700
    }
    model_options = {
    "dftbsk": {
            "skdata": skdatapath
        },
        "embedding": {
            "method": "se2",
            "rs": 2.5,
            "rc": 5.0,
            "radial_net": {
                "neurons": [10,20,30],
                "activation": "tanh",
                "if_batch_normalized": False
            },
            "n_axis": None
        },
        "prediction": {
            "method": "sktb",
            "neurons": [16,16,16],
            "activation": "tanh",
            "if_batch_normalized": False
        }
    }
    model = build_model(None, model_options, common_options)
    assert model.name == 'mix'
    assert model.nnenv.method == 'sktb'
    assert model.nnenv.name == 'nnenv'
    assert hasattr(model, "dftbsk")
    assert model.dftbsk.name == 'dftbsk'
    

def test_build_model_failure():
    run_options = {
        "init_model": None,
        "restart": None,
        "train_soc": False,
        "log_path": f"{rootdir}/log.txt",
        "log_level": "INFO"
    }
    model_options = {}
    common_options = {}

    with pytest.raises(ValueError) as excinfo:
        build_model(None, model_options, common_options)
    assert "You need to provide model_options and common_options" in str(excinfo.value)
    
    common_options = {"basis": {"Si": ["3s", "3p"]}}
    
    model_options = {"embedding":False, "prediction":False, "nnsk":True, "dftbsk":True}
    with pytest.raises(AssertionError) as excinfo:
        build_model(None, model_options, common_options)
    assert "There should only be one of the dftbsk and nnsk in model_options." in str(excinfo.value)

    # T F T
    model_options = {"embedding":True, "prediction":False, "nnsk":True}
    with pytest.raises(ValueError) as excinfo:
        build_model(None, model_options, common_options)
    assert "Model_options are not set correctly!" in str(excinfo.value)
    
    # F T T
    model_options = {"embedding":False,"prediction":True, "nnsk":True}
    with pytest.raises(ValueError) as excinfo:
        build_model(None, model_options, common_options)
    assert "Model_options are not set correctly!" in str(excinfo.value)

    # F T F 
    model_options = {"embedding":False,"prediction":True, "nnsk":False}
    with pytest.raises(ValueError) as excinfo:
        build_model(None, model_options, common_options)
    assert "Model_options are not set correctly!" in str(excinfo.value)

    # T F F
    model_options = {"embedding":True,"prediction":False, "nnsk":False}
    with pytest.raises(ValueError) as excinfo:
        build_model(None, model_options, common_options)
    assert "Model_options are not set correctly!" in str(excinfo.value)

    # F F F
    model_options = {"embedding":False,"prediction":False, "nnsk":False}
    with pytest.raises(ValueError) as excinfo:
        build_model(None, model_options, common_options)
    assert "Model_options are not set correctly!" in str(excinfo.value)


    model_options = {"embedding":{"method":"se2"},"prediction":{"method":"e3tb"}, "nnsk":True}
    with pytest.raises(ValueError) as excinfo:
        build_model(None, model_options, common_options)
    assert "The prediction method must be sktb for mix mode." in str(excinfo.value)

    model_options = {"embedding":{"method":"e3"},"prediction":{"method":"sktb"}, "nnsk":True}
    with pytest.raises(ValueError) as excinfo:
        build_model(None, model_options, common_options)
    assert "The embedding method must be se2 for mix mode." in str(excinfo.value)

    model_options = {"embedding":{"method":"e3"},"prediction":{"method":"sktb"}, "nnsk":False}
    with pytest.raises(ValueError) as excinfo:
        build_model(None, model_options, common_options)
    assert "The embedding method must be se2 for sktb prediction in deeptb mode." in str(excinfo.value)

    model_options = {"embedding":{"method":"se2"},"prediction":{"method":"e3tb"}, "nnsk":False}
    with pytest.raises(ValueError) as excinfo:
        build_model(None, model_options, common_options)
    assert "The embedding method can not be se2 for e3tb prediction in deeptb mode" in str(excinfo.value)



#TODO: add test for dptb-e3tb from scratch
#TODO: add test for all the cases from checkpoint, restart and init_model