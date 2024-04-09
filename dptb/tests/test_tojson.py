import pytest
from dptb.nn.build import build_model
from dptb.postprocess.bandstructure.band import Band
import os
from pathlib import Path
import numpy as np
import json

rootdir = os.path.join(Path(os.path.abspath(__file__)).parent, "data")

class TestV1Jsonuniform:
    run_opt = {
            "init_model": f"{rootdir}/json_model/AlAs_v1_nnsk_b5.200_c4.200_w0.200.json",
            "restart": None,
            "freeze": False,
            "train_soc": False,
            "log_path": None,
            "log_level": None
        }

    model_option = {
        "nnsk": {
            "onsite": {"method": "uniform"},
            "hopping": {"method": "powerlaw", "rs":4.2, "w": 0.2},
            "soc":{},
            "push": False,
            "freeze": False
        }
    }
    common_options = {
    "basis": {
        "As": ["4s","4p","d*"],
        "Al": ["3s","3p","d*"]
    },
    "device": "cpu",
    "dtype": "float32",
    "overlap": False
    }

    model = build_model(run_opt["init_model"], model_option, common_options)
    v1json = model.to_json(version=1)
    v2json = model.to_json()
    with open(run_opt['init_model'], 'r') as f:
        ckpt = json.load(f)
        
    def test_hopping(self):
        assert 'hopping' in self.v1json
        assert len(self.v1json['hopping']) == len(self.ckpt['hopping'])
        for key,val in self.ckpt['hopping'].items():
            assert key in self.v1json['hopping']
            val_tmp = np.array(val)
            val_tmp[0] *= 13.605662285137 * 2
            assert (np.isclose(val_tmp, self.v1json['hopping'][key], atol=1e-6)).all()
    
    def test_onsite(self):
        assert 'onsite' in self.v1json
        assert len(self.v1json['onsite']) == len(self.ckpt['onsite'])
        for key,val in self.ckpt['onsite'].items():
            assert key in self.v1json['onsite']
            val_tmp = np.array(val)
            val_tmp[0] *= 13.605662285137 * 2
            assert (np.isclose(val_tmp, self.v1json['onsite'][key], atol=1e-6)).all()

    def test_hopping_v2(self):
        assert "common_options" in self.v2json
        assert "model_options" in self.v2json
        assert 'model_params' in self.v2json

        assert 'hopping' in self.v2json['model_params']

        assert len(self.v2json['model_params']['hopping']) == len(self.ckpt['hopping'])
        for key,val in self.ckpt['hopping'].items():
            assert key in self.v2json['model_params']['hopping']
            val_tmp = np.array(val)
            val_tmp[0] *= 13.605662285137 * 2
            assert (np.isclose(val_tmp, self.v2json['model_params']['hopping'][key], atol=1e-6)).all()

    def test_onsite_v2(self):

        assert 'model_params' in self.v2json
        assert 'onsite' in self.v2json['model_params']

        assert len(self.v2json['model_params']['onsite']) == len(self.ckpt['onsite'])
        for key,val in self.ckpt['onsite'].items():
            val_tmp = np.array(val)
            val_tmp[0] *= 13.605662285137 * 2
            assert key in self.v2json['model_params']['onsite']
            assert (np.isclose(val_tmp, self.v2json['model_params']['onsite'][key], atol=1e-6)).all()



class Test2Jsonstrain:
    run_opt = {
            "init_model": f"{rootdir}/json_model/Si_v1_nnsk_b2.600_c2.600_w0.300.json",
            "restart": None,
            "freeze": False,
            "train_soc": False,
            "log_path": None,
            "log_level": None
        }

    model_option = {
        "nnsk": {
        "onsite": {"method": "strain","rs":6, "w": 0.1},
        "hopping": {"method": "powerlaw", "rs":2.6, "w": 0.3},
        "soc":{},
        "push": False,
        "freeze": False
        }
    }
    common_options = {
    "basis": {
    "Si": ["3s","3p","d*"]
    },
    "device": "cpu",
    "dtype": "float32",
    "overlap": False
    }

    model = build_model(run_opt["init_model"], model_option, common_options)
    v1json = model.to_json(version=1)
    v2json = model.to_json()

    with open(run_opt['init_model'], 'r') as f:
        ckpt = json.load(f)
    
    def test_hopping_v1(self):
        assert 'hopping' in self.v1json
        assert len(self.v1json['hopping']) == len(self.ckpt['hopping'])
        for key,val in self.ckpt['hopping'].items():
            assert key in self.v1json['hopping']
            val_tmp = np.array(val)
            val_tmp[0] *= 13.605662285137 * 2
            assert (np.isclose(val_tmp, self.v2json['model_params']['hopping'][key], atol=1e-6)).all()
    
    def test_onsite_v1(self):
        assert 'onsite' in self.v1json
        assert len(self.v1json['onsite']) == len(self.ckpt['onsite'])
        for key,val in self.ckpt['onsite'].items():
            assert key in self.v1json['onsite']
            val_tmp = np.array(val)
            val_tmp[0] *= 13.605662285137 * 2
            assert (np.isclose(val_tmp, self.v1json['onsite'][key], atol=1e-6)).all()

    def test_hopping_v2(self):
        assert "common_options" in self.v2json
        assert "model_options" in self.v2json
        assert 'model_params' in self.v2json

        assert 'hopping' in self.v2json['model_params']

        assert len(self.v2json['model_params']['hopping']) == len(self.ckpt['hopping'])
        for key,val in self.ckpt['hopping'].items():
            assert key in self.v2json['model_params']['hopping']
            val_tmp = np.array(val)
            v2tmp = np.array(self.v2json['model_params']['hopping'][key])
            val_tmp[0] *= 13.605662285137 * 2
            assert (np.isclose(val_tmp, self.v2json['model_params']['hopping'][key], atol=1e-6)).all()


    def test_onsite_v2(self):

        assert 'model_params' in self.v2json
        assert 'onsite' in self.v2json['model_params']

        assert len(self.v2json['model_params']['onsite']) == len(self.ckpt['onsite'])
        for key,val in self.ckpt['onsite'].items():
            val_tmp = np.array(val)
            val_tmp[0] *= 13.605662285137 * 2
            assert key in self.v2json['model_params']['onsite']
            assert (np.isclose(val_tmp, self.v2json['model_params']['onsite'][key], atol=1e-6)).all()



    