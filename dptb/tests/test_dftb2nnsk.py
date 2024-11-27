import pytest
import os
from pathlib import Path
from dptb.nn.dftb2nnsk import DFTB2NNSK
from dptb.nn.nnsk import NNSK
from dptb.entrypoints.collectskf import skf2nnsk


rootdir = os.path.join(Path(os.path.abspath(__file__)).parent, "data")
exampledir = os.path.join(Path(os.path.abspath(__file__)).parent, "..", "..", "examples")

class TestDFTB2NNSK:
    rootdir = f"{rootdir}"
    train_ops = {
        'nstep':10,
        'dis_freq':1,
        'nsample':256,
        'save_freq':1,
        'max_elmt_batch':4,
        "optimizer": {
            "lr": 0.05,
            "type": "RMSprop"
        },
        "lr_scheduler": {
            "type": "cos",
            "T_max": 4
        }
    }
    
    dftb2nnsk = DFTB2NNSK(
            basis={"B":["2s"], "N": ["2s"]}, 
            skdata=os.path.join(rootdir, "slakos"),
            train_options=train_ops,
            rs=6.0,
            w=1.0,
            method="powerlaw"
            )

    def test_optimize(self):
        self.dftb2nnsk.optimize(r_min=1,r_max=6,nstep=10)

    def test_tonnsk(self):
        nnsk = self.dftb2nnsk.to_nnsk()
        assert isinstance(nnsk, NNSK)
        
    def test_tojson(self):
        jdata = self.dftb2nnsk.to_json()
        assert isinstance(jdata, dict)

def test_skf2nnsk_run():
    input_file = os.path.join(exampledir, "skf2nnsk", "input_test.json")
    skf2nnsk(INPUT = input_file,
        init_model = None,
        output = os.path.join(rootdir, "..","test_temp"),
        log_level = 2
    )