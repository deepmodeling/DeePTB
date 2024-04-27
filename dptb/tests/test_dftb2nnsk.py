import pytest
import os
from pathlib import Path
from dptb.nn.dftb2nnsk import DFTB2NNSK
from dptb.nn.nnsk import NNSK

rootdir = os.path.join(Path(os.path.abspath(__file__)).parent, "data/")

class TestDFTB2NNSK:
    rootdir = f"{rootdir}"
    def test_init(self):
        dftb2nnsk = DFTB2NNSK(
            basis={"B":["2s"], "N": ["2s"]}, 
            skdata=os.path.join(rootdir, "slakos"),
            rs=6.0,
            rc=6.0,
            w=1.0,
            functype="powerlaw"
            )

    def test_optimize(self):
        dftb2nnsk = DFTB2NNSK(
            basis={"B":["2s"], "N": ["2s"]}, 
            skdata=os.path.join(rootdir, "slakos"),
            rs=6.0,
            rc=6.0,
            w=1.0,
            functype="powerlaw"
            )
        dftb2nnsk.optimize(nstep=10)




    def test_tonnsk(self):
        dftb2nnsk = DFTB2NNSK(
            basis={"B":["2s"], "N": ["2s"]}, 
            skdata=os.path.join(rootdir, "slakos"),
            rs=6.0,
            rc=6.0,
            w=1.0,
            functype="powerlaw"
            )
        nnsk = dftb2nnsk.to_nnsk()

        assert isinstance(nnsk, NNSK)
        

    def test_tojson(self):
        dftb2nnsk = DFTB2NNSK(
            basis={"B":["2s"], "N": ["2s"]}, 
            skdata=os.path.join(rootdir, "slakos"),
            rs=6.0,
            rc=6.0,
            w=1.0,
            functype="powerlaw"
            )
        jdata = dftb2nnsk.to_json()
        assert isinstance(jdata, dict)
        