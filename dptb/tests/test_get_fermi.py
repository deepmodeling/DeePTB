from dptb.postprocess.elec_struc_cal import ElecStruCal

# from dptb.postprocess.bandstructure.band import Band
from dptb.nn.build import build_model
import os
from pathlib import Path

rootdir = os.path.join(Path(os.path.abspath(__file__)).parent, "data")


def test_get_fermi():
    ckpt = f"{rootdir}/test_get_fermi/nnsk.best.pth"
    stru_data = f"{rootdir}/test_get_fermi/PRIMCELL.vasp"

    model = build_model(checkpoint=ckpt)
    AtomicData_options={
            "r_max": 5.50,
            "pbc": True
        }

    AtomicData_options = AtomicData_options
    nel_atom = {"Au":11}

    elec_cal = ElecStruCal(model=model,device='cpu')
    _, efermi =elec_cal.get_fermi_level(data=stru_data, 
                    nel_atom = nel_atom,
                kmesh=[30,30,30],
                AtomicData_options=AtomicData_options)
    
    assert abs(efermi  + 3.25725233554) < 1e-5

