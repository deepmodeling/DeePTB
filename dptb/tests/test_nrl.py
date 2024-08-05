import pytest
from dptb.nn.build import build_model
from dptb.postprocess.bandstructure.band import Band
import  matplotlib.pyplot  as plt
import numpy as np
from ase.io import read
from dptb.data import AtomicData, AtomicDataDict
from dptb.nn.nnsk import NNSK
import json
import torch
import os
from pathlib import Path
from dptb.entrypoints.nrl2json import nrl2json
from dptb.entrypoints.pth2json import pth2json

from dptb.utils.tools import flatten_dict


rootdir = os.path.join(Path(os.path.abspath(__file__)).parent, "data")


def test_nrl_json_band():
    model = NNSK.from_reference(checkpoint= f"{rootdir}/json_model/Si_nrl.json")

    #set the band structure task
    jdata={   
        "task_options": {
            "task": "band",
            "kline_type":"abacus",
            "kpath":[[0.0000000000,  0.0000000000,   0.0000000000,   2],   
                    [0.5000000000,   0.0000000000,   0.5000000000,   2],               
                    [0.6250000000,   0.2500000000,   0.6250000000,   1],    
                    [0.3750000000,   0.3750000000,   0.7500000000,   2],     
                    [0.0000000000,   0.0000000000,   0.0000000000,   2],    
                    [0.5000000000,   0.5000000000,   0.5000000000,   2],                
                    [0.5000000000,   0.2500000000,   0.7500000000,   2],               
                    [0.5000000000,   0.0000000000,   0.5000000000,   1 ]
                    ],
            "klabels":["G","X","X/U","K","G","L","W","X"],
            "E_fermi":-9.307,
            "emin":-15,
            "emax":10
        }
    }

    stru_data = f"{rootdir}/json_model/silicon.vasp"
    AtomicData_options = {"r_max": 5.0, "oer_max":6.6147151362875}
    kpath_kwargs = jdata["task_options"]
    bcal = Band(model=model, 
                use_gui=True, 
                results_path='./', 
                device=model.device)

    eigenstatus = bcal.get_bands(data=stru_data, 
                   kpath_kwargs=kpath_kwargs)
    
    expected_eigenvalues = np.array([[-6.1741133 ,  5.2992673 ,  5.299269  ,  5.2992706 ,  8.679379  ,
         8.67938   ,  8.679387  ,  9.836669  , 14.15181   , 14.151812  ,
        15.179906  , 15.179909  , 17.065308  , 17.065311  , 17.065315  ,
        23.384512  , 23.384514  , 23.384523  ],
       [-5.5645704 ,  2.1704118 ,  3.4521012 ,  3.4521055 ,  7.330651  ,
         9.427716  , 11.252065  , 11.252069  , 14.348874  , 14.904958  ,
        15.063788  , 15.08024   , 16.522131  , 16.522133  , 20.978777  ,
        20.97878   , 21.235731  , 28.363321  ],
       [-2.554551  , -2.5545506 ,  2.4126623 ,  2.4126637 ,  6.4693484 ,
         6.46935   , 14.620965  , 14.620967  , 14.736008  , 14.736009  ,
        14.747112  , 14.747118  , 15.574924  , 15.574924  , 17.599064  ,
        17.599068  , 38.834724  , 38.83473   ],
       [-2.6305206 , -2.3678906 ,  1.7033997 ,  2.5220068 ,  6.6189265 ,
         7.990758  , 13.693519  , 14.290318  , 14.706135  , 14.793065  ,
        14.83984   , 15.134137  , 15.58144   , 15.826494  , 17.384142  ,
        18.580969  , 35.741035  , 37.842724  ],
       [-2.990186  , -1.8204781 ,  0.89282584,  2.8375692 ,  7.0482116 ,
        10.63623   , 12.305137  , 13.244209  , 14.652864  , 14.841782  ,
        15.511359  , 15.560423  , 16.049604  , 16.561003  , 17.382034  ,
        20.065767  , 28.94485   , 34.8759    ],
       [-2.9901865 , -1.8204774 ,  0.8928198 ,  2.8375793 ,  7.0482135 ,
        10.636231  , 12.305133  , 13.244207  , 14.65287   , 14.8417845 ,
        15.51136   , 15.560425  , 16.049616  , 16.560993  , 17.38204   ,
        20.065779  , 28.944853  , 34.8759    ],
       [-5.47864   ,  1.5369629 ,  2.5553446 ,  4.5224996 ,  8.77259   ,
         9.497431  , 10.579291  , 11.207781  , 14.566749  , 14.716702  ,
        14.79304   , 15.098012  , 17.310905  , 17.856058  , 18.338556  ,
        22.161259  , 23.587708  , 24.457659  ],
       [-6.1741133 ,  5.2992673 ,  5.299269  ,  5.2992706 ,  8.679379  ,
         8.67938   ,  8.679387  ,  9.836669  , 14.15181   , 14.151812  ,
        15.179906  , 15.179909  , 17.065308  , 17.065311  , 17.065315  ,
        23.384512  , 23.384514  , 23.384523  ],
       [-5.7577815 ,  1.7521204 ,  4.532131  ,  4.532138  ,  8.251087  ,
         9.490989  ,  9.490992  , 11.636496  , 14.439946  , 14.439954  ,
        14.857351  , 14.857357  , 16.893194  , 16.8932    , 19.772648  ,
        22.80723   , 22.807241  , 23.869514  ],
       [-4.443464  , -1.6066544 ,  4.0123796 ,  4.012383  ,  7.202861  ,
         9.727551  ,  9.72756   , 13.991438  , 14.478059  , 14.478066  ,
        14.982593  , 14.982594  , 16.354605  , 16.354612  , 18.78879   ,
        23.787397  , 23.7874    , 27.123451  ],
       [-3.8896487 , -1.3633558 ,  1.791838  ,  3.1084414 ,  8.638544  ,
         9.726504  , 11.305393  , 12.739989  , 14.571893  , 14.913423  ,
        15.175632  , 15.675038  , 16.678635  , 17.124842  , 19.235235  ,
        21.592587  , 24.378092  , 28.834328  ],
       [-2.3344338 , -2.334433  ,  1.3823981 ,  1.3824005 , 10.204421  ,
        10.204421  , 12.187885  , 12.187888  , 14.745355  , 14.745364  ,
        15.736717  , 15.736728  , 15.787358  , 15.787364  , 18.632502  ,
        18.632504  , 31.974377  , 31.974384  ],
       [-2.449809  , -2.4498005 ,  1.8228805 ,  1.8228889 ,  8.088991  ,
         8.088991  , 13.572681  , 13.572683  , 14.7421055 , 14.742113  ,
        15.033702  , 15.03371   , 15.825112  , 15.825113  , 18.229067  ,
        18.229078  , 35.388714  , 35.388718  ],
       [-2.554551  , -2.5545506 ,  2.4126623 ,  2.4126637 ,  6.4693484 ,
         6.46935   , 14.620965  , 14.620967  , 14.736008  , 14.736009  ,
        14.747112  , 14.747118  , 15.574924  , 15.574924  , 17.599064  ,
        17.599068  , 38.834724  , 38.83473   ]], dtype=np.float32)

    assert np.allclose(eigenstatus["eigenvalues"], expected_eigenvalues, atol=1e-4)

@pytest.mark.order(3)
def test_nrl_train_freeze():
    model = NNSK.from_reference(checkpoint= f"{rootdir}/json_model/Si_nrl.json")
    model_nfz = NNSK.from_reference(checkpoint= f"{rootdir}/test_sktb/output/test_nrl/checkpoint/nnsk.best.pth")
    model_fz = NNSK.from_reference(checkpoint= f"{rootdir}/test_sktb/output/test_nrlfz/checkpoint/nnsk.best.pth")

    assert torch.any(torch.abs(model_nfz.hopping_param - model.hopping_param ) > 1e-5)
    assert torch.any(torch.abs(model_nfz.overlap_param - model.overlap_param )> 1e-5)
    assert torch.any(torch.abs(model_nfz.onsite_param - model.onsite_param )> 1e-5)

    assert torch.any(torch.abs(model_fz.hopping_param - model.hopping_param ) > 1e-5)
    assert torch.all(torch.abs(model_fz.overlap_param - model.overlap_param ) < 1e-6)
    assert torch.any(torch.abs(model_fz.onsite_param - model.onsite_param ) > 1e-5)

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def compare_dicts(d1, d2):
    return flatten_dict(d1) == flatten_dict(d2)

def test_nrl2json():
    INPUT_file = f"{rootdir}/test_sktb/input/input_nrl.json"
    nrlfile = f"{rootdir}/json_model/Si_spd.par"
    outdir = f"{rootdir}/out"
    nrl2json(INPUT_file, nrlfile, outdir,log_level=5, log_path=outdir+"/test_nrl2json.log")

    with open(f"{outdir}/nrl_ckpt.json",'r') as f:
        nrl1 = json.load(f)
    with open(f"{rootdir}/json_model/Si_nrl.json",'r') as f:
        nrl2 = json.load(f)
    nrl1_flat = flatten_dict(nrl1)
    nrl2_flat = flatten_dict(nrl2)
    for key, val in nrl1_flat.items():
        assert key in nrl2_flat
        if isinstance(val, list):
            if isinstance(val[0], float):
                assert np.allclose(val, nrl2_flat[key], atol=1e-5)
            else:
                assert val == nrl2_flat[key]
        else:
            assert val == nrl2_flat[key]

def test_pth2json():
    init_model = f"{rootdir}/test_sktb/output/test_nrl/checkpoint/nnsk.best.pth"
    outdir = f"{rootdir}/out"
    pth2json(init_model, outdir, log_level=5, log_path=outdir+"/test_pth2json.log")
