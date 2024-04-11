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
    AtomicData_options = {"r_max": 5.0, "oer_max":6.6147151362875, "pbc": True}
    kpath_kwargs = jdata["task_options"]
    bcal = Band(model=model, 
                use_gui=True, 
                results_path='./', 
                device=model.device)

    eigenstatus = bcal.get_bands(data=stru_data, 
                   kpath_kwargs=kpath_kwargs, 
                   AtomicData_options=AtomicData_options)
    
    expected_eigenvalues = np.array([[-6.1745434 ,  5.282297  ,  5.282303  ,  5.2823052 ,  8.658317  ,  8.6583185 ,  8.658324  ,  9.862869  , 14.152446  , 14.152451  , 15.180438  , 15.180452  , 16.983887  , 16.983889  , 16.983896  , 23.09491   , 23.094921  , 23.094925  ],
                                     [-5.5601606 ,  2.1920488 ,  3.4229636 ,  3.4229672 ,  7.347074  ,  9.382092  , 11.1772175 , 11.177221  , 14.349099  , 14.924912  , 15.062427  , 15.064081  , 16.540335  , 16.54034   , 20.871534  , 20.871536  , 21.472364  , 28.740482  ],
                                     [-2.556269  , -2.5562677 ,  2.3915231 ,  2.391524  ,  6.4689007 ,  6.468908  , 14.639398  , 14.6394005 , 14.734453  , 14.734456  , 14.747707  , 14.74771   , 15.57567   , 15.575676  , 17.403324  , 17.403334  , 38.39217   , 38.392174  ],
                                     [-2.6333795 , -2.367625  ,  1.6872846 ,  2.5042236 ,  6.6183453 ,  7.9818068 , 13.933364  , 14.267717  , 14.706404  , 14.793142  , 14.841357  , 15.211192  , 15.578381  , 15.838447  , 17.168877  , 18.059359  , 35.321945  , 37.87687   ],
                                     [-2.9967206 , -1.8161079 ,  0.88636655,  2.829976  ,  7.0469265 , 10.600885  , 12.648353  , 13.126463  , 14.653016  , 14.841116  , 15.541919  , 15.576077  , 16.276308  , 16.574654  , 17.213411  , 19.315798  , 28.62305   , 35.468586  ],
                                     [-2.996724  , -1.8161156 ,  0.88636786,  2.8299737 ,  7.046927  , 10.600888  , 12.648361  , 13.126465  , 14.653028  , 14.841116  , 15.541907  , 15.5760765 , 16.276312  , 16.574644  , 17.21341   , 19.315798  , 28.623045  , 35.46858   ],
                                     [-5.471941  ,  1.5238439 ,  2.5368657 ,  4.577535  ,  8.749301  ,  9.402245  , 10.557684  , 11.247256  , 14.576941  , 14.75164   , 14.775435  , 15.122616  , 17.103615  , 17.840292  , 18.390976  , 22.68788   , 23.806395  , 24.265633  ],
                                     [-6.1745434 ,  5.282297  ,  5.282303  ,  5.2823052 ,  8.658317  ,  8.6583185 ,  8.658324  ,  9.862869  , 14.152446  , 14.152451  , 15.180438  , 15.180452  , 16.983887  , 16.983889  , 16.983896  , 23.09491   , 23.094921  , 23.094925  ],
                                     [-5.749872  ,  1.7248219 ,  4.5455103 ,  4.545513  ,  8.227031  ,  9.438793  ,  9.4388    , 11.6675415 , 14.485937  , 14.485939  , 14.894153  , 14.894157  , 16.697474  , 16.697474  , 19.904425  , 23.02558   , 23.025585  , 23.831646  ],
                                     [-4.44458   , -1.6045983 ,  4.0464916 ,  4.046497  ,  7.2234683 ,  9.777258  ,  9.777259  , 14.115966  , 14.4775715 , 14.4775715 , 14.98191   , 14.9819145 , 16.346727  , 16.346727  , 18.716038  , 23.819721  , 23.819735  , 27.016748  ],
                                     [-3.8950639 , -1.3644799 ,  1.8130541 ,  3.112887  ,  8.6044655 ,  9.8463125 , 11.3755455 , 12.709737  , 14.566758  , 14.910749  , 15.183235  , 15.717886  , 16.694214  , 17.240337  , 19.386671  , 21.171314  , 23.601032  , 29.806623  ],
                                     [-2.3356187 , -2.3356178 ,  1.3771206 ,  1.3771234 , 10.240082  , 10.240085  , 12.212795  , 12.212798  , 14.746381  , 14.746386  , 15.778043  , 15.778048  , 15.790003  , 15.790005  , 18.402258  , 18.40226   , 31.99752   , 31.997526  ],
                                     [-2.4508858 , -2.4508843 ,  1.809629  ,  1.809632  ,  8.082377  ,  8.082378  , 13.7137    , 13.713703  , 14.742302  , 14.742307  , 15.081548  , 15.081549  , 15.864478  , 15.864485  , 17.778458  , 17.77847   , 35.317     , 35.317005  ],
                                     [-2.556269  , -2.5562677 ,  2.3915231 ,  2.391524  ,  6.4689007 ,  6.468908  , 14.639398  , 14.6394005 , 14.734453  , 14.734456  , 14.747707  , 14.74771   , 15.57567   , 15.575676  , 17.403324  , 17.403334  , 38.39217   , 38.392174  ]])
    

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
