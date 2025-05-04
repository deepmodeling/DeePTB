import pytest
import os 
from pathlib import Path
import numpy as np
from dptb.nn.build import build_model
from ase.io import read
from dptb.data import AtomicData, AtomicDataDict
import torch
from dptb.utils.auto_band_config import auto_band_config


rootdir = os.path.join(Path(os.path.abspath(__file__)).parent, "data")

def test_build_esk():
    structure = f'{rootdir}/mos2/struct.vasp'
    bandjdata, common_options = auto_band_config(structure=structure, kpathtype='vasp')
    in_common_options = common_options
    init_model = os.path.join(Path(os.path.abspath(__file__)).parent, '..', 'nn', 'dftb', 'base_poly2.pth')
    model = build_model(checkpoint=init_model)
    model = build_model(checkpoint=init_model, common_options=in_common_options)
