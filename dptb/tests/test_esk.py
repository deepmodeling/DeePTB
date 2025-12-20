import pytest
import os 
from pathlib import Path
import numpy as np
from dptb.nn.build import build_model
from ase.io import read
from dptb.data import AtomicData, AtomicDataDict
import torch
from dptb.utils.auto_band_config import auto_band_config
from dptb.entrypoints.emp_sk import to_empsk
import json

rootdir = os.path.join(Path(os.path.abspath(__file__)).parent, "data")

def test_build_esk():
    common_options ={'basis': {'S': ['s', 'p', 'd'], 'Mo': ['s', 'p', 'd']}}
    init_model = os.path.join(Path(os.path.abspath(__file__)).parent, '..', 'nn', 'dftb', 'base_poly2.pth')
    model = build_model(checkpoint=init_model)
    model = build_model(checkpoint=init_model, common_options=common_options)
    
def test_to_esk_json():
    infile = os.path.join(Path(os.path.abspath(__file__)).parent, 'data', 'esk_orb', 'orb.json')
    out = os.path.join(Path(os.path.abspath(__file__)).parent, 'data', 'esk_orb')
    to_empsk(
    infile,
    output=out, 
    basemodel='poly2',
    soc= None)

    to_empsk(
    infile,
    output=out, 
    basemodel='poly4',
    soc= None)
    
    out_json_model = os.path.join(Path(os.path.abspath(__file__)).parent, 'data', 'esk_orb','sktb.json')
    with open(out_json_model,'r') as f:
        json_dict = json.load(f)

    out_json_model = os.path.join(Path(os.path.abspath(__file__)).parent, 'data', 'esk_orb','sktb.json')
    with open(out_json_model,'r') as f:
        json_dict = json.load(f)
    assert json_dict['model_options']['nnsk']['onsite'].get('method') == "uniform_noref"
    assert json_dict['model_options']['nnsk']['soc'].get('method') is None
    
    to_empsk(
    infile,
    output=out, 
    basemodel='poly2',
    soc= 0.1)

    to_empsk(
    infile,
    output=out, 
    basemodel='poly4',
    soc= 0.1)

    out_json_model = os.path.join(Path(os.path.abspath(__file__)).parent, 'data', 'esk_orb','sktb.json')
    with open(out_json_model,'r') as f:
        json_dict = json.load(f)

    assert json_dict['model_options']['nnsk']['onsite'].get('method') == "uniform_noref"
    assert json_dict['model_options']['nnsk']['soc'].get('method') == "uniform_noref"
    

def test_to_esk_json_with_Shell():
    infile = os.path.join(Path(os.path.abspath(__file__)).parent, 'data', 'esk_orb', 'orb_shell.json')
    out = os.path.join(Path(os.path.abspath(__file__)).parent, 'data', 'esk_orb')
    to_empsk(
    infile,
    output=out, 
    basemodel='poly2',
    soc= None)
    to_empsk(
    infile,
    output=out, 
    basemodel='poly4',
    soc= None)

    out_json_model = os.path.join(Path(os.path.abspath(__file__)).parent, 'data', 'esk_orb','sktb.json')
    with open(out_json_model,'r') as f:
        json_dict = json.load(f)
    assert json_dict['model_options']['nnsk']['onsite'].get('method') == "uniform"
    assert json_dict['model_options']['nnsk']['soc'].get('method') is None

    to_empsk(
    infile,
    output=out, 
    basemodel='poly2',
    soc= 0.1)

    to_empsk(
    infile,
    output=out, 
    basemodel='poly4',
    soc= 0.1)

    out_json_model = os.path.join(Path(os.path.abspath(__file__)).parent, 'data', 'esk_orb','sktb.json')
    with open(out_json_model,'r') as f:
        json_dict = json.load(f)
    assert json_dict['model_options']['nnsk']['onsite'].get('method') == "uniform"
    assert json_dict['model_options']['nnsk']['soc'].get('method') == "uniform"