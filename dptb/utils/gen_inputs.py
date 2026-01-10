from dptb.nn.build import build_model
import json
import logging
from dptb.utils.config_sk import TrainFullConfigSK, TestFullConfigSK
from dptb.utils.config_skenv import TrainFullConfigSKEnv, TestFullConfigSKEnv
from dptb.utils.config_e3 import TrainFullConfigE3, TestFullConfigE3
import os
import torch

def gen_inputs(mode, task='train', model=None):
    assert task in ['train', 'test'], 'task should be train or test'
    assert mode in ['sk', 'skenv', 'e3'], 'mode should be sk, skenv or e3'
    
    if task == 'train':
        if mode == 'sk':
            input_dict = TrainFullConfigSK
        elif mode == 'skenv':
            input_dict = TrainFullConfigSKEnv
        else:
            input_dict = TrainFullConfigE3
    else:
        if mode =='sk':
            input_dict = TestFullConfigSK
        elif mode =='skenv':
            input_dict = TestFullConfigSKEnv
        else:
            input_dict = TestFullConfigE3
    
    if model is not None:
        # if model provided, update the input template
        if isinstance(model, str):
            model = build_model(model)
        if model.name == 'nnsk':
            assert mode in ['sk', 'skenv']
            is_overlap = hasattr(model, "overlap_param")
        elif model.name == 'nnenv':
            assert mode == 'e3'
            is_overlap = True
        elif model.name == 'mix':
            assert mode == 'skenv'
            is_overlap = hasattr(model.nnsk, "overlap_param")
        else:
            raise NotImplementedError
        basis = model.basis
        if isinstance(model.dtype, str):
            dtype = model.dtype
        else:
            dtype = model.dtype.__str__().split('.')[-1]

        if model.device == 'cpu' or model.device == torch.device("cpu"):
            dd = "cpu"
        else:
            dd = "cuda"

        common_options = {
            "basis": basis,
            "dtype": dtype,
            "device": dd,
            "overlap": is_overlap,
        }
        input_dict["common_options"].update(common_options)
        input_dict["model_options"].update(model.model_options)
        if is_overlap:
            if "nnsk" in input_dict["model_options"]:
                # for nnsk if there is overlap param, freeze the overlap param in the nnsk model.
                input_dict["model_options"]["nnsk"].update({"freeze": ["overlap"]})

    #with open(os.path.join(outdir,'input_template.json'), 'w') as f:
    #    json.dump(input_dict, f, indent=4)
    return input_dict
    