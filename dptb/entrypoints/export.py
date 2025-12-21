import json
import torch
import os
import argparse
import logging
from dptb.nn.build import build_model
from dptb.utils.tools import j_loader, j_must_have
from dptb.postprocess.interfaces import ToWannier90, ToPythTB

log = logging.getLogger(__name__)

def export(INPUT: str, init_model: str = None, structure: str = None, output: str = "./", format: str = "wannier90", **kwargs):
    """
    Export the DeePTB model to external formats.
    """
    
    # 1. Load config
    if INPUT and INPUT.endswith(".json"):
        jdata = j_loader(INPUT)
    else:
        # Fallback if no json provided (though usually required)
        # If user provides structure + model but no config? 
        # DeePTB typically needs config for model architecture parameters stored in it (or in checkpoint?)
        # build_model usually needs common_options.
        # If init_model is a .pth, it might contain config?
        # dptb.nn.build.build_model(checkpoint=...) handles loading from checkpoint which might have config.
        jdata = {}

    # Determine device
    device_str = jdata.get("device", "cpu")
    device = torch.device(device_str)
    
    # Load model
    ckpt_path = init_model if init_model else jdata.get("model_ckpt")
    if not ckpt_path:
        log.error("Model checkpoint not found. Provide via -i or in config json via 'model_ckpt'.")
        return
    
    # Common options for build_model
    # We might need to extract them from jdata or checkpoint
    # build_model(checkpoint=...) attempts to load config from checkpoint if available?
    # run.py passes in_common_options loaded from json.
    
    in_common_options = {}
    if jdata.get("device"): in_common_options["device"] = jdata["device"]
    if jdata.get("dtype"): in_common_options["dtype"] = jdata["dtype"]
    
    try:
        model = build_model(checkpoint=ckpt_path, common_options=in_common_options)
        model.to(device)
        model.eval()
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        return
    
    # Determine structure input
    struct_path = structure if structure else jdata.get("structure")
    if not struct_path:
        log.error("Structure file not specified. Provide via -stu or in config json.")
        return
        
    # Ensure output directory exists
    if not os.path.exists(output):
        os.makedirs(output)
        
    log.info(f"Loaded model from {ckpt_path}")
    log.info(f"Processing structure {struct_path}")
    log.info(f"Exporting to format: {format}")
    
    # 2. Dispatch to Interface
    
    if format.lower() in ["wannier90", "w90", "tb2j", "wannierberri"]:
        exporter = ToWannier90(model, device=device)
        
        prefix = os.path.splitext(os.path.basename(struct_path))[0]
        hr_file = os.path.join(output, f"{prefix}_hr.dat")
        win_file = os.path.join(output, f"{prefix}.win")
        cen_file = os.path.join(output, f"{prefix}_centres.xyz")
        
        ad_options = jdata.get("AtomicData_options", {})
        e_fermi = jdata.get("e_fermi", 0.0)
        
        try:
            exporter.write_hr(struct_path, filename=hr_file, AtomicData_options=ad_options, e_fermi=e_fermi)
            exporter.write_win(struct_path, filename=win_file, e_fermi=e_fermi)
            exporter.write_centres(struct_path, filename=cen_file)
        except Exception as e:
            log.error(f"Error during export: {e}")
        
    elif format.lower() in ["pythtb"]:
        try:
            exporter = ToPythTB(model, device=device)
        except ImportError:
            return # Logged in __init__
            
        ad_options = jdata.get("AtomicData_options", {})
        e_fermi = jdata.get("e_fermi", 0.0)
        
        try:
            tb_model = exporter.get_model(struct_path, AtomicData_options=ad_options, e_fermi=e_fermi)
            
            prefix = os.path.splitext(os.path.basename(struct_path))[0]
            pkl_file = os.path.join(output, f"{prefix}_pythtb.pkl")
            
            import pickle
            with open(pkl_file, 'wb') as f:
                pickle.dump(tb_model, f)
                
            log.info(f"Saved PythTB model object to {pkl_file}")
        except Exception as e:
            log.error(f"Error during PythTB export: {e}")
        
    else:
        log.error(f"Unknown format: {format}")
