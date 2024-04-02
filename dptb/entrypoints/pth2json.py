from typing import Dict, List, Optional, Any
import ase.io as io
from pathlib import Path
from dptb.nn.build import build_model
import os
import logging, json
log = logging.getLogger(__name__)

def pth2json(
        init_model: str,
        outdir: str,
        log_level: int,
        log_path: Optional[str],
        **kwargs
):
    
    if os.path.exists(outdir):
        log.warning('Warning! the outdir exists, will overwrite the file.')
    else:
        os.makedirs(outdir)

    run_options = {
        "init_model": init_model,
        "log_level": log_level,
        "log_path": log_path,
    }
    
    model = build_model(run_options["init_model"])
    
    if model.name == "nnsk":
        nnsk = model
    elif model.name == "mix":
        nnsk = model.nnsk
        log.warning("The model is a mixed model. The nnsk model is extracted. But the env correction in dptb model can not be transfered to json!")
    else:
        log.error("The model is not a nnsk model.")
        raise ValueError("The model is not a nnsk model.")

    json_dict = nnsk.to_json()

    # dump the json file
    json_file = Path(outdir) / "ckpt.json"
    with open(json_file, "w") as f:
        json.dump(json_dict, f, indent=4)

    log.info(f'pth ckpt {init_model} has been converted to {outdir}/ckpt.json.')
