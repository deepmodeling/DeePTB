import logging
import json
import os
import struct
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dptb.plugins.train_logger import Logger
from dptb.plugins.init_nnsk import InitSKModel
from dptb.plugins.init_dptb import InitDPTBModel
from dptb.utils.argcheck import normalize
from dptb.utils.tools import j_loader
from dptb.utils.loggers import set_log_handles
from dptb.utils.tools import j_must_have
from dptb.nnops.apihost import NNSKHost, DPTBHost
from dptb.nnops.NN2HRK import NN2HRK
from ase.io import read,write
from dptb.postprocess.bandstructure.band import bandcalc

__all__ = ["run"]

log = logging.getLogger(__name__)

def postrun(
        INPUT: str,
        model_ckpt: str,
        output: str,
        run_sk: bool,
        structure: str,
        log_level: int,
        log_path: Optional[str],
        use_correction: Optional[str],
        **kwargs
    ):
    
    run_opt = {
        "run_sk": run_sk,
        "model_ckpt":model_ckpt,
        "structure":structure,
        "log_path": log_path,
        "log_level": log_level,
        "use_correction":use_correction
    }

    # output folder.
    if output:
        Path(output).parent.mkdir(exist_ok=True, parents=True)
        Path(output).mkdir(exist_ok=True, parents=True)
        results_path = os.path.join(str(output), "results")
        Path(results_path).mkdir(exist_ok=True, parents=True)
        if not log_path:
            log_path = os.path.join(str(output), "log/log.txt")
        Path(log_path).parent.mkdir(exist_ok=True, parents=True)

        run_opt.update({
                        "output": str(Path(output).absolute()),
                        "results_path": str(Path(results_path).absolute()),
                        "log_path": str(Path(log_path).absolute())
                        })
    
    set_log_handles(log_level, Path(log_path) if log_path else None)
    jdata = j_loader(INPUT)
    # jdata = normalize(jdata)
    
    task = j_must_have(jdata, "task")

    if  run_opt['structure'] == None:
        log.info(msg="structure is not set in run option, read from input config file.")
        structure = j_must_have(jdata, "structure")
        run_opt.update({"structure":structure})

    if run_opt['model_ckpt'] == None:
        log.info(msg="model_ckpt is not set in run option, read from input config file.")
        model_ckpt = j_must_have(jdata, "model_ckpt")
    
    if run_opt['use_correction'] == None and jdata.get('use_correction',None) != None:
        use_correction = jdata['use_correction']
        run_opt.update({"use_correction":use_correction})
        log.info(msg="use_correction is set in run option, read from input config file.")


    if run_sk:
        apihost = NNSKHost(checkpoint=model_ckpt)
        apihost.register_plugin(InitSKModel())
        apihost.build()
        apiHrk = NN2HRK(apihost=apihost, mode='nnsk')
    else:
        apihost = DPTBHost(dptbmodel=model_ckpt,use_correction=use_correction)
        apihost.register_plugin(InitDPTBModel())
        apihost.build()
        apiHrk = NN2HRK(apihost=apihost, mode='dptb')    
        
    # one can just add his own function to calculate properties by add a task, and its code to calculate.

    if task=='eigenvalues':
        bcal = bandcalc(apiHrk, run_opt, jdata)
        bcal.get_bands()
        bcal.band_plot()
        log.info(msg='band calculation successfully completed.')



    