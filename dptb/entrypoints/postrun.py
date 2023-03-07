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
from dptb.utils.argcheck import normalize, normalize_bandplot, host_normalize
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
        init_model: str,
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
        "init_model":init_model,
        "structure":structure,
        "log_path": log_path,
        "log_level": log_level,
        "use_correction":use_correction
    }

    if all((use_correction, run_sk)):
        log.error(msg="--use-correction and --train_sk should not be set at the same time")
        raise RuntimeError
    
    jdata = j_loader(INPUT)

    jdata = host_normalize(jdata)

    #if run_sk:
    if run_opt["init_model"] is None:
        log.info(msg="model_ckpt is not set in run option, read from input config file.")

        if "init_model" in jdata and "path" in jdata["init_model"] and jdata["init_model"]["path"] is not None:
            run_opt["init_model"] = jdata["init_model"]["path"]
        else:
            log.error(msg="Error! init_model is not set in config file and command line.")
            raise RuntimeError

    task = j_must_have(jdata, "task")

    if  run_opt['structure'] is None:
        log.warning(msg="Warning! structure is not set in run option, read from input config file.")
        structure = j_must_have(jdata, "structure")
        run_opt.update({"structure":structure})

    model_ckpt = run_opt["init_model"]

    if run_opt['use_correction'] is None and jdata.get('use_correction',None) != None:
        use_correction = jdata['use_correction']
        run_opt.update({"use_correction":use_correction})
        log.info(msg="use_correction is not set in run option, read from input config file.")
    
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

    init_type = model_ckpt.split(".")[-1]
    if init_type == "json":
        jdata.update({"init_model": {"path": model_ckpt,"interpolate": False}})

    if run_sk:
        apihost = NNSKHost(checkpoint=model_ckpt, config=jdata)
        apihost.register_plugin(InitSKModel())
        apihost.build()
        apiHrk = NN2HRK(apihost=apihost, mode='nnsk')
    else:
        apihost = DPTBHost(dptbmodel=model_ckpt,use_correction=use_correction)
        apihost.register_plugin(InitDPTBModel())
        apihost.build()
        apiHrk = NN2HRK(apihost=apihost, mode='dptb')    
        
    # one can just add his own function to calculate properties by add a task, and its code to calculate.

    if task=='bandstructure':
        plot_opt = j_must_have(jdata, "bandstructure")
        plot_opt = normalize_bandplot(plot_opt)
        plot_jdata = {"bandstructure":plot_opt}
        # plot_jdata = normalize_bandplot(plot_jdata)
        jdata.update(plot_jdata)
        
        with open(os.path.join(output, "run_config.json"), "w") as fp:
            json.dump(jdata, fp, indent=4)

        bcal = bandcalc(apiHrk, run_opt, plot_jdata)
        bcal.get_bands()
        bcal.band_plot()
        log.info(msg='band calculation successfully completed.')


    