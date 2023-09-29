import logging
import json
import os
import struct
import time
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
from dptb.plugins.train_logger import Logger
from dptb.plugins.init_nnsk import InitSKModel
from dptb.plugins.init_dptb import InitDPTBModel
from dptb.utils.argcheck import normalize, normalize_run
from dptb.utils.tools import j_loader
from dptb.utils.loggers import set_log_handles
from dptb.utils.tools import j_must_have
from dptb.utils.constants import dtype_dict
from dptb.nnops.apihost import NNSKHost, DPTBHost
from dptb.nnops.NN2HRK import NN2HRK
from ase.io import read,write
from dptb.postprocess.bandstructure.band import bandcalc
from dptb.postprocess.bandstructure.dos import doscalc, pdoscalc
from dptb.postprocess.bandstructure.fermisurface import fs2dcalc, fs3dcalc
from dptb.postprocess.bandstructure.ifermi_api import ifermiapi, ifermi_installed, pymatgen_installed
from dptb.postprocess.write_skparam import WriteNNSKParam
from dptb.postprocess.NEGF import NEGF

__all__ = ["run"]

log = logging.getLogger(__name__)

def run(
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
    jdata = normalize_run(jdata)

    if all((jdata["init_model"]["path"], run_opt["init_model"])):
        raise RuntimeError(
            "init-model in config and command line is in conflict, turn off one of then to avoid this error !"
        )
    
    if run_opt["init_model"] is None:
        log.info(msg="model_ckpt is not set in command line, read from input config file.")

        if run_sk:
            if jdata["init_model"]["path"] is not None:
                run_opt["init_model"] = jdata["init_model"]
            else:
                log.error(msg="Error! init_model is not set in config file and command line.")
                raise RuntimeError
            if isinstance(run_opt["init_model"]["path"], list):
                if len(run_opt["init_model"]["path"])==0:
                    log.error("Error! list mode init_model in config file cannot be empty!")
                    raise RuntimeError
        else:
            if jdata["init_model"]["path"] is not None:
                run_opt["init_model"] = jdata["init_model"]
            else:
                log.error(msg="Error! init_model is not set in config file and command line.")
                raise RuntimeError
            if isinstance(run_opt["init_model"]["path"], list):
                raise RuntimeError(
                "loading lists of checkpoints is only supported in init_nnsk!"
            )
        if isinstance(run_opt["init_model"]["path"], list):
            if len(run_opt["init_model"]["path"]) == 1:
                run_opt["init_model"]["path"] = run_opt["init_model"]["path"][0]
    else:
        path = run_opt["init_model"]
        run_opt["init_model"] = jdata["init_model"]
        run_opt["init_model"]["path"] = path

    
    task_options = j_must_have(jdata, "task_options")
    task = task_options["task"]
    model_ckpt = run_opt["init_model"]["path"]
    # init_type = model_ckpt.split(".")[-1]
    # if init_type not in ["json", "pth"]:
    #     log.error(msg="Error! the model file should be a json or pth file.")
    #     raise RuntimeError
    
    # if init_type == "json":
    #     jdata = host_normalize(jdata)
    #     if run_sk:
    #         jdata.update({"init_model": {"path": model_ckpt,"interpolate": False}})
    #     else:
    #         jdata.update({"init_model": model_ckpt})

    if  run_opt['structure'] is None:
        log.warning(msg="Warning! structure is not set in run option, read from input config file.")
        structure = j_must_have(jdata, "structure")
        run_opt.update({"structure":structure})

        print(run_opt["structure"])

    if not run_sk:
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

    # if jdata.get("common_options", None):
    #     # in this case jdata must have common options
    #     str_dtype = jdata["common_options"]["dtype"]
    #     jdata["common_options"]["dtype"] = dtype_dict[jdata["common_options"]["dtype"]]

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

    if task=='band':
        # TODO: add argcheck for bandstructure, with different options. see, kline_mode: ase, vasp, abacus, etc. 
        bcal = bandcalc(apiHrk, run_opt, task_options)
        bcal.get_bands()
        bcal.band_plot()
        log.info(msg='band calculation successfully completed.')

    if task=='dos':
        bcal = doscalc(apiHrk, run_opt, task_options)
        bcal.get_dos()
        bcal.dos_plot()
        log.info(msg='dos calculation successfully completed.')

    if task=='pdos':
        bcal = pdoscalc(apiHrk, run_opt, task_options)
        bcal.get_pdos()
        bcal.pdos_plot()
        log.info(msg='pdos calculation successfully completed.')
    
    if task=='FS2D':
        fs2dcal = fs2dcalc(apiHrk, run_opt, task_options)
        fs2dcal.get_fs()
        fs2dcal.fs2d_plot()
        log.info(msg='2dFS calculation successfully completed.')
    
    if task == 'FS3D':
        fs3dcal = fs3dcalc(apiHrk, run_opt, task_options)
        fs3dcal.get_fs()
        fs3dcal.fs_plot()
        log.info(msg='3dFS calculation successfully completed.')
    
    if task == 'ifermi':
        if not(ifermi_installed and pymatgen_installed):
            log.error(msg="ifermi and pymatgen are required to perform ifermi calculation !")
            raise RuntimeError

        ifermi = ifermiapi(apiHrk, run_opt, task_options)
        bs = ifermi.get_band_structure()
        fs = ifermi.get_fs(bs)
        ifermi.fs_plot(fs)
        log.info(msg='Ifermi calculation successfully completed.')
    if task == 'write_sk':
        if not run_sk:
            raise RuntimeError("write_sk can only perform on nnsk model !")
        write_sk = WriteNNSKParam(apiHrk, run_opt, task_options)
        write_sk.write()
        log.info(msg='write_sk calculation successfully completed.')

    if task == 'negf':
        negf = NEGF(apiHrk, run_opt, task_options)
        negf.compute()
        log.info(msg='NEGF calculation successfully completed.')


    if output:
        with open(os.path.join(output, "run_config.json"), "w") as fp:
            if jdata.get("common_options", None):
                jdata["common_options"]["dtype"] = str_dtype
            json.dump(jdata, fp, indent=4)
