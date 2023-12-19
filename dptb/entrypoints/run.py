import logging
import json
import os
import time
import torch
from pathlib import Path
from typing import Optional
from dptb.plugins.train_logger import Logger
from dptb.utils.argcheck import normalize_run
from dptb.utils.tools import j_loader
from dptb.utils.loggers import set_log_handles
from dptb.utils.tools import j_must_have
from dptb.nn.build import build_model
from dptb.postprocess.bandstructure.band import bandcalc
from dptb.postprocess.bandstructure.dos import doscalc, pdoscalc
from dptb.postprocess.bandstructure.fermisurface import fs2dcalc, fs3dcalc
from dptb.postprocess.bandstructure.ifermi_api import ifermiapi, ifermi_installed, pymatgen_installed
from dptb.postprocess.write_skparam import WriteNNSKParam
from dptb.postprocess.NEGF import NEGF
from dptb.postprocess.tbtrans_init import TBTransInputSet,sisl_installed

__all__ = ["run"]

log = logging.getLogger(__name__)

def run(
        INPUT: str,
        init_model: str,
        output: str,
        structure: str,
        log_level: int,
        log_path: Optional[str],
        **kwargs
    ):
    
    run_opt = {
        "init_model":init_model,
        "structure":structure,
        "log_path": log_path,
        "log_level": log_level,
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
    
    jdata = j_loader(INPUT)
    jdata = normalize_run(jdata)

    set_log_handles(log_level, Path(log_path) if log_path else None)

    f = torch.load(run_opt["init_model"])
    jdata["model_options"] = f["config"]["model_options"]
    del f

    
    task_options = j_must_have(jdata, "task_options")
    task = task_options["task"]
    use_gui = jdata.get("use_gui", False)
    task_options.update({"use_gui": use_gui})

    if  run_opt['structure'] is None:
        log.warning(msg="Warning! structure is not set in run option, read from input config file.")
        structure = j_must_have(jdata, "structure")
        run_opt.update({"structure":structure})
    else:
        structure = run_opt["structure"]

    print(run_opt["structure"])

    if jdata.get("common_options", None):
        # in this case jdata must have common options
        str_dtype = jdata["common_options"]["dtype"]
    #     jdata["common_options"]["dtype"] = dtype_dict[jdata["common_options"]["dtype"]]

    model = build_model(run_options=run_opt, model_options=jdata["model_options"], common_options=jdata["common_options"])
    
    # one can just add his own function to calculate properties by add a task, and its code to calculate.
    if task=='band':
        # TODO: add argcheck for bandstructure, with different options. see, kline_mode: ase, vasp, abacus, etc. 
        bcal = bandcalc(model, structure, task_options)
        bcal.get_bands()
        bcal.band_plot()
        log.info(msg='band calculation successfully completed.')

    if task=='dos':
        bcal = doscalc(model, structure, task_options)
        bcal.get_dos()
        bcal.dos_plot()
        log.info(msg='dos calculation successfully completed.')

    if task=='pdos':
        bcal = pdoscalc(model, structure, task_options)
        bcal.get_pdos()
        bcal.pdos_plot()
        log.info(msg='pdos calculation successfully completed.')
    
    if task=='FS2D':
        fs2dcal = fs2dcalc(model, structure, task_options)
        fs2dcal.get_fs()
        fs2dcal.fs2d_plot()
        log.info(msg='2dFS calculation successfully completed.')
    
    if task == 'FS3D':
        fs3dcal = fs3dcalc(model, structure, task_options)
        fs3dcal.get_fs()
        fs3dcal.fs_plot()
        log.info(msg='3dFS calculation successfully completed.')
    
    if task == 'ifermi':
        if not(ifermi_installed and pymatgen_installed):
            log.error(msg="ifermi and pymatgen are required to perform ifermi calculation !")
            raise RuntimeError

        ifermi = ifermiapi(model, structure, task_options)
        bs = ifermi.get_band_structure()
        fs = ifermi.get_fs(bs)
        ifermi.fs_plot(fs)
        log.info(msg='Ifermi calculation successfully completed.')
    if task == 'write_sk':
        if not jdata["model_options"].keys()[0] == "nnsk" or len(jdata["model_options"].keys()) > 1:
            raise RuntimeError("write_sk can only perform on nnsk model !")
        write_sk = WriteNNSKParam(model, structure, task_options)
        write_sk.write()
        log.info(msg='write_sk calculation successfully completed.')

    if task == 'negf':
        negf = NEGF(model, structure, task_options)
        negf.compute()
        log.info(msg='NEGF calculation successfully completed.')

    if task == 'tbtrans_negf':
        if not(sisl_installed):
            log.error(msg="sisl is required to perform tbtrans calculation !")
            raise RuntimeError

        tbtrans_init = TBTransInputSet(apiHrk, run_opt, task_options)
        tbtrans_init.hamil_get_write(write_nc=True)
        log.info(msg='TBtrans input files are successfully generated.')

    if output:
        with open(os.path.join(output, "run_config.json"), "w") as fp:
            json.dump(jdata, fp, indent=4)
