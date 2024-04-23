import os
import logging
import json
from typing import Optional
from pathlib import Path
from dptb.nn.build import build_model
from dptb.postprocess.bandstructure.band import Band
from dptb.utils.loggers import set_log_handles
from dptb.utils.argcheck import normalize_run
from dptb.utils.tools import j_loader
from dptb.utils.tools import j_must_have
from dptb.postprocess.NEGF import NEGF
from dptb.postprocess.tbtrans_init import TBTransInputSet,sisl_installed

from dptb.postprocess.write_ham import write_ham
import torch
import h5py
import pytest

log = logging.getLogger(__name__)

def run(
        INPUT: str,
        init_model: str,
        structure: str,
        output: str,
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
    jdata = normalize_run(jdata)

    task_options = j_must_have(jdata, "task_options")
    task = task_options["task"]
    use_gui = jdata.get("use_gui", False)
    task_options.update({"use_gui": use_gui})
    results_path = run_opt.get("results_path", None)

    in_common_options = {}
    if jdata.get("device", None):
        in_common_options.update({"device": jdata["device"]})
    
    if jdata.get("dtype", None):
        in_common_options.update({"dtype": jdata["dtype"]})

    model = build_model(checkpoint=init_model, common_options=in_common_options)
    
    if  run_opt['structure'] is None:
        log.warning(msg="Warning! structure is not set in run option, read from input config file.")
        structure = j_must_have(jdata, "structure")
        run_opt.update({"structure":structure})

    struct_file = run_opt["structure"]

    if task=='band':        
        bcal = Band(model=model, results_path=results_path, use_gui=use_gui)
        bcal.get_bands( data=struct_file, 
                        kpath_kwargs=jdata["task_options"], 
                        AtomicData_options=jdata['AtomicData_options'])
        
        bcal.band_plot( ref_band=jdata["task_options"].get("ref_band", None),
                        E_fermi=jdata["task_options"].get("E_fermi", None),
                        emin=jdata["task_options"].get("emin", None),
                        emax=jdata["task_options"].get("emax", None))
        log.info(msg='band calculation successfully completed.')

    elif task=='negf':

        # try:
        #     from pyinstrument import Profiler
        # except ImportWarning:
        #     log.warning(msg="pyinstrument is not installed, no profiling will be done.")
        #     Profiler = None
        # if Profiler is not None:
        #     profiler = Profiler()
        #     profiler.start()
        
        negf = NEGF(
            model=model,
            AtomicData_options=jdata['AtomicData_options'],
            structure=structure,
            results_path=results_path,  
            **task_options
            )
   
        negf.compute()
        log.info(msg='negf calculation successfully completed.')

        # if Profiler is not None:
        #     profiler.stop()
        #     with open(results_path+'/profile_report.html', 'w') as report_file:
        #         report_file.write(profiler.output_html())

    elif task == 'tbtrans_negf':
        if not(sisl_installed):
            log.error(msg="sisl is required to perform tbtrans calculation !")
            raise RuntimeError
        basis_dict = json.load(open(init_model))['common_options']['basis']
        tbtrans_init = TBTransInputSet(
            model=model, 
            AtomicData_options=jdata['AtomicData_options'],
            structure=structure,
            basis_dict=basis_dict,
            results_path=results_path,
            **task_options)
        tbtrans_init.hamil_get_write(write_nc=True)
        log.info(msg='TBtrans input files are successfully generated.')
    
    elif task=='write_block':
        task = torch.load(init_model)["task"]
        block = write_ham(data=struct_file, AtomicData_options=jdata['AtomicData_options'], model=model, device=jdata["device"])
        # write to h5 file, block is a dict, write to a h5 file
        with h5py.File(os.path.join(results_path, task+".h5"), 'w') as fid:
            default_group = fid.create_group("1")
            for key_str, value in block.items():
                default_group[key_str] = value.detach().cpu().numpy()
        log.info(msg='write block successfully completed.')
