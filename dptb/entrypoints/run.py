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
from dptb.postprocess.write_block import write_block, write_block_parallel_hdf5
import torch
import h5py
from dptb.utils.auto_band_config import auto_band_config
from dptb.utils.distributed import initialize_distributed, is_distributed

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

    in_common_options = {}
    if INPUT.endswith(".json"):
        jdata = j_loader(INPUT)
        jdata = normalize_run(jdata)  
    else:
        assert run_opt["structure"] is not None, "Please provide a structure file or a json file."
        jdata, com_opts = auto_band_config(structure=structure, kpathtype='vasp')
        assert jdata['task_options']["task"] == "band", "No Input json is provided, then only band task is supported."
        jdata = normalize_run(jdata)

        if init_model in ['poly2','poly4']:
            modelname = f'base_{init_model}.pth'
            init_model = os.path.join(os.path.dirname(__file__), '..', 'nn', 'dftb', modelname)
            in_common_options.update(com_opts)
        elif not(init_model.endswith(".pth") or init_model.endswith(".json")):
            raise ValueError(f'init_model {init_model} is not supported.')
        run_opt.update({'init_model': init_model})
            
    task_options = j_must_have(jdata, "task_options")
    task = task_options["task"]
    use_gui = jdata.get("use_gui", False)
    task_options.update({"use_gui": use_gui})
    results_path = run_opt.get("results_path", None)

    dist_options = jdata.get("distributed_options", {})
    distributed = bool(dist_options.get("enabled", False))
    rank = 0
    if distributed:
        if task != "write_block":
            raise ValueError("Distributed run currently supports only the write_block task")
        requested_device = jdata.get("device")
        if requested_device is None:
            checkpoint_data = torch.load(init_model, map_location="cpu", weights_only=False)
            requested_device = (
                checkpoint_data.get("config", {})
                .get("common_options", {})
                .get("device", "cpu")
            )
            del checkpoint_data
            jdata["device"] = requested_device
        rank, _, local_rank = initialize_distributed(
            device=requested_device, backend=dist_options.get("backend", "auto")
        )
        if str(jdata.get("device", "")).startswith("cuda"):
            jdata["device"] = f"cuda:{local_rank}"
        jdata["AtomicData_options"]["partition_options"] = {
            "method": dist_options.get("partition", "rib"),
        }

    set_log_handles(
        log_level if rank == 0 else logging.CRITICAL + 1,
        Path(log_path) if log_path and rank == 0 else None,
    )

    
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
        bcal = Band(model=model, results_path=results_path, use_gui=use_gui, device=model.device)
        bcal.get_bands( data=struct_file, 
                        kpath_kwargs=jdata["task_options"], 
                        pbc=jdata["pbc"],
                        AtomicData_options=jdata['AtomicData_options'])
        
        bcal.band_plot( ref_band=jdata["task_options"].get("ref_band", None),
                        E_fermi=jdata["task_options"].get("E_fermi", None),
                        emin=jdata["task_options"].get("emin", None),
                        emax=jdata["task_options"].get("emax", None))
        log.info(msg='band calculation successfully completed.')

    elif task=='write_block':
        task = torch.load(init_model, map_location="cpu", weights_only=False)["task"]
        block = write_block(data=struct_file, AtomicData_options=jdata['AtomicData_options'], model=model, device=jdata["device"])
        output_file = os.path.join(results_path, task+".h5")
        if is_distributed():
            write_block_parallel_hdf5(output_file, block)
        else:
            # write to h5 file, block is a dict, write to a h5 file
            with h5py.File(output_file, 'w') as fid:
                default_group = fid.create_group("0")
                for key_str, value in block.items():
                    default_group[key_str] = value.detach().cpu().numpy()
        log.info(msg='write block successfully completed.')
