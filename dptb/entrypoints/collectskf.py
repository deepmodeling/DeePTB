from typing import Dict, List, Optional, Any, Union
import json
from pathlib import Path
import os
import torch
import glob
from dptb.nn.dftb.sk_param import SKParam
from dptb.nn.dftb2nnsk import DFTB2NNSK
import logging
from dptb.utils.loggers import set_log_handles
from dptb.utils.tools import j_loader, setup_seed, j_must_have
from dptb.utils.argcheck import normalize, collect_cutoffs, normalize_skf2nnsk


__all__ = ["skf2pth", "skf2nnsk"]


log = logging.getLogger(__name__)


def skf2pth(
        dir_path: str = "./",
        output: str = "skparams.pth",
        log_level: int = logging.INFO,
        log_path: Optional[str] = None,
        **kwargs
):
    skfiles = glob.glob(f"{dir_path}/*.skf") 
    skfile_dict = {}
    for ifile in skfiles:
        ifile_name = ifile.split('/')[-1]
        bond_type = ifile_name.split('.')[0]
        skfile_dict[bond_type] = ifile

    skdict = SKParam.read_skfiles(skfile_dict)

    if output.split('.')[-1] != 'pth':
        output += '/skparams.pth'

    output_path = Path(output)
    if not output_path.parent.exists():
        log.info(f"Directory {output_path.parent} does not exist. Creating it now.")
        output_path.parent.mkdir(parents=True)  # 创建所有必需的中间目录

    # 现在目录已存在，可以安全地保存文件
    if os.path.exists(output):
        log.warning(f"Overwriting {output}")

    torch.save(skdict, output)
    

def skf2nnsk(
        INPUT:str,
        init_model: Optional[str],
        output:str,
        log_level: int,
        log_path: Optional[str] = None,
        **kwargs
):
    run_opt = {
        "init_model": init_model,
        "log_path": log_path,
        "log_level": log_level
    }

        # setup output path
    if output:
        Path(output).parent.mkdir(exist_ok=True, parents=True)
        Path(output).mkdir(exist_ok=True, parents=True)
        if not log_path:
            log_path = os.path.join(str(output), "log.txt")
        Path(log_path).parent.mkdir(exist_ok=True, parents=True)

        run_opt.update({
            "output": str(Path(output).absolute()),
            "log_path": str(Path(log_path).absolute())
        })
    set_log_handles(log_level, Path(log_path) if log_path else None)

    jdata = j_loader(INPUT)
    jdata = normalize_skf2nnsk(jdata)

    common_options = jdata['common_options']
    model_options = jdata['model_options']
    train_options = jdata['train_options']

    basis = j_must_have(common_options, "basis")
    skdata_file = j_must_have(common_options, "skdata")

    if skdata_file.split('.')[-1] != 'pth':
        log.error("The skdata file should be a pth file.")
        raise ValueError("The skdata file should be a pth file.")
    log.info(f"Loading skdata from {skdata_file}")
    skdata = torch.load(skdata_file, weights_only=False)

    if isinstance(basis, str) and basis == "auto":
        log.info("Automatically determining basis")
        basis = dict(zip(skdata['OnsiteE'], [['s', 'p', 'd']] * len(skdata['OnsiteE'])))
    else:
        assert isinstance(basis, dict), "basis must be a dict or 'auto'"
    
    train_options = jdata['train_options']

    if init_model:
        dftb2nn = DFTB2NNSK.load(ckpt=init_model, 
                                 skdata=skdata,
                                 train_options=train_options,
                                 output=run_opt.get('output', './')
                                 )
    
    else:
        dftb2nn = DFTB2NNSK(
                        basis = basis,
                        skdata = skdata,
                        method=j_must_have(model_options, "method"),
                        rs=model_options.get('rs', None),
                        w = j_must_have(model_options, "w"),
                        cal_rcuts= model_options.get('rs', None) is None,
                        atomic_radius= model_options.get('atomic_radius', 'cov'),
                        train_options=train_options,
                        output=run_opt.get('output', './')
                    )
        
    dftb2nn.optimize()
        
        


