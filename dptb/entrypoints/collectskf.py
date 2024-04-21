from typing import Dict, List, Optional, Any
import json
from pathlib import Path
import os
import torch
import glob
from dptb.nn.dftb.sk_param import SKParam

import logging

__all__ = ["skf2pth"]


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
    

