from typing import Dict, List, Optional, Any
import ase.io as io
from pathlib import Path
from dptb.utils.read_NRL_tojson import read_nrl_file,  nrl2dptb, save2json
import os
import logging
log = logging.getLogger(__name__)

def nrl2json(
        INPUT: str,
        nrl_file: str,
        outdir: str,
        log_level: int,
        log_path: Optional[str],
        **kwargs
):

    NRL_data = read_nrl_file(nrl_file)
    input_dict, nrl_tb_dict =  nrl2dptb(INPUT, NRL_data)
    save2json(input_dict, nrl_tb_dict, outdir)
    log.info(f'NRL file {nrl_file} has been converted to {outdir}/nrl_ckpt.json.')
    log.info(f'INPUT json is updated to {outdir}/input_nrl_auto.json.')
    log.info('Please check the json file and modify the parameters if necessary.')
