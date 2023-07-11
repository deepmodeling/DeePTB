import numpy as np
from dptb.utils.tools import j_must_have, get_neighbours
from dptb.utils.make_kpoints  import ase_kpath, abacus_kpath, vasp_kpath
from ase.io import read
import ase
import matplotlib.pyplot as plt
import matplotlib
import logging
from dptb.utils.tools import write_skparam

log = logging.getLogger(__name__)

try:
    import tbplas as tb
    ifermi_tbplus = True
except ImportError:
    log.error('TBPLaS is not installed. Thus the ifermiaip is not available, Please install it first.')
    ifermi_installed = False


class WriteTBPLaS(object):
    def __init__(self, apiHrk, run_opt, jdata):
        self.apiH = apiHrk

        if isinstance(run_opt['structure'],str):
            self.structase = read(run_opt['structure'])
        elif isinstance(run_opt['structure'],ase.Atoms):
            self.structase = run_opt['structure']
        else:
            raise ValueError('structure must be ase.Atoms or str')
        
        self.results_path = run_opt.get('results_path')
        self.apiH.update_struct(self.structase)
        self.model_config = self.apiH.apihost.model_config
        self.jdata = jdata

    def write(self):
        # 1. lattice vector
        # 2. coordinates

        all_bonds, hamil_blocks, overlap_blocks = self.apiH.get_HR()
        