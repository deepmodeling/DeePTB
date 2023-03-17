import numpy as np
from dptb.utils.tools import j_must_have
from dptb.utils.make_kpoints  import ase_kpath, abacus_kpath, vasp_kpath
from ase.io import read
import ase
import matplotlib.pyplot as plt
import matplotlib
import logging
log = logging.getLogger(__name__)

class WriteSKParam(object):
    def __init__(self, apiHrk, run_opt, jdata):
        self.apiH = apiHrk

        if isinstance(run_opt['structure'],str):
            self.structase = read(run_opt['structure'])
        elif isinstance(run_opt['structure'],ase.Atoms):
            self.structase = run_opt['structure']
        else:
            raise ValueError('structure must be ase.Atoms or str')
        
        self.jdata = jdata
        self.results_path = run_opt.get('results_path')
        self.apiH.update_struct(self.structase)

    def write(self):
        # step 1: get neighbours of structure.
        # step 2: fake bonds onsite and bond hopping fake it !
        # step 3: get nnsk and dptb calculation result, do the correction
        # step 4: format the output
        pass