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

class WriteNNSKParam(object):
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
        # step 1: get neighbours of structure.
        onsite_cutoff = self.model_config["onsite_cutoff"]
        bond_cutoff = self.model_config["bond_cutoff"]
        onsite_index_dict = self.apiH.apihost.model.onsite_index_dict
        sk_cutoff = self.model_config["skfunction"]["sk_cutoff"]
        sk_decay_w = self.model_config["skfunction"]["sk_decay_w"]
        onsitemode = self.model_config["onsitemode"]
        functype = self.model_config["skfunction"]["skformula"]
        format = self.jdata["format"]
        outPath = self.results_path
        thr = self.jdata["thr"]
        
        nn_onsiteE, onsite_coeffdict = self.apiH.apihost.model(mode='onsite')
        if onsitemode == "strain":
            onsite_coeff = onsite_coeffdict
        elif onsitemode in ["uniform", "split"]:
            onsite_coeff = nn_onsiteE
        elif onsitemode == "none":
            onsite_coeff = None
        else:
            raise ValueError("onsite mode have wrong value, \
                             should be setted within [strain, uniform, split, none]")
        
        hopping_coeff = self.apiH.apihost.model(mode='hopping')

        if self.model_config["soc"]:
            soc_coeff, _ = self.apiH.apihost.model(mode='soc')
        else:
            soc_coeff = None

        
        write_skparam(
            onsite_coeff=onsite_coeff,
            hopping_coeff=hopping_coeff,
            soc_coeff=soc_coeff,
            onsite_index_dict=onsite_index_dict, 
            rcut=sk_cutoff, 
            w=sk_decay_w,
            atom=self.structase,
            onsite_cutoff=onsite_cutoff, 
            bond_cutoff=bond_cutoff,
            thr=thr, 
            onsitemode=onsitemode, 
            functype=functype, 
            format=format,
            outPath=outPath
        )