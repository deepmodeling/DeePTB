import torch
import numpy as np
from dptb.nn.build import build_model
import json
import logging
from dptb.nn.sktb.onsiteDB import onsite_energy_database
import re
import os
from dptb.utils.gen_inputs import gen_inputs
import json
from collections import OrderedDict
log = logging.getLogger(__name__)

def to_empsk(
    INPUT,
    output='./', 
    basemodel='poly2',
    soc= None,
    **kwargs):
    """
    Convert the model to empirical SK parameters.
    """
    if INPUT is None:
        raise ValueError('INPUT is None.')
    with open(INPUT, 'r') as f:
        input = json.load(f)
    common_options = input['common_options']
    EmpSK(common_options, basemodel=basemodel).to_json(outdir=output, soc=soc)

class EmpSK(object):
    """
    Empirical SK parameters.
    """
    def __init__(self, common_options, basemodel='poly2'):
        """
        Args:
            common_options: common options for the model. especially contain the basis information.
            basemodel: base model type for the empirical SK parameters  either 'poly2' or 'poly4'.
        """
        self.common_options,self.basisref = self.format_common_options(common_options)
        if basemodel == 'poly2':
            model_ckpt = os.path.join(os.path.dirname(__file__), '..', 'nn', 'dftb', "base_poly2.pth")
        elif basemodel == 'poly4':
            model_ckpt = os.path.join(os.path.dirname(__file__), '..', 'nn', 'dftb', "base_poly4.pth")
        else:
            raise ValueError(f'basemodel {basemodel} is not supported.')

        self.model = build_model(model_ckpt, common_options=common_options, no_check=True)

    def to_json(self, outdir='./', soc=None):
        """
        Convert the model to json format.
        """
        # 判断是否存在输出目录
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        json_dict = self.model.to_json(basisref=self.basisref)\
        
        if soc is not None:
            mp = json_dict.setdefault("model_params", {})
            onsite = mp.get("onsite", {})

            # build soc block based on onsite
            soc_block = {}
            for key, val in onsite.items():
                parts = key.split("-")
                if len(parts) < 3:
                    continue
                elem, orb = parts[0], parts[1]
                # s and * orbitals -> 0, others -> soc value
                if orb.lower() == "s" or "*" in orb:
                    v = 0.0
                else:
                    v = float(soc)
                soc_block[key] = [v]

            # insert soc block after overlap
            if isinstance(mp, dict):
                new_mp = OrderedDict()
                inserted = False
                for k, v in mp.items():
                    new_mp[k] = v
                    if k == "overlap":
                        new_mp["soc"] = soc_block
                        inserted = True
                if not inserted:
                    new_mp["soc"] = soc_block
                json_dict["model_params"] = new_mp

            # update model_options for nnsk.soc.method
            mo = json_dict.setdefault("model_options", {})
            nnsk = mo.setdefault("nnsk", {})
            soc_opt = nnsk.setdefault("soc", {})
            soc_opt["method"] = "uniform_noref"

        # write final file
        with open(os.path.join(outdir, 'sktb.json'), 'w') as f:
            json.dump(json_dict, f, indent=4)

        # save input template
        # input_template = gen_inputs(model=self.model, task='train', mode=mode)
        
        #with open(os.path.join(outdir,'input_template.json'), 'w') as f:
        #    json.dump(input_template, f, indent=4)
        log.info(f'Empirical SK parameters are saved in {os.path.join(outdir,"sktb.json")}')
        log.info('If you want to further train the model, please use `dptb config` command to generate input template.')
        return json_dict

    def format_common_options(self, common_options):
        """
        Format the common options for the model. and construct the mapping between two kind of basis definition.
        The two kind of basis definition are:
            1. common_options = {'basis': {'C': ['s','p','d']}}
            2. common_options = {'basis': {'C': ['2s','2p','d*']}}
        
        Args:
            common_options: common options for the model. especially contain the basis information.
            e.g. common_options = {'basis': {'C': ['s','p','d']}} or common_options = {'basis': {'C': ['2s','2p','d*']}}
        
        Returns:
            common_options: common options for the model.
            basisref: basis reference for the model.
        """        
        # check basis in common_options
        if 'basis' not in common_options:
            raise ValueError('basis information is not given in common_options.')
        # check basis type
        assert isinstance(common_options['basis'], dict), 'basis information is not a dictionary.'
        basis = common_options['basis'] 
        sys_ele =  "".join(list(basis.keys()))
        log.info(f'Extracting empirical SK parameters for {sys_ele}')

        use_basis_ref = False
        basisref = {}
        for ie in basis.keys():
            basisref[ie] = {}
            assert isinstance(basis[ie], list), f'basis information for {ie} is not a list.'
            for ieorb in basis[ie]:
                assert isinstance(ieorb, str), f'basis information for {ie} is not a string.'
                if len(ieorb) == 1:
                    assert use_basis_ref is False, 'Invalid basis setting: cannot mix s, p, d with ns, np, d*.'
                    continue
                else:
                    use_basis_ref = True
                    assert ieorb in onsite_energy_database[ie], f'basis information for {ie} is not in onsite_energy_database : {onsite_energy_database[ie].keys()}.'
                    orbsymb = re.findall(r'[A-Za-z]', ieorb)[0]
                    basisref[ie][orbsymb] = ieorb

        if use_basis_ref:
            std_basis = {}
            for ie in basis.keys():
                std_basis[ie] = []
                for ieorb in basis[ie]:
                    std_basis[ie].append(re.findall(r'[A-Za-z]', ieorb)[0])
            common_options['basis'].update(std_basis)
        
            return common_options, basisref
        else:
            return common_options, None
