import torch
from dptb.plugins.base_plugin import Plugin
import logging
from dptb.nnsktb.sknet import SKNet
from dptb.nnsktb.skintTypes import all_skint_types, all_onsite_intgrl_types
from dptb.utils.index_mapping import Index_Mapings
from dptb.nnsktb.loadparas import load_paras
from dptb.utils.constants import dtype_dict
from dptb.utils.tools import get_uniq_bond_type, get_uniq_env_bond_type

log = logging.getLogger(__name__)

class InitIntpSKModel(Plugin):
    def __init__(self, interval=None):
        if interval is None:
            interval = [(-1, 'initintpmodel')]
        super(InitIntpSKModel, self).__init__(interval)
    def register(self, host):
        self.host = host

    def initintpmodel(self,  models_dict, **common_and_model_options):
        # models_dict is a dictionary of models, e.g. {'Si': Si.model, 'C':C.model} ...
        # paras directly imported from inputs.
        # ----------------------------------------------------------------------------------------------------------
        device = common_and_model_options['device']
        dtype = dtype_dict[common_and_model_options['dtype']]
        proj_atom_anglr_m = common_and_model_options['proj_atom_anglr_m']
        onsitemode = common_and_model_options['onsitemode']
        atom_types = common_and_model_options['atom_types']
        skformula = common_and_model_options['skfunction'].get('skformula')
        sk_cutoff = common_and_model_options['skfunction'].get('sk_cutoff')
        sk_decay_w = common_and_model_options['skfunction'].get('sk_decay_w')
        onsite_cutoff = common_and_model_options['onsite_cutoff']
        # ----------------------------------------------------------------------------------------------------------
        
        proj_atom_types = proj_atom_anglr_m.keys()
        bond_types = get_uniq_bond_type(proj_atom_type = proj_atom_types)
        if onsitemode == 'strain':
            onsite_envbond_types = get_uniq_env_bond_type(proj_atom_type = proj_atom_types, atom_type=atom_types) 
        assert set(proj_atom_types).issubset(set(models_dict.keys()))

        


    def init_intra_model(self, model1, **kwargs):
        # init the nn models  for  onsite and intra atom type hoppings.
        pass
    
    def init_inter_model(self, model1, model2, **kwargs):
        # init the nn models for inter types hoppings.
        pass


