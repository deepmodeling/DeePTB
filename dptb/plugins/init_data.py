import os
import time
import torch
from dptb.plugins.base_plugin import Plugin
import logging
from dptb.nnsktb.sknet import SKNet
from dptb.nnsktb.skintTypes import all_skint_types, all_onsite_intgrl_types
from dptb.utils.index_mapping import Index_Mapings
from dptb.nnsktb.onsiteFunc import onsiteFunc, loadOnsite
from dptb.nnsktb.integralFunc import SKintHops
from dptb.nnsktb.loadparas import load_paras
from dptb.dataprocess.datareader import get_data
from dptb.utils.constants import dtype_dict


log = logging.getLogger(__name__)

class InitData(Plugin):
    def __init__(self, interval=None):
        if interval is None:
            interval = [(-1, 'inidata')]
        super(InitData, self).__init__(interval)
        
    def register(self, host):
        self.host = host

    def inidata(self, **common_and_data_options):
        # ----------------------------------------------------------------------------------------------------------
        use_reference = common_and_data_options['use_reference']
        self.bond_cutoff = common_and_data_options['bond_cutoff']
        self.env_cutoff = common_and_data_options['env_cutoff']
        self.onsite_cutoff = common_and_data_options['onsite_cutoff']
        self.proj_atom_anglr_m = common_and_data_options['proj_atom_anglr_m']
        self.proj_atom_neles = common_and_data_options['proj_atom_neles']
        self.onsitemode = common_and_data_options['onsitemode']
        self.time_symm = common_and_data_options['time_symm']
        self.device = common_and_data_options['device']
        self.dtype = dtype_dict[common_and_data_options['dtype']]
        # ----------------------------------------------------------------------------------------------------------

        
        self.init_training_data(**common_and_data_options)

        self.init_validation_dara(**common_and_data_options)

        if use_reference:
            self.init_reference_data(**common_and_data_options)

    
    def init_training_data(self,  **data_options):
        # paras directly imported from inputs.
        # ----------------------------------------------------------------------------------------------------------
        train_data_path = data_options['train'].get('train_data_path')
        train_data_prefix = data_options['train'].get('train_data_prefix')
        train_batch_size = data_options['train'].get('train_batch_size')
        # ----------------------------------------------------------------------------------------------------------

        self.host.train_processor_list = get_data(
            path=train_data_path, 
            prefix=train_data_prefix,
            batch_size=train_batch_size, 
            bond_cutoff=self.bond_cutoff, 
            env_cutoff= self.env_cutoff, 
            onsite_cutoff= self.onsite_cutoff, 
            proj_atom_anglr_m= self.proj_atom_anglr_m, 
            proj_atom_neles= self.proj_atom_neles, 
            sorted_onsite="st", 
            sorted_bond="st", 
            sorted_env="st", 
            onsitemode= self.onsitemode, 
            time_symm= self.time_symm, 
            device= self.device, 
            dtype= self.dtype
        )

    def init_reference_data(self, **data_options):
        # paras directly imported from inputs.
        # ----------------------------------------------------------------------------------------------------------
        reference_data_path = data_options['reference'].get('reference_data_path')
        reference_data_prefix = data_options['reference'].get('reference_data_prefix')
        reference_batch_size = data_options['reference'].get('reference_batch_size')
        # ----------------------------------------------------------------------------------------------------------

        self.host.ref_processor_list = get_data(
            path= reference_data_path, 
            prefix= reference_data_prefix,
            batch_size= reference_batch_size, 
            bond_cutoff= self.bond_cutoff, 
            env_cutoff= self.env_cutoff, 
            onsite_cutoff= self.onsite_cutoff, 
            proj_atom_anglr_m= self.proj_atom_anglr_m, 
            proj_atom_neles= self.proj_atom_neles, 
            sorted_onsite="st", 
            sorted_bond="st", 
            sorted_env="st", 
            onsitemode= self.onsitemode, 
            time_symm= self.time_symm, 
            device= self.device, 
            dtype= self.dtype
            )
    
    def init_validation_dara(self,**data_options):
        # paras directly imported from inputs.
        # ----------------------------------------------------------------------------------------------------------
        validation_data_path = data_options['validation'].get('validation_data_path')
        validation_data_prefix = data_options['validation'].get('validation_data_prefix')
        validation_batch_size = data_options['validation'].get('validation_batch_size')
        # ----------------------------------------------------------------------------------------------------------

        self.host.test_processor_list = get_data(
            path= validation_data_path, 
            prefix= validation_data_prefix,
            batch_size= validation_batch_size, 
            bond_cutoff= self.bond_cutoff, 
            env_cutoff= self.env_cutoff, 
            onsite_cutoff= self.onsite_cutoff, 
            proj_atom_anglr_m= self.proj_atom_anglr_m, 
            proj_atom_neles= self.proj_atom_neles, 
            sorted_onsite="st", 
            sorted_bond="st", 
            sorted_env="st", 
            onsitemode= self.onsitemode, 
            time_symm= self.time_symm, 
            device= self.device, 
            dtype= self.dtype
        )
