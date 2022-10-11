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


log = logging.getLogger(__name__)

class InitData(Plugin):
    def __init__(self, interval=None):
        if interval is None:
            interval = [(-1, 'inidata')]
        super(InitData, self).__init__(interval)
        
    def register(self, host):
        self.host = host

    def inidata(self, **kwargs):
        self.init_training_data()

        self.init_validation_dara()
        
        if self.host.use_reference:
            self.init_reference_data()

    
    def init_training_data(self):
        self.host.train_processor_list = get_data(
            path=self.host.train_data_path, 
            prefix=self.host.train_data_prefix,
            batch_size=self.host.batch_size, 
            bond_cutoff=self.host.bond_cutoff, 
            env_cutoff=self.host.env_cutoff, 
            onsite_cutoff=self.host.onsite_cutoff, 
            proj_atom_anglr_m=self.host.proj_atom_anglr_m, 
            proj_atom_neles=self.host.proj_atom_neles, 
            sorted_onsite="st", 
            sorted_bond="st", 
            sorted_env="st", 
            onsitemode=self.host.onsitemode, 
            time_symm=self.host.time_symm, 
            device=self.host.device, 
            dtype=self.host.dtype
        )

    def init_reference_data(self):
        self.host.ref_processor_list = get_data(
            path=self.host.ref_data_path, 
            prefix=self.host.ref_data_prefix,
            batch_size=self.host.ref_batch_size, 
            bond_cutoff=self.host.bond_cutoff, 
            env_cutoff=self.host.env_cutoff, 
            onsite_cutoff=self.host.onsite_cutoff, 
            proj_atom_anglr_m=self.host.proj_atom_anglr_m, 
            proj_atom_neles=self.host.proj_atom_neles, 
            sorted_onsite="st", 
            sorted_bond="st", 
            sorted_env="st", 
            onsitemode=self.host.onsitemode, 
            time_symm=self.host.time_symm, 
            device=self.host.device, 
            dtype=self.host.dtype
            )
    
    def init_validation_dara(self):
        self.host.test_processor_list = get_data(
            path=self.host.test_data_path, 
            prefix=self.host.test_data_prefix,
            batch_size=self.host.test_batch_size, 
            bond_cutoff=self.host.bond_cutoff, 
            env_cutoff=self.host.env_cutoff, 
            onsite_cutoff=self.host.onsite_cutoff, 
            proj_atom_anglr_m=self.host.proj_atom_anglr_m, 
            proj_atom_neles=self.host.proj_atom_neles, 
            sorted_onsite="st", 
            sorted_bond="st", 
            sorted_env="st", 
            onsitemode=self.host.onsitemode, 
            time_symm=self.host.time_symm, 
            device=self.host.device, 
            dtype=self.host.dtype
        )
