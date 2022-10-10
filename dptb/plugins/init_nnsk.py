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


log = logging.getLogger(__name__)

class InitSKModel(Plugin):
    def __init__(self, interval=None):
        if interval is None:
            interval = [(-1, 'inimodel')]
        super(InitSKModel, self).__init__(interval)
    def register(self, host):
        self.trainer = host

    def inimodel(self, mode=None, **kwargs):
        if mode == "from_scratch":
            self.init_from_scratch()
        elif mode == "init_model":
            self.init_from_model()
        else:
            log.info(msg="Haven't assign a initializing mode, training from scratch as default.")
            self.init_from_scratch()


    def init_from_scratch(self):
        IndMap = Index_Mapings()
        IndMap.update(proj_atom_anglr_m=self.trainer.proj_atom_anglr_m)
        bond_index_map, bond_num_hops = IndMap.Bond_Ind_Mapings()
        onsite_strain_index_map, onsite_strain_num, onsite_index_map, onsite_num = \
                IndMap.Onsite_Ind_Mapings(self.trainer.onsitemode, atomtype=self.trainer.atom_type)

        self.trainer.onsite_fun = onsiteFunc
        self.trainer.onsite_db  = loadOnsite(onsite_index_map)
        self.trainer.hops_fun   = SKintHops(mode=self.trainer.sk_options.get('skformula',"varTang96"))
        
        _, reducted_skint_types, self.trainer.sk_bond_ind_dict = all_skint_types(bond_index_map)
        bond_neurons = {"nhidden": self.trainer.model_options.get('sk_hop_nhidden',1), "nout":self.trainer.hops_fun.num_paras}
    


        options = {"onsitemode":self.trainer.onsitemode}
        if self.trainer.onsitemode == 'strain':
            onsite_neurons = {"nhidden":self.trainer.model_options.get('sk_onsite_nhidden',1),"nout":self.trainer.hops_fun.num_paras}
            _, reducted_onsiteint_types, _ = all_onsite_intgrl_types(onsite_strain_index_map)
            options.update({"onsiteint_types":reducted_onsiteint_types})
        else:
            onsite_neurons = {"nhidden":self.trainer.model_options.get('sk_onsite_nhidden',1)}

        self.trainer.model = SKNet(skint_types=reducted_skint_types,
                                   onsite_num=onsite_num,
                                   bond_neurons=bond_neurons,
                                   onsite_neurons=onsite_neurons,
                                   device=self.trainer.device,
                                   dtype=self.trainer.dtype,
                                   **options)
        
        
        self.trainer.model_config = ({  
                                        "proj_atom_anglr_m":self.trainer.proj_atom_anglr_m,
                                        "atom_type":self.trainer.atom_type,
                                        "skformula":self.trainer.sk_options.get('skformula',"varTang96"),
                                        "sk_cutoff":self.trainer.sk_cutoff,
                                        "sk_decay_w":self.trainer.sk_decay_w,
                                        "sk_hop_nhidden":self.trainer.model_options.get('sk_hop_nhidden',1),
                                        "sk_onsite_nhidden":self.trainer.model_options.get('sk_onsite_nhidden',1),
                                        "onsitemode":self.trainer.onsitemode,
                                        "onsite_cutoff":self.trainer.onsite_cutoff
                                    })

    def init_from_model(self):
        f = torch.load(self.trainer.run_opt["init_model"])
        model_config = f["model_config"]

        IndMap = Index_Mapings()
        IndMap.update(proj_atom_anglr_m=self.trainer.proj_atom_anglr_m)
        bond_index_map, bond_num_hops = IndMap.Bond_Ind_Mapings()
        onsite_strain_index_map, onsite_strain_num, onsite_index_map, onsite_num = \
                IndMap.Onsite_Ind_Mapings(self.trainer.onsitemode, atomtype=self.trainer.atom_type)

        self.trainer.onsite_fun = onsiteFunc
        self.trainer.onsite_db = loadOnsite(onsite_index_map)
        self.trainer.hops_fun = SKintHops(mode=model_config['skformula'])

        _, reducted_skint_types, self.trainer.sk_bond_ind_dict = all_skint_types(bond_index_map)
        bond_neurons = {"nhidden": model_config['sk_hop_nhidden'], "nout":self.trainer.hops_fun.num_paras}



        options = {"onsitemode":self.trainer.onsitemode}
        if self.trainer.onsitemode == 'strain':
            onsite_neurons = {"nhidden":model_config['sk_onsite_nhidden'],"nout":self.trainer.hops_fun.num_paras}
            _, reducted_onsiteint_types, _ = all_onsite_intgrl_types(onsite_strain_index_map)
            options.update({"onsiteint_types":reducted_onsiteint_types})
        else:
            onsite_neurons = {"nhidden":model_config['sk_onsite_nhidden']}

        _, state_dict = load_paras(model_config=model_config, state_dict=f['state_dict'],
                                                proj_atom_anglr_m=self.trainer.proj_atom_anglr_m, onsitemode=self.trainer.onsitemode)

        
        self.trainer.model = SKNet(skint_types=reducted_skint_types,
                                   onsite_num=onsite_num,
                                   bond_neurons=bond_neurons,
                                   onsite_neurons=onsite_neurons,
                                   device=self.trainer.device,
                                   dtype=self.trainer.dtype,
                                   **options)
        
        self.trainer.model_config.update({"proj_atom_anglr_m":self.trainer.proj_atom_anglr_m,
                                  "sk_cutoff":self.trainer.sk_cutoff,
                                  "sk_decay_w":self.trainer.sk_decay_w,
                                  "onsitemode":self.trainer.onsitemode,
                                  "onsite_cutoff":self.trainer.onsite_cutoff})
        
        self.trainer.model.load_state_dict(state_dict)
        self.trainer.model.train()

        


