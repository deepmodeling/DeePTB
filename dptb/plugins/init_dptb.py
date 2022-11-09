from pdb import Restart
from pyexpat import model
import torch
from dptb.plugins.base_plugin import Plugin
import logging
from dptb.nnet.nntb import NNTB
from dptb.nnsktb.sknet import SKNet
from dptb.nnsktb.onsiteFunc import onsiteFunc, loadOnsite
from dptb.nnsktb.skintTypes import all_skint_types, all_onsite_intgrl_types
from dptb.utils.index_mapping import Index_Mapings
from dptb.nnsktb.integralFunc import SKintHops
from dptb.nnsktb.loadparas import load_paras
from dptb.utils.constants import dtype_dict
from dptb.utils.tools import get_uniq_symbol, \
    get_lr_scheduler, get_uniq_bond_type, get_uniq_env_bond_type, \
    get_env_neuron_config, get_bond_neuron_config, get_onsite_neuron_config, \
    get_optimizer, nnsk_correction, j_must_have

log = logging.getLogger(__name__)

class InitDPTBModel(Plugin):
    def __init__(self, interval=None):
        if interval is None:
            interval = [(-1, 'disposable')]
        super(InitDPTBModel, self).__init__(interval)

    def register(self, host):
        self.host = host

    def disposable(self, mode=None, use_correction=False, time=0, **common_and_model_options):
        self.mode = mode
        if mode == "from_scratch":
            self.init_from_scratch(**common_and_model_options)
        elif mode =="init_model" or mode == "restart":
            self.init_from_model(**common_and_model_options)
        else:
            log.info(msg="Haven't assign a initializing mode, training from scratch as default.")
            self.init_from_scratch(**common_and_model_options)
        
        if use_correction:
            self.init_correction_model(use_correction=use_correction, **common_and_model_options)
    
    def init_from_scratch(self, **options):
        env_nnl = options["dptb"]['env_net_neuron']
        env_axisnn = options["dptb"]['axis_neuron']
        onsite_nnl = options["dptb"]['onsite_net_neuron']
        bond_nnl = options["dptb"]['bond_net_neuron']
        proj_atom_anglr_m = options["proj_atom_anglr_m"]
        onsitemode = options["onsitemode"]
        skformula = options['skfunction']['skformula']
        atomtype = options["atomtype"]
        proj_atomtype = get_uniq_symbol(list(proj_atom_anglr_m.keys()))
        bond_type = get_uniq_bond_type(proj_atomtype)

        IndMap = Index_Mapings()
        IndMap.update(proj_atom_anglr_m=proj_atom_anglr_m)
        bond_index_map, bond_num_hops = IndMap.Bond_Ind_Mapings()
        onsite_strain_index_map, onsite_strain_num, onsite_index_map, onsite_num = \
                IndMap.Onsite_Ind_Mapings(onsitemode, atomtype=atomtype)
        

        env_net_config = get_env_neuron_config(env_nnl)
        onsite_net_config = get_onsite_neuron_config(onsite_nnl, onsite_num, proj_atomtype, env_axisnn,
                                                        env_nnl[-1])
        bond_net_config = get_bond_neuron_config(bond_nnl, bond_num_hops, bond_type, env_axisnn,
                                                    env_nnl[-1])
    
        self.host.nntb = NNTB(proj_atomtype=proj_atomtype, env_net_config=env_net_config, 
                    bond_net_config=bond_net_config, onsite_net_config=onsite_net_config, **options, **options["dptb"])
        self.host.model = self.host.nntb.tb_net
        self.host.model_config = options

    def init_from_model(self, **options):
        # TODO: env_cutoff 按照input file 更新checkpoint.
        if self.mode == "init_model":
            checkpoint = options['init_model']
        elif self.mode == "restart":
            checkpoint = options["restart"]
        ckpt = torch.load(checkpoint)
        model_config = ckpt["model_config"]
        model_config["dtype"] = dtype_dict[model_config["dtype"]]

        env_nnl = model_config["dptb"]['env_net_neuron']
        env_axisnn = model_config["dptb"]['axis_neuron']
        onsite_nnl = model_config["dptb"]['onsite_net_neuron']
        bond_nnl = model_config["dptb"]['bond_net_neuron']
        proj_atom_anglr_m = model_config["proj_atom_anglr_m"]
        onsitemode = model_config["onsitemode"]
        skformula = model_config['skfunction']['skformula']
        atomtype = model_config["atomtype"]
        proj_atomtype = get_uniq_symbol(list(proj_atom_anglr_m.keys()))
        bond_type = get_uniq_bond_type(proj_atomtype)

        IndMap = Index_Mapings()
        IndMap.update(proj_atom_anglr_m=proj_atom_anglr_m)
        bond_index_map, bond_num_hops = IndMap.Bond_Ind_Mapings()
        onsite_strain_index_map, onsite_strain_num, onsite_index_map, onsite_num = \
                IndMap.Onsite_Ind_Mapings(onsitemode, atomtype=atomtype)
        
        env_net_config = get_env_neuron_config(env_nnl)
        onsite_net_config = get_onsite_neuron_config(onsite_nnl, onsite_num, proj_atomtype, env_axisnn,
                                                        env_nnl[-1])
        bond_net_config = get_bond_neuron_config(bond_nnl, bond_num_hops, bond_type, env_axisnn,
                                                    env_nnl[-1])
        
        self.host.nntb = NNTB(proj_atomtype=proj_atomtype, env_net_config=env_net_config, 
                    bond_net_config=bond_net_config, onsite_net_config=onsite_net_config, **model_config, **model_config["dptb"])
        self.host.nntb.tb_net.load_state_dict(ckpt["model_state_dict"])
        self.host.model = self.host.nntb.tb_net
        self.host.model_config = model_config
        self.host.model.train()

    def init_correction_model(self, **options):
        ckpt = torch.load(options["use_correction"])
        model_config = ckpt["model_config"]
        
        # -------------------------------------------------------------------------------------------
        device = options['device']
        atomtype = options["atomtype"]
        dtype = options['dtype']
        proj_atom_anglr_m = options['proj_atom_anglr_m']
        onsitemode = options['onsitemode']
        skformula = options['skfunction']['skformula']
        # -------------------------------------------------------------------------------------------

        num_hopping_hideen = model_config['sknetwork']['sk_hop_nhidden']
        num_onsite_hidden = model_config['sknetwork']['sk_onsite_nhidden']
        assert skformula == model_config['skfunction'].get('skformula')

        IndMap = Index_Mapings()
        IndMap.update(proj_atom_anglr_m=proj_atom_anglr_m)
        bond_index_map, bond_num_hops = IndMap.Bond_Ind_Mapings()
        onsite_strain_index_map, onsite_strain_num, onsite_index_map, onsite_num = \
                IndMap.Onsite_Ind_Mapings(onsitemode, atomtype=atomtype)

        onsite_fun = onsiteFunc
        hops_fun = SKintHops(mode='hopping',functype=skformula,proj_atom_anglr_m=proj_atom_anglr_m)
        if onsitemode == 'strain':
            onsitestrain_fun = SKintHops(mode='onsite', functype=skformula,proj_atom_anglr_m=proj_atom_anglr_m, atomtype=atomtype)

        _, reducted_skint_types, _ = all_skint_types(bond_index_map)
        bond_neurons = {"nhidden": num_hopping_hideen,  "nout": hops_fun.num_paras}

        if onsitemode == 'strain':
            onsite_neurons = {"nhidden":num_onsite_hidden,"nout":onsitestrain_fun.num_paras}
            _, reducted_onsiteint_types, _ = all_onsite_intgrl_types(onsite_strain_index_map)
        else:
            onsite_neurons = {"nhidden":num_onsite_hidden}
            reducted_onsiteint_types = False

        _, state_dict = load_paras(model_config=model_config, state_dict=ckpt['model_state_dict'], proj_atom_anglr_m=proj_atom_anglr_m, onsitemode=onsitemode)

        self.host.sknet = SKNet(skint_types=reducted_skint_types,
                                   onsite_num=onsite_num,
                                   bond_neurons=bond_neurons,
                                   onsite_neurons=onsite_neurons,
                                   device=device,
                                   dtype=dtype,
                                   onsitemode=onsitemode,
                                   onsiteint_types=reducted_onsiteint_types
                                   )
        self.host.sknet_config  = model_config
        
        self.host.sknet.load_state_dict(state_dict)
        self.host.sknet.train()

        self.host.onsite_fun = onsite_fun
        self.host.hops_fun = hops_fun
        self.host.onsite_db = loadOnsite(onsite_index_map)
        if onsitemode == 'strain':
            self.host.onsitestrain_fun = onsitestrain_fun
        
        if options['freeze']:
            self.host.sknet.eval()
            for p in self.host.sknet.parameters():
                p.requires_grad = False
