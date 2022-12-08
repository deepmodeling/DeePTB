from pyexpat import model
import torch
from dptb.plugins.base_plugin import Plugin
import logging
from dptb.nnsktb.sknet import SKNet
from dptb.nnsktb.onsiteFunc import onsiteFunc, loadOnsite
from dptb.nnsktb.socFunc import socFunc, loadSoc
from dptb.nnsktb.skintTypes import all_skint_types, all_onsite_intgrl_types
from dptb.utils.index_mapping import Index_Mapings
from dptb.nnsktb.integralFunc import SKintHops
from dptb.nnsktb.loadparas import load_paras
from dptb.utils.constants import dtype_dict
from dptb.utils.tools import update_dict
from dptb.utils.argcheck import model_config_checklist

log = logging.getLogger(__name__)

class InitSKModel(Plugin):
    def __init__(self, interval=None):
        if interval is None:
            interval = [(-1, 'disposable')]
        super(InitSKModel, self).__init__(interval)
    def register(self, host):
        self.host = host

    def disposable(self, mode=None, time=0, **common_and_model_options):
        self.mode = mode
        if mode == "from_scratch":
            self.init_from_scratch(**common_and_model_options)
        elif mode == "init_model" or mode == "restart":
            self.init_from_model(**common_and_model_options)
        else:
            log.info(msg="Haven't assign a initializing mode, training from scratch as default.")
            self.init_from_scratch(**common_and_model_options)


    def init_from_scratch(self, **common_and_model_options):
        # paras directly imported from inputs.
        # ----------------------------------------------------------------------------------------------------------
        device = common_and_model_options['device']
        atomtype = common_and_model_options["atomtype"]
        dtype = common_and_model_options['dtype']
        num_hopping_hideen = common_and_model_options['sknetwork']['sk_hop_nhidden']
        num_onsite_hidden = common_and_model_options['sknetwork']['sk_onsite_nhidden']
        num_soc_hidden = common_and_model_options['sknetwork']['num_soc_hidden']
        proj_atom_anglr_m = common_and_model_options['proj_atom_anglr_m']
        onsitemode = common_and_model_options['onsitemode']
        skformula = common_and_model_options['skfunction']['skformula']
        soc = common_and_model_options['skfunction']["soc"]
        # ----------------------------------------------------------------------------------------------------------
        
        IndMap = Index_Mapings()
        IndMap.update(proj_atom_anglr_m=proj_atom_anglr_m)
        bond_index_map, bond_num_hops = IndMap.Bond_Ind_Mapings()
        onsite_strain_index_map, onsite_strain_num, onsite_index_map, onsite_num = \
                IndMap.Onsite_Ind_Mapings(onsitemode, atomtype=atomtype)

        onsite_fun = onsiteFunc
        hops_fun = SKintHops(mode='hopping',functype=skformula,proj_atom_anglr_m=proj_atom_anglr_m)
        if soc:
            soc_fun = socFunc
        if onsitemode == 'strain':
            onsitestrain_fun = SKintHops(mode='onsite', functype=skformula,proj_atom_anglr_m=proj_atom_anglr_m, atomtype=atomtype)
        
        _, reducted_skint_types, _ = all_skint_types(bond_index_map)
        bond_neurons = {"nhidden": num_hopping_hideen,  "nout": hops_fun.num_paras}


        options = {"onsitemode": onsitemode}
        if onsitemode == 'strain':
            onsite_neurons = {"nhidden": num_onsite_hidden, "nout": onsitestrain_fun.num_paras}
            _, reducted_onsiteint_types, onsite_strain_ind_dict = all_onsite_intgrl_types(onsite_strain_index_map)
            options.update({"onsiteint_types":reducted_onsiteint_types})
        else:
            onsite_neurons = {"nhidden":num_onsite_hidden}
            reducted_onsiteint_types = False

        if soc:
            soc_neurons = {"nhidden":num_soc_hidden}

        self.host.model = SKNet(skint_types=reducted_skint_types,
                                onsite_num=onsite_num,
                                bond_neurons=bond_neurons,
                                onsite_neurons=onsite_neurons,
                                soc_neurons=soc_neurons,
                                device=device,
                                dtype=dtype,
                                onsitemode=onsitemode,
                                onsiteint_types=reducted_onsiteint_types)

        self.host.onsite_fun = onsite_fun
        self.host.hops_fun = hops_fun
        #self.host.onsite_index_map = onsite_index_map
        self.host.onsite_db = loadOnsite(onsite_index_map)
        if soc:
            self.host.soc_fun = soc_fun
            self.host.soc_db = loadSoc(onsite_index_map)
        if onsitemode == 'strain':
            self.host.onsitestrain_fun = onsitestrain_fun

        self.host.model_config = common_and_model_options

    def init_from_model(self, **common_and_model_and_run_options):
        # load checkpoint
        if self.mode == "init_model":
            checkpoint = common_and_model_and_run_options['init_model']
        elif self.mode == "restart":
            checkpoint = common_and_model_and_run_options["restart"]

        ckpt = torch.load(checkpoint)
        model_config = ckpt["model_config"]
        model_config["dtype"] = dtype_dict[model_config["dtype"]]
        
        # paras directly imported from inputs.
        # ----------------------------------------------------------------------------------------------------------
        device = common_and_model_and_run_options['device']
        atomtype = common_and_model_and_run_options["atomtype"]
        dtype = common_and_model_and_run_options['dtype']
        proj_atom_anglr_m = common_and_model_and_run_options['proj_atom_anglr_m']
        onsitemode = common_and_model_and_run_options['onsitemode']
        skformula = common_and_model_and_run_options['skfunction']['skformula']
        
        # ----------------------------------------------------------------------------------------------------------
        
        # load params from model_config
        assert skformula == model_config['skfunction'].get('skformula')        
        num_hopping_hideen = model_config['sknetwork']['sk_hop_nhidden']
        num_onsite_hidden = model_config['sknetwork']['sk_onsite_nhidden']
        num_soc_hidden = model_config['sknetwork']['num_soc_hidden']
        soc = model_config['skfunction']['soc']

        IndMap = Index_Mapings()
        IndMap.update(proj_atom_anglr_m=proj_atom_anglr_m)
        bond_index_map, bond_num_hops = IndMap.Bond_Ind_Mapings()
        onsite_strain_index_map, onsite_strain_num, onsite_index_map, onsite_num = \
                IndMap.Onsite_Ind_Mapings(onsitemode, atomtype=atomtype)

        onsite_fun = onsiteFunc
        hops_fun = SKintHops(mode='hopping',functype=skformula,proj_atom_anglr_m=proj_atom_anglr_m)
        if soc:
            soc_fun = socFunc
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

        if soc:
            soc_neurons = {"nhidden":num_soc_hidden}

        _, state_dict = load_paras(model_config=model_config, state_dict=ckpt['model_state_dict'], proj_atom_anglr_m=proj_atom_anglr_m, onsitemode=onsitemode)

        
        self.host.model = SKNet(skint_types=reducted_skint_types,
                                   onsite_num=onsite_num,
                                   bond_neurons=bond_neurons,
                                   onsite_neurons=onsite_neurons,
                                   soc_neurons=soc_neurons,
                                   device=device,
                                   dtype=dtype,
                                   onsitemode=onsitemode,
                                   onsiteint_types=reducted_onsiteint_types
                                   )
        self.host.onsite_fun = onsite_fun
        self.host.hops_fun = hops_fun
        #self.host.onsite_index_map = onsite_index_map
        self.host.onsite_db = loadOnsite(onsite_index_map)
        if soc:
            self.host.soc_fun = soc_fun
            self.host.soc_db = loadSoc(onsite_index_map)
        if onsitemode == 'strain':
            self.host.onsitestrain_fun = onsitestrain_fun
        
        model_config.update(common_and_model_and_run_options)
        self.host.model_config = update_dict(temp_dict=model_config, update_dict=common_and_model_and_run_options, checklist=model_config_checklist)
        
        self.host.model.load_state_dict(state_dict)            
        self.host.model.train()