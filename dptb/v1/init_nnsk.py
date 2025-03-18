from pyexpat import model
import torch
from dptb.plugins.base_plugin import Plugin
import logging
from dptb.nnsktb.sknet import SKNet
from dptb.nnsktb.onsiteFunc import onsiteFunc, loadOnsite, orbitalEs
from dptb.nnsktb.socFunc import socFunc, loadSoc
from dptb.nnsktb.skintTypes import all_skint_types, all_onsite_intgrl_types, all_onsite_ene_types
from dptb.utils.index_mapping import Index_Mapings
from dptb.nnsktb.integralFunc import SKintHops
from dptb.nnsktb.loadparas import load_paras
from dptb.nnsktb.init_from_model import init_from_model_, init_from_json_
from dptb.utils.constants import dtype_dict
from dptb.utils.tools import update_dict, checkdict, update_dict_with_warning
from dptb.utils.argcheck import nnsk_model_config_checklist, nnsk_model_config_updatelist
from dptb.utils.tools import j_loader

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
        num_soc_hidden = common_and_model_options['sknetwork']['sk_soc_nhidden']
        proj_atom_anglr_m = common_and_model_options['proj_atom_anglr_m']
        onsitemode = common_and_model_options['onsitemode']
        skformula = common_and_model_options['skfunction']['skformula']
        soc = common_and_model_options["soc"]
        unit=common_and_model_options["unit"]
        # ----------------------------------------------------------------------------------------------------------
        # new add for NRL

        onsite_func_cutoff = common_and_model_options['onsitefuncion']['onsite_func_cutoff']
        onsite_func_decay_w = common_and_model_options['onsitefuncion']['onsite_func_decay_w']
        onsite_func_lambda = common_and_model_options['onsitefuncion']['onsite_func_lambda']

        overlap = common_and_model_options.get('overlap',False)

        #-----------------------------------------------------------------------------------------------------------
        IndMap = Index_Mapings()
        IndMap.update(proj_atom_anglr_m=proj_atom_anglr_m)
        bond_index_map, bond_num_hops = IndMap.Bond_Ind_Mapings()
        onsite_strain_index_map, onsite_strain_num, onsite_index_map, onsite_num = \
                IndMap.Onsite_Ind_Mapings(onsitemode, atomtype=atomtype)

        # onsite_fun = onsiteFunc
        hops_fun = SKintHops(mode='hopping',functype=skformula,proj_atom_anglr_m=proj_atom_anglr_m)
        if overlap:
            overlap_fun = SKintHops(mode='hopping',functype=skformula,proj_atom_anglr_m=proj_atom_anglr_m,overlap=overlap)

        if soc:
            soc_fun = socFunc
        if onsitemode == 'strain':
            onsitestrain_fun = SKintHops(mode='onsite', functype=skformula,proj_atom_anglr_m=proj_atom_anglr_m, atomtype=atomtype)
            # for strain mode the onsite_fun will use none mode to add the onsite_db.
            onsite_fun = orbitalEs(proj_atom_anglr_m=proj_atom_anglr_m,atomtype=atomtype,functype='none',unit=unit)
        elif onsitemode == 'NRL':
            onsite_fun = orbitalEs(proj_atom_anglr_m=proj_atom_anglr_m,atomtype=atomtype,functype=onsitemode,unit=unit,
                    onsite_func_cutoff=onsite_func_cutoff,onsite_func_decay_w=onsite_func_decay_w,onsite_func_lambda=onsite_func_lambda)
        else:
            onsite_fun = orbitalEs(proj_atom_anglr_m=proj_atom_anglr_m,atomtype=atomtype,functype=onsitemode,unit=unit)

            

        _, reducted_skint_types, _ = all_skint_types(bond_index_map)
        _, reduced_onsiteE_types, onsiteE_ind_dict = all_onsite_ene_types(onsite_index_map)
        if overlap:
            hopping_neurons = {"nhidden": num_hopping_hideen,  "nout": hops_fun.num_paras, "nout_overlap": overlap_fun.num_paras}
        else:
            hopping_neurons = {"nhidden": num_hopping_hideen,  "nout": hops_fun.num_paras}


# TODO: modify onsite_neurons, to have nout for other modes.

        options = {"onsitemode": onsitemode}
        if onsitemode == 'strain':
            onsite_neurons = {"nhidden": num_onsite_hidden, "nout": onsitestrain_fun.num_paras}
            _, reducted_onsiteint_types, onsite_strain_ind_dict = all_onsite_intgrl_types(onsite_strain_index_map)
            onsite_types = reducted_onsiteint_types
        else:
            onsite_neurons = {"nhidden":num_onsite_hidden, "nout": onsite_fun.num_paras}
            onsite_types = reduced_onsiteE_types
        
        options.update({"onsite_types":onsite_types})

        # TODO: generate soc types. here temporarily use the same as onsite types.
        if soc:
            if num_soc_hidden is not None:
                soc_neurons = {"nhidden":num_soc_hidden}
            else:
                log.err(msg="Please specify the number of hidden layers for soc network. please set the key `sk_soc_nhidden` in `sknetwork` in `model_options`.")
                raise ValueError
        else:
            soc_neurons=None

        self.host.model = SKNet(skint_types=reducted_skint_types,
                                onsite_types=onsite_types,
                                soc_types=reduced_onsiteE_types,
                                hopping_neurons=hopping_neurons,
                                onsite_neurons=onsite_neurons,
                                soc_neurons=soc_neurons,
                                device=device,
                                dtype=dtype,
                                onsitemode=onsitemode,
                                onsite_index_dict=onsiteE_ind_dict,
                                overlap=overlap)

        self.host.onsite_fun = onsite_fun
        self.host.hops_fun = hops_fun
        self.host.overlap = overlap
        if overlap:
            self.host.overlap_fun = overlap_fun
        # self.host.onsite_index_map = onsite_index_map
        # self.host.onsite_db = loadOnsite(onsite_index_map, unit=common_and_model_options["unit"])
        if soc:
            self.host.soc_fun = soc_fun
            self.host.soc_db = loadSoc(onsite_index_map)
        if onsitemode == 'strain':
            self.host.onsitestrain_fun = onsitestrain_fun

        self.host.model_config = common_and_model_options
        self.host.model_config.update({"types_list": [self.host.model.onsite_types, self.host.model.skint_types, self.host.model.soc_types]})

        if common_and_model_options["train_soc"]:
            for k,v in self.host.model.named_parameters():
                if "soc" not in k:
                    v.requires_grad = False

    def init_from_model(self, **common_and_model_and_run_options):
        # load checkpoint
        if self.mode == "init_model":
            checkpoint = common_and_model_and_run_options['init_model']["path"]
            interpolate = common_and_model_and_run_options['init_model']["interpolate"]
        elif self.mode == "restart":
            checkpoint = common_and_model_and_run_options["restart"]
            interpolate = False
        if not isinstance(checkpoint, list):
            checkpoint = [checkpoint]

        
        # paras directly imported from inputs.
        # ----------------------------------------------------------------------------------------------------------
        device = common_and_model_and_run_options['device']
        atomtype = common_and_model_and_run_options["atomtype"]
        dtype = common_and_model_and_run_options['dtype']
        proj_atom_anglr_m = common_and_model_and_run_options['proj_atom_anglr_m']
        onsitemode = common_and_model_and_run_options['onsitemode']
        skformula = common_and_model_and_run_options['skfunction']['skformula']
        soc = common_and_model_and_run_options['soc']
        
        if checkpoint[0].split('.')[-1] == 'json':
            modeltype = "json"
        elif checkpoint[0].split('.')[-1] == 'pth':
            modeltype = "ckpt"
        else:
            raise NotImplementedError("Only support json and ckpt file as checkpoint")
        
        #modeltype = common_and_model_and_run_options['modeltype']

        # ----------------------------------------------------------------------------------------------------------
        json_model_types = ["onsite", "hopping", "overlap", "soc"]
        if modeltype == "ckpt":
            ckpt_list = [torch.load(ckpt, weights_only=False) for ckpt in checkpoint]
        elif modeltype == "json":
            # 只用一个文件包含所有的键积分参数：
            assert len(checkpoint) ==1
            json_dict = j_loader(checkpoint[0])
            assert 'onsite'  in json_dict, "onsite paras is not in the json file, or key err, check the key onsite in json fle"
            assert 'hopping' in json_dict, "hopping paras is not in the json file, or key err, check the key hopping in json file"
            if soc:
                assert 'soc' in json_dict, "soc parameters not found in json file, or key err, check the soc key in json file"
            
            json_model_list = {}
            for ikey in json_model_types:
                json_model_i ={}
                if ikey in json_dict.keys():
                    for itype in json_dict[ikey]:
                        json_model_i[itype] = torch.tensor(json_dict[ikey][itype],dtype=dtype,device=device)
                    json_model_list[ikey] = json_model_i

            assert 'onsite' in json_model_list and 'hopping' in json_model_list, "onsite and hopping must be in json_model_list"
            if 'overlap' in json_model_list:
                for ikey in json_model_list['hopping']:
                    json_model_list['hopping'][ikey] = torch.cat((json_model_list['hopping'][ikey],json_model_list['overlap'][ikey]),dim=0)
                json_model_list.pop('overlap')
        else:
            raise NotImplementedError("modeltype {} not implemented".format(modeltype))

        # ----------------------------------------------------------------------------------------------------------
        # load params from model_config and make sure the key param doesn't conflict
        # ----------------------------------------------------------------------------------------------------------
        num_hopping_hidden = common_and_model_and_run_options['sknetwork']['sk_hop_nhidden']
        num_onsite_hidden = common_and_model_and_run_options['sknetwork']['sk_onsite_nhidden']
        num_soc_hidden = common_and_model_and_run_options['sknetwork']['sk_soc_nhidden']
        unit = common_and_model_and_run_options["unit"]

        # ----------------------------------------------------------------------------------------------------------
        if onsitemode == 'NRL':
            onsite_func_cutoff = common_and_model_and_run_options['onsitefuncion']['onsite_func_cutoff']
            onsite_func_decay_w = common_and_model_and_run_options['onsitefuncion']['onsite_func_decay_w']
            onsite_func_lambda = common_and_model_and_run_options['onsitefuncion']['onsite_func_lambda']
        
        
        overlap = common_and_model_and_run_options.get('overlap',False)

        #-----------------------------------------------------------------------------------------------------------

        if soc and num_soc_hidden is None:
            log.err(msg="Please specify the number of hidden layers for soc network. please set the key `sk_soc_nhidden` in `sknetwork` in `model_options`.")
            raise ValueError
        
        if modeltype == "ckpt":
            for ckpt in ckpt_list:
                model_config = ckpt["model_config"]
                checkdict(
                    dict_prototype=common_and_model_and_run_options,
                    dict_update=model_config,
                    checklist=nnsk_model_config_checklist
                    )

                num_hopping_hidden = max(num_hopping_hidden, model_config['sknetwork']['sk_hop_nhidden'])
                num_onsite_hidden = max(num_onsite_hidden ,model_config['sknetwork']['sk_onsite_nhidden'])
                if soc:
                    if  model_config.get('soc',False):
                        num_soc_hidden = max(num_soc_hidden ,model_config['sknetwork']['sk_soc_nhidden'])
                    else:
                        log.warning('Warning, the model is non-soc. Transferring it into soc case.')
                        
                else:
                    if 'soc' in model_config.keys() and model_config['soc']:
                        log.warning('Warning, the model is with soc, but this run job soc is turned off. Transferring it into non-soc case.')
        
            # update common_and_model_and_run_options
            # ----------------------------------------------------------------------------------------------------------
            common_and_model_and_run_options = update_dict_with_warning(
                dict_input=common_and_model_and_run_options,
                update_list=nnsk_model_config_updatelist,
                update_value=[num_hopping_hidden, num_onsite_hidden, num_soc_hidden]
                )
        
        # computing onsite/hopping/soc types to init model
        IndMap = Index_Mapings()
        IndMap.update(proj_atom_anglr_m=proj_atom_anglr_m)
        bond_index_map, bond_num_hops = IndMap.Bond_Ind_Mapings()
        onsite_strain_index_map, onsite_strain_num, onsite_index_map, onsite_num = \
                IndMap.Onsite_Ind_Mapings(onsitemode, atomtype=atomtype)

        # onsite_fun = onsiteFunc
        hops_fun = SKintHops(mode='hopping',functype=skformula,proj_atom_anglr_m=proj_atom_anglr_m)
        if overlap:
            overlap_fun = SKintHops(mode='hopping',functype=skformula,proj_atom_anglr_m=proj_atom_anglr_m,overlap=overlap)
        if soc:
            soc_fun = socFunc
        if onsitemode == 'strain':
            onsitestrain_fun = SKintHops(mode='onsite', functype=skformula,proj_atom_anglr_m=proj_atom_anglr_m, atomtype=atomtype)
            onsite_fun = orbitalEs(proj_atom_anglr_m=proj_atom_anglr_m,atomtype=atomtype,functype='none',unit=unit)
        elif onsitemode == 'NRL':
            onsite_fun = orbitalEs(proj_atom_anglr_m=proj_atom_anglr_m,atomtype=atomtype,functype=onsitemode,unit=unit,
                                   onsite_func_cutoff=onsite_func_cutoff,onsite_func_decay_w=onsite_func_decay_w,onsite_func_lambda=onsite_func_lambda)
        else:
            onsite_fun = orbitalEs(proj_atom_anglr_m=proj_atom_anglr_m,atomtype=atomtype,functype=onsitemode,unit=unit)

            
        _, reducted_skint_types, _ = all_skint_types(bond_index_map)
        _, reduced_onsiteE_types, onsiteE_ind_dict = all_onsite_ene_types(onsite_index_map)
        if overlap:
            hopping_neurons = {"nhidden": num_hopping_hidden,  "nout": hops_fun.num_paras, "nout_overlap": overlap_fun.num_paras}
        else:
            hopping_neurons = {"nhidden": num_hopping_hidden,  "nout": hops_fun.num_paras}
        if onsitemode == 'strain':
            onsite_neurons = {"nhidden":num_onsite_hidden,"nout":onsitestrain_fun.num_paras}
            _, reducted_onsiteint_types, _ = all_onsite_intgrl_types(onsite_strain_index_map)
            onsite_types = reducted_onsiteint_types
        else:
            # onsite_neurons = {"nhidden":num_onsite_hidden}
            onsite_neurons = {"nhidden":num_onsite_hidden, "nout": onsite_fun.num_paras}
            onsite_types = reduced_onsiteE_types

        if soc:
            soc_neurons = {"nhidden":num_soc_hidden}
        else:
            soc_neurons = None

        model = SKNet(skint_types=reducted_skint_types,
                                   onsite_types=onsite_types,
                                   soc_types=reduced_onsiteE_types,
                                   hopping_neurons=hopping_neurons,
                                   onsite_neurons=onsite_neurons,
                                   soc_neurons=soc_neurons,
                                   device=device,
                                   dtype=dtype,
                                   onsitemode=onsitemode,
                                   # Onsiteint_types is a list of onsite integral types, which is used
                                   # to determine the number of output neurons of the onsite network.
                                   onsite_index_dict=onsiteE_ind_dict,
                                   overlap=overlap
                                   )
    
        if modeltype == 'ckpt':
            self.host.model = init_from_model_(SKNet=model, checkpoint_list=ckpt_list, interpolate=interpolate)
        elif modeltype == 'json':
            self.host.model = init_from_json_(SKNet=model, json_model=json_model_list)
        else:
            raise NotImplementedError("modeltype {} not implemented".format(modeltype))

        

        self.host.onsite_fun = onsite_fun
        self.host.hops_fun = hops_fun
        #self.host.onsite_index_map = onsite_index_map
        #self.host.onsite_db = loadOnsite(onsite_index_map, unit=unit)
        self.host.overlap = overlap

        if overlap:
            self.host.overlap_fun = overlap_fun
        if soc:
            self.host.soc_fun = soc_fun
            self.host.soc_db = loadSoc(onsite_index_map)
        if onsitemode == 'strain':
            self.host.onsitestrain_fun = onsitestrain_fun
        
        # model_config.update(common_and_model_and_run_options)
        # self.host.model_config = update_dict(temp_dict=model_config, update_dict=common_and_model_and_run_options, checklist=nnsk_model_config_checklist)
        
        # self.host.model.load_state_dict(state_dict)            
        self.host.model.train()
        self.host.model_config = common_and_model_and_run_options
        self.host.model_config.update({"types_list": [self.host.model.onsite_types, self.host.model.skint_types, self.host.model.soc_types]})

        if common_and_model_and_run_options["freeze"]:
            for k,v in self.host.model.named_parameters():
                if "onsite" in k:
                    v.requires_grad = False

        if common_and_model_and_run_options["train_soc"]:
            for k,v in self.host.model.named_parameters():
                if "soc" not in k:
                    v.requires_grad = False