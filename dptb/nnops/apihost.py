import logging
import torch
from dptb.utils.tools import get_uniq_bond_type,  j_must_have
from dptb.utils.index_mapping import Index_Mapings
from dptb.nnsktb.integralFunc import SKintHops
from dptb.nnsktb.onsiteFunc import onsiteFunc, loadOnsite
from dptb.plugins.base_plugin import PluginUser

log = logging.getLogger(__name__)

# TODO: add a entrypoints for api.
# TODO: 优化structure的传入方式。

class NNSKHost(PluginUser):
    def __init__(self, checkpoint):
        super(NNSKHost, self).__init__()
        ckpt = torch.load(checkpoint)
        model_config = ckpt["model_config"]
        model_config.update({'init_model':checkpoint})
        self.__init_params(**model_config)

    def __init_params(self, **model_config):
        self.model_config = model_config
        self.proj_atom_anglr_m = model_config['proj_atom_anglr_m']
        self.onsitemode = model_config['onsitemode']
        self.atomtype = model_config['atomtype']
        self.skformula = model_config["skfunction"]["skformula"]
        self.sk_cutoff = model_config["skfunction"]["sk_cutoff"]
        self.sk_decay_w = model_config["skfunction"]["sk_decay_w"]

    def build(self):
        # ---------------------------------------------------------------- init onsite and hopping functions  ----------------------------------------------------------------
        self.IndMap = Index_Mapings()
        self.IndMap.update(proj_atom_anglr_m=self.proj_atom_anglr_m)
        self.bond_index_map, self.bond_num_hops = self.IndMap.Bond_Ind_Mapings()
        self.onsite_strain_index_map, self.onsite_strain_num, self.onsite_index_map, self.onsite_num = self.IndMap.Onsite_Ind_Mapings(self.onsitemode, atomtype=self.atomtype)

        self.onsite_fun = onsiteFunc
        self.onsite_db = loadOnsite(self.onsite_index_map)
        self.hops_fun = SKintHops(mode='hopping',functype=self.skformula, proj_atom_anglr_m=self.proj_atom_anglr_m)
        if self.onsitemode == 'strain':
            self.onsitestrain_fun = SKintHops(mode='onsite',functype=self.skformula, proj_atom_anglr_m=self.proj_atom_anglr_m, atomtype=self.atomtype)
        
        # ----------------------------------------------------------------         init network model         ----------------------------------------------------------------
        self.call_plugins(queue_name='disposable', time=0, mode='init_model', **self.model_config)
        