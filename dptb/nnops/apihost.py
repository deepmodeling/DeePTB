import logging
import torch
from dptb.utils.tools import get_uniq_bond_type,  j_must_have
from dptb.utils.index_mapping import Index_Mapings
from dptb.nnsktb.integralFunc import SKintHops
from dptb.utils.constants import dtype_dict
from dptb.nnsktb.onsiteFunc import onsiteFunc, loadOnsite
from dptb.plugins.base_plugin import PluginUser

log = logging.getLogger(__name__)

# TODO: add a entrypoints for api.
# TODO: 优化structure的传入方式。

class DPTBHost(PluginUser):
    def __init__(self, dptbmodel, use_correction=False):
        super(DPTBHost, self).__init__()
        ckpt = torch.load(dptbmodel)
        model_config = ckpt["model_config"]
        model_config["dtype"] = dtype_dict[model_config["dtype"]]
        model_config.update({'init_model':dptbmodel,'use_correction':use_correction})
        self.use_correction = use_correction
        self.__init_params(**model_config)
    
    def __init_params(self, **model_config):
        self.model_config = model_config      

    
    def build(self):
        if not 'soc' in self.model_config.keys():
            self.model_config.update({'soc':False})
        self.call_plugins(queue_name='disposable', time=0, mode='init_model', **self.model_config)
        self.model_config.update({'use_correction':self.use_correction})

class NNSKHost(PluginUser):
    def __init__(self, checkpoint):
        super(NNSKHost, self).__init__()
        ckpt = torch.load(checkpoint)
        model_config = ckpt["model_config"]
        model_config["dtype"] = dtype_dict[model_config["dtype"]]
        model_config.update({"init_model": {"path": checkpoint,"interpolate": False}})
        self.__init_params(**model_config)

    def __init_params(self, **model_config):
        self.model_config = model_config        
        
    def build(self):
        if not 'soc' in self.model_config.keys():
            self.model_config.update({'soc':False})
        # ---------------------------       init network model        -----------------------
        self.call_plugins(queue_name='disposable', time=0, mode='init_model', **self.model_config)
        