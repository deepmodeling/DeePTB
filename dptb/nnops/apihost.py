import logging
import torch
from dptb.utils.tools import get_uniq_bond_type,  j_must_have, j_loader
from dptb.utils.index_mapping import Index_Mapings
from dptb.nnsktb.integralFunc import SKintHops
from dptb.utils.argcheck import normalize, host_normalize
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
        init_type = checkpoint.split(".")[-1]
        if init_type == "json":
            jdata = j_loader(checkpoint)
            jdata = host_normalize(jdata)
            #self.call_plugins(queue_name='disposable', time=0, **self.model_options, **self.common_options, **self.data_options, **self.run_opt)

            common_options = j_must_have(jdata, "common_options")
            # data_options = j_must_have(jdata,"data_options")
            model_options = j_must_have(jdata, "model_options")
            init_opts = j_must_have(jdata, "init_model")
            run_opt = {
                "init_model": init_opts,
                "freeze": False,
                "train_soc": False}
            model_config={}
            model_config.update(common_options)
            # model_config.update(data_options)
            model_config.update(model_options)
            model_config.update(run_opt)


        else:
            ckpt = torch.load(checkpoint)
            model_config = ckpt["model_config"]
            model_config.update({"init_model": {"path": checkpoint,"interpolate": False}})
        model_config["dtype"] = dtype_dict[model_config["dtype"]]
        
        self.__init_params(**model_config)

    def __init_params(self, **model_config):
        self.model_config = model_config        
        
    def build(self):
        if not 'soc' in self.model_config.keys():
            self.model_config.update({'soc':False})
        # ---------------------------       init network model        -----------------------
        self.call_plugins(queue_name='disposable', time=0, mode='init_model', **self.model_config)
        