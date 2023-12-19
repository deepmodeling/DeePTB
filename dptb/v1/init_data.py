from dptb.plugins.base_plugin import Plugin
import logging
from dptb.nnsktb.sknet import SKNet
from dptb.nnsktb.skintTypes import all_skint_types, all_onsite_intgrl_types
from dptb.utils.index_mapping import Index_Mapings
from dptb.utils.tools import get_uniq_symbol
from dptb.nnsktb.integralFunc import SKintHops
from dptb.nnsktb.loadparas import load_paras
from dptb.dataprocess.datareader import get_data
from dptb.utils.constants import dtype_dict


log = logging.getLogger(__name__)

class InitData(Plugin):
    def __init__(self, interval=None):
        if interval is None:
            interval = [(-2, 'disposable')]
        super(InitData, self).__init__(interval)
        
    def register(self, host):
        self.host = host

    def disposable(self, **common_and_data_options):
        # ----------------------------------------------------------------------------------------------------------
        self.use_reference = common_and_data_options['use_reference']
        # ----------------------------------------------------------------------------------------------------------
        
        self.host.train_processor_list = get_data(
            sorted_onsite="st", 
            sorted_bond="st", 
            sorted_env="itype-jtype",
            **common_and_data_options,
            **common_and_data_options["train"]
        )

        self.host.validation_processor_list = get_data(
            sorted_onsite="st", 
            sorted_bond="st", 
            sorted_env="itype-jtype",
            **common_and_data_options,
            **common_and_data_options["validation"]
        )


        if self.use_reference:
            self.host.ref_processor_list = get_data(
            sorted_onsite="st", 
            sorted_bond="st", 
            sorted_env="itype-jtype", 
            **common_and_data_options["reference"],
            **common_and_data_options
            )

        self.data_stats()

    def data_stats(self):
        self.host.use_reference = self.use_reference
        self.host.n_train_sets = len(self.host.train_processor_list)
        self.host.n_validation_sets = len(self.host.validation_processor_list)
        if self.host.use_reference:
            self.host.n_reference_sets = len(self.host.ref_processor_list)
        
        atomtype, proj_atomtype = [], []
        for ips in self.host.train_processor_list:
            atomtype += ips.atomtype
            proj_atomtype += ips.proj_atomtype
        
        atomtype = get_uniq_symbol(list(set(atomtype)))
        proj_atomtype = get_uniq_symbol(list(set(proj_atomtype)))

        assert atomtype == self.host.atomtype
        assert proj_atomtype == self.host.proj_atomtype

class InitTestData(Plugin):
    def __init__(self, interval=None):
        if interval is None:
            interval = [(-2, 'disposable')]
        super(InitTestData, self).__init__(interval)
        
    def register(self, host):
        self.host = host

    def disposable(self, **common_and_data_options):
        self.host.test_processor_list = get_data(
            sorted_onsite="st", 
            sorted_bond="st", 
            sorted_env="itype-jtype",
            if_shuffle = False,
            **common_and_data_options,
            **common_and_data_options["test"]
        )

        self.data_stats()

    def data_stats(self):
        self.host.n_test_sets = len(self.host.test_processor_list)
        
        atomtype, proj_atomtype = [], []
        for ips in self.host.test_processor_list:
            atomtype += ips.atomtype
            proj_atomtype += ips.proj_atomtype
        
        atomtype = get_uniq_symbol(list(set(atomtype)))
        proj_atomtype = get_uniq_symbol(list(set(proj_atomtype)))

        assert atomtype == self.host.atomtype
        assert proj_atomtype == self.host.proj_atomtype