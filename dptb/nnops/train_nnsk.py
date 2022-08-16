import torch
from dptb.nnops.trainer import Trainer
from dptb.utils.tools import get_uniq_symbol,  Index_Mapings, \
    get_lr_scheduler, get_uniq_bond_type, get_uniq_env_bond_type, \
    get_env_neuron_config, get_bond_neuron_config, get_onsite_neuron_config, \
    get_optimizer, nnsk_correction, j_must_have
from dptb.hamiltonian.hamil_eig_sk import HamilEig
from dptb.dataprocess.processor import Processor
from dptb.dataprocess.datareader import read_data
from dptb.nnsktb.skintTypes import all_skint_types
from dptb.nnsktb.sknet import SKNet
import logging

log = logging.getLogger(__name__)

class NNSKTrainer(Trainer):
    def __init__(self, run_opt, jdata) -> None:
        super(NNSKTrainer, self).__init__(jdata)
        self._init_param(jdata)

    def _init_param(self, jdata):
        train_options = j_must_have(jdata, "train_options")
        opt_options = j_must_have(jdata, "optimizer_options")
        sch_options = j_must_have(jdata, "sch_options")
        data_options = j_must_have(jdata,"data_options")
        model_options = j_must_have(jdata, "model_options")

        self.train_options = train_options
        self.opt_options = opt_options
        self.sch_options = sch_options
        self.data_options = data_options
        self.model_options = model_options

        self.num_epoch = train_options.get('num_epoch')
        self.display_epoch = train_options.get('display_epoch')

        # initialize data options
        # ----------------------------------------------------------------------------------------------------------------------------------------------
        self.batch_size = data_options.get('batch_size')
        self.test_batch_size = data_options.get('test_batch_size', self.batch_size)

        self.train_data_path = data_options.get('train_data_path')
        self.train_data_prefix = data_options.get('train_data_prefix')
        self.test_data_path = data_options.get('test_data_path')
        self.test_data_prefix = data_options.get('test_data_prefix')
        self.proj_atom_anglr_m = data_options.get('proj_atom_anglr_m')
        self.proj_atom_neles = data_options.get('proj_atom_neles')

        if data_options['time_symm'] is True:
            self.time_symm = True
        else:
            self.time_symm = False

        self.band_min = data_options.get('band_min', 0)
        self.band_max = data_options.get('band_max', None)

        # init the dataset
        # -----------------------------------init training set------------------------------------------
        struct_list_sets, kpoints_sets, eigens_sets = read_data(self.train_data_path, self.train_data_prefix,
                                                                      self.bond_cutoff, self.proj_atom_anglr_m,
                                                                      self.proj_atom_neles,
                                                                      self.time_symm)
        self.n_train_sets = len(struct_list_sets)
        assert self.n_train_sets == len(kpoints_sets) == len(eigens_sets)

        self.train_processor_list = []
        for i in range(len(struct_list_sets)):
            self.train_processor_list.append(
                Processor(struct_list_sets[i], batchsize=self.batch_size, env_cutoff=self.env_cutoff,
                          kpoint=kpoints_sets[i], eigen_list=eigens_sets[i], device=self.device, dtype=self.dtype))
        # --------------------------------init testing set----------------------------------------------
        struct_list_sets, kpoints_sets, eigens_sets = read_data(self.test_data_path,
                                                                      self.test_data_prefix,
                                                                      self.bond_cutoff, self.proj_atom_anglr_m,
                                                                      self.proj_atom_neles,
                                                                      self.time_symm)
        self.n_test_sets = len(struct_list_sets)
        assert self.n_test_sets == len(kpoints_sets) == len(eigens_sets)

        self.test_processor_list = []
        for i in range(len(struct_list_sets)):
            self.test_processor_list.append(
                Processor(struct_list_sets[i], batchsize=self.test_batch_size, env_cutoff=self.env_cutoff,
                          kpoint=kpoints_sets[i], eigen_list=eigens_sets[i], device=self.device, dtype=self.dtype))

        # ---------------------------------init index map------------------------------------------------
        # since training and testing set contains same atom type and proj_atom type, we may expect the maps are the same in train and test.
        atom_type = []
        proj_atom_type = []
        for ips in self.train_processor_list:
            atom_type += ips.atom_type
            proj_atom_type += ips.proj_atom_type
        self.atom_type = get_uniq_symbol(list(set(atom_type)))
        self.proj_atom_type = get_uniq_symbol(list(set(proj_atom_type)))
        self.IndMap = Index_Mapings()
        self.IndMap.update(envtype=self.atom_type, bondtype=self.proj_atom_type,
                           proj_atom_anglr_m=self.proj_atom_anglr_m)
        self.bond_index_map, self.bond_num_hops = self.IndMap.Bond_Ind_Mapings()
        self.onsite_index_map, self.onsite_num = self.IndMap.Onsite_Ind_Mapings()
        self.bond_type = get_uniq_bond_type(proj_atom_type)

        # # ------------------------------------initialize model options----------------------------------
        self._init_model()
        self.hamileig = HamilEig(dtype='tensor')

        self.optimizer = get_optimizer(model_param=self.nntb.tb_net.parameters(), **opt_options)
        self.lr_scheduler = get_lr_scheduler(optimizer=self.optimizer, **sch_options)  # add optmizer

        self.criterion = torch.nn.MSELoss(reduction='mean')

    def _init_model(self):
        mode = self.train_options.get("mode", "from_scratch")
        if mode is None:
            mode = 'from_scratch'
            log.info(msg="Haven't assign a initializing mode, training from scratch as default.")

        if mode == "from_scratch":
            all_skint_types_dict, reducted_skint_types, sk_bond_ind_dict = all_skint_types(self.bond_index_map)
            self.model_config = self.model_options.update({"skint_types":reducted_skint_types, "device":self.device, "dtype":self.dtype})
            self.model = SKNet(skint_types=reducted_skint_types, device=self.device, dtype=self.dtype, **self.model_options)
        elif mode == "from_model":
            # read configuration from checkpoint path.
            f = torch.load(self.train_options["init_path"])
            self.model_config = f["model_config"]
            for kk in self.model_config:
                if self.model_options.get(kk) is not None and self.model_options.get(kk) != self.model_config.get(kk):
                    log.warning(msg="The configure in checkpoint is mismatch with the input configuration {}, init from checkpoint temporarily\n, ".format(kk) +
                                    "but this might cause conflict.")
                    break
            self.model = SKNet(**self.model_config)
            self.model.load_state_dict(f['state_dict'])
            self.model.eval()

        else:
            raise RuntimeError("init_mode should be from_scratch/from_model/..., not {}".format(mode))


    def calc(self, batch_bond, batch_env, structs, kpoints):
        assert len(kpoints.shape) == 2, "kpoints should have shape of [num_kp, 3]."
        self.model()



        pass

    def train(self) -> None:
        pass

    def validation(self, **kwargs):
        pass