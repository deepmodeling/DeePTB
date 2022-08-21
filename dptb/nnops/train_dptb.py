import torch
import heapq
import logging
import numpy as np
from dptb.nnet.nntb import NNTB
from dptb.sktb.skIntegrals import SKIntegrals
from dptb.sktb.struct_skhs import SKHSLists
from dptb.hamiltonian.hamil_eig_sk import HamilEig
from dptb.dataprocess.processor import Processor
from dptb.nnsktb.sknet import SKNet
from dptb.nnsktb.integralFunc import SKintHops
from dptb.nnsktb.onsiteFunc import onsiteFunc, loadOnsite
from dptb.nnsktb.skintTypes import all_skint_types
from dptb.dataprocess.datareader import read_data
from dptb.nnops.loss import loss_type1, loss_spectral
from dptb.utils.tools import get_uniq_symbol,  Index_Mapings, \
    get_lr_scheduler, get_uniq_bond_type, get_uniq_env_bond_type, \
    get_env_neuron_config, get_bond_neuron_config, get_onsite_neuron_config, \
    get_optimizer, nnsk_correction, j_must_have
from dptb.nnops.trainer import Trainer

log = logging.getLogger(__name__)

class DPTBTrainer(Trainer):

    def __init__(self, run_opt, jdata) -> None:
        super(DPTBTrainer, self).__init__(jdata)
        self.name = "dptb"
        self.run_opt = run_opt
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
        self.use_reference = train_options.get('use_reference', False)

        # initialize data options
        # ----------------------------------------------------------------------------------------------------------------------------------------------
        self.require_dict = False
        self.batch_size = data_options.get('batch_size')
        self.test_batch_size = data_options.get('test_batch_size', self.batch_size)
        self.ref_batch_size = data_options.get('ref_batch_size', 1)

        self.sk_file_path = data_options.get('sk_file_path')
        self.bond_cutoff = data_options.get('bond_cutoff')
        self.env_cutoff = data_options.get('env_cutoff')
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

        if self.use_reference:
            self.ref_data_path = data_options.get('ref_data_path')
            self.ref_data_prefix = data_options.get('ref_data_prefix')

            self.ref_band_min = data_options.get('ref_band_min', 0)
            self.ref_band_max = data_options.get('ref_band_max', None)

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
                Processor(mode='dptb', structure_list=struct_list_sets[i], batchsize=self.batch_size, env_cutoff=self.env_cutoff,
                          kpoint=kpoints_sets[i], eigen_list=eigens_sets[i], device=self.device, dtype=self.dtype, require_dict=self.require_dict))

        # init reference dataset
        # -----------------------------------init reference set------------------------------------------
        if self.use_reference:
            struct_list_sets, kpoints_sets, eigens_sets = read_data(self.ref_data_path, self.ref_data_prefix,
                                                                          self.bond_cutoff, self.proj_atom_anglr_m,
                                                                          self.proj_atom_neles,
                                                                          self.time_symm)
            self.n_ref_sets = len(struct_list_sets)
            assert self.n_ref_sets == len(kpoints_sets) == len(eigens_sets)
            self.ref_processor_list = []
            for i in range(len(struct_list_sets)):
                self.ref_processor_list.append(
                    Processor(mode='dptb', structure_list=struct_list_sets[i], batchsize=self.ref_batch_size, env_cutoff=self.env_cutoff,
                              kpoint=kpoints_sets[i], eigen_list=eigens_sets[i], device=self.device, dtype=self.dtype, require_dict=self.require_dict))

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
                Processor(mode='dptb', structure_list=struct_list_sets[i], batchsize=self.test_batch_size, env_cutoff=self.env_cutoff,
                          kpoint=kpoints_sets[i], eigen_list=eigens_sets[i], device=self.device, dtype=self.dtype, require_dict=self.require_dict))

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
        self.IndMap.update(proj_atom_anglr_m=self.proj_atom_anglr_m)
        self.bond_index_map, self.bond_num_hops = self.IndMap.Bond_Ind_Mapings()
        self.onsite_index_map, self.onsite_num = self.IndMap.Onsite_Ind_Mapings()
        self.bond_type = get_uniq_bond_type(proj_atom_type)

        # # ------------------------------------initialize model options----------------------------------
        self._init_model()

        self.skint = SKIntegrals(proj_atom_anglr_m=self.proj_atom_anglr_m, sk_file_path=self.sk_file_path)
        self.skhslist = SKHSLists(self.skint, dtype='tensor')
        self.hamileig = HamilEig(dtype='tensor')

        self.optimizer = get_optimizer(model_param=self.model.parameters(), **opt_options)
        self.lr_scheduler = get_lr_scheduler(optimizer=self.optimizer, **sch_options)  # add optmizer


        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.emin = self.model_options["emin"]
        self.emax = self.model_options["emax"]

    def _init_model(self):
        '''
        initialize the model, the following things need to be taken into account:
        -1- whether to load model from checkpoint or init model from scratch
            -1.1- if init from checkpoint, do we need to frozen the parameter ?
        -2- whether to init nnsktb model for correction

        Parameters
        ----------

        Returns
        -------
        '''

        mode = self.run_opt.get("mode", None)
        if mode is None:
            mode = 'from_scratch'
            log.info(msg="Haven't assign a initializing mode, training from scratch as default.")

        # neural list for env onsite and bond.
        if mode == "from_scratch":
            env_nnl = self.model_options['env_net_neuron']
            env_axisnn = self.model_options['axis_neuron']
            onsite_nnl = self.model_options['onsite_net_neuron']
            bond_nnl = self.model_options['bond_net_neuron']
            env_net_config = get_env_neuron_config(env_nnl)
            onsite_net_config = get_onsite_neuron_config(onsite_nnl, self.onsite_num, self.proj_atom_type, env_axisnn,
                                                         env_nnl[-1])
            bond_net_config = get_bond_neuron_config(bond_nnl, self.bond_num_hops, self.bond_type, env_axisnn,
                                                     env_nnl[-1])
            def pack(**options):
                return options

            self.model_config = pack(
                atom_type=self.atom_type,
                proj_atom_type=self.proj_atom_type,
                axis_neuron=self.model_options['axis_neuron'],
                env_net_config=env_net_config,
                onsite_net_config=onsite_net_config,
                bond_net_config=bond_net_config,
                onsite_net_activation=self.model_options['onsite_net_activation'],
                env_net_activation=self.model_options['env_net_activation'],
                bond_net_activation=self.model_options['bond_net_activation'],
                onsite_net_type=self.model_options['onsite_net_type'],
                env_net_type=self.model_options['env_net_type'],
                bond_net_type=self.model_options['bond_net_type'],
                if_batch_normalized=self.model_options['if_batch_normalized'],
                device=self.device,
                dtype=self.dtype
            )
            self.nntb = NNTB(**self.model_config)
            self.model = self.nntb.tb_net
        elif mode == "init_model":
            # read configuration from checkpoint path.
            f = torch.load(self.run_opt["init_model"])
            self.model_config = f["model_config"]
            for kk in self.model_config:
                if self.model_options.get(kk) is not None and self.model_options.get(kk) != self.model_config.get(kk):
                    log.warning(msg="The configure in checkpoint is mismatch with the input configuration {}, init from checkpoint temporarily\n, ".format(kk) +
                                    "but this might cause conflict.")
                    break

            self.nntb = NNTB(**self.model_config)
            self.model = self.nntb.tb_net
            self.model.load_state_dict(f['state_dict'])
            self.model.train()

        else:
            raise RuntimeError("init_mode should be from_scratch/from_model/..., not {}".format(mode))

        if self.run_opt["use_correction"]:
            all_skint_types_dict, reducted_skint_types, self.sk_bond_ind_dict = all_skint_types(self.bond_index_map)
            f = torch.load(self.run_opt["skcheckpoint_path"])
            model_config = f["model_config"]
            for kk in self.model_config:
                if self.model_options.get(kk) is not None and self.model_options.get(kk) != self.model_config.get(kk):
                    log.warning(
                        msg="The configure in checkpoint is mismatch with the input configuration {}, init from checkpoint temporarily\n, ".format(
                            kk) +
                            "but this might cause conflict.")
                    break
            self.sknet = SKNet(**model_config)
            self.sknet.load_state_dict(f['state_dict'])
            self.sknet.train()
            if self.run_opt['freeze']:
                self.sknet.eval()
                for p in self.sknet.parameters():
                    p.requires_grad = False
            self.hops_fun = SKintHops()
            self.onsite_fun = onsiteFunc
            self.onsite_db = loadOnsite(self.onsite_index_map)

    def calc(self, batch_bond, batch_bond_onsite, batch_env, structs, kpoints):
        '''
        conduct one step forward computation, used in train, test and validation.
        '''

        assert len(kpoints.shape) == 2, "kpoints should have shape of [num_kp, 3]."

        batch_bond_hoppings, batch_hoppings, \
        batch_bond_onsites, batch_onsiteEs = self.nntb.calc(batch_bond, batch_env)

        if self.run_opt.get("use_correction", False):
            coeffdict = self.sknet()
            sktb_onsiteEs = self.onsite_fun(batch_bond_onsites, self.onsite_db)
            sktb_hoppings = self.hops_fun.get_skhops(batch_bond_hoppings, coeffdict, self.sk_bond_ind_dict)

        # call sktb to get the sktb hoppings and onsites
        eigenvalues_pred = []
        for ii in range(len(structs)):
            if not self.run_opt.get("use_correction", False):
                self.skhslist.update_struct(structs[ii])
                self.skhslist.get_HS_list(bonds_onsite=np.asarray(batch_bond_onsites[ii][:,1:]),
                                          bonds_hoppings=np.asarray(batch_bond_hoppings[ii][:,1:]))
                onsiteEs, hoppings, onsiteSs, overlaps = \
                    nnsk_correction(nn_onsiteEs=batch_onsiteEs[ii], nn_hoppings=batch_hoppings[ii],
                                    sk_onsiteEs=self.skhslist.onsiteEs, sk_hoppings=self.skhslist.hoppings,
                                    sk_onsiteSs=self.skhslist.onsiteSs, sk_overlaps=self.skhslist.overlaps)
            else:
                onsiteEs, hoppings, onsiteSs, overlaps = \
                    nnsk_correction(nn_onsiteEs=batch_onsiteEs[ii], nn_hoppings=batch_hoppings[ii],
                                    sk_onsiteEs=sktb_onsiteEs[ii], sk_hoppings=sktb_hoppings[ii],
                                    sk_onsiteSs=None, sk_overlaps=None)
            # call hamiltonian block
            self.hamileig.update_hs_list(struct=structs[ii], hoppings=hoppings, onsiteEs=onsiteEs, overlaps=overlaps,
                                         onsiteSs=onsiteSs)
            self.hamileig.get_hs_blocks(bonds_onsite=np.asarray(batch_bond_onsites[ii][:,1:]),
                                        bonds_hoppings=np.asarray(batch_bond_hoppings[ii][:,1:]))
            eigenvalues_ii = self.hamileig.Eigenvalues(kpoints=kpoints, time_symm=self.time_symm, dtype='tensor')
            eigenvalues_pred.append(eigenvalues_ii)
        eigenvalues_pred = torch.stack(eigenvalues_pred)

        return eigenvalues_pred


    def train(self) -> None:

        data_set_seq = np.random.choice(self.n_train_sets, size=self.n_train_sets, replace=False)
        for iset in data_set_seq:
            processor = self.train_processor_list[iset]
            # iter with different structure
            for data in processor:
                # iter with samples from the same structure


                def closure():
                    # calculate eigenvalues.
                    self.optimizer.zero_grad()
                    batch_bond, batch_bond_onsite, batch_env, structs, kpoints, eigenvalues = data[0], data[1], data[2], \
                                                                                              data[3], data[4], data[5]
                    eigenvalues_pred = self.calc(batch_bond, batch_bond_onsite, batch_env, structs, kpoints)
                    eigenvalues_lbl = torch.from_numpy(eigenvalues.astype(float)).float()

                    num_kp = kpoints.shape[0]
                    num_el = np.sum(structs[0].proj_atom_neles_per)

                    if self.use_reference:
                        ref_eig = []
                        ref_kp_el = []

                        for irefset in range(self.n_ref_sets):
                            ref_processor = self.ref_processor_list[irefset]
                            for refdata in ref_processor:
                                batch_bond, _, batch_env, structs, kpoints, eigenvalues = refdata[0], refdata[1], \
                                                                                          refdata[2], refdata[3], \
                                                                                          refdata[4], refdata[5]
                                ref_eig_pred = self.calc(batch_bond, batch_bond_onsite, batch_env, structs, kpoints)
                                ref_eig_lbl = torch.from_numpy(eigenvalues.astype(float)).float()
                                num_kp_ref = kpoints.shape[0]
                                num_el_ref = np.sum(structs[0].proj_atom_neles_per)

                                ref_eig.append([ref_eig_pred, ref_eig_lbl])
                                ref_kp_el.append([num_kp_ref, num_el_ref])

                    loss = loss_type1(self.criterion, eigenvalues_pred, eigenvalues_lbl, num_el, num_kp, self.band_min, self.band_max)
                    if self.use_reference:
                        for irefset in range(self.n_ref_sets):
                            ref_eig_pred, ref_eig_lbl = ref_eig[irefset]
                            num_kp_ref, num_el_ref = ref_kp_el[irefset]
                            loss += loss_type1(self.criterion, ref_eig_pred, ref_eig_lbl, num_el_ref, num_kp_ref, self.ref_band_min, self.ref_band_max)
                            loss += loss_spectral(self.criterion, eigenvalues_pred, eigenvalues_lbl, self.emin,
                                                  self.emax)
                    loss.backward()

                    self.train_loss = loss
                    return loss

                self.optimizer.step(closure)
                state = {'field':'iteration', "train_loss": self.train_loss, "lr": self.optimizer.state_dict()["param_groups"][0]['lr']}

                self.call_plugins(queue_name='iteration', time=self.iteration, **state)
                # self.lr_scheduler.step() # 在epoch 加入 scheduler.


                self.iteration += 1

    def validation(self, quick=False):

        with torch.no_grad():
            total_loss = torch.scalar_tensor(0., dtype=self.dtype, device=self.device)
            for processor in self.test_processor_list:
                for data in processor:
                    batch_bond, batch_bond_onsite, batch_env, structs, kpoints, eigenvalues = data[0],data[1],data[2], data[3], data[4], data[5]
                    eigenvalues_pred = self.calc(batch_bond, batch_bond_onsite, batch_env, structs, kpoints)
                    eigenvalues_lbl = torch.from_numpy(eigenvalues.astype(float)).float()

                    num_kp = kpoints.shape[0]
                    num_el = np.sum(structs[0].proj_atom_neles_per)

                    total_loss += loss_type1(self.criterion, eigenvalues_pred, eigenvalues_lbl, num_el, num_kp, self.band_min,
                                      self.band_max)
                    if quick:
                        break

            return total_loss





if __name__ == '__main__':
    a = [1,2,3]

    print(list(enumerate(a, 2)))