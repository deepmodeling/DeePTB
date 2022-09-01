import torch
from dptb.nnops.trainer import Trainer
from dptb.utils.tools import get_uniq_symbol,  Index_Mapings, \
    get_lr_scheduler, get_uniq_bond_type, get_uniq_env_bond_type, \
    get_env_neuron_config, get_bond_neuron_config, get_onsite_neuron_config, \
    get_optimizer, nnsk_correction, j_must_have
from dptb.sktb.struct_skhs import SKHSLists
from dptb.hamiltonian.hamil_eig_sk import HamilEig
from dptb.nnops.loss import loss_type1, loss_spectral, loss_soft_sort, loss_proj_env
from dptb.dataprocess.processor import Processor
from dptb.dataprocess.datareader import read_data
from dptb.nnsktb.skintTypes import all_skint_types
from dptb.nnsktb.sknet import SKNet
from dptb.nnsktb.integralFunc import SKintHops
from dptb.nnsktb.onsiteFunc import onsiteFunc, loadOnsite
import logging
import heapq
import numpy as np

log = logging.getLogger(__name__)

class NNSKTrainer(Trainer):
    def __init__(self, run_opt, jdata) -> None:
        super(NNSKTrainer, self).__init__(jdata)
        self.run_opt = run_opt
        self.name = "nnsk"
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

        self.bond_cutoff = data_options.get('bond_cutoff')
        # self.env_cutoff = data_options.get('env_cutoff')
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

        proj_wei = np.load(self.train_data_path + '/set.0/' + 'project_weights.npy')
        self.projwei_label = torch.from_numpy(proj_wei).float()

        self.train_processor_list = []
        for i in range(len(struct_list_sets)):
            self.train_processor_list.append(
                Processor(mode='nnsk', structure_list=struct_list_sets[i], batchsize=self.batch_size,
                          kpoint=kpoints_sets[i], eigen_list=eigens_sets[i], device=self.device, dtype=self.dtype, require_dict=True))
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
                Processor(mode='nnsk', structure_list=struct_list_sets[i], batchsize=self.test_batch_size, 
                          kpoint=kpoints_sets[i], eigen_list=eigens_sets[i], device=self.device, dtype=self.dtype, require_dict=True))

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

        self.hamileig = HamilEig(dtype='tensor')



        self.hops_fun = SKintHops()
        self.onsite_fun = onsiteFunc
        self.onsite_db = loadOnsite(self.onsite_index_map)
        self._init_model()

        self.optimizer = get_optimizer(model_param=self.model.parameters(), **opt_options)
        self.lr_scheduler = get_lr_scheduler(optimizer=self.optimizer, **sch_options)  # add optmizer

        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.emin = self.model_options["emin"]
        self.emax = self.model_options["emax"]
        self.sigma = self.model_options.get('sigma', 0.1)
        self.num_omega = self.model_options.get('num_omega',None)
        self.sortstrength = self.model_options.get('sortstrength',[0.1,0.1])
        self.sortstrength_epoch = np.linspace(self.sortstrength[0],self.sortstrength[1],self.num_epoch)

    def _init_model(self):
        mode = self.run_opt.get("mode", None)
        if mode is None:
            mode = 'from_scratch'
            log.info(msg="Haven't assign a initializing mode, training from scratch as default.")

        if mode == "from_scratch":
            all_skint_types_dict, reducted_skint_types, self.sk_bond_ind_dict = all_skint_types(self.bond_index_map)
            # self.model_options.update({"skint_types":reducted_skint_types,"nout":self.hops_fun.num_paras, "device":self.device, "dtype":self.dtype})
            self.model_config = self.model_options.copy()
            self.model_config.update({"skint_types":reducted_skint_types,"nout":self.hops_fun.num_paras, "device":self.device, "dtype":self.dtype})
            self.model = SKNet(**self.model_config)
        elif mode == "init_model":
            # read configuration from checkpoint path.
            all_skint_types_dict, reducted_skint_types, self.sk_bond_ind_dict = all_skint_types(self.bond_index_map)
            f = torch.load(self.run_opt["init_model"])
            self.model_config = f["model_config"]
            for kk in self.model_config:
                if self.model_options.get(kk) is not None and self.model_options.get(kk) != self.model_config.get(kk):
                    log.warning(msg="The configure in checkpoint is mismatch with the input configuration {}, init from checkpoint temporarily\n, ".format(kk) +
                                    "but this might cause conflict.")
                    break
            self.model = SKNet(**self.model_config)
            self.model.load_state_dict(f['state_dict'])
            self.model.train()

        else:
            raise RuntimeError("init_mode should be from_scratch/from_model/..., not {}".format(mode))


    def calc(self, batch_bond, batch_bond_onsites, structs, kpoints):
        assert len(kpoints.shape) == 2, "kpoints should have shape of [num_kp, 3]."
        coeffdict = self.model()

        batch_onsiteEs = self.onsite_fun(batch_bond_onsites, self.onsite_db)
        batch_hoppings = self.hops_fun.get_skhops(batch_bond, coeffdict, self.sk_bond_ind_dict)

        # call sktb to get the sktb hoppings and onsites
        eigenvalues_pred = []
        eigenvector_pred = []
        for ii in range(len(structs)):
            onsiteEs, hoppings = batch_onsiteEs[ii], batch_hoppings[ii]
            # call hamiltonian block
            self.hamileig.update_hs_list(struct=structs[ii], hoppings=hoppings, onsiteEs=onsiteEs)
            self.hamileig.get_hs_blocks(bonds_onsite=np.asarray(batch_bond_onsites[ii][:,1:]),
                                        bonds_hoppings=np.asarray(batch_bond[ii][:,1:]))
            eigenvalues_ii, eigvec = self.hamileig.Eigenvalues(kpoints=kpoints, time_symm=self.time_symm, dtype='tensor')
            eigenvalues_pred.append(eigenvalues_ii)
            eigenvector_pred.append(eigvec)
        eigenvalues_pred = torch.stack(eigenvalues_pred)
        eigenvector_pred = torch.stack(eigenvector_pred)


        return eigenvalues_pred, eigenvector_pred
    
    def run(self, epochs=1):
        for q in self.plugin_queues.values():
            '''对四个事件调用序列进行最小堆排序。'''
            heapq.heapify(q)

        for i in range(1, epochs + 1):
            self.sortstrength_current = self.sortstrength_epoch[i-1]
            self.train()
            # run plugins of epoch events.
            self.call_plugins(queue_name='epoch', time=i)
            self.lr_scheduler.step()  # modify the lr at each epoch (should we add it to pluggins so we could record the lr scheduler process?)

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
                    batch_bond, batch_bond_onsites, _, structs, kpoints, eigenvalues = data[0], data[1], data[2], data[
                        3], data[4], data[5]
                    eigenvalues_pred, eigenvector_pred = self.calc(batch_bond, batch_bond_onsites, structs, kpoints)
                    eigenvalues_lbl = torch.from_numpy(eigenvalues.astype(float)).float()

                    num_kp = kpoints.shape[0]
                    num_el = np.sum(structs[0].proj_atom_neles_per)
                    #loss = loss_type1(self.criterion, eigenvalues_pred, eigenvalues_lbl, num_el, num_kp, self.band_min,
                    #                  self.band_max)
                    #loss = loss_spectral(self.criterion, eigenvalues_pred, eigenvalues_lbl, self.emin, self.emax, self.num_omega, self.sigma)

                    loss1 = loss_soft_sort(self.criterion, eigenvalues_pred, eigenvalues_lbl ,num_el,num_kp, self.sortstrength_current, self.band_min, self.band_max)
                    loss = loss_proj_env(self.criterion, eigenvalues_pred, eigenvalues_lbl, eigenvector_pred, self.projwei_label, self.band_min, self.band_max)
                    loss = loss1
                    loss.backward()

                    self.train_loss = loss
                    return loss

                self.optimizer.step(closure)
                #print('sortstrength_current:', self.sortstrength_current)
                state = {'field': 'iteration', "train_loss": self.train_loss,
                         "lr": self.optimizer.state_dict()["param_groups"][0]['lr']}

                self.call_plugins(queue_name='iteration', time=self.iteration, **state)
                # self.lr_scheduler.step() # 在epoch 加入 scheduler.

                self.iteration += 1

    def validation(self, **kwargs):
        with torch.no_grad():
            total_loss = torch.scalar_tensor(0., dtype=self.dtype, device=self.device)
            for processor in self.test_processor_list:
                for data in processor:
                    batch_bond, batch_bond_onsites, _, structs, kpoints, eigenvalues = data[0], data[1], data[2], data[
                        3], data[4], data[5]
                    eigenvalues_pred,eigenvector_pred = self.calc(batch_bond, batch_bond_onsites, structs, kpoints)
                    eigenvalues_lbl = torch.from_numpy(eigenvalues.astype(float)).float()

                    num_kp = kpoints.shape[0]
                    num_el = np.sum(structs[0].proj_atom_neles_per)

                    total_loss += loss_type1(self.criterion, eigenvalues_pred, eigenvalues_lbl, num_el, num_kp,
                                             self.band_min,
                                             self.band_max)
                    if kwargs.get('quick'):
                        break

            return total_loss