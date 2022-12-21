import torch
import logging
import numpy as np
from dptb.hamiltonian.hamil_eig_sk_crt import HamilEig
from dptb.utils.tools import get_uniq_symbol, get_lr_scheduler, \
get_optimizer, nnsk_correction, j_must_have

from dptb.nnops.trainloss import lossfunction
from dptb.nnops.base_trainer import Trainer

log = logging.getLogger(__name__)

class DPTBTrainer(Trainer):

    def __init__(self, run_opt, jdata) -> None:
        super(DPTBTrainer, self).__init__(jdata)
        self.name = "dptb"
        self.run_opt = run_opt
        self._init_param(jdata)


    def _init_param(self, jdata):
        common_options = j_must_have(jdata, "common_options")
        train_options = j_must_have(jdata, "train_options")
        data_options = j_must_have(jdata,"data_options")
        model_options = j_must_have(jdata, "model_options")
        loss_options = j_must_have(jdata, "loss_options")

        self.common_options = common_options
        self.train_options = train_options
        self.data_options = data_options
        self.model_options = model_options
        self.loss_options = loss_options

        self.num_epoch = train_options['num_epoch']
        self.use_reference = data_options['use_reference']

        # initialize data options
        # ------------------------------------------------------------------------
        self.batch_size = data_options["train"]['batch_size']
        self.reference_batch_size = data_options["reference"]['batch_size']

        self.proj_atom_anglr_m = common_options.get('proj_atom_anglr_m')
        self.proj_atom_neles = common_options.get('proj_atom_neles')
        self.onsitemode = common_options.get('onsitemode','none')
        self.atomtype = common_options["atomtype"]
        self.soc = common_options['soc']
        self.proj_atomtype = get_uniq_symbol(list(self.proj_atom_anglr_m.keys()))

        self.band_min = loss_options.get('band_min', 0)
        self.band_max = loss_options.get('band_max', None)

        self.validation_loss_options = loss_options.copy()
        if loss_options['val_band_min'] != None:
            self.validation_loss_options.update({'band_min': loss_options['val_band_min']})
        if loss_options['val_band_max'] != None:
            self.validation_loss_options.update({'band_max': loss_options['val_band_max']})
        if self.use_reference:
            self.reference_loss_options = loss_options.copy()
            if loss_options['ref_band_min'] != None:
                self.reference_loss_options.update({'band_min': loss_options['ref_band_min']})
            if loss_options['ref_band_max'] != None:
                self.reference_loss_options.update({'band_max': loss_options['ref_band_max']})
            if loss_options['ref_gap_penalty'] != None:
                self.reference_loss_options.update({'gap_penalty': loss_options['ref_gap_penalty']})
            if loss_options['ref_fermi_band'] != None:
                self.reference_loss_options.update({'fermi_band': loss_options['ref_fermi_band']})
            if loss_options['ref_loss_gap_eta'] != None:
                self.reference_loss_options.update({'loss_gap_eta': loss_options['ref_loss_gap_eta']})

        sortstrength = loss_options['sortstrength']
        self.sortstrength_epoch = torch.exp(torch.linspace(start=np.log(sortstrength[0]), end=np.log(sortstrength[1]), steps=self.num_epoch))


    def build(self):
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
        self.call_plugins(queue_name='disposable', time=0, **self.model_options, **self.common_options, **self.data_options, **self.run_opt)

        self.optimizer = get_optimizer(model_param=self.model.parameters(), **self.train_options["optimizer"])
        self.lr_scheduler = get_lr_scheduler(optimizer=self.optimizer, **self.train_options["lr_scheduler"])

        if self.run_opt["mode"] == "restart":
            ckpt = torch.load(self.run_opt["restart"])
            self.epoch = ckpt["epoch"]
            self.iteration = ckpt["iteration"]
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.lr_scheduler = get_lr_scheduler(optimizer=self.optimizer, last_epoch=self.epoch, **self.train_options["lr_scheduler"])  # add optmizer
            self.stats = ckpt["stats"]
            
            queues_name = list(self.plugin_queues.keys())
            for unit in queues_name:
                for plugin in self.plugin_queues[unit]:
                    plugin = (getattr(self, unit) + plugin[0], plugin[1], plugin[2])

        self.criterion = torch.nn.MSELoss(reduction='mean')

        self.train_lossfunc = getattr(lossfunction(self.criterion), self.loss_options['losstype'])
        self.validation_lossfunc = getattr(lossfunction(self.criterion), 'l2eig')

        self.hamileig = HamilEig(dtype=self.dtype, device=self.device)

    def calc(self, batch_bond, batch_bond_onsites, batch_env, batch_onsitenvs, structs, kpoints):
        '''
        conduct one step forward computation, used in train, test and validation.
        '''
        assert len(kpoints.shape) == 2, "kpoints should have shape of [num_kp, 3]."

        batch_bond_hoppings, batch_hoppings, \
        batch_bond_onsites, batch_onsiteEs, batch_soc_lambdas = self.nntb.calc(batch_bond, batch_env)

        if self.run_opt.get("use_correction", False):
            coeffdict = self.sknet(mode='hopping')
            batch_nnsk_hoppings = self.hops_fun.get_skhops(
                batch_bond_hoppings, coeffdict, rcut=self.model_options["skfunction"]["sk_cutoff"],
                w=self.model_options["skfunction"]["sk_decay_w"])
            nnsk_onsiteE, onsite_coeffdict = self.sknet(mode='onsite')
            batch_nnsk_onsiteEs = self.onsite_fun(batch_bonds_onsite=batch_bond_onsites, onsite_db=self.onsite_db, nn_onsiteE=nnsk_onsiteE)
            
            if self.onsitemode == "strain":
                batch_nnsk_onsiteVs = self.onsitestrain_fun.get_skhops(batch_bonds=batch_onsitenvs, coeff_paras=onsite_coeffdict)
            if self.soc:
                nnsk_soc_lambdas, _ = self.sknet(mode="soc")
                batch_nnsk_soc_lambdas = self.soc_fun(batch_bonds_onsite=batch_bond_onsites, soc_db=self.soc_db, nn_soc=nnsk_soc_lambdas)

        # ToDo: Advance the correction process before onsite_fun and hops_fun
        # call sktb to get the sktb hoppings and onsites
        eigenvalues_pred = []
        eigenvector_pred = []
        onsiteVs = None
        onsitenvs = None
        for ii in range(len(structs)):
            if not self.run_opt.get("use_correction", False):
                onsiteEs, hoppings = batch_onsiteEs[ii], batch_hoppings[ii]
                soc_lambdas = None
                if self.soc:
                    log.error(msg="ValueError: Soc mode can only be used with nnsk correction.")
                    raise ValueError
            else:
                if self.soc and self.model_options["dptb"]["soc_env"]:
                    nn_soc_lambdas = batch_soc_lambdas[ii]
                    sk_soc_lambdas = batch_nnsk_soc_lambdas[ii]
                else:
                    nn_soc_lambdas = None
                    if self.soc:
                        sk_soc_lambdas = batch_nnsk_soc_lambdas[ii]
                    else:
                        sk_soc_lambdas = None

                onsiteEs, hoppings, _, _, soc_lambdas = nnsk_correction(nn_onsiteEs=batch_onsiteEs[ii], nn_hoppings=batch_hoppings[ii],
                                    sk_onsiteEs=batch_nnsk_onsiteEs[ii], sk_hoppings=batch_nnsk_hoppings[ii],
                                    sk_onsiteSs=None, sk_overlaps=None, 
                                    nn_soc_lambdas=nn_soc_lambdas, 
                                    sk_soc_lambdas=sk_soc_lambdas)

                if self.onsitemode == "strain":
                    onsiteVs = batch_nnsk_onsiteVs[ii]
                    onsitenvs = np.asarray(batch_onsitenvs[ii][:,1:])
            # call hamiltonian block
            self.hamileig.update_hs_list(struct=structs[ii], hoppings=hoppings, onsiteEs=onsiteEs, onsiteVs=onsiteVs, soc_lambdas=soc_lambdas)
            self.hamileig.get_hs_blocks(bonds_onsite=np.asarray(batch_bond_onsites[ii][:,1:]),
                                        bonds_hoppings=np.asarray(batch_bond_hoppings[ii][:,1:]),
                                        onsite_envs=onsitenvs)
            eigenvalues_ii, eigvec = self.hamileig.Eigenvalues(kpoints=kpoints, time_symm=self.common_options["time_symm"])
            eigenvalues_pred.append(eigenvalues_ii)
            eigenvector_pred.append(eigvec)
        eigenvalues_pred = torch.stack(eigenvalues_pred)
        eigenvector_pred = torch.stack(eigenvector_pred)

        return eigenvalues_pred, eigenvector_pred


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
                    batch_bond, batch_bond_onsite, batch_env, batch_onsitenvs, structs, kpoints, eigenvalues = data[0], data[1], data[2], \
                                                                                              data[3], data[4], data[5], data[6]
                    eigenvalues_pred, eigenvector_pred = self.calc(batch_bond, batch_bond_onsite, batch_env, batch_onsitenvs, structs, kpoints)
                    eigenvalues_lbl = torch.from_numpy(eigenvalues.astype(float)).float()

                    num_kp = kpoints.shape[0]
                    num_el = np.sum(structs[0].proj_atom_neles_per)

                    if self.use_reference:
                        ref_eig = []
                        ref_kp_el = []

                        for irefset in range(self.n_reference_sets):
                            ref_processor = self.ref_processor_list[irefset]
                            for refdata in ref_processor:
                                batch_bond, _, batch_env, batch_onsitenvs, structs, kpoints, eigenvalues = refdata[0], refdata[1], \
                                                                                          refdata[2], refdata[3], \
                                                                                          refdata[4], refdata[5], refdata[6]
                                ref_eig_pred = self.calc(batch_bond, batch_bond_onsite, batch_env, batch_onsitenvs, structs, kpoints)
                                ref_eig_lbl = torch.from_numpy(eigenvalues.astype(float)).float()
                                num_kp_ref = kpoints.shape[0]
                                num_el_ref = np.sum(structs[0].proj_atom_neles_per)

                                ref_eig.append([ref_eig_pred, ref_eig_lbl])
                                ref_kp_el.append([num_kp_ref, num_el_ref])

                    self.loss_options.update({'num_el':num_el, 'strength':self.sortstrength_epoch[self.epoch-1]})
                    loss = self.train_lossfunc(eig_pred=eigenvalues_pred, eig_label=eigenvalues_lbl, **self.loss_options)
                    
                    if self.use_reference:
                        for irefset in range(self.n_reference_sets):
                            ref_eig_pred, ref_eig_lbl = ref_eig[irefset]
                            num_kp_ref, num_el_ref = ref_kp_el[irefset]
                            self.reference_loss_options.update({'num_el':num_el_ref, 'strength':self.sortstrength_epoch[self.epoch-1]})
                            loss += (self.batch_size * 1.0 / (self.reference_batch_size * (1+self.n_reference_sets))) * \
                                            self.train_lossfunc(eig_pred=eigenvalues_pred, eig_label=eigenvalues_lbl, **self.reference_loss_options)

                    loss.backward()

                    self.train_loss = loss.detach()
                    return loss

                self.optimizer.step(closure)
                state = {'field':'iteration', "train_loss": self.train_loss, "lr": self.optimizer.state_dict()["param_groups"][0]['lr']}

                self.call_plugins(queue_name='iteration', time=self.iteration, **state)
                # self.lr_scheduler.step() # 在epoch 加入 scheduler.


                self.iteration += 1

    def validation(self, quick=False):
        with torch.no_grad():
            total_loss = torch.scalar_tensor(0., dtype=self.dtype, device=self.device)
            for processor in self.validation_processor_list:
                for data in processor:
                    batch_bond, batch_bond_onsite, batch_env, batch_onsitenvs, structs, kpoints, eigenvalues = data[0],data[1],data[2], data[3], data[4], data[5], data[6]
                    eigenvalues_pred, _ = self.calc(batch_bond, batch_bond_onsite, batch_env, batch_onsitenvs, structs, kpoints)
                    eigenvalues_lbl = torch.from_numpy(eigenvalues.astype(float)).float()

                    num_kp = kpoints.shape[0]
                    num_el = np.sum(structs[0].proj_atom_neles_per)

                    self.validation_loss_options.update({'num_el':num_el})
                    total_loss += self.validation_lossfunc(eig_pred=eigenvalues_pred,eig_label=eigenvalues_lbl,**self.validation_loss_options)
                    if quick:
                        break

            return total_loss



if __name__ == '__main__':
    a = [1,2,3]

    print(list(enumerate(a, 2)))
