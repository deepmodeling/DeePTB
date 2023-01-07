import torch
import logging
import numpy as np
from dptb.nnops.base_trainer import Trainer
from dptb.utils.tools import get_uniq_symbol, \
    get_lr_scheduler, get_optimizer, j_must_have
from dptb.hamiltonian.hamil_eig_sk_crt import HamilEig
from dptb.nnops.trainloss import lossfunction

log = logging.getLogger(__name__)

class NNSKTrainer(Trainer):
    def __init__(self, run_opt, jdata) -> None:
        super(NNSKTrainer, self).__init__(jdata)
        self.name = "nnsk"
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

        # initialize loss options
        # ----------------------------------------------------------------------------------------------------------------------------------------------
        self.batch_size = data_options["train"]['batch_size']
        self.reference_batch_size = data_options["reference"]['batch_size']
        
        self.proj_atom_anglr_m = common_options.get('proj_atom_anglr_m')
        self.proj_atom_neles = common_options.get('proj_atom_neles')
        self.onsitemode = common_options.get('onsitemode','none')
        self.atomtype = common_options["atomtype"]
        self.proj_atomtype = get_uniq_symbol(list(self.proj_atom_anglr_m.keys()))

        self.soc = common_options['soc']
        
        self.validation_loss_options = loss_options.copy()
        if self.use_reference:
            self.reference_loss_options = loss_options.copy()

        sortstrength = loss_options['sortstrength']
        self.sortstrength_epoch = torch.exp(torch.linspace(start=np.log(sortstrength[0]), end=np.log(sortstrength[1]), steps=self.num_epoch))


        
    def build(self):
        
        # ---------------------------------------------------------------- init onsite and hopping functions  ----------------------------------------------------------------
        self.call_plugins(queue_name='disposable', time=0, **self.model_options, **self.common_options, **self.data_options, **self.run_opt)
        # ----------------------------------------------------------------         init network model         ----------------------------------------------------------------
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
        if self.loss_options['losstype'].startwith("eigs"):
            self.decompose = True
        else:
            self.decompose = False

        self.validation_lossfunc = getattr(lossfunction(self.criterion), 'l2eig')

        self.hamileig = HamilEig(dtype=self.dtype, device=self.device)
    

    def calc(self, batch_bonds, batch_bond_onsites, batch_envs, batch_onsitenvs, structs, kpoints, eigenvalues, wannier_blocks, decompose=True):
        assert len(kpoints.shape) == 2, "kpoints should have shape of [num_kp, 3]."
        coeffdict = self.model(mode='hopping')
        batch_hoppings = self.hops_fun.get_skhops(batch_bonds=batch_bonds, coeff_paras=coeffdict, 
            rcut=self.model_options["skfunction"]["sk_cutoff"], w=self.model_options["skfunction"]["sk_decay_w"])
        
        nn_onsiteE, onsite_coeffdict = self.model(mode='onsite')
        batch_onsiteEs = self.onsite_fun(batch_bonds_onsite=batch_bond_onsites, onsite_db=self.onsite_db, nn_onsiteE=nn_onsiteE)
        if self.onsitemode == 'strain':
            batch_onsiteVs = self.onsitestrain_fun.get_skhops(batch_bonds=batch_onsitenvs, coeff_paras=onsite_coeffdict)
        else:
            batch_onsiteVs = None

        if self.soc:
            nn_soc_lambdas, _ = self.model(mode='soc')
            batch_soc_lambdas = self.soc_fun(batch_bonds_onsite=batch_bond_onsites, soc_db=self.soc_db, nn_soc=nn_soc_lambdas)
        else:
            batch_soc_lambdas = None

        if decompose:
            # call sktb to get the sktb hoppings and onsites
            eigenvalues_pred = []
            for ii in range(len(structs)):
                if self.onsitemode == 'strain':
                    onsiteEs, onsiteVs, hoppings = batch_onsiteEs[ii], batch_onsiteVs[ii], batch_hoppings[ii]
                    # TODO: 这里的numpy 是否要改为tensor 方便之后为了GPU的加速。
                    onsitenvs = np.asarray(batch_onsitenvs[ii][:,1:])
                    # call hamiltonian block
                else:
                    onsiteEs, hoppings = batch_onsiteEs[ii], batch_hoppings[ii]
                    onsiteVs = None
                    onsitenvs = None
                    # call hamiltonian block

                if self.soc:
                    soc_lambdas = batch_soc_lambdas[ii]
                else:
                    soc_lambdas = None

                self.hamileig.update_hs_list(struct=structs[ii], hoppings=hoppings, onsiteEs=onsiteEs, onsiteVs=onsiteVs,soc_lambdas=soc_lambdas)
                self.hamileig.get_hs_blocks(bonds_onsite=np.asarray(batch_bond_onsites[ii][:,1:]),
                                            bonds_hoppings=np.asarray(batch_bonds[ii][:,1:]), 
                                            onsite_envs=onsitenvs)
                eigenvalues_ii, _ = self.hamileig.Eigenvalues(kpoints=kpoints, time_symm=self.common_options["time_symm"])
                eigenvalues_pred.append(eigenvalues_ii)
            eigenvalues_pred = torch.stack(eigenvalues_pred)

            return eigenvalues_pred, torch.from_numpy(eigenvalues.astype(float)).float()
        else:
            # directly return batch_onsiteEs, batch_hoppings, batch_onsiteVs, batch_soc_lambdas
            # matching the wannier block with shape as output of onsite E/V, hopping

            return (batch_onsiteEs, batch_hoppings, batch_onsiteVs, batch_soc_lambdas), ()
        
    
    def train(self) -> None:
        data_set_seq = np.random.choice(self.n_train_sets, size=self.n_train_sets, replace=False)
        for iset in data_set_seq:
            processor = self.train_processor_list[iset]
            
            # iter with different structure
            self.loss_options.update(processor.bandinfo)
            for data in processor:
                # iter with samples from the same structure


                def closure():
                    # calculate eigenvalues.
                    self.optimizer.zero_grad()
                    # eigenvalues_pred, eigenvector_pred = self.calc(batch_bond, batch_bond_onsites, batch_envs, batch_onsitenvs, structs, kpoints)
                    # eigenvalues_lbl = torch.from_numpy(eigenvalues.astype(float)).float()
                    pred, label = self.calc(*data, decompose=self.decompose)
                    loss = self.train_lossfunc(pred, label, **self.loss_options)

                    if self.use_reference:
                        ref_eig=[]
                        ref_kp_el=[]
                        for irefset in range(self.n_reference_sets):
                            ref_processor = self.ref_processor_list[irefset]
                            for refdata in ref_processor:
                                ref_pred, ref_label = self.calc(*refdata, decompose=self.decompose)
                                ref_eig.append([ref_eig_pred, ref_eig_lbl])
                                ref_kp_el.append([num_kp_ref, num_el_ref])
                    #loss_soft_sort(criterion=self.criterion, eig_pred=eigenvalues_pred, eig_label=eigenvalues_lbl, num_el=num_el,num_kp=num_kp, 
                    #                                    sort_strength=self.sortstrength_epoch[self.epoch-1], band_min=self.band_min, band_max=self.band_max, 
                    #                                    gap_penalty=self.gap_penalty, fermi_band=self.fermi_band,eta=self.loss_gap_eta)
                    if self.use_reference:
                        for irefset in range(self.n_reference_sets):
                            ref_eig_pred, ref_eig_lbl = ref_eig[irefset]
                            num_kp_ref, num_el_ref = ref_kp_el[irefset]
                            
                            self.reference_loss_options.update({'num_el':num_el_ref, 'strength':self.sortstrength_epoch[self.epoch-1]})
                            self.reference_loss_options.update(self.ref_processor_list[irefset].bandinfo)

                            loss += (self.batch_size * 1.0 / (self.reference_batch_size * (1+self.n_reference_sets))) * \
                                            self.train_lossfunc(eig_pred=eigenvalues_pred, eig_label=eigenvalues_lbl, **self.reference_loss_options)    
                    
                    loss.backward()

                    self.train_loss = loss.detach()
                    return loss

                self.optimizer.step(closure)
                state = {'field': 'iteration', "train_loss": self.train_loss,
                         "lr": self.optimizer.state_dict()["param_groups"][0]['lr']}

                self.call_plugins(queue_name='iteration', time=self.iteration, **state)
                # self.lr_scheduler.step() # 在epoch 加入 scheduler.

                self.iteration += 1

    def validation(self, **kwargs):
        with torch.no_grad():
            total_loss = torch.scalar_tensor(0., dtype=self.dtype, device=self.device)
            for processor in self.validation_processor_list:
                self.validation_loss_options.update(processor.bandinfo)
                for data in processor:
                    batch_bond, batch_bond_onsites, batch_envs, batch_onsitenvs, structs, kpoints, eigenvalues, wan = data[0], data[1], data[2], data[
                        3], data[4], data[5], data[6]
                    eigenvalues_pred, eigenvector_pred = self.calc(batch_bond, batch_bond_onsites, batch_envs, batch_onsitenvs, structs, kpoints)
                    eigenvalues_lbl = torch.from_numpy(eigenvalues.astype(float)).float()

                    num_kp = kpoints.shape[0]
                    num_el = np.sum(structs[0].proj_atom_neles_per)
                    self.validation_loss_options.update({'num_el':num_el})
                    total_loss += self.validation_lossfunc(eig_pred=eigenvalues_pred,eig_label=eigenvalues_lbl,**self.validation_loss_options)
                    #total_loss += loss_type1(self.criterion, eigenvalues_pred, eigenvalues_lbl, num_el, num_kp,
                    #                         self.band_min, self.band_max)
                    if kwargs.get('quick'):
                        break

            return total_loss