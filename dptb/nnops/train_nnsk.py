import torch
import logging
import numpy as np
from dptb.nnops.base_trainer import Trainer
from dptb.utils.tools import get_uniq_symbol, \
    get_lr_scheduler, get_optimizer, j_must_have
from dptb.hamiltonian.hamil_eig_sk_crt import HamilEig
from dptb.nnops.trainloss import lossfunction
import json

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
        self.atomtype = get_uniq_symbol(common_options["atomtype"])
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
        if self.loss_options['losstype'].startswith("eigs"):
            self.decompose = True
        else:
            self.decompose = False

        self.validation_lossfunc = getattr(lossfunction(self.criterion), 'eigs_l2')

        self.hamileig = HamilEig(dtype=self.dtype, device=self.device)
    

    def calc(self, batch_bonds, batch_bond_onsites, batch_envs, batch_onsitenvs, structs, kpoints, eigenvalues, wannier_blocks, decompose=True):
        if len(kpoints.shape) != 2: 
            log.error(msg="kpoints should have shape of [num_kp, 3].")
            raise ValueError
        if (wannier_blocks[0] is None) and not (self.decompose):
            log.error(msg="The wannier_blocks from processor is None, but the losstype wannier, please check the input data, maybe the wannier.npy is not there.")
            raise ValueError

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
        # call sktb to get the sktb hoppings and onsites
        self.onsite_index_dict = self.model.onsite_index_dict
        self.hopping_coeff = coeffdict
        if self.onsitemode == 'strain':
            self.onsite_coeff = onsite_coeffdict
        else:
            self.onsite_coeff = nn_onsiteE
        if self.soc:    
            self.soc_coeff = nn_soc_lambdas

        pred = []
        label = []
        for ii in range(len(structs)):
            l = []
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

            bond_onsites = np.asarray(batch_bond_onsites[ii][:,1:])
            bond_hoppings = np.asarray(batch_bonds[ii][:,1:])

            self.hamileig.update_hs_list(struct=structs[ii], hoppings=hoppings, onsiteEs=onsiteEs, onsiteVs=onsiteVs,soc_lambdas=soc_lambdas)
            self.hamileig.get_hs_blocks(bonds_onsite=bond_onsites,
                                        bonds_hoppings=bond_hoppings, 
                                        onsite_envs=onsitenvs)
            if decompose:
                #if self.run_opt["freeze"]:
                #    kpoints = np.array([[0,0,0]])
                eigenvalues_ii, _ = self.hamileig.Eigenvalues(kpoints=kpoints, time_symm=self.common_options["time_symm"], unit=self.common_options["unit"])
                pred.append(eigenvalues_ii)
            else:
                assert not self.soc, "soc should not open when using wannier blocks to fit."
                pred.append(self.hamileig.hamil_blocks) # in order of [batch_bond_onsite, batch_bonds]
                for bo in bond_onsites:
                    key = str(int(bo[1])) +"_"+ str(int(bo[3])) +"_"+ str(int(bo[4])) +"_"+ str(int(bo[5])) +"_"+ str(int(bo[6]))
                    l.append(torch.tensor(wannier_blocks[ii][key], dtype=self.dtype, device=self.device))
                for bo in bond_hoppings:
                    key = str(int(bo[1])) +"_"+ str(int(bo[3])) +"_"+ str(int(bo[4])) +"_"+ str(int(bo[5])) +"_"+ str(int(bo[6]))
                    l.append(torch.tensor(wannier_blocks[ii][key], dtype=self.dtype, device=self.device))
                label.append(l)

        if decompose:
            #if self.run_opt["freeze"]:
            #    label = torch.from_numpy(eigenvalues.astype(float))[:,[0],:].float()
            #else:
            label = torch.from_numpy(eigenvalues.astype(float)).float()
            pred = torch.stack(pred)
        return pred, label
    
    def train(self) -> None:

        total_batch = 0
        # reset processor:
        data_set_seq = []
        for ip in range(self.n_train_sets):
            self.train_processor_list[ip] = iter(self.train_processor_list[ip])
            data_set_seq += [ip] * self.train_processor_list[ip].n_batch
            total_batch += self.train_processor_list[ip].n_batch

        data_set_seq = np.array(data_set_seq)[np.random.choice(total_batch, size=total_batch, replace=False)]

        for iset in data_set_seq:
            processor = self.train_processor_list[iset]
            
            data = next(processor)
            # iter with different structure
            self.loss_options.update(processor.bandinfo)


            def closure():
                # calculate eigenvalues.
                self.optimizer.zero_grad()
                pred, label = self.calc(*data, decompose=self.decompose)

                loss = self.train_lossfunc(pred, label, **self.loss_options)

                if self.use_reference:
                    for irefset in range(self.n_reference_sets):
                        ref_processor = self.ref_processor_list[irefset]
                        self.reference_loss_options.update(self.ref_processor_list[irefset].bandinfo)
                        for refdata in ref_processor:
                            ref_pred, ref_label = self.calc(*refdata, decompose=self.decompose)
                            loss += (self.batch_size * 1.0 / (self.reference_batch_size * (1+self.n_reference_sets))) * \
                                        self.train_lossfunc(ref_pred, ref_label, **self.reference_loss_options)
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
                    eigenvalues_pred, eigenvalues_lbl = self.calc(*data)

                    total_loss += self.validation_lossfunc(eig_pred=eigenvalues_pred,eig_label=eigenvalues_lbl,**self.validation_loss_options)
                    #total_loss += loss_type1(self.criterion, eigenvalues_pred, eigenvalues_lbl, num_el, num_kp,
                    #                         self.band_min, self.band_max)
                    if kwargs.get('quick'):
                        break
        with torch.enable_grad():
            return total_loss.detach()