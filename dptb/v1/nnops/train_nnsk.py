import torch
import logging
import numpy as np
from dptb.nnops.base_trainer import BaseTrainer
from dptb.utils.tools import get_uniq_symbol, \
    get_lr_scheduler, get_optimizer, j_must_have
from dptb.hamiltonian.hamil_eig_sk_crt import HamilEig
from dptb.nnops.v1.trainloss import lossfunction
import json

log = logging.getLogger(__name__)

class NNSKTrainer(BaseTrainer):
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
        self.overlap = common_options['overlap']
        
        self.validation_loss_options = loss_options.copy()
        if self.use_reference:
            self.reference_loss_options = loss_options.copy()

        # sortstrength = loss_options['sortstrength']
        # self.sortstrength_epoch = torch.exp(torch.linspace(start=np.log(sortstrength[0]), end=np.log(sortstrength[1]), steps=self.num_epoch))
        self.sk_cutoff = self.model_options["skfunction"]["sk_cutoff"]
        self.sk_decay_w = self.model_options["skfunction"]["sk_decay_w"]
        


        
    def build(self):
        
        # ---------------------------------------------------------------- init onsite and hopping functions  ----------------------------------------------------------------
        self.call_plugins(queue_name='disposable', time=0, **self.model_options, **self.common_options, **self.data_options, **self.run_opt)
        # ----------------------------------------------------------------         init network model         ----------------------------------------------------------------
        self.optimizer = get_optimizer(model_param=self.model.parameters(), **self.train_options["optimizer"])
        self.lr_scheduler = get_lr_scheduler(optimizer=self.optimizer, **self.train_options["lr_scheduler"])
        
        if self.run_opt["mode"] == "restart":
            ckpt = torch.load(self.run_opt["restart"], weights_only=False)
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
        if isinstance(self.sk_cutoff, list):
            assert len(self.sk_cutoff) == 2
            self.skcut_step = (self.sk_cutoff[1] - self.sk_cutoff[0]) / (self.num_epoch - self.epoch + 1)
            self.model_options["skfunction"]["sk_cutoff"] = self.sk_cutoff[0]
        else:
            self.skcut_step = 0
        if isinstance(self.sk_decay_w, list):
            assert len(self.sk_decay_w) == 2
            self.skdecay_step = (self.sk_decay_w[1] - self.sk_decay_w[0]) / (self.num_epoch - self.epoch + 1)
            self.model_options["skfunction"]["sk_decay_w"] = self.sk_decay_w[0]
        else:
            self.skdecay_step = 0

        self.train_lossfunc = getattr(lossfunction(self.criterion), self.loss_options['losstype'])
        if self.loss_options['losstype'].startswith("eigs"):
            self.decompose = True
        elif self.loss_options['losstype'].startswith("block"):
            self.decompose = False
        else:
            log.error(msg="loss function is defined wrongly.")
            raise ValueError("loss function is defined wrongly.")
        

        self.validation_lossfunc = getattr(lossfunction(self.criterion), 'eigs_l2')

        self.hamileig = HamilEig(dtype=self.dtype, device=self.device)
    

    def calc(self, batch_bonds, batch_bond_onsites, batch_envs, batch_onsitenvs, structs, kpoints, eigenvalues, wannier_blocks, decompose=True):
        if len(kpoints.shape) != 2: 
            log.error(msg="kpoints should have shape of [num_kp, 3].")
            raise ValueError
        if (wannier_blocks[0] is None) and not (decompose):
            log.error(msg="The wannier_blocks from processor is None, but the losstype wannier, please check the input data, maybe the wannier.npy is not there.")
            raise ValueError

        # get sk param (model format)
        coeffdict, overlap_coeffdict = self.model(mode='hopping')
        nn_onsiteE, onsite_coeffdict = self.model(mode='onsite')


        # get sk param (of each bond or onsite)
        batch_hoppings = self.hops_fun.get_skhops(batch_bonds=batch_bonds, coeff_paras=coeffdict, 
            rcut=self.model_options["skfunction"]["sk_cutoff"], w=self.model_options["skfunction"]["sk_decay_w"])
        batch_onsiteEs = self.onsite_fun.get_onsiteEs(batch_bonds_onsite=batch_bond_onsites, onsite_env=batch_onsitenvs, nn_onsite_paras=nn_onsiteE)
        
        if self.overlap:
            assert overlap_coeffdict is not None, "The overlap_coeffdict should be provided if overlap is True."
            batch_overlaps = self.overlap_fun.get_skoverlaps(batch_bonds=batch_bonds, coeff_paras=overlap_coeffdict, 
                rcut=self.model_options["skfunction"]["sk_cutoff"], w=self.model_options["skfunction"]["sk_decay_w"])

        if self.onsitemode == 'strain':
            batch_onsiteVs = self.onsitestrain_fun.get_skhops(batch_bonds=batch_onsitenvs, coeff_paras=onsite_coeffdict)
        else:
            batch_onsiteVs = None

        if self.soc:
            nn_soc_lambdas, _ = self.model(mode='soc')
            batch_soc_lambdas = self.soc_fun(batch_bonds_onsite=batch_bond_onsites, soc_db=self.soc_db, nn_soc=nn_soc_lambdas)
        else:
            batch_soc_lambdas = None


        # copy sk param for writing json checkpoint
        self.onsite_index_dict = self.model.onsite_index_dict
        self.hopping_coeff = coeffdict
        self.overlap_coeff = overlap_coeffdict
        if self.onsitemode == 'strain':
            self.onsite_coeff = onsite_coeffdict
        else:
            self.onsite_coeff = nn_onsiteE
        if self.soc:    
            self.soc_coeff = nn_soc_lambdas

        # constructing hamiltonians and decomposition
        pred = []
        label = []
        for ii in range(len(structs)):
            l = []
            if self.onsitemode == 'strain':
                onsiteEs, onsiteVs, hoppings = batch_onsiteEs[ii], batch_onsiteVs[ii], batch_hoppings[ii]
                onsitenvs = batch_onsitenvs[ii][:,1:]
            else:
                onsiteEs, hoppings = batch_onsiteEs[ii], batch_hoppings[ii]
                onsiteVs = None
                onsitenvs = None
            if self.overlap:
                overlaps = batch_overlaps[ii]
            else:
                overlaps = None

            if self.soc:
                soc_lambdas = batch_soc_lambdas[ii]
            else:
                soc_lambdas = None

            bond_onsites = batch_bond_onsites[ii][:,1:]
            bond_hoppings = batch_bonds[ii][:,1:]

            self.hamileig.update_hs_list(struct=structs[ii], hoppings=hoppings, onsiteEs=onsiteEs, onsiteVs=onsiteVs,overlaps=overlaps, soc_lambdas=soc_lambdas)
            self.hamileig.get_hs_blocks(bonds_onsite=bond_onsites,
                                        bonds_hoppings=bond_hoppings, 
                                        onsite_envs=onsitenvs)
            if decompose:
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

    def update(self, **kwargs):
        self.model_options["skfunction"]["sk_cutoff"] += self.skcut_step
        self.model_options["skfunction"]["sk_decay_w"] += self.skdecay_step

        self.model_config["skfunction"]["sk_cutoff"] = self.model_options["skfunction"]["sk_cutoff"]
        self.model_config["skfunction"]["sk_decay_w"] = self.model_options["skfunction"]["sk_decay_w"]


    def validation(self, **kwargs):
        
        with torch.no_grad():

            total_loss = torch.scalar_tensor(0., dtype=self.dtype, device=self.device)
            data_set_seq = []
            total_batch = 0
            for ip in range(len(self.validation_processor_list)):
                data_set_seq += [ip] * self.train_processor_list[ip].n_batch
                total_batch += self.train_processor_list[ip].n_batch
                
            data_set_seq = np.array(data_set_seq)[np.random.choice(total_batch, size=total_batch, replace=False)]

            for iset in data_set_seq:
                processor  = self.validation_processor_list[iset]
                self.validation_loss_options.update(processor.bandinfo)
                for data in processor:
                    eigenvalues_pred, eigenvalues_lbl = self.calc(*data)

                    total_loss += self.validation_lossfunc(eig_pred=eigenvalues_pred,eig_label=eigenvalues_lbl,**self.validation_loss_options)
                    if kwargs.get('quick'):
                        break
                if kwargs.get('quick'):
                    break

        with torch.enable_grad():
            return total_loss.detach()