import torch
import logging
import numpy as np
from dptb.hamiltonian.hamil_eig_sk_crt import HamilEig
from dptb.utils.tools import get_uniq_symbol, get_lr_scheduler, \
get_optimizer, nnsk_correction, j_must_have

from dptb.nnops.v1.trainloss import lossfunction
from dptb.nnops.base_trainer import BaseTrainer

log = logging.getLogger(__name__)

class DPTBTrainer(BaseTrainer):

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
        self.atomtype = get_uniq_symbol(common_options["atomtype"])
        self.soc = common_options['soc']
        self.overlap = common_options['overlap']
        self.proj_atomtype = get_uniq_symbol(list(self.proj_atom_anglr_m.keys()))

        self.band_min = loss_options.get('band_min', 0)
        self.band_max = loss_options.get('band_max', None)

        self.validation_loss_options = loss_options.copy()
        if self.use_reference:
            self.reference_loss_options = loss_options.copy()

        # sortstrength = loss_options['sortstrength']
        # self.sortstrength_epoch = torch.exp(torch.linspace(start=np.log(sortstrength[0]), end=np.log(sortstrength[1]), steps=self.num_epoch))
        self.sk_cutoff = self.model_options["skfunction"]["sk_cutoff"]
        self.sk_decay_w = self.model_options["skfunction"]["sk_decay_w"]
        
        

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

        if self.run_opt.get("use_correction", False):
            model_param = [{"params":self.model.parameters()}, {"params":self.sknet.parameters()}]
        else:
            model_param = self.model.parameters()
        self.optimizer = get_optimizer(model_param=model_param, **self.train_options["optimizer"])
        self.lr_scheduler = get_lr_scheduler(optimizer=self.optimizer, **self.train_options["lr_scheduler"])

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

    def calc(self, batch_bond, batch_bond_onsites, batch_env, batch_onsitenvs, structs, kpoints, eigenvalues, wannier_blocks, decompose=True):
        '''
        conduct one step forward computation, used in train, test and validation.
        '''
        if len(kpoints.shape) != 2: 
            log.error(msg="kpoints should have shape of [num_kp, 3].")
            raise ValueError
        if (wannier_blocks[0] is None) and not (decompose):
            log.error(msg="The wannier_blocks from processor is None, but the losstype wannier, please check the input data, maybe the wannier.npy is not there.")
            raise ValueError

        # get sk param (of each bond or onsite)
        batch_bond_hoppings, batch_hoppings, \
        batch_bond_onsites, batch_onsiteEs, batch_soc_lambdas = self.nntb.calc(batch_bond, batch_env)

        if self.run_opt.get("use_correction", False):
            # get sk param (dptb-0)
            coeffdict, overlap_coeffdict = self.sknet(mode='hopping')
            nnsk_onsiteE, onsite_coeffdict = self.sknet(mode='onsite')

            # get sk param (of each bond or onsite, dptb-0)
            batch_nnsk_hoppings = self.hops_fun.get_skhops(
                batch_bonds=batch_bond_hoppings, coeff_paras=coeffdict, rcut=self.model_options["skfunction"]["sk_cutoff"],
                w=self.model_options["skfunction"]["sk_decay_w"])
            
            if self.overlap:
                batch_nnsk_overlaps = self.overlap_fun.get_skoverlaps(
                    batch_bonds=batch_bond_hoppings, coeff_paras=overlap_coeffdict, rcut=self.model_options["skfunction"]["sk_cutoff"],
                    w=self.model_options["skfunction"]["sk_decay_w"])
                
            batch_nnsk_onsiteEs = self.onsite_fun.get_onsiteEs(batch_bonds_onsite=batch_bond_onsites, onsite_env=batch_onsitenvs, nn_onsite_paras=nnsk_onsiteE)
            
            if self.onsitemode == "strain":
                batch_nnsk_onsiteVs = self.onsitestrain_fun.get_skhops(batch_bonds=batch_onsitenvs, coeff_paras=onsite_coeffdict)
            if self.soc:
                nnsk_soc_lambdas, _ = self.sknet(mode="soc")
                batch_nnsk_soc_lambdas = self.soc_fun(batch_bonds_onsite=batch_bond_onsites, soc_db=self.soc_db, nn_soc=nnsk_soc_lambdas)

        # ToDo: Advance the correction process before onsite_fun and hops_fun (this seems impossible?)
        # call sktb to get the sktb hoppings and onsites
        onsiteVs = None
        onsitenvs = None
        pred = []
        label = []
        for ii in range(len(structs)):
            l = []
            if not self.run_opt.get("use_correction", False):
                onsiteEs, hoppings = batch_onsiteEs[ii], batch_hoppings[ii]
                soc_lambdas = None
                overlaps = None
                if self.overlap:
                    log.error(msg="ValueError: Overlap mode can only be used with nnsk correction.")
                    raise ValueError
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
                if self.overlap:
                    nnsk_overlaps = batch_nnsk_overlaps[ii]
                else:
                    nnsk_overlaps = None
                onsiteEs, hoppings, onsiteSs, overlaps, soc_lambdas = nnsk_correction(
                    nn_onsiteEs=batch_onsiteEs[ii], nn_hoppings=batch_hoppings[ii],
                    sk_onsiteEs=batch_nnsk_onsiteEs[ii], sk_hoppings=batch_nnsk_hoppings[ii],
                    sk_onsiteSs=None, sk_overlaps=nnsk_overlaps, 
                    nn_soc_lambdas=nn_soc_lambdas, 
                    sk_soc_lambdas=sk_soc_lambdas
                    )

                if self.onsitemode == "strain":
                    onsiteVs = batch_nnsk_onsiteVs[ii]
                    onsitenvs = batch_onsitenvs[ii][:,1:]
            # call hamiltonian block

            bond_onsites = batch_bond_onsites[ii][:,1:]
            bond_hoppings = batch_bond_hoppings[ii][:,1:]

            self.hamileig.update_hs_list(struct=structs[ii], hoppings=hoppings, onsiteEs=onsiteEs, onsiteVs=onsiteVs, overlaps=overlaps, soc_lambdas=soc_lambdas)
            self.hamileig.get_hs_blocks(bonds_onsite=bond_onsites,
                                        bonds_hoppings=bond_hoppings,
                                        onsite_envs=onsitenvs)
            
            if decompose:
                eigenvalues_ii, _ = self.hamileig.Eigenvalues(kpoints=kpoints, time_symm=self.common_options["time_symm"],unit=self.common_options["unit"])
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

# 

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
            self.loss_options.update(processor.bandinfo)
            # iter with different structure
            data = next(processor)
                # iter with samples from the same structure

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
            state = {'field':'iteration', "train_loss": self.train_loss, "lr": self.optimizer.state_dict()["param_groups"][0]['lr']}

            self.call_plugins(queue_name='iteration', time=self.iteration, **state)
            # self.lr_scheduler.step() # 在epoch 加入 scheduler.


            self.iteration += 1

    def update(self, **kwargs):
        self.model_options["skfunction"]["sk_cutoff"] += self.skcut_step
        self.model_options["skfunction"]["sk_decay_w"] += self.skdecay_step

        self.model_config["skfunction"]["sk_cutoff"] = self.model_options["skfunction"]["sk_cutoff"]
        self.model_config["skfunction"]["sk_decay_w"] = self.model_options["skfunction"]["sk_decay_w"]

    def validation(self, quick=False):
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
                    if quick:
                        break
                if quick:
                    break
        with torch.enable_grad():
            return total_loss.detach()



if __name__ == '__main__':
    a = [1,2,3]

    print(list(enumerate(a, 2)))
