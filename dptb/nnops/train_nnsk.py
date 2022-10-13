import torch
from dptb.nnops.trainer import Trainer
from dptb.utils.tools import get_uniq_symbol, \
    get_lr_scheduler, get_uniq_bond_type, get_optimizer, j_must_have
from dptb.utils.index_mapping import Index_Mapings
from dptb.sktb.struct_skhs import SKHSLists
from dptb.hamiltonian.hamil_eig_sk_crt import HamilEig
from dptb.nnops.loss import loss_type1, loss_soft_sort
from dptb.dataprocess.processor import Processor
from dptb.dataprocess.datareader import read_data, get_data
from dptb.nnsktb.skintTypes import all_skint_types, all_onsite_intgrl_types
from dptb.nnsktb.sknet import SKNet
from dptb.nnsktb.integralFunc import SKintHops
from dptb.nnsktb.onsiteFunc import onsiteFunc, loadOnsite
import logging
import numpy as np
from dptb.nnsktb.loadparas import load_paras
from dptb.plugins.base_plugin import PluginUser

log = logging.getLogger(__name__)

class NNSKTrainer(Trainer):
    def __init__(self, run_opt, jdata) -> None:
        super(NNSKTrainer, self).__init__(jdata)
        self.run_opt = run_opt
        self.name = "nnsk"
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

        self.band_min = loss_options.get('band_min', 0)
        self.band_max = loss_options.get('band_max', None)
        self.gap_penalty = loss_options.get('gap_penalty',False)
        self.fermi_band = loss_options.get('fermi_band', 0)
        self.loss_gap_eta = loss_options.get('loss_gap_eta',1e-2)    

        if self.use_reference:
            self.ref_band_min = loss_options.get('ref_band_min', 0)
            self.ref_band_max = loss_options.get('ref_band_max', None)

            self.ref_gap_penalty = loss_options.get('ref_gap_penalty', self.gap_penalty)
            self.ref_fermi_band = loss_options.get('ref_fermi_band',self.fermi_band)
            self.ref_loss_gap_eta = loss_options.get('ref_loss_gap_eta',self.loss_gap_eta)

        self.emin = self.loss_options["emin"]
        self.emax = self.loss_options["emax"]
        self.sigma = self.loss_options.get('sigma', 0.1)
        self.num_omega = self.loss_options.get('num_omega',None)
        self.sortstrength = self.loss_options.get('sortstrength',[0.1,0.1])
        self.sortstrength_epoch = torch.exp(torch.linspace(start=np.log(self.sortstrength[0]), end=np.log(self.sortstrength[1]), steps=self.num_epoch))
        
        
    def build(self):
        
        
        # ---------------------------------------------------------------- init onsite and hopping functions  ----------------------------------------------------------------
        self.IndMap = Index_Mapings()
        self.IndMap.update(proj_atom_anglr_m=self.proj_atom_anglr_m)
        self.bond_index_map, self.bond_num_hops = self.IndMap.Bond_Ind_Mapings()
        self.onsite_strain_index_map, self.onsite_strain_num, self.onsite_index_map, self.onsite_num = self.IndMap.Onsite_Ind_Mapings(self.onsitemode, atomtype=self.atomtype)

        self.bond_type = get_uniq_bond_type(self.proj_atomtype)
        self.onsite_fun = onsiteFunc
        self.onsite_db = loadOnsite(self.onsite_index_map)
        self.hops_fun = SKintHops(mode='hopping',functype=self.model_options["skfunction"]["skformula"],proj_atom_anglr_m=self.proj_atom_anglr_m)
        if self.onsitemode == 'strain':
            self.onsitestrain_fun = SKintHops(mode='onsite', functype=self.model_options["skfunction"]["skformula"],proj_atom_anglr_m=self.proj_atom_anglr_m, atomtype=self.atomtype)
        self.call_plugins(queue_name='disposable', time=0, mode=self.run_opt["mode"], **self.model_options, **self.common_options, **self.data_options)
        # ----------------------------------------------------------------         init network model         ----------------------------------------------------------------
        self.optimizer = get_optimizer(model_param=self.model.parameters(), **self.train_options["optimizer"])
        self.lr_scheduler = get_lr_scheduler(optimizer=self.optimizer, **self.train_options["lr_scheduler"])  # add optmizer
        self.criterion = torch.nn.MSELoss(reduction='mean')

        self.hamileig = HamilEig(dtype='tensor')
    

    def calc(self, batch_bonds, batch_bond_onsites, batch_envs, batch_onsitenvs, structs, kpoints):
        assert len(kpoints.shape) == 2, "kpoints should have shape of [num_kp, 3]."
        coeffdict = self.model(mode='hopping')
        batch_hoppings = self.hops_fun.get_skhops(batch_bonds=batch_bonds, coeff_paras=coeffdict, 
            rcut=self.model_options["skfunction"]["sk_cutoff"], w=self.model_options["skfunction"]["sk_decay_w"])
        
        nn_onsiteE, onsite_coeffdict = self.model(mode='onsite')
        batch_onsiteEs = self.onsite_fun(batch_bonds_onsite=batch_bond_onsites, onsite_db=self.onsite_db, nn_onsiteE=nn_onsiteE)
        if self.onsitemode == 'strain':
            batch_onsiteVs = self.onsitestrain_fun.get_skhops(batch_bonds=batch_onsitenvs, coeff_paras=onsite_coeffdict)

        # call sktb to get the sktb hoppings and onsites
        eigenvalues_pred = []
        eigenvector_pred = []
        for ii in range(len(structs)):
            if self.onsitemode == 'strain':
                onsiteEs, onsiteVs, hoppings = batch_onsiteEs[ii], batch_onsiteVs[ii], batch_hoppings[ii]
                onsitenvs = np.asarray(batch_onsitenvs[ii][:,1:])
                # call hamiltonian block
            else:
                onsiteEs, hoppings = batch_onsiteEs[ii], batch_hoppings[ii]
                onsiteVs = None
                onsitenvs = None
                # call hamiltonian block

            self.hamileig.update_hs_list(struct=structs[ii], hoppings=hoppings, onsiteEs=onsiteEs, onsiteVs=onsiteVs)
            self.hamileig.get_hs_blocks(bonds_onsite=np.asarray(batch_bond_onsites[ii][:,1:]),
                                        bonds_hoppings=np.asarray(batch_bonds[ii][:,1:]), 
                                        onsite_envs=onsitenvs)
            eigenvalues_ii, eigvec = self.hamileig.Eigenvalues(kpoints=kpoints, time_symm=self.common_options["time_symm"], dtype='tensor')
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
                    batch_bond, batch_bond_onsites, batch_envs, batch_onsitenvs, structs, kpoints, eigenvalues = data[0], data[1], data[2], data[
                        3], data[4], data[5], data[6]
                    eigenvalues_pred, eigenvector_pred = self.calc(batch_bond, batch_bond_onsites, batch_envs, batch_onsitenvs, structs, kpoints)
                    eigenvalues_lbl = torch.from_numpy(eigenvalues.astype(float)).float()

                    num_kp = kpoints.shape[0]
                    num_el = np.sum(structs[0].proj_atom_neles_per)

                    if self.use_reference:
                        ref_eig=[]
                        ref_kp_el=[]
                        for irefset in range(self.n_reference_sets):
                            ref_processor = self.ref_processor_list[irefset]
                            for refdata in ref_processor:
                                batch_bond, batch_bond_onsites, batch_envs, batch_onsitenvs, structs, kpoints, eigenvalues = refdata[0], refdata[1], refdata[2], \
                                                                                              refdata[3], refdata[4], refdata[5], refdata[6]
                                ref_eig_pred, ref_eigv_pred = self.calc(batch_bond, batch_bond_onsites, batch_envs, batch_onsitenvs, structs, kpoints)
                                ref_eig_lbl = torch.from_numpy(eigenvalues.astype(float)).float()
                                num_kp_ref = kpoints.shape[0]
                                num_el_ref = np.sum(structs[0].proj_atom_neles_per)
                                ref_eig.append([ref_eig_pred, ref_eig_lbl])
                                ref_kp_el.append([num_kp_ref, num_el_ref])
             
                    loss = loss_soft_sort(criterion=self.criterion, eig_pred=eigenvalues_pred, eig_label=eigenvalues_lbl, num_el=num_el,num_kp=num_kp, 
                                                        sort_strength=self.sortstrength_epoch[self.epoch-1], band_min=self.band_min, band_max=self.band_max, 
                                                        gap_penalty=self.gap_penalty, fermi_band=self.fermi_band,eta=self.loss_gap_eta)

                    if self.use_reference:
                        for irefset in range(self.n_reference_sets):
                            ref_eig_pred, ref_eig_lbl = ref_eig[irefset]
                            num_kp_ref, num_el_ref = ref_kp_el[irefset]
                            loss += (self.batch_size * 1.0 / (self.reference_batch_size * (1+self.n_reference_sets))) * loss_soft_sort(criterion=  self.criterion, 
                                        eig_pred=ref_eig_pred, eig_label=ref_eig_lbl,num_el=num_el_ref, num_kp=num_kp_ref, sort_strength=self.sortstrength_epoch[self.epoch-1], 
                                        band_min=self.ref_band_min, band_max=self.ref_band_max,
                                        gap_penalty=self.ref_gap_penalty, fermi_band=self.ref_fermi_band,eta=self.ref_loss_gap_eta)           
                    
                    loss.backward()

                    self.train_loss = loss.detach()
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
            for processor in self.validation_processor_list:
                for data in processor:
                    batch_bond, batch_bond_onsites, batch_envs, batch_onsitenvs, structs, kpoints, eigenvalues = data[0], data[1], data[2], data[
                        3], data[4], data[5], data[6]
                    eigenvalues_pred, eigenvector_pred = self.calc(batch_bond, batch_bond_onsites, batch_envs, batch_onsitenvs, structs, kpoints)
                    eigenvalues_lbl = torch.from_numpy(eigenvalues.astype(float)).float()

                    num_kp = kpoints.shape[0]
                    num_el = np.sum(structs[0].proj_atom_neles_per)

                    total_loss += loss_type1(self.criterion, eigenvalues_pred, eigenvalues_lbl, num_el, num_kp,
                                             self.band_min, self.band_max)
                    if kwargs.get('quick'):
                        break

            return total_loss