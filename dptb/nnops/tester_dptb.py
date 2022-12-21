import torch
import logging
import numpy as np
from dptb.hamiltonian.hamil_eig_sk_crt import HamilEig
from dptb.nnsktb.onsiteFunc import loadOnsite
from dptb.nnops.loss import loss_type1, loss_spectral
from dptb.utils.tools import get_uniq_symbol, get_lr_scheduler, \
get_optimizer, nnsk_correction, j_must_have

from dptb.nnops.trainloss import lossfunction
from dptb.nnops.base_tester import Tester


class DPTBTester(Tester):

    def __init__(self, run_opt, jdata) -> None:
        super(DPTBTester, self).__init__(jdata)
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
        self.results_path = self.run_opt['results_path']



        self.batch_size = data_options["train"]['batch_size']

        self.proj_atom_anglr_m = common_options.get('proj_atom_anglr_m')
        self.proj_atom_neles = common_options.get('proj_atom_neles')
        self.onsitemode = common_options.get('onsitemode','none')
        self.atomtype = common_options["atomtype"]
        self.soc = common_options['soc']
        self.proj_atomtype = get_uniq_symbol(list(self.proj_atom_anglr_m.keys()))

        self.band_min = loss_options.get('band_min', 0)
        self.band_max = loss_options.get('band_max', None)

    def build(self):
        # ---------------------------------------------------------------- init onsite and hopping functions  ----------------------------------------------------------------
        self.call_plugins(queue_name='disposable', time=0, **self.model_options, **self.common_options, **self.data_options, **self.run_opt)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.test_lossfunc = getattr(lossfunction(self.criterion), 'l2eig')
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

    def test(self) -> None:
        with torch.no_grad():
            iprocess =0
            for processor in self.test_processor_list:
                idata = 0
                for data in processor:
                    batch_bond, batch_bond_onsite, batch_env, batch_onsitenvs, structs, kpoints, eigenvalues = \
                                                            data[0],data[1],data[2], data[3], data[4], data[5], data[6]
                    eigenvalues_pred, _ = self.calc(batch_bond, batch_bond_onsite, batch_env, batch_onsitenvs, structs, kpoints)
                    eigenvalues_lbl = torch.from_numpy(eigenvalues.astype(float)).float()
                    
                    if idata ==0:
                        eigenvalues_pred_collect = eigenvalues_pred.clone()
                        eigenvalues_lbel_collect = eigenvalues_lbl.clone()
                    else:
                        eigenvalues_pred_collect = torch.cat([eigenvalues_pred_collect,eigenvalues_pred],dim=0)
                        eigenvalues_lbel_collect = torch.cat([eigenvalues_lbel_collect,eigenvalues_lbl],dim=0)

                    num_kp = kpoints.shape[0]
                    num_el = np.sum(structs[0].proj_atom_neles_per)

                    self.loss_options.update({'num_el':num_el})
                    loss = self.test_lossfunc(eig_pred=eigenvalues_pred,eig_label=eigenvalues_lbl,**self.loss_options)

                    self.test_loss = loss.detach()
                    state = {'field': 'iteration', "test_loss": self.test_loss}
                    
                    self.call_plugins(queue_name='iteration', time=self.iteration, **state)
                    self.iteration += 1
                    
                    idata += 1

                torch.save(eigenvalues_pred_collect, self.results_path + '/eigenvalues_pred_ips' + str(iprocess))
                torch.save(eigenvalues_lbel_collect, self.results_path + '/eigenvalues_lbel_ips'+ str(iprocess))
                iprocess += 1

