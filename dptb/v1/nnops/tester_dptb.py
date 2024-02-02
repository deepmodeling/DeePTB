import torch
import logging
import numpy as np
from dptb.hamiltonian.hamil_eig_sk_crt import HamilEig
from dptb.nnsktb.onsiteFunc import loadOnsite
from dptb.nnops.v1.loss import loss_type1, loss_spectral
from dptb.utils.tools import get_uniq_symbol, get_lr_scheduler, \
get_optimizer, nnsk_correction, j_must_have

from dptb.nnops.v1.trainloss import lossfunction
from dptb.nnops.base_tester import Tester

log = logging.getLogger(__name__)

class DPTBTester(Tester):

    def __init__(self, run_opt, jdata) -> None:
        super(DPTBTester, self).__init__(jdata)
        self.name = "dptb"
        self.run_opt = run_opt
        self._init_param(jdata)
    
    def _init_param(self, jdata):
        common_options = j_must_have(jdata, "common_options")
        data_options = j_must_have(jdata,"data_options")
        model_options = j_must_have(jdata, "model_options")
        loss_options = j_must_have(jdata, "loss_options")

        self.common_options = common_options
        self.data_options = data_options
        self.model_options = model_options
        self.loss_options = loss_options
        self.results_path = self.run_opt['results_path']



        self.batch_size = data_options["test"]['batch_size']

        self.proj_atom_anglr_m = common_options.get('proj_atom_anglr_m')
        self.proj_atom_neles = common_options.get('proj_atom_neles')
        self.onsitemode = common_options.get('onsitemode','none')
        self.atomtype = get_uniq_symbol(common_options["atomtype"])
        self.soc = common_options['soc']
        self.overlap = common_options['overlap']
        self.proj_atomtype = get_uniq_symbol(list(self.proj_atom_anglr_m.keys()))

        self.band_min = loss_options.get('band_min', 0)
        self.band_max = loss_options.get('band_max', None)

    def build(self):
        # ---------------------------------------------------------------- init onsite and hopping functions  ----------------------------------------------------------------
        self.call_plugins(queue_name='disposable', time=0, **self.model_options, **self.common_options, **self.data_options, **self.run_opt)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.test_lossfunc = getattr(lossfunction(self.criterion), 'eigs_l2')
        self.decompose = True
        self.hamileig = HamilEig(dtype=self.dtype, device=self.device)
    

    def calc(self, batch_bond, batch_bond_onsites, batch_env, batch_onsitenvs, structs, kpoints, eigenvalues, wannier_blocks, decompose=True):
        '''
        conduct one step forward computation, used in train, test and validation.
        '''
        assert len(kpoints.shape) == 2, "kpoints should have shape of [num_kp, 3]."

        batch_bond_hoppings, batch_hoppings, \
        batch_bond_onsites, batch_onsiteEs, batch_soc_lambdas = self.nntb.calc(batch_bond, batch_env)

        if self.run_opt.get("use_correction", False):
            coeffdict, overlap_coeffdict = self.sknet(mode='hopping')
            batch_nnsk_hoppings = self.hops_fun.get_skhops(
                batch_bonds=batch_bond_hoppings, coeff_paras=coeffdict, rcut=self.model_options["skfunction"]["sk_cutoff"],
                w=self.model_options["skfunction"]["sk_decay_w"])
            nnsk_onsiteE, onsite_coeffdict = self.sknet(mode='onsite')
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

        # ToDo: Advance the correction process before onsite_fun and hops_fun
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

                onsiteEs, hoppings, onsiteSs, overlaps, soc_lambdas = nnsk_correction(nn_onsiteEs=batch_onsiteEs[ii], nn_hoppings=batch_hoppings[ii],
                                    sk_onsiteEs=batch_nnsk_onsiteEs[ii], sk_hoppings=batch_nnsk_hoppings[ii],
                                    sk_onsiteSs=None, sk_overlaps=nnsk_overlaps, 
                                    nn_soc_lambdas=nn_soc_lambdas, 
                                    sk_soc_lambdas=sk_soc_lambdas)

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

    def test(self) -> None:
        with torch.no_grad():
            iprocess =0
            for processor in self.test_processor_list:
                idata = 0
                self.loss_options.update(processor.bandinfo)
                for data in processor:
                    pred, label = self.calc(*data, decompose=self.decompose)
                    
                    if idata ==0:
                        eigenvalues_pred_collect = pred.clone()
                        eigenvalues_lbel_collect = label.clone()
                    else:
                        eigenvalues_pred_collect = torch.cat([eigenvalues_pred_collect,pred],dim=0)
                        eigenvalues_lbel_collect = torch.cat([eigenvalues_lbel_collect,label],dim=0)

                    # num_kp = kpoints.shape[0]
                    # num_el = np.sum(structs[0].proj_atom_neles_per)

                    #self.loss_options.update({'num_el':num_el})
                    loss = self.test_lossfunc(eig_pred=pred, eig_label=label,**self.loss_options)

                    self.test_loss = loss.detach()
                    state = {'field': 'iteration', "test_loss": self.test_loss}
                    
                    self.call_plugins(queue_name='iteration', time=self.iteration, **state)
                    self.iteration += 1
                    
                    idata += 1

                torch.save(eigenvalues_pred_collect, self.results_path + '/eigenvalues_pred_ips' + str(iprocess))
                torch.save(eigenvalues_lbel_collect, self.results_path + '/eigenvalues_lbel_ips'+ str(iprocess))
                iprocess += 1

