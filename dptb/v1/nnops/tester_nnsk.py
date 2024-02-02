from cProfile import run
import torch
import logging
import numpy as np
from dptb.nnops.base_tester import Tester
from dptb.utils.tools import get_uniq_symbol, \
    get_lr_scheduler, get_optimizer, j_must_have
from dptb.hamiltonian.hamil_eig_sk_crt import HamilEig
from dptb.nnsktb.onsiteFunc import loadOnsite
from dptb.nnops.v1.trainloss import lossfunction

log = logging.getLogger(__name__)

class NNSKTester(Tester):
    def __init__(self, run_opt, jdata) -> None:
        super(NNSKTester, self).__init__(jdata)
        self.name = "nnsk"
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

        

        # initialize loss options
        # ----------------------------------------------------------------------------------------------------------------------------------------------
        self.batch_size = data_options["test"]['batch_size']
        
        self.soc = common_options['soc']
        self.overlap = common_options['overlap']
        self.proj_atom_anglr_m = common_options.get('proj_atom_anglr_m')
        self.proj_atom_neles = common_options.get('proj_atom_neles')
        self.onsitemode = common_options.get('onsitemode','none')
        self.atomtype = get_uniq_symbol(common_options["atomtype"])
        self.proj_atomtype = get_uniq_symbol(list(self.proj_atom_anglr_m.keys()))
 
    def build(self):
        # ---------------------------------------------------------------- init onsite and hopping functions  ----------------------------------------------------------------
        self.call_plugins(queue_name='disposable', time=0, **self.model_options, **self.common_options, **self.data_options, **self.run_opt)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.test_lossfunc = getattr(lossfunction(self.criterion), 'eigs_l2')
        self.decompose = True
        self.hamileig = HamilEig(dtype=self.dtype, device=self.device)
    
    def calc(self, batch_bonds, batch_bond_onsites, batch_envs, batch_onsitenvs, structs, kpoints, eigenvalues, wannier_blocks, decompose=True):
        if len(kpoints.shape) != 2: 
            log.error(msg="kpoints should have shape of [num_kp, 3].")
            raise ValueError
        if (wannier_blocks[0] is None) and not (self.decompose):
            log.error(msg="The wannier_blocks from processor is None, but the losstype wannier, please check the input data, maybe the wannier.npy is not there.")
            raise ValueError

        coeffdict, overlap_coeffdict = self.model(mode='hopping')
        batch_hoppings = self.hops_fun.get_skhops(batch_bonds=batch_bonds, coeff_paras=coeffdict, 
            rcut=self.model_options["skfunction"]["sk_cutoff"], w=self.model_options["skfunction"]["sk_decay_w"])
        
        nn_onsiteE, onsite_coeffdict = self.model(mode='onsite')
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
       
        # call sktb to get the sktb hoppings and onsites
        self.onsite_index_dict = self.model.onsite_index_dict
        self.hopping_coeff = coeffdict
        self.overlap_coeff = overlap_coeffdict
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
                onsitenvs = batch_onsitenvs[ii][:,1:]
                # call hamiltonian block
            else:
                onsiteEs, hoppings = batch_onsiteEs[ii], batch_hoppings[ii]
                onsiteVs = None
                onsitenvs = None
                # call hamiltonian block
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
        else:
            # directly return batch_onsiteEs, batch_hoppings, batch_onsiteVs, batch_soc_lambdas
            return batch_onsiteEs, batch_hoppings, batch_onsiteVs, batch_soc_lambdas


        
    
    def test(self):
        with torch.no_grad():
            iprocess =0
            for processor in self.test_processor_list:
                idata = 0
                self.loss_options.update(processor.bandinfo)

                for data in processor:
                    pred, label = self.calc(*data, decompose=self.decompose)
                    
                    #eigenvalues_lbl = torch.from_numpy(eigenvalues.astype(float)).float()
                    if idata ==0:
                        eigenvalues_pred_collect = pred.clone()
                        eigenvalues_lbel_collect = label.clone()
                    else:
                        eigenvalues_pred_collect = torch.cat([eigenvalues_pred_collect, pred],dim=0)
                        eigenvalues_lbel_collect = torch.cat([eigenvalues_lbel_collect, label],dim=0)

                    # num_kp = kpoints.shape[0]
                    # num_el = np.sum(structs[0].proj_atom_neles_per)
                    # self.loss_options.update({'num_el':num_el})
                    loss = self.test_lossfunc(eig_pred=pred,eig_label=label,**self.loss_options)
            

                    self.test_loss = loss.detach()
                    state = {'field': 'iteration', "test_loss": self.test_loss}

                    self.call_plugins(queue_name='iteration', time=self.iteration, **state)
                    self.iteration += 1

                    idata += 1
                
                torch.save(eigenvalues_pred_collect, self.results_path + '/eigenvalues_pred_ips' + str(iprocess))
                torch.save(eigenvalues_lbel_collect, self.results_path + '/eigenvalues_lbel_ips'+ str(iprocess))
                
                iprocess += 1
                # save the eigenvalues_pred.