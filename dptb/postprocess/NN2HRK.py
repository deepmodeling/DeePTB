import torch
import numpy as np
from dptb.structure.structure import BaseStruct
from dptb.dataprocess.processor import Processor
from dptb.hamiltonian.hamil_eig_sk_crt import HamilEig

class NN2HRK(object):
    def __init__(self, apihost, mode):
        assert mode in ['nnsk', 'dptb']
        self.apihost = apihost
        self.mode = mode
        self.hamileig = HamilEig(dtype='tensor')
        
        self.if_nn_HR_ready = False
        self.if_dp_HR_ready = False
    
        ## parameters.
        self.device = apihost.model_config['device']
        self.dtype =  apihost.model_config['dtype']
        self.env_cutoff = self.apihost.model_config['env_cutoff']
        self.onsitemode = self.apihost.model_config['onsitemode']
        self.onsite_cutoff = self.apihost.model_config['onsite_cutoff']
        self.sk_cutoff = self.apihost.model_config['skfunction']['sk_cutoff']
        self.sk_decay_w = self.apihost.model_config['skfunction']['sk_decay_w']


    def update_struct(self,structure):
        # update status is the structure is update.
        self.structure = structure
        self.time_symm = self.structure.time_symm
        self.if_dp_HR_ready = False
        self.if_nn_HR_ready = False

    def get_HR(self):
        if self.mode == 'nnsk' and not self.if_nn_HR_ready:
            self._get_nnsk_HR()
            
        if self.mode == 'dptb' and not self.if_dp_HR_ready:
            self._get_dptb_HR()

        return self.allbonds, self.hamil_blocks, self.overlap_blocks 
    
    
    def get_HK(self, kpoints):
        assert self.if_nn_HR_ready or self.if_dp_HR_ready, "The HR shoule be calcualted before call for HK." 

        if not self.use_orthogonal_basis:
            hkmat =  self.hamileig.hs_block_R2k(kpoints=kpoints, HorS='H', time_symm=self.time_symm, dtype=self.hamileig.dtype)
            skmat =  self.hamileig.hs_block_R2k(kpoints=kpoints, HorS='S', time_symm=self.time_symm, dtype=self.hamileig.dtype)
        else:
            hkmat =  self.hamileig.hs_block_R2k(kpoints=kpoints, HorS='H', time_symm=self.time_symm, dtype=self.hamileig.dtype)
            skmat = torch.eye(hkmat.shape[1], dtype=torch.complex64).unsqueeze(0).repeat(hkmat.shape[0], 1, 1)
        return hkmat, skmat
    
    def get_eigenvalues(self,kpoints,spindeg=2):
        assert self.if_nn_HR_ready or self.if_dp_HR_ready, "The HR shoule be calcualted before call for HK." 
        eigenvalues,_ = self.hamileig.Eigenvalues(kpoints, time_symm=self.time_symm,dtype=self.hamileig.dtype)
        eigks = eigenvalues.detach().numpy()

        num_el = np.sum(self.structure.proj_atom_neles_per)
        nk = len(kpoints)
        numek = num_el * nk // spindeg
        sorteigs =  np.sort(np.reshape(eigks,[-1]))
        EF=(sorteigs[numek] + sorteigs[numek-1])/2
        return eigks, EF

    def _get_nnsk_HR(self):
        assert isinstance(self.structure, BaseStruct)
        assert self.structure.onsitemode == self.onsitemode
        # TODO: 注意检查 processor 关于 env_cutoff 和 onsite_cutoff.
        predict_process = Processor(structure_list=self.structure, batchsize=1, kpoint=None, eigen_list=None, device=self.device, dtype=self.dtype, 
                                        env_cutoff=self.env_cutoff, onsitemode=self.onsitemode, onsite_cutoff=self.onsite_cutoff, sorted_onsite="st", sorted_bond="st", sorted_env="st")

        batch_bonds, batch_bond_onsites = predict_process.get_bond(sorted="st")
        coeffdict = self.apihost.model(mode='hopping')
        batch_hoppings = self.apihost.hops_fun.get_skhops(batch_bonds=batch_bonds, coeff_paras=coeffdict, rcut=self.sk_cutoff, w=self.sk_decay_w)
        nn_onsiteE, onsite_coeffdict = self.apihost.model(mode='onsite')
        batch_onsiteEs = self.apihost.onsite_fun(batch_bonds_onsite=batch_bond_onsites, onsite_db=self.apihost.onsite_db, nn_onsiteE=nn_onsiteE)
        
        if self.onsitemode == 'strain':
            # TODO: 注意检查 processor get_env 以及get_onsite_env 涉及的 env_cutoff 和 onsite_cutoff.
            batch_onsite_envs = predict_process.get_env(sorted="st")
            batch_onsiteVs = self.apihost.onsitestrain_fun.get_skhops(batch_bonds=batch_onsite_envs, coeff_paras=onsite_coeffdict)

        if self.onsitemode == 'strain':
            onsiteEs, hoppings, onsiteVs = batch_onsiteEs[0], batch_hoppings[0], batch_onsiteVs[0]
            onsitenvs = np.asarray(batch_onsite_envs[0][:,1:])
        else:
            onsiteEs, hoppings, onsiteVs = batch_onsiteEs[0], batch_hoppings[0],  None
            onsitenvs = None

        self.hamileig.update_hs_list(struct=self.structure, hoppings=hoppings, onsiteEs=onsiteEs, onsiteVs=onsiteVs)
        self.hamileig.get_hs_blocks(bonds_onsite=np.asarray(batch_bond_onsites[0][:,1:]), bonds_hoppings=np.asarray(batch_bonds[0][:,1:]), 
                                    onsite_envs=onsitenvs)
        
        # 同一个类实例, 只能计算一种TB hamiltonian. 
        self.if_nn_HR_ready = True
        self.if_dp_HR_ready = False
        self.use_orthogonal_basis = self.hamileig.use_orthogonal_basis
        self.allbonds, self.hamil_blocks = self.hamileig.all_bonds, self.hamileig.hamil_blocks
        
        if not self.hamileig.use_orthogonal_basis:
            self.overlap_blocks = None
        else:
            self.overlap_blocks = self.hamileig.overlap_blocks
    
    def _get_dptb_HR(self):
        self.allbonds, self.hamil_blocks, self.overlap_blocks = None, None, None
        pass

