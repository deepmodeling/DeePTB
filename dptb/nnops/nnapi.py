import torch
import numpy  as np
from dptb.dataprocess.processor import Processor
from dptb.nnet.nntb import NNTB
from dptb.sktb.skIntegrals import SKIntegrals
from dptb.sktb.struct_skhs import SKHSLists
from dptb.hamiltonian.hamil_eig_sk import HamilEig
from dptb.structure.structure import BaseStruct
from dptb.utils.tools import nnsk_correction

class DeePTB(object):
    def __init__(self,checkpoint,sk_file_path, proj_atom_anglr_m):
        f=torch.load(checkpoint)
        model_config = f["model_config"]
        self.nntb = NNTB(**model_config)
        self.nntb.tb_net.load_state_dict(f['state_dict'])
        self.nntb.tb_net.eval()
        skint = SKIntegrals(proj_atom_anglr_m = proj_atom_anglr_m, sk_file_path=sk_file_path)
        self.skhslist = SKHSLists(skint,dtype='numpy')
        self.hamileig = HamilEig(dtype='tensor')
        self.if_HR_ready=False

    def get_HR(self, structure, env_cutoff,device='cpu', dtype=torch.float32):
        assert isinstance(structure,BaseStruct)
        self.structure = structure
        self.time_symm = structure.time_symm
        predict_process = Processor(structure_list=structure, batchsize=1, kpoint=None, eigen_list=None, env_cutoff=env_cutoff, device=device, dtype=dtype)
        
        bond  = predict_process.get_bond()
        env = predict_process.get_env()
        
        batched_dcp = self.nntb.get_desciptor(env)
        # get hoppings (SK type bond integrals.)    
        batch_bond_hoppings, batch_hoppings = self.nntb.hopping(batched_dcp=batched_dcp, batch_bond=bond)
        # get onsite energies
        batch_bond_onsites, batch_onsiteEs = self.nntb.onsite(batched_dcp=batched_dcp)    

        # get the sk parameters.
        self.skhslist.update_struct(structure)
        self.skhslist.get_HS_list(bonds_onsite=np.asarray(batch_bond_onsites[0]),
                                                bonds_hoppings=np.asarray(batch_bond_hoppings[0]))
        # combine the nn and sk part for the hamiltonian.
        onsiteEs, hoppings, onsiteSs, overlaps = \
                            nnsk_correction(nn_onsiteEs=batch_onsiteEs[0],
                                            nn_hoppings=batch_hoppings[0],
                                            sk_onsiteEs=self.skhslist.onsiteEs, 
                                            sk_hoppings=self.skhslist.hoppings,
                                            sk_onsiteSs=self.skhslist.onsiteSs, 
                                            sk_overlaps=self.skhslist.overlaps)
        
        self.hamileig.update_hs_list(structure, hoppings, onsiteEs, overlaps, onsiteSs)
        self.hamileig.get_hs_blocks(bonds_onsite = np.asarray(batch_bond_onsites[0]), 
                                    bonds_hoppings=np.asarray(batch_bond_hoppings[0]))

        self.if_HR_ready=True
        if not self.hamileig.use_orthogonal_basis:
            return self.hamileig.all_bonds, self.hamileig.hamil_blocks, None
        else:
            return self.hamileig.all_bonds, self.hamileig.hamil_blocks, self.hamileig.overlap_blocks

    # ToDo 现在版本的程序对于正交基和非正交基组的情况有些不兼容的地方。后续要修改！
    def get_HK(self, kpoints):
        assert self.if_HR_ready

        hkmat =  self.hamileig.hs_block_R2k(kpoints=kpoints, HorS='H', time_symm=self.time_symm, dtype=self.hamileig.dtype)
        skmat =  self.hamileig.hs_block_R2k(kpoints=kpoints, HorS='S', time_symm=self.time_symm, dtype=self.hamileig.dtype)

        return hkmat, skmat
    
    def get_eigenvalues(self,kpoints,spindeg=2):
        assert self.if_HR_ready
        eigenvalues = self.hamileig.Eigenvalues(kpoints, time_symm=self.time_symm,dtype=self.hamileig.dtype)
        eigks = eigenvalues.detach().numpy()

        num_el = np.sum(self.structure.proj_atom_neles_per)
        nk = len(kpoints)
        numek = num_el * nk // spindeg
        sorteigs =  np.sort(np.reshape(eigks,[-1]))
        EF=(sorteigs[numek] + sorteigs[numek-1])/2
        return eigks, EF
