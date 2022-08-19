import torch
import numpy  as np
from dptb.dataprocess.processor import Processor
from dptb.nnet.nntb import NNTB
from dptb.nnsktb.sknet import SKNet
from dptb.sktb.skIntegrals import SKIntegrals
from dptb.sktb.struct_skhs import SKHSLists
from dptb.hamiltonian.hamil_eig_sk import HamilEig
from dptb.structure.structure import BaseStruct
from dptb.utils.tools import nnsk_correction
from abc import ABC, abstractmethod

from dptb.utils.tools import Index_Mapings

from dptb.nnsktb.integralFunc import SKintHops
from dptb.nnsktb.onsiteFunc import onsiteFunc, loadOnsite
from dptb.nnsktb.skintTypes import all_skint_types

class ModelAPI(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_HR(self,**kwargs):
        pass

    @abstractmethod
    def get_HK(self, **kwargs):
        pass

    @abstractmethod
    def get_eigenvalues(self, **kwargs):
        pass



class DeePTB(ModelAPI):
    def __init__(self, dptb_checkpoint:str, proj_atom_anglr_m:dict, 
                            sktbmode:str='nnsk', nnsk_checkpoint:str = None, sk_file_path=None):
        f=torch.load(dptb_checkpoint)
        model_config = f["model_config"]
        self.nntb = NNTB(**model_config)
        self.nntb.tb_net.load_state_dict(f['state_dict'])
        self.nntb.tb_net.eval()

        self.sktbmode = sktbmode

        if sktbmode == 'nnsk':
            f = torch.load(nnsk_checkpoint)
            model_config = f["model_config"]
            self.sknet = SKNet(**model_config)
            self.sknet.load_state_dict(f['state_dict'])
            self.sknet.eval()
            #for p in self.sknet.parameters():
            #    p.requires_grad = False

            indmap = Index_Mapings(proj_atom_anglr_m)
            bond_index_map, bond_num_hops =  indmap.Bond_Ind_Mapings()
            onsite_index_map, onsite_num = indmap.Onsite_Ind_Mapings()

            self.hops_fun = SKintHops()
            self.onsite_db = loadOnsite(onsite_index_map)
            all_skint_types_dict, reducted_skint_types, self.sk_bond_ind_dict = all_skint_types(bond_index_map)

        else:
            skint = SKIntegrals(proj_atom_anglr_m = proj_atom_anglr_m, sk_file_path=sk_file_path)
            self.skhslist = SKHSLists(skint,dtype='tensor')
        self.hamileig = HamilEig(dtype='tensor')
        
        self.if_HR_ready=False

    def get_HR(self, structure, env_cutoff, device='cpu', dtype=torch.float32):
        assert isinstance(structure, BaseStruct)
        self.structure = structure
        self.time_symm = structure.time_symm
        predict_process = Processor(mode='dptb', structure_list=structure, batchsize=1, kpoint=None, eigen_list=None, env_cutoff=env_cutoff, require_dict=False, device=device, dtype=dtype)
        bond, bond_onsite = predict_process.get_bond()
        env = predict_process.get_env()
        batched_dcp = self.nntb.get_desciptor(env)
        # get hoppings (SK type bond integrals.)    
        batch_bond_hoppings, batch_hoppings = self.nntb.hopping(batched_dcp=batched_dcp, batch_bond=bond)
        # get onsite energies
        batch_bond_onsites, batch_onsiteEs = self.nntb.onsite(batched_dcp=batched_dcp)    

        if self.sktbmode == 'nnsk':
            coeffdict = self.sknet()
            sktb_onsiteEs = onsiteFunc(batch_bond_onsites, self.onsite_db)
            sktb_hoppings = self.hops_fun.get_skhops(batch_bond_hoppings, coeffdict, self.sk_bond_ind_dict)
            
            # combine the nn and sk part for the hamiltonian.
            onsiteEs, hoppings, onsiteSs, overlaps = \
                            nnsk_correction(nn_onsiteEs=batch_onsiteEs[0],
                                            nn_hoppings=batch_hoppings[0],
                                            sk_onsiteEs=sktb_onsiteEs[0], 
                                            sk_hoppings=sktb_hoppings[0])
        else:
            # get the sk parameters.
            self.skhslist.update_struct(structure)
            self.skhslist.get_HS_list(bonds_onsite=np.asarray(batch_bond_onsites[0][:,1:]),
                                                bonds_hoppings=np.asarray(batch_bond_hoppings[0][:,1:]))
            # combine the nn and sk part for the hamiltonian.
            onsiteEs, hoppings, onsiteSs, overlaps = \
                            nnsk_correction(nn_onsiteEs=batch_onsiteEs[0],
                                            nn_hoppings=batch_hoppings[0],
                                            sk_onsiteEs=self.skhslist.onsiteEs, 
                                            sk_hoppings=self.skhslist.hoppings,
                                            sk_onsiteSs=self.skhslist.onsiteSs, 
                                            sk_overlaps=self.skhslist.overlaps)
        
        self.hamileig.update_hs_list(structure, hoppings, onsiteEs, overlaps, onsiteSs)
        self.hamileig.get_hs_blocks(bonds_onsite=np.asarray(batch_bond_onsites[0][:,1:]),
                                        bonds_hoppings=np.asarray(batch_bond_hoppings[0][:,1:]))

        self.if_HR_ready=True
        if not self.hamileig.use_orthogonal_basis:
            return self.hamileig.all_bonds, self.hamileig.hamil_blocks, None
        else:
            return self.hamileig.all_bonds, self.hamileig.hamil_blocks, self.hamileig.overlap_blocks

    # ToDo 现在版本的程序对于正交基和非正交基组的情况有些不兼容的地方。后续要修改！
    def get_HK(self, kpoints):
        assert self.if_HR_ready

        if not self.hamileig.use_orthogonal_basis:
            hkmat =  self.hamileig.hs_block_R2k(kpoints=kpoints, HorS='H', time_symm=self.time_symm, dtype=self.hamileig.dtype)
            skmat =  self.hamileig.hs_block_R2k(kpoints=kpoints, HorS='S', time_symm=self.time_symm, dtype=self.hamileig.dtype)
        else:
            hkmat =  self.hamileig.hs_block_R2k(kpoints=kpoints, HorS='H', time_symm=self.time_symm, dtype=self.hamileig.dtype)
            skmat = torch.eye(hkmat.shape[1], dtype=torch.complex64).unsqueeze(0).repeat(hkmat.shape[0], 1, 1)
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

class NNSK(ModelAPI):
    def __init__(self,checkpoint, proj_atom_anglr_m):
        f=torch.load(checkpoint)
        model_config = f["model_config"]

        self.model = SKNet(**model_config)
        self.model.load_state_dict(f['state_dict'])
        self.model.eval()

        indmap = Index_Mapings(proj_atom_anglr_m)
        bond_index_map, bond_num_hops =  indmap.Bond_Ind_Mapings()
        onsite_index_map, onsite_num = indmap.Onsite_Ind_Mapings()

        self.hops_fun = SKintHops()
        self.onsite_db = loadOnsite(onsite_index_map)
        all_skint_types_dict, reducted_skint_types, self.sk_bond_ind_dict = all_skint_types(bond_index_map)
        
        self.hamileig = HamilEig(dtype='tensor')
        self.if_HR_ready=False


    def get_HR(self, structure, device='cpu', dtype=torch.float32):
        assert isinstance(structure, BaseStruct)
        self.structure = structure
        self.time_symm = structure.time_symm
        predict_process = Processor(mode='nnsk', structure_list=structure, batchsize=1, kpoint=None, eigen_list=None, require_dict=True, device=device, dtype=dtype)
        batch_bond, batch_bond_onsites = predict_process.get_bond()

        coeffdict = self.model()
        batch_onsiteEs = onsiteFunc(batch_bond_onsites, self.onsite_db)
        batch_hoppings = self.hops_fun.get_skhops(batch_bond, coeffdict, self.sk_bond_ind_dict)
        onsiteEs, hoppings = batch_onsiteEs[0], batch_hoppings[0]

        self.hamileig.update_hs_list(struct=structure, hoppings=hoppings, onsiteEs=onsiteEs)
        self.hamileig.get_hs_blocks(bonds_onsite=np.asarray(batch_bond_onsites[0][:,1:]),
                                        bonds_hoppings=np.asarray(batch_bond[0][:,1:]))
        
        self.if_HR_ready=True

        if not self.hamileig.use_orthogonal_basis:
            return self.hamileig.all_bonds, self.hamileig.hamil_blocks, None
        else:
            return self.hamileig.all_bonds, self.hamileig.hamil_blocks, self.hamileig.overlap_blocks

    def get_HK(self, kpoints):
        assert self.if_HR_ready

        if not self.hamileig.use_orthogonal_basis:
            hkmat =  self.hamileig.hs_block_R2k(kpoints=kpoints, HorS='H', time_symm=self.time_symm, dtype=self.hamileig.dtype)
            skmat =  self.hamileig.hs_block_R2k(kpoints=kpoints, HorS='S', time_symm=self.time_symm, dtype=self.hamileig.dtype)
        else:
            hkmat =  self.hamileig.hs_block_R2k(kpoints=kpoints, HorS='H', time_symm=self.time_symm, dtype=self.hamileig.dtype)
            skmat = torch.eye(hkmat.shape[1], dtype=torch.complex64).unsqueeze(0).repeat(hkmat.shape[0], 1, 1)
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