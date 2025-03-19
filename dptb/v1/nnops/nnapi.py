import torch
import numpy  as np
from dptb.dataprocess.processor import Processor
from dptb.nnet.nntb import NNTB
from dptb.nnsktb.sknet import SKNet
from dptb.sktb.skIntegrals import SKIntegrals
from dptb.sktb.struct_skhs import SKHSLists
from dptb.hamiltonian.hamil_eig_sk_crt import HamilEig
from dptb.utils.constants import dtype_dict
from dptb.structure.structure import BaseStruct
from dptb.utils.tools import nnsk_correction
from abc import ABC, abstractmethod

from dptb.utils.index_mapping import Index_Mapings

from dptb.nnsktb.integralFunc import SKintHops
from dptb.nnsktb.onsiteFunc import onsiteFunc, loadOnsite
from dptb.nnsktb.skintTypes import all_skint_types, all_onsite_intgrl_types

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
        f=torch.load(dptb_checkpoint, weights_only=False)
        model_config = f["model_config"]
        self.nntb = NNTB(**model_config)
        self.nntb.tb_net.load_state_dict(f['state_dict'])
        self.nntb.tb_net.eval()

        self.sktbmode = sktbmode
        self.unitenergy = model_config.get('unit','Hartree') 

        if sktbmode == 'nnsk':
            f = torch.load(nnsk_checkpoint, weights_only=False)
            model_config = f["model_config"]
            self.sknet = SKNet(**model_config)
            self.sknet.load_state_dict(f['state_dict'])
            self.sknet.eval()
            #for p in self.sknet.parameters():
            #    p.requires_grad = False

            indmap = Index_Mapings(proj_atom_anglr_m)
            bond_index_map, bond_num_hops =  indmap.Bond_Ind_Mapings()
            onsite_index_map, onsite_num = indmap.Onsite_Ind_Mapings()

            self.hops_fun = SKintHops(mode=model_config.get("skformula", "powerlaw"))
            self.onsite_db = loadOnsite(onsite_index_map)
            all_skint_types_dict, reducted_skint_types, self.sk_bond_ind_dict = all_skint_types(bond_index_map)

        else:
            skint = SKIntegrals(proj_atom_anglr_m = proj_atom_anglr_m, sk_file_path=sk_file_path)
            self.skhslist = SKHSLists(skint,dtype='tensor')
        self.hamileig = HamilEig(dtype=dtype_dict[model_config["dtype"]])
        
        self.if_HR_ready=False

    def get_HR(self, structure, env_cutoff, device='cpu', dtype=torch.float32):
        assert isinstance(structure, BaseStruct)
        self.structure = structure
        self.time_symm = structure.time_symm
        predict_process = Processor(mode='dptb', structure_list=structure, batchsize=1, kpoint=None, eigen_list=None, env_cutoff=env_cutoff, device=device, dtype=dtype)
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
        eigenvalues,_ = self.hamileig.Eigenvalues(kpoints, time_symm=self.time_symm,dtype=self.hamileig.dtype, unit=self.unitenergy)
        eigks = eigenvalues.detach().numpy()

        num_el = np.sum(self.structure.proj_atom_neles_per)
        nk = len(kpoints)
        numek = num_el * nk // spindeg
        sorteigs =  np.sort(np.reshape(eigks,[-1]))
        EF=(sorteigs[numek] + sorteigs[numek-1])/2
        return eigks, EF

class NNSK(ModelAPI):
    def __init__(self,checkpoint, proj_atom_anglr_m):
        f=torch.load(checkpoint, weights_only=False)
        model_config = f["model_config"]
        self.onsitemode =  model_config['onsitemode']
        self.onsite_cutoff = model_config.get('onsite_cutoff',0.)
        self.model = SKNet(**model_config)
        self.model.load_state_dict(f['state_dict'])
        self.sk_options = f.get("sk_options", None)
        self.model.eval()
        self.unitenergy = model_config.get('unit', 'Hartree')
        
        if self.sk_options is not None:
            self.skformula = self.sk_options["skformula"]
            self.sk_cutoff = self.sk_options["sk_cutoff"]
            self.sk_decay_w = self.sk_options["sk_decay_w"]
        else:
            self.skformula = "varTang96"
            self.sk_cutoff = torch.tensor(6.0)
            self.sk_decay_w = torch.tensor(0.1)

        self.indmap = Index_Mapings(proj_atom_anglr_m)
        bond_index_map, bond_num_hops =  self.indmap.Bond_Ind_Mapings()
        self.onsite_strain_index_map, onsite_strain_num, onsite_index_map, onsite_num = self.indmap.Onsite_Ind_Mapings(onsitemode=self.onsitemode, atomtype=model_config.get("atom_type"))
            
        self.hops_fun = SKintHops(mode=self.skformula)
        self.onsite_db = loadOnsite(onsite_index_map)
        all_skint_types_dict, reducted_skint_types, self.sk_bond_ind_dict = all_skint_types(bond_index_map)
        
        self.hamileig = HamilEig(dtype=dtype_dict[model_config["dtype"]])
        self.if_HR_ready=False

    def get_HR(self, structure, device='cpu', dtype=torch.float32):
        assert isinstance(structure, BaseStruct)
        assert structure.onsitemode == self.onsitemode
        self.structure = structure
        self.time_symm = structure.time_symm
        predict_process = Processor(structure_list=structure, batchsize=1, kpoint=None, eigen_list=None, device=device, dtype=dtype, 
        env_cutoff=self.onsite_cutoff, onsitemode=self.onsitemode, onsite_cutoff=self.onsite_cutoff, sorted_onsite="st", sorted_bond="st", sorted_env="st")

        batch_bond, batch_bond_onsites = predict_process.get_bond(sorted="st")
        if self.onsitemode == 'strain':
            batch_envs = predict_process.get_env(sorted="st")
            nn_onsiteE, onsite_coeffdict = self.model(mode='onsite')
            all_onsiteint_types_dcit, reducted_onsiteint_types, self.onsite_strain_ind_dict = all_onsite_intgrl_types(self.onsite_strain_index_map)
            batch_onsiteVs = self.hops_fun.get_skhops(batch_bonds=batch_envs, coeff_paras=onsite_coeffdict, sk_bond_ind=self.onsite_strain_ind_dict)
            batch_onsiteEs = onsiteFunc(batch_bonds_onsite=batch_bond_onsites, onsite_db=self.onsite_db, nn_onsiteE=None)

        else:
            nn_onsiteE, onsite_coeffdict = self.model(mode='onsite')
            batch_onsiteEs = onsiteFunc(batch_bonds_onsite=batch_bond_onsites, onsite_db=self.onsite_db, nn_onsiteE=nn_onsiteE)


        coeffdict = self.model(mode='hopping')
        batch_hoppings = self.hops_fun.get_skhops(batch_bonds=batch_bond, coeff_paras=coeffdict, sk_bond_ind=self.sk_bond_ind_dict, rcut=self.sk_cutoff, w=self.sk_decay_w)
        onsiteEs, hoppings = batch_onsiteEs[0], batch_hoppings[0]
        if self.onsitemode == 'strain':
            onsiteVs = batch_onsiteVs[0]
            self.hamileig.update_hs_list(struct=structure, hoppings=hoppings, onsiteEs=onsiteEs, onsiteVs=onsiteVs)
            self.hamileig.get_hs_blocks(bonds_onsite=np.asarray(batch_bond_onsites[0][:,1:]),
                                        bonds_hoppings=np.asarray(batch_bond[0][:,1:]), 
                                        onsite_envs=np.asarray(batch_envs[0][:,1:]))
        else:
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
            hkmat =  self.hamileig.hs_block_R2k(kpoints=kpoints, HorS='H', time_symm=self.time_symm)
            skmat =  self.hamileig.hs_block_R2k(kpoints=kpoints, HorS='S', time_symm=self.time_symm)
        else:
            hkmat =  self.hamileig.hs_block_R2k(kpoints=kpoints, HorS='H', time_symm=self.time_symm)
            skmat = torch.eye(hkmat.shape[1], dtype=torch.complex64).unsqueeze(0).repeat(hkmat.shape[0], 1, 1)
        return hkmat, skmat

    def get_eigenvalues(self,kpoints,spindeg=2):
        assert self.if_HR_ready
        eigenvalues,_ = self.hamileig.Eigenvalues(kpoints, time_symm=self.time_symm, dtype=self.hamileig.dtype, unit=self.unitenergy)
        eigks = eigenvalues.detach().numpy()

        num_el = np.sum(self.structure.proj_atom_neles_per)
        nk = len(kpoints)
        numek = num_el * nk // spindeg
        sorteigs =  np.sort(np.reshape(eigks,[-1]))
        EF=(sorteigs[numek] + sorteigs[numek-1])/2
        return eigks, EF