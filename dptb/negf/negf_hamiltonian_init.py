from typing import List
import torch
from dptb.negf.areshkin_pole_sum import pole_maker
from dptb.negf.recursive_green_cal import recursive_gf
from dptb.negf.surface_green import selfEnergy
from dptb.negf.negf_utils import quad, gauss_xw,update_kmap,leggauss
from dptb.negf.ozaki_res_cal import ozaki_residues
from dptb.negf.areshkin_pole_sum import pole_maker
from ase.io import read,write
from dptb.negf.poisson import Density2Potential, getImg
from dptb.negf.scf_method import SCFMethod
import logging
import os
import torch.optim as optim
from dptb.utils.tools import j_must_have
import numpy as np
from dptb.utils.make_kpoints import kmesh_sampling

import ase
from dptb.data import AtomicData, AtomicDataDict
from typing import Optional, Union
from dptb.nn.energy import Eigenvalues
from dptb.nn.hamiltonian import E3Hamiltonian
from dptb.nn.hr2hk import HR2HK
from ase import Atoms
from ase.build import sort
from dptb.negf.bloch import Bloch
from dptb.negf.sort_btd import sort_lexico, sort_projection, sort_capacitance
from dptb.negf.split_btd import show_blocks,split_into_subblocks,split_into_subblocks_optimized
from scipy.spatial import KDTree
'''
a Hamiltonian object  that initializes and manipulates device and  lead Hamiltonians for NEGF
'''

log = logging.getLogger(__name__)

class NEGFHamiltonianInit(object):
    '''The Class for Hamiltonian object in negf module. 
    
        It is used to initialize and manipulate device and lead Hamiltonians for negf.
        It is different from the Hamiltonian object in the dptb module.
        
        Property
        ----------
        apiH: the API object for Hamiltonian
        unit: the unit of energy
        structase: the structure object for the device and leads
        stru_options: the options for structure from input file
        results_path: the path to store the results

        device_id: the start-atom id and end-atom id of the device in the structure file
        lead_ids: the start-atom id and end-atom id of the leads in the structure file


        Methods
        ----------
        initialize: initializes the device and lead Hamiltonians
        get_hs_device: get the device Hamiltonian and overlap matrix at a specific kpoint
        get_hs_lead: get the lead Hamiltonian and overlap matrix at a specific kpoint
        
    '''

    def __init__(self, 
                 model: torch.nn.Module,
                 AtomicData_options: dict, 
                 structure: ase.Atoms,
                 block_tridiagonal: bool,
                 pbc_negf: List[bool],
                 stru_options:dict, 
                 unit: str,
                 results_path:Optional[str]=None,
                 torch_device: Union[str, torch.device]=torch.device('cpu')
                 ) -> None:
        
        # TODO: add dtype and device setting to the model
        torch.set_default_dtype(torch.float64)

        if isinstance(torch_device, str):
            torch_device = torch.device(torch_device)
        self.torch_device = torch_device   
        self.model = model
        self.AtomicData_options = AtomicData_options
        log.info(msg="The AtomicData_options is {}".format(AtomicData_options))
        self.model.eval()
        
        # get bondlist with pbc in all directions for complete chemical environment
        # around atoms in the two ends of device when predicting HR 
        if isinstance(structure,str):
            self.structase = read(structure)           
        elif isinstance(structure,ase.Atoms):
            self.structase = structure
        else:
            raise ValueError('structure must be ase.Atoms or str')
        
        self.unit = unit
        self.stru_options = stru_options
        self.pbc_negf = pbc_negf
        assert len(self.pbc_negf) == 3
        self.results_path = results_path
        self.saved_HS_path = None
        self.subblocks = None

        self.h2k = HR2HK(
            idp=model.idp, 
            edge_field=AtomicDataDict.EDGE_FEATURES_KEY, 
            node_field=AtomicDataDict.NODE_FEATURES_KEY, 
            out_field=AtomicDataDict.HAMILTONIAN_KEY, 
            dtype= model.dtype, 
            device=self.torch_device,
            )

        # if overlap:
        #     self.s2k = HR2HK(
        #         idp=model.idp, 
        #         overlap=True, 
        #         edge_field=AtomicDataDict.EDGE_OVERLAP_KEY, 
        #         node_field=AtomicDataDict.NODE_OVERLAP_KEY, 
        #         out_field=AtomicDataDict.OVERLAP_KEY, 
        #         dtype=model.dtype, 
        #         device=self.torch_device,
        #         )   
        
        self.device_id = [int(x) for x in self.stru_options['device']["id"].split("-")]
        self.lead_ids = {}
        for kk in self.stru_options:
            if kk.startswith("lead"):
                self.lead_ids[kk] = [int(x) for x in self.stru_options.get(kk)["id"].split("-")]

        # sort the atoms in device region lexicographically
        if block_tridiagonal:
            self.structase.positions[self.device_id[0]:self.device_id[1]] =\
            self.structase.positions[self.device_id[0]:self.device_id[1]][sort_lexico(self.structase.positions[self.device_id[0]:self.device_id[1]])]
            log.info(msg="The structure is sorted lexicographically in this version!")

        if self.unit == "Hartree":
            self.h_factor = 13.605662285137 * 2
        elif self.unit == "eV":
            self.h_factor = 1.
        elif self.unit == "Ry":
            self.h_factor = 13.605662285137
        else:
            log.error("The unit name is not correct !")
            raise ValueError

    def initialize(self, kpoints, block_tridiagnal=False,useBloch=False,bloch_factor=None,\
                   use_saved_HS=False, saved_HS_path=None):
        '''This function initializes the structure and Hamiltonian for a system with optional block tridiagonal
        and Bloch factor parameters.
        
        Parameters
        ----------
        kpoints
            Kpoints in the Brillouin zone
        block_tridiagnal, optional
             a boolean flag that determines whether the Hamiltonian matrix should be stored in a block-tridiagonal form. 
        useBloch, optional
            a boolean flag that determines whether Bloch boundary conditions should be used in the lead self energy calculations. 
        bloch_factor
            a list of integers that determines the Bloch factor for the lead self energy calculations.
        
        Returns
        -------
            The `initialize` method returns the following variables in this order:
        - `structure_device`
        - `structure_leads`
        - `structure_leads_fold`
        - `bloch_sorted_indices`
        - `bloch_R_lists`
        - `subblocks`
        
        '''

        # structure initialization       
        self.structase.set_pbc(self.pbc_negf)
        self.structase.pbc[2] = True
        structure_device = self.structase[self.device_id[0]:self.device_id[1]]
        structure_device.pbc = self.pbc_negf

        lead_atom_range = {}
        structure_leads = {};structure_leads_fold = {}
        bloch_sorted_indices={};bloch_R_lists = {}
        for kk in self.stru_options:
            if kk.startswith("lead"):
                n_proj_atom_pre = np.array([1]*len(self.structase))[:self.lead_ids[kk][0]].sum()
                n_proj_atom_lead = np.array([1]*len(self.structase))[self.lead_ids[kk][0]:self.lead_ids[kk][1]].sum()
                lead_atom_range[kk] = [n_proj_atom_pre, n_proj_atom_pre + n_proj_atom_lead]
                structure_leads[kk],structure_leads_fold[kk],\
                bloch_sorted_indices[kk],bloch_R_lists[kk] = self.get_lead_structure(kk,n_proj_atom_lead,\
                                useBloch=useBloch,bloch_factor=bloch_factor) 

        # Hamiltonian initialization
        if use_saved_HS:
            if saved_HS_path is None:
                saved_HS_path = self.results_path
                log.warning(msg="The saved_HS_path is not provided, use the results path by default.")
            self.saved_HS_path = saved_HS_path

            log.info(msg="--"*40)
            log.info(msg=f"The Hamiltonian is initialized from the saved path {self.saved_HS_path}.")
            log.info(msg="=="*40)
        else:
            self.saved_HS_path = self.results_path
            self.Hamiltonian_initialized(kpoints,useBloch,bloch_factor,block_tridiagnal,\
                                                 lead_atom_range,structure_leads,structure_leads_fold)
            log.info(msg="--"*40)
            log.info(msg=f"The Hamiltonian has been initialized by model.")
            log.info(msg="=="*40)

        torch.set_default_dtype(torch.float32)
        return  structure_device, structure_leads, structure_leads_fold, \
                bloch_sorted_indices, bloch_R_lists


    def Hamiltonian_initialized(self,kpoints:List[List[float]],useBloch:bool,bloch_factor:List[int],\
                                block_tridiagnal:bool,lead_atom_range:dict,structure_leads:Atoms,structure_leads_fold:Atoms):
        '''This function initializes the Hamiltonian for a device with leads, handling various calculations
        and checks along the way.
        
        Parameters
        ----------
        kpoints : List[List[float]]
            the k-points in Brillouin zone
        useBloch : bool
            The `useBloch` parameter is a boolean flag that indicates whether to use Bloch boundary conditions
        for the lead self energy calculations. If `useBloch` is set to `True`, the Bloch boundary conditions
        will be used. Otherwise, the calculations will be performed without Bloch boundary conditions.
        bloch_factor : List[int]
            The `bloch_factor` parameter is a list of integers that determines the Bloch factor for the lead
        self energy calculations. The Bloch factor is used to fold the lead structures in the context of the
        larger device structure. The Bloch factor specifies the number of times the lead structure is
        replicated along each direction to create a periodic structure.
        block_tridiagnal : bool
            The `block_tridiagnal` parameter is a boolean flag that determines whether the Hamiltonian matrix
        should be stored in a block-tridiagonal form. If `block_tridiagnal` is set to `True`, the Hamiltonian
        matrix will be stored in a block-tridiagonal form. Otherwise, the Hamiltonian matrix will be stored in
        a full matrix format.
        lead_atom_range : dict
            The `lead_atom_range` parameter indicates the range of leads. The key of the dictionary is the
        lead name, and the value is a list containing the start and end indices of the lead atoms.
        structure_leads : Atoms
            The `structure_leads` parameter is an Atoms object containing the structures of the leads. 
        structure_leads_fold : Atoms
            The `structure_leads_fold` parameter is an Atoms object containing the folded structures of the leads 
        by the Bloch theorem.
            
        
        Returns
        -------
        subblocks : List[int]
        
        '''
                                
        
        HS_device = {}
        assert len(np.array(kpoints).shape) == 2
        HS_device["kpoints"] = kpoints
        alldata = AtomicData.from_ase(self.structase, **self.AtomicData_options)
        alldata[AtomicDataDict.PBC_KEY][2] = True # force pbc in z-axis to get reasonable chemical environment in two ends
        alldata = AtomicData.to_AtomicDataDict(alldata.to(self.torch_device))
        alldata = self.model.idp(alldata)
        alldata[AtomicDataDict.KPOINT_KEY] = \
            torch.nested.as_nested_tensor([torch.as_tensor(HS_device["kpoints"], dtype=self.model.dtype, device=self.torch_device)])        
        alldata = self.model(alldata)
        
        if alldata.get(AtomicDataDict.EDGE_OVERLAP_KEY,None) is not None:
            self.overlap = True
            self.s2k = HR2HK(
                idp=self.model.idp, 
                overlap=True, 
                edge_field=AtomicDataDict.EDGE_OVERLAP_KEY, 
                node_field=AtomicDataDict.NODE_OVERLAP_KEY, 
                out_field=AtomicDataDict.OVERLAP_KEY, 
                dtype=self.model.dtype, 
                device=self.torch_device,
                )
        else: 
            self.overlap = False   

        self.remove_bonds_nonpbc(data=alldata,pbc=self.pbc_negf,overlap=self.overlap)  
        alldata = self.h2k(alldata)
        HK = alldata[AtomicDataDict.HAMILTONIAN_KEY]

        if self.overlap: 
            alldata = self.s2k(alldata)
            SK = alldata[AtomicDataDict.OVERLAP_KEY]
        else:
            SK = torch.eye(HK.shape[1], dtype=self.model.dtype, device=self.torch_device).unsqueeze(0).repeat(HK.shape[0], 1, 1)          
      
        # H, S = self.apiH.get_HK(kpoints=kpoints)
        d_start = int(np.sum(self.h2k.atom_norbs[:self.device_id[0]]))
        d_end = int(np.sum(self.h2k.atom_norbs)-np.sum(self.h2k.atom_norbs[self.device_id[1]:]))
        HD, SD = HK[:,d_start:d_end, d_start:d_end], SK[:, d_start:d_end, d_start:d_end]
        Hall, Sall = HK, SK

        coupling_width = {}
        for kk in self.stru_options:
            if kk.startswith("lead"):
                HS_leads = {}
                if useBloch:
                    bloch_unfolder = Bloch(bloch_factor)
                    k_unfolds_list = []
                    for kp in kpoints:
                        k_unfolds_list.append(bloch_unfolder.unfold_points(kp))
                    kpoints_bloch = np.concatenate(k_unfolds_list,axis=0)
                    HS_leads["kpoints"] = kpoints
                    HS_leads["kpoints_bloch"] = kpoints_bloch
                    HS_leads["bloch_factor"] = bloch_factor
                else:
                    HS_leads["kpoints"] = kpoints
                    HS_leads["kpoints_bloch"] = None
                    HS_leads["bloch_factor"] = None
                        
                l_start = int(np.sum(self.h2k.atom_norbs[:lead_atom_range[kk][0]]))
                l_end = int(l_start + np.sum(self.h2k.atom_norbs[lead_atom_range[kk][0]:lead_atom_range[kk][1]]) / 2)
                # lead hamiltonian in the first principal layer(the layer close to the device)
                HL, SL = HK[:,l_start:l_end, l_start:l_end], SK[:, l_start:l_end, l_start:l_end]
                # device and lead's hopping
                HDL, SDL = HK[:,d_start:d_end, l_start:l_end], SK[:,d_start:d_end, l_start:l_end]
                nonzero_indice = torch.nonzero(HDL)
                coupling_width[kk] = max(torch.max(nonzero_indice[:,1]).item()-torch.min(nonzero_indice[:,1]).item() +1,\
                                         torch.max(nonzero_indice[:,2]).item()-torch.min(nonzero_indice[:,2]).item() +1)
                log.info(msg="The coupling width of {} is {}.".format(kk,coupling_width[kk]))

                # get lead_data
                if useBloch:
                    lead_data = AtomicData.from_ase(structure_leads_fold[kk], **self.AtomicData_options)
                else:
                    lead_data = AtomicData.from_ase(structure_leads[kk], **self.AtomicData_options)
                lead_data = AtomicData.to_AtomicDataDict(lead_data.to(self.torch_device))
                lead_data = self.model.idp(lead_data)
                if useBloch:
                    lead_data[AtomicDataDict.KPOINT_KEY] = \
                    torch.nested.as_nested_tensor([torch.as_tensor(HS_leads["kpoints_bloch"], dtype=self.model.dtype, device=self.torch_device)])
                else:
                    lead_data[AtomicDataDict.KPOINT_KEY] = \
                    torch.nested.as_nested_tensor([torch.as_tensor(HS_leads["kpoints"], dtype=self.model.dtype, device=self.torch_device)])
                lead_data = self.model(lead_data)               
             
                self.remove_bonds_nonpbc(data=lead_data,pbc=self.pbc_negf,overlap=self.overlap)
                lead_data = self.h2k(lead_data)
                HK_lead = lead_data[AtomicDataDict.HAMILTONIAN_KEY]
                if self.overlap: 
                    lead_data = self.s2k(lead_data)
                    S_lead = lead_data[AtomicDataDict.OVERLAP_KEY]
                else:
                    S_lead = torch.eye(HK_lead.shape[1], dtype=self.model.dtype, device=self.torch_device).unsqueeze(0).repeat(HK_lead.shape[0], 1, 1)


                nL = int(HK_lead.shape[1] / 2)
                hLL, sLL = HK_lead[:, :nL, nL:], S_lead[:, :nL, nL:] # H_{L_first2L_second}
                hL, sL = HK_lead[:,:nL,:nL], S_lead[:,:nL,:nL] # lead hamiltonian in one principal layer
                if not useBloch:
                    err_l_HK = (hL - HL).abs().max()
                    err_l_SK = (sL - SL).abs().max()
                    rmse_l_HK = abs(torch.sqrt(torch.mean((hL - HL) ** 2)))
                    rmse_l_SK = abs(torch.sqrt(torch.mean((sL - SL) ** 2)))

                else: #TODO: add err_l_Hk and err_l_SK check in bloch case
                    err_l_HK = 0
                    err_l_SK = 0
                    rmse_l_HK = 0
                    rmse_l_SK = 0
                
                # if  max(err_l_HK,err_l_SK) >= 1e-3: 
                if max(rmse_l_HK,rmse_l_SK) >= 1e-4:
                    # check the lead hamiltonian get from device and lead calculation matches each other
                    # a standard check to see the lead environment is bulk-like or not
                    # Here we use RMSE to check the difference between the lead's hamiltonian and overlap
                    log.info(msg="The lead's hamiltonian or overlap attained from device and lead calculation does not match. RMSE for HK is {:.7f} and for SK is {:.7f} ".format(rmse_l_HK,rmse_l_SK))
                    log.error(msg="ERROR, the lead's hamiltonian attained from diffferent methods does not match.")
                # elif 1e-7 <= max(err_l_HK,err_l_SK) <= 1e-4:
                elif 1e-7 <= max(rmse_l_HK,rmse_l_SK) <= 1e-4:
                    log.warning(msg="WARNING, the lead's hamiltonian attained from diffferent methods have slight differences   RMSE = {:.7f}.".format(max(rmse_l_HK,rmse_l_SK)))

                # ensure the locality of the lead's Hamiltonian to stablize the self energy algorithm
                h_lead_threshold = 1e-6
                for ik in range(HK.shape[0]):
                    hL[ik][torch.abs(hL[ik])<h_lead_threshold] = 0
                    hLL[ik][torch.abs(hLL[ik])<h_lead_threshold] = 0

                HS_leads.update({
                    "HL":hL.cdouble()*self.h_factor, 
                    "SL":sL.cdouble(), 
                    "HDL":HDL.cdouble()*self.h_factor, 
                    "SDL":SDL.cdouble(),
                    "HLL":hLL.cdouble()*self.h_factor, 
                    "SLL":sLL.cdouble(),
                    "useBloch":useBloch
                    })                
                                
                torch.save(HS_leads, os.path.join(self.results_path, "HS_"+kk+".pth"))

        
        if not block_tridiagnal:
            # change HD format to ( k_index,block_index=0, orb, orb)
            subblocks = [HD.shape[1]]
            HD = torch.unsqueeze(HD,dim=1)
            SD = torch.unsqueeze(SD,dim=1)
            HS_device.update({"HD":HD.cdouble()*self.h_factor, "SD":SD.cdouble()})
            HS_device.update({"Hall":Hall.cdouble()*self.h_factor, "Sall":Sall.cdouble()})
            HS_device.update({"subblocks":subblocks, "block_tridiagonal":False})
        else:
            leftmost_size = coupling_width['lead_L']
            rightmost_size = coupling_width['lead_R']
            hd, hu, hl, sd, su, sl, subblocks = self.get_block_tridiagonal(HD*self.h_factor,SD.cdouble(),self.structase,\
                                                                leftmost_size,rightmost_size)
            HS_device.update({"hd":hd, "hu":hu, "hl":hl, "sd":sd, "su":su, "sl":sl, \
                              "subblocks":subblocks, "block_tridiagonal":True})

        self.subblocks = subblocks
        torch.save(HS_device, os.path.join(self.results_path, "HS_device.pth"))



    @staticmethod
    def remove_bonds_nonpbc(data,pbc,overlap):

        for ip,p in enumerate(pbc):
            if not p:
                mask = abs(data[AtomicDataDict.EDGE_CELL_SHIFT_KEY][:,ip])<1e-7
                data[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = data[AtomicDataDict.EDGE_CELL_SHIFT_KEY][mask]
                data[AtomicDataDict.EDGE_INDEX_KEY] = data[AtomicDataDict.EDGE_INDEX_KEY][:,mask]
                data[AtomicDataDict.EDGE_FEATURES_KEY] = data[AtomicDataDict.EDGE_FEATURES_KEY][mask]
                if overlap:
                    data[AtomicDataDict.EDGE_OVERLAP_KEY] = data[AtomicDataDict.EDGE_OVERLAP_KEY][mask]

    def get_lead_structure(self,kk,natom,useBloch=False,bloch_factor=None):       
        stru_lead = self.structase[self.lead_ids[kk][0]:self.lead_ids[kk][1]]
        cell = np.array(stru_lead.cell)[:2]
        
        R_vec = stru_lead[int(natom/2):].positions - stru_lead[:int(natom/2)].positions
        assert np.abs(R_vec[0] - R_vec[-1]).sum() < 1e-5
        R_vec = R_vec.mean(axis=0) * 2
        cell = np.concatenate([cell, R_vec.reshape(1,-1)])
        pbc_lead = self.pbc_negf.copy()
        pbc_lead[2] = True

        # get lead structure in ase format
        stru_lead = Atoms(str(stru_lead.symbols), 
                            positions=stru_lead.positions, 
                            cell=cell, 
                            pbc=pbc_lead)
        stru_lead.set_chemical_symbols(stru_lead.get_chemical_symbols())
        write(os.path.join(self.results_path, "stru_leadall_"+kk+".xyz"),stru_lead,format='extxyz')

        if useBloch:
            assert bloch_factor is not None
            bloch_reduce_cell = np.array([
                [cell[0][0]/bloch_factor[0], 0, 0],
                [0, cell[1][1]/bloch_factor[1], 0],
                [0, 0, cell[2][2]]
            ])
            bloch_reduce_cell = torch.from_numpy(bloch_reduce_cell)
            new_positions = []
            new_atomic_numbers = []
            delta = 1e-4
            for ip,pos in enumerate(stru_lead.get_positions()):
                if pos[0]<bloch_reduce_cell[0][0]-delta and pos[1]<bloch_reduce_cell[1][1]-delta:
                    new_positions.append(pos)
                    new_atomic_numbers.append(stru_lead.get_atomic_numbers()[ip])           
            
            stru_lead_fold = Atoms(numbers=new_atomic_numbers,
                              positions=new_positions,
                              cell=bloch_reduce_cell,
                              pbc=pbc_lead)
            stru_lead_fold.set_chemical_symbols(stru_lead_fold.get_chemical_symbols())
            stru_lead_fold = sort(stru_lead_fold,tags=stru_lead_fold.positions[:,0])
            stru_lead_fold = sort(stru_lead_fold,tags=stru_lead_fold.positions[:,1])
            stru_lead_fold = sort(stru_lead_fold,tags=stru_lead_fold.positions[:,2])

            if kk == "lead_L":
                natom_1PL = int(len(stru_lead_fold)/2)
                stru_lead_fold.positions[:natom_1PL,2],stru_lead_fold.positions[natom_1PL:,2] = \
                    stru_lead_fold.positions[natom_1PL:,2],stru_lead_fold.positions[:natom_1PL,2].copy()
            
            write(os.path.join(self.results_path, "stru_lead_fold_"+kk+".xyz"),stru_lead_fold,format='extxyz')
            log.info(msg="The lead structure is folded by Bloch theorem!")

            stru_lead_fold_1PL = stru_lead_fold[:int(len(stru_lead_fold)/2)]
            stru_lead_fold_minz = stru_lead_fold_1PL.positions[:,2].min() 
            bloch_R_list = []; expand_pos = []
            for rz in range(bloch_factor[2]):
                for ry in range(bloch_factor[1]):
                    for rx in range(bloch_factor[0]):     
                        R = torch.tensor([rx,ry,rz],dtype=torch.float64)
                        bloch_R_list.append(R)
                        for id in range(len(stru_lead_fold_1PL)):
                            pos = torch.tensor(stru_lead_fold_1PL.positions[id]) + \
                                R[0]*bloch_reduce_cell[0] + R[1]*bloch_reduce_cell[1] - stru_lead_fold_minz*torch.tensor([0,0,1])
                            expand_pos.append(pos)            
            expand_pos = np.stack(expand_pos) # expand_pos is for 1 PL
            assert len(expand_pos) == int(len(stru_lead)/2)

            # get the corresponding indices of the expanded structure in the original structure by KD tree
            struct_lead_minz = stru_lead.positions[:int(len(stru_lead)/2),2].min()
            struct_lead_pos = np.array([pos - np.array([0,0,1])*struct_lead_minz for pos in stru_lead.positions[:int(len(stru_lead)/2)]])
            kdtree = KDTree(expand_pos)
            _, sorted_indices = kdtree.query(struct_lead_pos,k=1,eps=1e-3)

            
            self.model.idp.get_orbital_maps()
            orb_dict = self.model.idp.norbs
            orb_list = np.array([ orb_dict[el] for el in stru_lead_fold_1PL.get_chemical_symbols()]*len(bloch_R_list))
            expand_basis_index = np.cumsum(orb_list)
            bloch_sorted_indice = []
            for ia in sorted_indices:
                for k in range(orb_list[ia]): 
                    bloch_sorted_indice.append(expand_basis_index[ia]-orb_list[ia]+k)
            bloch_sorted_indice = np.stack(bloch_sorted_indice)
            bloch_sorted_indice = torch.from_numpy(bloch_sorted_indice)
            # torch.save(bloch_sorted_indice, os.path.join(self.results_path, "bloch_sorted_indice_"+kk+".pth"))

        else:
            stru_lead_fold = None
            bloch_sorted_indice = None
            bloch_R_list = None

        
        return stru_lead, stru_lead_fold, bloch_sorted_indice, bloch_R_list

    def get_block_tridiagonal(self,HK,SK,structase:ase.Atoms,leftmost_size:int,rightmost_size:int):


        # return hd in format: (k_index,block_index, orb, orb)
        hd,hu,hl,sd,su,sl = [],[],[],[],[],[]

        if leftmost_size is None:
            leftmost_atoms_index = np.where(structase.positions[:,2]==min(structase.positions[:,2]))[0]
            leftmost_size = sum([self.h2k.atom_norbs[leftmost_atoms_index[i]] for i in range(len(leftmost_atoms_index))])
        if rightmost_size is None:   
            rightmost_atoms_index = np.where(structase.positions[:,2]==max(structase.positions[:,2]))[0]
            rightmost_size = sum([self.h2k.atom_norbs[rightmost_atoms_index[i]] for i in range(len(rightmost_atoms_index))])
        
        subblocks = split_into_subblocks_optimized(HK[0],leftmost_size,rightmost_size)
        if subblocks[0] < leftmost_size or subblocks[-1] < rightmost_size:
            subblocks = split_into_subblocks(HK[0],leftmost_size,rightmost_size)
            log.info(msg="The optimized block tridiagonalization is not successful, \
                     the original block tridiagonalization is used.")
        subblocks = [0]+subblocks
        
        
        for ik in range(HK.shape[0]):
            hd_k,hu_k,hl_k,sd_k,su_k,sl_k = [],[],[],[],[],[]
            counted_block = 0
            for id in range(len(subblocks)-1):
                counted_block+=subblocks[id]
                d_slice = slice(counted_block,counted_block+subblocks[id+1])
                hd_k.append(HK[ik,d_slice,d_slice])
                sd_k.append(SK[ik,d_slice,d_slice])
                if id < len(subblocks)-2:
                    u_slice = slice(counted_block+subblocks[id+1],counted_block+subblocks[id+1]+subblocks[id+2])
                    hu_k.append(HK[ik,d_slice,u_slice])
                    su_k.append(SK[ik,d_slice,u_slice])
                if id > 0:
                    l_slice = slice(counted_block-subblocks[id],counted_block)
                    hl_k.append(HK[ik,d_slice,l_slice])
                    sl_k.append(SK[ik,d_slice,l_slice]) 
            hd.append(hd_k);hu.append(hu_k);hl.append(hl_k)
            sd.append(sd_k);su.append(su_k);sl.append(sl_k)

            
        num_diag = sum([i**2 for i in subblocks])
        num_updiag = sum([subblocks[i]*subblocks[i+1] for i in range(len(subblocks)-1)])
        num_lowdiag = num_updiag
        num_total = num_diag+num_updiag+num_lowdiag
        log.info(msg="The Hamiltonian is block tridiagonalized into {} subblocks.".format(len(hd[0])))
        log.info(msg="   the number of elements in subblocks: {}".format(num_total))
        log.info(msg="               occupation of subblocks: {} %".format(num_total/(HK[0].shape[0]**2)*100))     

        subblocks = subblocks[1:]
        show_blocks(subblocks,HK[0],self.results_path)
        
        return hd, hu, hl, sd, su, sl, subblocks

    def get_hs_device(self, kpoint=[0,0,0], V=None, block_tridiagonal=False, only_subblocks=False):
        """ get the device Hamiltonian and overlap matrix at a specific kpoint

        In diagonalization mode, the Hamiltonian and overlap matrix are block tridiagonalized,
        and hd,hu,hl refers to the diagnonal, upper and lower blocks of the Hamiltonian, respectively.
        The same rules apply to sd, su, sl.
        
        Args:
            kpoints: k-points in the Brillouin zone with three coordinates (kx, ky, kz)
            V: voltage bias
            block_tridiagonal:  a boolean flag that shows whether Hamiltonian has been diagonalized or not
        
        Returns:
            if not diagonalized, return the whole Hamiltonian and Overlap HD-V*SD, SD
            if diagonalized, return the block tridiagonalized Hamiltonian and Overlap component hd, hu, hl,
            sd, su, sl.
        """
        if self.saved_HS_path is None:
            self.saved_HS_path = self.results_path
        
        HS_device_path = os.path.join(self.saved_HS_path, "HS_device.pth")
        if not os.path.exists(HS_device_path):
            log.error(msg="The HS_device.pth does not exist in the saved path {}.".format(self.saved_HS_path))
            raise FileNotFoundError
        f = torch.load(HS_device_path)

        if only_subblocks:
            if "subblocks" not in f:
                log.warning(msg=" 'subblocks' might not be saved in the HS_device.pth for old version.")
                log.error(msg="The subblocks are not saved in the HS_device.pth.")
                
                raise ValueError
            subblocks = f["subblocks"]
            return subblocks


        kpoints = f["kpoints"]

        ik = None
        for i, k in enumerate(kpoints):
            if np.abs(np.array(k) - np.array(kpoint)).sum() < 1e-8:
                ik = i
                break

        assert ik is not None

            
        
        if block_tridiagonal:
            # hd format: ( k_index,block_index, orb, orb)
            hd_k, sd_k, hl_k, su_k, sl_k, hu_k = f["hd"][ik], f["sd"][ik], f["hl"][ik], f["su"][ik], f["sl"][ik], f["hu"][ik]

            if V.shape == torch.Size([]):
                allorb = sum([hd_k[i].shape[0] for i in range(len(hd_k))])
                V = V.repeat(allorb)
            V = torch.diag(V).cdouble()
            counted = 0
            for i in range(len(hd_k)): # TODO: this part may have probelms when V!=0
                l_slice = slice(counted, counted+hd_k[i].shape[0])
                hd_k[i] = hd_k[i] - V[l_slice,l_slice]@sd_k[i]
                if i<len(hd_k)-1: 
                    hu_k[i] = hu_k[i] - V[l_slice,l_slice]@su_k[i]
                if i > 0:
                    hl_k[i-1] = hl_k[i-1] - V[l_slice,l_slice]@sl_k[i-1]
                counted += hd_k[i].shape[0]
            
            return hd_k , sd_k, hl_k , su_k, sl_k, hu_k
        else:
            HD_k, SD_k = f["HD"][ik], f["SD"][ik]
            return HD_k - V*SD_k, SD_k, [], [], [], []
    
    def get_hs_lead(self, kpoint, tab, v):
        """get the lead Hamiltonian and overlap matrix at a specific kpoint
        
        In diagonalization mode, the Hamiltonian and overlap matrix are block tridiagonalized,
        and hd,hu,hl refers to the diagnonal, upper and lower blocks of the Hamiltonian, respectively.
        The same rules apply to sd, su, sl.
        
        Args:
            kpoints: k-points in the Brillouin zone with three coordinates (kx, ky, kz)
            V: voltage bias
            block_tridiagonal:  a boolean flag that shows whether Hamiltonian has been diagonalized or not
        
        Returns:
            if not diagonalized, return the whole Hamiltonian and Overlap HD-V*SD, SD
            if diagonalized, return the block tridiagonalized Hamiltonian and Overlap component hd, hu, hl,
            sd, su, sl.
        """
        if self.saved_HS_path is None:
            self.saved_HS_path = self.results_path

        HS_lead_path = os.path.join(self.saved_HS_path, "HS_{0}.pth".format(tab))
        if not os.path.exists(HS_lead_path):
            log.error(msg="The HS_{0}.pth does not exist in the saved path {1}.".format(tab, self.saved_HS_path))
            raise FileNotFoundError
        f = torch.load(HS_lead_path)
        kpoints = f["kpoints"]
        kpoints_bloch = f["kpoints_bloch"]
        bloch_factor = f["bloch_factor"]

        if kpoints_bloch is None:
            ik = None
            for i, k in enumerate(kpoints):
                if np.abs(np.array(k) - np.array(kpoint)).sum() < 1e-8:
                    ik = i
                    break

            assert ik is not None
            assert len(kpoints) == f['HL'].shape[0], "The number of kpoints in the lead Hamiltonian file does not match the number of kpoints."
            hL, hLL, sL, sLL = f["HL"][ik], f["HLL"][ik],f["SL"][ik], f["SLL"][ik]
            hDL,sDL = f["HDL"][ik], f["SDL"][ik]

        else:
            multi_k_num = int(bloch_factor[0]*bloch_factor[1])
            ik = None; ik_bloch = None
            for i, k in enumerate(kpoints_bloch):
                if np.abs(np.array(k) - np.array(kpoint)).sum() < 1e-8:
                    ik_bloch = i
                    ik = int(i/multi_k_num)
                    break
            assert ik is not None
            assert len(kpoints_bloch) == f['HL'].shape[0], "The number of kpoints in the lead Hamiltonian file does not match the number of kpoints."
            assert len(kpoints) == f['HDL'].shape[0], "The number of kpoints in the lead Hamiltonian file does not match the number of kpoints."
            hL, hLL, sL, sLL = f["HL"][ik_bloch], f["HLL"][ik_bloch],f["SL"][ik_bloch], f["SLL"][ik_bloch]
            hDL,sDL = f["HDL"][ik], f["SDL"][ik]

        return hL-v*sL, hLL-v*sLL, hDL, sL, sLL, sDL 

    def attach_potential():
        pass

    def write(self):
        pass

    @property
    def device_norbs(self):
        """ 
        return the number of atoms in the device Hamiltonian
        """
        return self.h2k.atom_norbs[self.device_id[0]:self.device_id[1]]

    # def get_hs_block_tridiagonal(self, HD, SD):

    #     return hd, hu, hl, sd, su, sl



# class _NEGFHamiltonianInit(object):
#     '''The Class for Hamiltonian object in negf module. 
    
#         It is used to initialize and manipulate device and lead Hamiltonians for negf.
#         It is different from the Hamiltonian object in the dptb module.
        
#         Property
#         ----------
#         apiH: the API object for Hamiltonian
#         unit: the unit of energy
#         structase: the structure object for the device and leads
#         stru_options: the options for structure from input file
#         results_path: the path to store the results

#         device_id: the start-atom id and end-atom id of the device in the structure file
#         lead_ids: the start-atom id and end-atom id of the leads in the structure file


#         Methods
#         ----------
#         initialize: initializes the device and lead Hamiltonians
#         get_hs_device: get the device Hamiltonian and overlap matrix at a specific kpoint
#         get_hs_lead: get the lead Hamiltonian and overlap matrix at a specific kpoint
        
#     '''

#     def __init__(self, apiH, structase, stru_options, results_path) -> None:
#         self.apiH = apiH
#         self.unit = apiH.unit
#         self.structase = structase
#         self.stru_options = stru_options
#         self.results_path = results_path
        
#         self.device_id = [int(x) for x in self.stru_options.get("device")["id"].split("-")]
#         self.lead_ids = {}
#         for kk in self.stru_options:
#             if kk.startswith("lead"):
#                 self.lead_ids[kk] = [int(x) for x in self.stru_options.get(kk)["id"].split("-")]

#         if self.unit == "Hartree":
#             self.h_factor = 13.605662285137 * 2
#         elif self.unit == "eV":
#             self.h_factor = 1.
#         elif self.unit == "Ry":
#             self.h_factor = 13.605662285137
#         else:
#             log.error("The unit name is not correct !")
#             raise ValueError

#     def initialize(self, kpoints, block_tridiagnal=False):
#         """initializes the device and lead Hamiltonians 
        
#         construct device and lead Hamiltonians and return the structures respectively.The lead Hamiltonian 
#         is k-resolved due to the transverse k point sampling.

#         Args: 
#                 kpoints: k-points in the Brillouin zone with three coordinates (kx, ky, kz)
#                 block_tridiagnal: A boolean parameter that determines whether to block-tridiagonalize the
#                     device Hamiltonian or not. 

#         Returns: 
#                 structure_device and structure_leads corresponding to the structure of device and leads.

#         Raises:
#                 RuntimeError: if the lead hamiltonian attained from device and lead calculation does not match.                
        
#         """
#         assert len(np.array(kpoints).shape) == 2

#         HS_device = {}
#         HS_leads = {}
#         HS_device["kpoints"] = kpoints

#         self.apiH.update_struct(self.structase, mode="device", stru_options=j_must_have(self.stru_options, "device"), pbc=self.stru_options["pbc"])
#         # change parameters to match the structure projection
#         n_proj_atom_pre = np.array([1]*len(self.structase))[:self.device_id[0]][self.apiH.structure.projatoms[:self.device_id[0]]].sum()
#         n_proj_atom_device = np.array([1]*len(self.structase))[self.device_id[0]:self.device_id[1]][self.apiH.structure.projatoms[self.device_id[0]:self.device_id[1]]].sum()
#         proj_device_id = [0,0]
#         proj_device_id[0] = n_proj_atom_pre
#         proj_device_id[1] = n_proj_atom_pre + n_proj_atom_device
#         self.proj_device_id = proj_device_id
#         projatoms = self.apiH.structure.projatoms

#         self.atom_norbs = [self.apiH.structure.proj_atomtype_norbs[i] for i in self.apiH.structure.proj_atom_symbols]
#         self.apiH.get_HR()
#         # output the allbonds and hamil_block for check
#         # allbonds,hamil_block,_ =self.apiH.get_HR()
#         # torch.save(allbonds, os.path.join(self.results_path, "allbonds"+".pth"))
#         # torch.save(hamil_block, os.path.join(self.results_path, "hamil_block"+".pth"))

#         H, S = self.apiH.get_HK(kpoints=kpoints)
#         d_start = int(np.sum(self.atom_norbs[:proj_device_id[0]]))
#         d_end = int(np.sum(self.atom_norbs)-np.sum(self.atom_norbs[proj_device_id[1]:]))
#         HD, SD = H[:,d_start:d_end, d_start:d_end], S[:, d_start:d_end, d_start:d_end]
        
#         if not block_tridiagnal:
#             HS_device.update({"HD":HD.cdouble()*self.h_factor, "SD":SD.cdouble()})
#         else:
#             hd, hu, hl, sd, su, sl = self.get_block_tridiagonal(HD*self.h_factor, SD)
#             HS_device.update({"hd":hd, "hu":hu, "hl":hl, "sd":sd, "su":su, "sl":sl})

#         torch.save(HS_device, os.path.join(self.results_path, "HS_device.pth"))
#         structure_device = self.apiH.structure.projected_struct[self.device_id[0]:self.device_id[1]]
        
#         structure_leads = {}
#         for kk in self.stru_options:
#             if kk.startswith("lead"):
#                 HS_leads = {}
#                 stru_lead = self.structase[self.lead_ids[kk][0]:self.lead_ids[kk][1]]
#                 # write(os.path.join(self.results_path, "stru_"+kk+".vasp"), stru_lead)
#                 self.apiH.update_struct(stru_lead, mode="lead", stru_options=self.stru_options.get(kk), pbc=self.stru_options["pbc"])
#                 # update lead id
#                 n_proj_atom_pre = np.array([1]*len(self.structase))[:self.lead_ids[kk][0]][projatoms[:self.lead_ids[kk][0]]].sum()
#                 n_proj_atom_lead = np.array([1]*len(self.structase))[self.lead_ids[kk][0]:self.lead_ids[kk][1]][projatoms[self.lead_ids[kk][0]:self.lead_ids[kk][1]]].sum()
#                 proj_lead_id = [0,0]
#                 proj_lead_id[0] = n_proj_atom_pre
#                 proj_lead_id[1] = n_proj_atom_pre + n_proj_atom_lead

#                 l_start = int(np.sum(self.atom_norbs[:proj_lead_id[0]]))
#                 l_end = int(l_start + np.sum(self.atom_norbs[proj_lead_id[0]:proj_lead_id[1]]) / 2)
#                 HL, SL = H[:,l_start:l_end, l_start:l_end], S[:, l_start:l_end, l_start:l_end] # lead hamiltonian in one principal layer
#                 HDL, SDL = H[:,d_start:d_end, l_start:l_end], S[:,d_start:d_end, l_start:l_end] # device and lead's hopping
#                 HS_leads.update({
#                     "HL":HL.cdouble()*self.h_factor, 
#                     "SL":SL.cdouble(), 
#                     "HDL":HDL.cdouble()*self.h_factor, 
#                     "SDL":SDL.cdouble()}
#                     )

                
#                 structure_leads[kk] = self.apiH.structure.struct
#                 self.apiH.get_HR()
#                 # output the allbonds and hamil_block for check
#                 # allbonds_lead,hamil_block_lead,_ = self.apiH.get_HR()
#                 # torch.save(allbonds_lead, os.path.join(self.results_path, "allbonds_"+kk+".pth"))
#                 # torch.save(hamil_block_lead, os.path.join(self.results_path, "hamil_block_"+kk+".pth"))

#                 h, s = self.apiH.get_HK(kpoints=kpoints)
#                 nL = int(h.shape[1] / 2)
#                 HLL, SLL = h[:, :nL, nL:], s[:, :nL, nL:] # H_{L_first2L_second}
#                 err_l = (h[:, :nL, :nL] - HL).abs().max()
#                 if  err_l >= 1e-4: # check the lead hamiltonian get from device and lead calculation matches each other
#                     log.error(msg="ERROR, the lead's hamiltonian attained from diffferent methods does not match.")
#                     raise RuntimeError
#                 elif 1e-7 <= err_l <= 1e-4:
#                     log.warning(msg="WARNING, the lead's hamiltonian attained from diffferent methods have slight differences {:.7f}.".format(err_l))

#                 HS_leads.update({
#                     "HLL":HLL.cdouble()*self.h_factor, 
#                     "SLL":SLL.cdouble()}
#                     )
                
#                 HS_leads["kpoints"] = kpoints
                
#                 torch.save(HS_leads, os.path.join(self.results_path, "HS_"+kk+".pth"))
        
#         return structure_device, structure_leads
    
#     def get_hs_device(self, kpoint, V, block_tridiagonal=False):
#         """ get the device Hamiltonian and overlap matrix at a specific kpoint

#         In diagonalization mode, the Hamiltonian and overlap matrix are block tridiagonalized,
#         and hd,hu,hl refers to the diagnonal, upper and lower blocks of the Hamiltonian, respectively.
#         The same rules apply to sd, su, sl.
        
#         Args:
#             kpoints: k-points in the Brillouin zone with three coordinates (kx, ky, kz)
#             V: voltage bias
#             block_tridiagonal:  a boolean flag that shows whether Hamiltonian has been diagonalized or not
        
#         Returns:
#             if not diagonalized, return the whole Hamiltonian and Overlap HD-V*SD, SD
#             if diagonalized, return the block tridiagonalized Hamiltonian and Overlap component hd, hu, hl,
#             sd, su, sl.
#         """
#         f = torch.load(os.path.join(self.results_path, "HS_device.pth"))
#         kpoints = f["kpoints"]

#         ix = None
#         for i, k in enumerate(kpoints):
#             if np.abs(np.array(k) - np.array(kpoint)).sum() < 1e-8:
#                 ix = i
#                 break

#         assert ix is not None

#         if not block_tridiagonal:
#             HD, SD = f["HD"][ix], f["SD"][ix]
#         else:
#             hd, sd, hl, su, sl, hu = f["hd"][ix], f["sd"][ix], f["hl"][ix], f["su"][ix], f["sl"][ix], f["hu"][ix]
        
#         if block_tridiagonal:
#             return hd, sd, hl, su, sl, hu
#         else:
#             # print('HD shape:', HD.shape)
#             # print('SD shape:', SD.shape)
#             # print('V shape:', V.shape)
#             log.info(msg='Device Hamiltonian shape: {0}x{0}'.format(HD.shape[0], HD.shape[1]))
            
#             return [HD - V*SD], [SD], [], [], [], []
    
#     def get_hs_lead(self, kpoint, tab, v):
#         """get the lead Hamiltonian and overlap matrix at a specific kpoint
        
#         In diagonalization mode, the Hamiltonian and overlap matrix are block tridiagonalized,
#         and hd,hu,hl refers to the diagnonal, upper and lower blocks of the Hamiltonian, respectively.
#         The same rules apply to sd, su, sl.
        
#         Args:
#             kpoints: k-points in the Brillouin zone with three coordinates (kx, ky, kz)
#             V: voltage bias
#             block_tridiagonal:  a boolean flag that shows whether Hamiltonian has been diagonalized or not
        
#         Returns:
#             if not diagonalized, return the whole Hamiltonian and Overlap HD-V*SD, SD
#             if diagonalized, return the block tridiagonalized Hamiltonian and Overlap component hd, hu, hl,
#             sd, su, sl.
#         """
#         f = torch.load(os.path.join(self.results_path, "HS_{0}.pth".format(tab)))
#         kpoints = f["kpoints"]

#         ix = None
#         for i, k in enumerate(kpoints):
#             if np.abs(np.array(k) - np.array(kpoint)).sum() < 1e-8:
#                 ix = i
#                 break

#         assert ix is not None

#         hL, hLL, hDL, sL, sLL, sDL = f["HL"][ix], f["HLL"][ix], f["HDL"][ix], \
#                          f["SL"][ix], f["SLL"][ix], f["SDL"][ix]


#         return hL-v*sL, hLL-v*sLL, hDL, sL, sLL, sDL 

#     def attach_potential():
#         pass

#     def write(self):
#         pass

#     @property
#     def device_norbs(self):
#         """ 
#         return the number of atoms in the device Hamiltonian
#         """
#         return self.atom_norbs[self.device_id[0]:self.device_id[1]]

#     # def get_hs_block_tridiagonal(self, HD, SD):

#     #     return hd, hu, hl, sd, su, sl
