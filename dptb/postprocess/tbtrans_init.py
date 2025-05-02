import numpy as np
import logging
import shutil
import re
from dptb.utils.tools import j_loader,j_must_have
from typing import Optional, Union
from ase.io import read,write
from ase.build import sort
import ase.atoms
import torch

from dptb.utils.constants import atomic_num_dict_r
from dptb.data import AtomicData, AtomicDataDict
from  dptb.data.interfaces.ham_to_feature import feature_to_block

log = logging.getLogger(__name__)

try:
    import sisl
    from sisl.orbital import Orbital
    
    sisl_installed = True

except ImportError:
    log.error('sisl is not installed.Thus the input for TBtrans can not be generated, please install it first!')
    sisl_installed = False


if  shutil.which('tbtrans') is None:
    log.error('tbtrans is not in the Environment PATH. Thus the input for TBtrans can be generated but not run.')
 
#  TBTransInputSet is used to transform input data for DeePTB-negf into TBtrans input files.
#  TBtrans (Tight-Binding transport) is a generic computer program which calculates transport and other physical quantities
#  using the Green function formalism. It is a stand-alone program which allows extreme scale tight-binding calculations. 
#  For details, see https://www.sciencedirect.com/science/article/pii/S001046551630306X?via%3Dihub.
#  To run TBTransInputSet, user need sisl package(https://zerothi.github.io/sisl/index.html)


class TBTransInputSet(object):


    def __init__(self, 
                model: torch.nn.Module,
                AtomicData_options: dict, 
                structure: Union[AtomicData, ase.Atoms, str],
                stru_options: dict,
                basis_dict:dict,
                results_path: Optional[str]=None,
                unit: str='eV',
                nel_atom: Optional[dict]=None,
                **kwargs): 
        '''
        This function initializes  properties and calculations for TBtrans input files.
        
        Parameters
        ----------
        model : torch.nn.Module
            The `model` parameter in the `__init__` method is expected to be an instance of
        `torch.nn.Module`. This parameter is used to store a neural network model that has been loaded
        in the `run.py` script.
        AtomicData_options : dict
            The `AtomicData_options` parameter is a dictionary containing options for atomic data.
         These options are used to provide necessary atomic information for the model.
        structure : Union[AtomicData, ase.Atoms, str]
            The `structure` parameter in the `__init__` method is used to specify the structure of the
        system. 
        stru_options : dict
            The `stru_options` parameter in the `__init__` method is a dictionary containing options for
        the structure. This dictionary includes pbc, kmesh and the regions division of the structure.
        basis_dict : dict
            The `basis_dict` parameter in the `__init__` method is used in the `orbitals_get`
        method. It is  a dictionary that contains the basis functions used in the calculations. 
        results_path : Optional[str]
            The `results_path` parameter is a string that specifies the path where the results of the
        calculations will be saved. 
        unit : str, optional
            The `unit` parameter in the `__init__` function is used to specify the energy unit for TBtrans
        calculation. It can take two possible values: eV or Hartree. The default value is eV.
        nel_atom : Optional[dict]
            The `nel_atom` parameter in the `__init__` function is an optional dictionary that represents
        the number of electrons per atom. 
        '''  
        
        self.model = model  #apiHrk has been loaded in run.py
        self.AtomicData_options = AtomicData_options
        # self.jdata = jdata    #jdata has been loaded in run.py, jdata is written in negf.json    

        self.results_path = results_path
        if not self.results_path.endswith('/'):self.results_path += '/'             
        self.stru_options = stru_options
        self.energy_unit_option = unit  # enenrgy unit for TBtrans calculation
        self.nel_atom = nel_atom

       
        self.geom_all,self.geom_lead_L,self.geom_lead_R,self.all_tbtrans_stru,self.lead_L_tbtrans_stru,self.lead_R_tbtrans_stru\
                   = self.read_rewrite_structure(structure,self.stru_options,self.results_path)
        
        if nel_atom is not None:
            self.orbitals_get(self.geom_all,self.geom_lead_L,self.geom_lead_R,basis_dict)
            log.warning('nel_atom is none!')
        
        self.H_all = sisl.Hamiltonian(self.geom_all)
        self.H_lead_L = sisl.Hamiltonian(self.geom_lead_L)
        self.H_lead_R = sisl.Hamiltonian(self.geom_lead_R)


        #important properties for later use

        ##allbonds matrx, hamiltonian matrix, overlap matrix for the whole structure
        self.allbonds_all = None
        self.hamil_block_all = None
        self.overlap_block_all = None
        ##allbonds matrx, hamiltonian matrix, overlap matrix for lead_L
        self.allbonds_lead_L = None
        self.hamil_block_lead_L = None
        self.overlap_block_lead_L = None
        ##allbonds matrx, hamiltonian matrix, overlap matrix for lead_R
        self.allbonds_lead_R = None
        self.hamil_block_lead_R = None
        self.overlap_block_lead_R = None

        if self.energy_unit_option=='Hartree':
            self.unit_constant = 1.0000/13.605662285137 /2
           
        elif self.energy_unit_option=='eV':
            self.unit_constant = 1.0000000000
            
        else:
            raise RuntimeError("energy_unit_option should be 'Hartree' or 'eV'")

   
 
    def hamil_get_write(self,write_nc:bool=True):
        
        '''The function `hamil_get_write` loads models for different structure.retrieves the Hamiltonian and overlap matrices /
        for the device and left and right leads, then writes the contents of `self.H_all`, `self.H_lead_L`, and `self.H_lead_R` to nc files for
        TBtrans calculations.

            `all` refers to the entire system, including the device and leads.


        Returns
        -------
        - allbonds_all: all of the bond information 
        - hamil_block_all: Hamiltonian block for the entire system, which is a tensor that contains 
                            the values of the Hamiltonian matrix elements for each specific bond in allbonds_all
        - overlap_block_all: overlap block for the entire system,  which is a tensor that contains 
                             the values of the overlap matrix elements for each specific basis
        '''


        # get the Hamiltonian matrix for the entire system
        self.allbonds_all,self.hamil_block_all,self.overlap_block_all\
                    =self._load_model(self.model,self.AtomicData_options,self.all_tbtrans_stru)
        self.hamiltonian_get(self.allbonds_all,self.hamil_block_all,self.overlap_block_all,self.H_all)    

        # get the Hamiltonian matrix for the left lead
        self.allbonds_lead_L,self.hamil_block_lead_L,self.overlap_block_lead_L\
                        =self._load_model(self.model,self.AtomicData_options,self.lead_L_tbtrans_stru)
        self.hamiltonian_get(self.allbonds_lead_L,self.hamil_block_lead_L,self.overlap_block_lead_L,self.H_lead_L)
        
        # get the Hamiltonian matrix for the right lead
        self.allbonds_lead_R,self.hamil_block_lead_R,self.overlap_block_lead_R\
                        =self._load_model(self.model,self.AtomicData_options,self.lead_R_tbtrans_stru)       
        self.hamiltonian_get(self.allbonds_lead_R,self.hamil_block_lead_R,self.overlap_block_lead_R,self.H_lead_R)

        if write_nc:
            self.H_all.write(self.results_path+'structure.nc')
            self.H_lead_L.write(self.results_path+'lead_L.nc')
            self.H_lead_L.write(self.results_path+'lead_R.nc')
        else:
            print('Hamiltonian matrices have been generated, but not written to nc files(TBtrans input file).')



    def read_rewrite_structure(self,structure_file:str,struct_options:dict,results_path:str):
        '''The function `read_rewrite_structure` reads a structure file, extracts specific regions of the structure,
        sorts the atoms in the structure, and outputs the sorted structures in XYZ and VASP file formats for later operations.
        
        Parameters
        ----------
        structure_file
            The `structure_file` parameter is the path to the file containing the structure information of the
        system you want to analyze. It can be in either VASP format (.vasp) or XYZ format (.xyz).
        struct_options
            The `struct_options` parameter is a dictionary that contains various options for the structure. 
        result_path
            The `result_path` parameter is the path where the output files will be saved. 
        
        Returns
        -------
            
        - geom_all: the geometry of the entire structure,including the device and leads
        - geom_lead_L: the geometry of the left lead
        - geom_lead_R: the geometry of the right lead
        - all_tbtrans_stru: the path to the sorted xyz file for the entire structure
        - lead_L_tbtrans_stru: the path to the sorted xyz file for the left lead
        - lead_R_tbtrans_stru: the path to the sorted xyz file for the right lead
        
        '''
    
    
        lead_L_id=struct_options['lead_L']['id'].split('-')
        lead_R_id=struct_options['lead_R']['id'].split('-')
        # device_id=struct_options['device']['id'].split('-') 

        lead_L_range=[i for i in range(int(lead_L_id[0]), int(lead_L_id[1]))]
        lead_R_range=[i for i in range(int(lead_R_id[0]), int(lead_R_id[1]))]


        # Structure input: read vasp file
        # structure_vasp = sisl.io.carSileVASP(structure_file)
        # geom_device = structure_vasp.read_geometry()
        if structure_file.split('.')[-1]=='vasp':
            structure_vasp = sisl.io.carSileVASP(structure_file)
            geom_all = structure_vasp.read_geometry()
            if geom_all.atoms.nspecie>1:
                raise RuntimeError('ERROR! In transport calculation, VASP structure file is only valid for materials with one single element!')        
        elif structure_file.split('.')[-1]=='xyz':
            structure_xyz = sisl.io.xyzSile(structure_file)
            geom_all = structure_xyz.read_geometry()
        else:
            raise RuntimeError('Structure file format is not supported. Only support vasp and xyz format')
    # structure_xyz = sisl.io.xyzSile(structure_file)
    # geom_device = structure_xyz.read_geometry()
    #define lead geometry structure
        geom_lead_L = geom_all.sub(lead_L_range) # subset of the geometry
        geom_lead_R = geom_all.sub(lead_R_range)

        #sort sturcture atoms according to z-direction: it's easier for later coding
        #2,1,0 refers to sort by axis=0 firstly, then axis=1, last for axis=2
        geom_lead_R = geom_lead_R.sort(axis=(2,1,0));geom_lead_L=geom_lead_L.sort(axis=(2,1,0))  
        geom_all=geom_all.sort(axis=(2,1,0))
    

        lead_L_cor = geom_lead_L.axyz() #Return the atomic coordinates in the supercell of a given atom.
        cell = np.array(geom_lead_L.lattice.cell)[:2]
        Natom_PL = int(len(lead_L_cor)/2)
        first_PL_leadL = lead_L_cor[Natom_PL:];second_PL_leadL =lead_L_cor[:Natom_PL]
        R_vec = first_PL_leadL - second_PL_leadL
        # assert np.abs(R_vec[0] - R_vec[-1]).sum() < 1e-5
        assert np.abs(R_vec[0] - R_vec.mean(axis=0)).sum() < 1e-5
        R_vec = R_vec.mean(axis=0) * 2
        cell = np.concatenate([cell, R_vec.reshape(1,-1)])
        # PL_leadL_zspace = first_PL_leadL[0][2]-second_PL_leadL[-1][2] # the distance between Principal layers
        geom_lead_L.lattice.cell=cell
        # assert geom_lead_L.lattice.cell[2,2]>0

        #TODO: This version tbtrans_init only supports double lead case. Code for more leads will be added later.

        lead_R_cor = geom_lead_R.axyz()
        cell = np.array(geom_lead_R.lattice.cell)[:2]
        Natom_PL = int(len(lead_R_cor)/2)
        first_PL_leadR = lead_R_cor[:Natom_PL];second_PL_leadR = lead_R_cor[Natom_PL:]
        R_vec = first_PL_leadR - second_PL_leadR
        assert np.abs(R_vec[0] - R_vec.mean(axis=0)).sum() < 1e-5
        R_vec = -1*R_vec.mean(axis=0) * 2
        cell = np.concatenate([cell, R_vec.reshape(1,-1)])
        # PL_leadR_zspace = second_PL_leadR[0][2]-first_PL_leadR[-1][2]
        geom_lead_R.lattice.cell = cell

        all_tbtrans_stru=results_path+'structure_tbtrans.xyz'
        sorted_structure = sisl.io.xyzSile(all_tbtrans_stru,'w')
        geom_all.write(sorted_structure)

        lead_L_tbtrans_stru=results_path+'lead_L_tbtrans.xyz'
        sorted_lead_L = sisl.io.xyzSile(lead_L_tbtrans_stru,'w')
        geom_lead_L.write(sorted_lead_L)

        lead_R_tbtrans_stru=results_path+'lead_R_tbtrans.xyz'
        sorted_lead_R = sisl.io.xyzSile(lead_R_tbtrans_stru,'w')
        geom_lead_R.write(sorted_lead_R)

        # output sorted geometry into vasp Structure file: rewrite xyz files
        ## writen as VASP for VESTA view
        all_struct = read(all_tbtrans_stru)
        all_vasp_struct = results_path+'structure_tbtrans.vasp'
        write(all_vasp_struct,sort(all_struct),format='vasp')

        return geom_all,geom_lead_L,geom_lead_R,all_tbtrans_stru,lead_L_tbtrans_stru,lead_R_tbtrans_stru



    def orbitals_get(self,geom_all, geom_lead_L,geom_lead_R,basis_dict:dict):
        '''The function `orbitals_get` takes in various inputs such as geometric devices, leads, deeptb model, and
        configurations, and assigns orbitals number, orbital names, shell-electron numbers to the atoms in the given sisl geometries .

            Here the geometry class is sisl.geometry, which is different from the structure class in dptb-negf.
            We initialize sisl.geometry from structure files directly, therefore there is no orbital information in sisl.geometry.
        
        Parameters
        ----------
        geom_all
            The `geom_all` parameter is the geometry of the whole structure. It contains information about the
        atoms and their positions.
        geom_lead_L
            The `geom_lead_L` parameter represents the geometry of the left lead. It contains
        information about the atoms and their positions in the lead.
        geom_lead_R
            The `geom_lead_R` parameter represents the geometry of the right lead. It contains 
            information about the atoms in the lead, such as their positions and chemical symbols.
        apiHrk
            apiHrk has been loaded in the run.py file. It is used as an API for
            performing certain operations or accessing certain functionalities when loading dptb model.
        
        '''        
        n_species_lead_L = geom_lead_L.atoms.nspecie
        n_species_lead_R = geom_lead_R.atoms.nspecie
        n_species_all = geom_all.atoms.nspecie
        n_species_list = [n_species_lead_L,n_species_lead_R,n_species_all]
        geom_list = [geom_lead_L,geom_lead_R,geom_all]  


        dict_element_orbital = basis_dict
        dict_shell_electron = self.nel_atom

        for n_species, geom_part in zip(n_species_list,geom_list):
            # species_symbols = split_string(geom_part.atoms.formula())
            ## get the chemical symbol of the part
            # species_symbols = ''.join(char for char in geom_part.atoms.formula() if char.isalpha())
            
            uni_symbol_index = np.unique(geom_part.atoms.Z)
            species_symbols=[atomic_num_dict_r[i] for i in uni_symbol_index]
            
            assert len(species_symbols)==n_species # number of chemical elements in this part

            for i  in range(n_species): #determine the orbitals number for each species
                element_orbital_list = dict_element_orbital[species_symbols[i]]
                # Examples of elemet_orbital_list: ['3s', '3p', 'd*'] 
                element_orbital_name = self._orbitals_name_get(element_orbital_list) 
                # Examples of element_orbital_name: ['3s', '3py', '3pz', '3px', 'dxy*', 'dyz*', 'dz2*', 'dxz*', 'dx2-y2*']
                
                # shell_elec_num =  self._shell_electrons(species_symbols[i])
                shell_elec_num =  dict_shell_electron[species_symbols[i]]
                shell_elec_list = np.zeros(len(element_orbital_name))
                shell_elec_list[:shell_elec_num]=1 #occupation number for each orbital, here we assume the system is in ground state

                for atom_index in range(geom_part.na):
                    if geom_part.atoms[atom_index].symbol == species_symbols[i]:
                        geom_part.atoms[atom_index]._orbitals =[Orbital(-1, q0=q,tag=tag) for q,tag in zip(shell_elec_list,element_orbital_name)]
                    # attribute sisl.Orbital object to each atom in sisl.geometry.atoms[x]._orbitals

        geom_lead_L.atoms._update_orbitals()  #sisl use ._update_orbitals() to ensure the order of orbitals 
        geom_lead_R.atoms._update_orbitals()
        geom_all.atoms._update_orbitals()


    def _orbitals_name_get(self,element_orbital_class:str):
        '''The `_orbitals_name_get` function takes a list of element orbital classes and returns a list of
        orbital names.
        
        Parameters
        ----------
        element_orbital_class
            A list of strings representing the orbital classes of an element. Each string in the list
        represents a different orbital class.
        
        Returns
        -------
            a list of orbital names.

        Examples
        --------
        >>>element_orbital_name = self._orbitals_name_get(['2s', '2p'])
        >>>element_orbital_name
        ['2s', '2py', '2pz', '2px']
        >>>element_orbital_name = self._orbitals_name_get(['3s', '3p', 'd*'])
        >>>element_orbital_name
        ['3s', '3py', '3pz', '3px', 'dxy*', 'dyz*', 'dz2*', 'dxz*', 'dx2-y2*']
        '''
        orbital_name_list=[] 
        for orb_cla in element_orbital_class:  # ['3s', '3p', 'd*']          
            orbital_name = re.findall(r'\d+|[a-zA-Z]+|\*',orb_cla)
            orbital_name = sorted(orbital_name, key=lambda x: (x == '*',x.isnumeric(),x))# put s,p,d in the 1st position
            
            assert len(orbital_name)>1 #example 3d*:  orbital_name: ['d', '3', '*']
            if orbital_name[0]=='s':
                orbital_name_list += [orbital_name[1]+'s']
            elif orbital_name[0]=='p':
                if orbital_name[1].isnumeric(): # not polarized orbital
                    orbital_name_list += [orbital_name[1]+'py',orbital_name[1]+'pz',orbital_name[1]+'px']
                else:#polarized orbital
                    orbital_name_list += ['py*','pz*','px*']
            elif orbital_name[0]=='d':
                if orbital_name[1].isnumeric(): # not polarized orbital
                    orbital_name_list += [orbital_name[1]+'dxy',orbital_name[1]+'dyz',\
                                            orbital_name[1]+'dz2',orbital_name[1]+'dxz',orbital_name[1]+'dx2-y2']
                else:#polarized orbital
                    orbital_name_list += ['dxy*','dyz*','dz2*','dxz*','dx2-y2*']
            else:
                raise RuntimeError("At this stage dptb-negf only supports s, p, d orbitals")
        # print(orbital_name_list)
        # raise RuntimeError('stop here')
        return orbital_name_list      


    def _load_model(self,model,AtomicData_options,structure_tbtrans_file:str):
        '''The `load_dptb_model` function loads model and returns the Hamiltonian elements.
        Parameters
        ----------
        checkfile : str
            The `checkfile` parameter is the file path to the model checkpoint file. 
        config : str
            The `config` parameter is a string that represents the configuration file for the model. It
        contains information such as the model architecture, hyperparameters, and other settings that are
        necessary for loading and building the model.
        structure_tbtrans_file : str
            The `structure_tbtrans_file` parameter is the file path to the structure file in the TBtrans
        format.
        struct_option : dict
            The `struct_option` parameter is a dictionary that contains various options for the structure. It
        includes the following keys:
        run_sk : bool
            The `run_sk` parameter is a boolean flag that determines whether to run the model using the NNSK
        (Neural Network Schrödinger-Kohn) method. If `run_sk` is set to `True`, the model will be run using
        the NNSK method. If
        use_correction : Optional[str]
            The `use_correction` parameter is an optional parameter that specifies whether to use correction
        terms in the model. It can be set to either `None` or a string value. If it is set to `None`, the
        model will not use any correction terms. If it is set to a string value
        
        Returns
        -------
            The function `load_dptb_model` returns three variables: `allbonds`, `hamil_block`, and
        `overlap_block`.
        
        '''
        structase = read(structure_tbtrans_file)

        data = AtomicData.from_ase(structase,**AtomicData_options)
        data = AtomicData.to_AtomicDataDict(data)
        data = model.idp(data)

        data = model(data)
        HR_dict = feature_to_block(data,model.idp) # get HR
        allbonds = [] # a list contain bond info in the form like (type1,atom1_index,type2,atom2_index,displacement)
        hamil_block = []
        overlap_block = []

        for key,value in HR_dict.items():
            bond = key.split('_')
            bond = torch.as_tensor([int(bond[i]) for i in range(len(bond))])
            iatom,jatom = model.idp.untransform(data[AtomicDataDict.ATOM_TYPE_KEY][bond[0]-1]),\
                            model.idp.untransform(data[AtomicDataDict.ATOM_TYPE_KEY][bond[1]-1])
            allbonds.append(torch.cat([iatom,torch.tensor([bond[0]-1]),jatom,torch.tensor([bond[1]-1]),bond[2:]]))
            hamil_block.append(value)
        
        allbonds = torch.stack(allbonds)
        hamil_block = torch.stack(hamil_block)
        overlap_block = torch.stack([torch.eye(hamil_block.shape[-1]) for i in range(hamil_block.shape[0])])
        # TODO: check tbtrans support overlap matrix or not in TB-NEGF

        return allbonds,hamil_block,overlap_block

    def hamiltonian_get(self,allbonds:torch.tensor,hamil_block:torch.tensor,overlap_block:torch.tensor,Hamil_sisl):
        '''The function `hamiltonian_get` takes in various parameters and calculates the Hamiltonian matrix
        for a given set of bonds, storing the result in the `Hamil_sisl` matrix.
        
        Parameters
        ----------
        allbonds
            A torch.tensor containing information about the bonds in the system. Each element of the list
        corresponds to a bond and contains information such as the indices of the atoms involved in the
        bond and the displacement vector between the atoms.
        hamil_block
            The `hamil_block` parameter is a block of the Hamiltonian matrix. It is a tensor that contains the
        values of the Hamiltonian matrix elements for a specific bond in the system.
        overlap_block
            The `overlap_block` parameter is a block matrix representing the overlap between orbitals in the
        Hamiltonian. It is used to calculate the Hamiltonian matrix elements.
        Hamil_sisl
            Hamil_sisl is sisl.hamiltonian that represents the Hamiltonian matrix. The first two
        dimensions correspond to the orbital indices, and the third dimension represents the supercell
        indices. The Hamiltonian matrix elements are stored in this array.
            Inertially, Hamil_sisl is a scipy.sparse.csr_matrix.
        energy_unit_option
            The `energy_unit_option` parameter is a string that specifies the unit of energy for the
        calculation. It can be either "Hartree" or "eV".
        
        '''   
        x_max = abs(allbonds[:,-3].numpy()).max()
        y_max = abs(allbonds[:,-2].numpy()).max()
        z_max = abs(allbonds[:,-1].numpy()).max()
        Hamil_sisl.set_nsc(a=2*abs(x_max)+1,b=2*abs(y_max)+1,c=2*abs(z_max)+1)
        # set the number of super-cells in Hamiltonian object in sisl, which is based on allbonds results
        

        for i in range(len(allbonds)):
            # if i%100==0:print('bond_index: ',i)
            orb_first_a = Hamil_sisl.geometry.a2o(allbonds[i,1])
            orb_last_a = Hamil_sisl.geometry.a2o(allbonds[i,1]+1)
            orb_first_b = Hamil_sisl.geometry.a2o(allbonds[i,3])
            orb_last_b = Hamil_sisl.geometry.a2o(allbonds[i,3]+1)


            if allbonds[i][-3:].equal(torch.tensor([0,0,0])): #allbonds[i,1] is atom index
                
                for orb_a in range(orb_first_a,orb_last_a):
                    for orb_b in range(orb_first_b,orb_last_b):
                        Hamil_sisl[orb_a,orb_b]=hamil_block[i].detach().numpy()[orb_a-orb_first_a,orb_b-orb_first_b]*self.unit_constant
                        # Hamil_sisl[orb_b,orb_a]=hamil_block[i].detach().numpy()[orb_b-orb_first_b,orb_a-orb_first_a]*unit_constant
                        Hamil_sisl[orb_b,orb_a]=np.conjugate(Hamil_sisl[orb_a,orb_b])
            else: 
                x = allbonds[i,-3].numpy().tolist()
                y = allbonds[i,-2].numpy().tolist()
                z = allbonds[i,-1].numpy().tolist()

                for orb_a in range(orb_first_a,orb_last_a):
                    for orb_b in range(orb_first_b,orb_last_b):
                        H_value = hamil_block[i].detach().numpy()[orb_a-orb_first_a,orb_b-orb_first_b]*self.unit_constant
                        if H_value != 0:
                            Hamil_sisl[orb_a,orb_b,(x,y,z)]=H_value
                            Hamil_sisl[orb_b,orb_a,(-1*x,-1*y,-1*z)]=np.conjugate(Hamil_sisl[orb_a,orb_b,(x,y,z)])
                        # Hamil_sisl[orb_b,orb_a,(-1*x,-1*y,-1*z)]=hamil_block[i].detach().numpy()[orb_b-orb_first_b,orb_a-orb_first_a]*unit_constant
                
                #TODO: At this stage, there is some problem using slice operation in sisl. I'm fixing it with the developer of sisl.
                # I believe that the slice operation will be take soon.
            
        
