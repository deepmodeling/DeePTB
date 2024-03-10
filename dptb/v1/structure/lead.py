import ase
import warnings
import logging
import ase.neighborlist
from ase import Atoms
import numpy  as np
import re
from itertools import accumulate
import ase.io
from dptb.utils.constants import anglrMId,atomic_num_dict
from dptb.utils.tools import get_uniq_symbol, env_smoth
from dptb.utils.index_mapping import Index_Mapings
from dptb.structure.abstract_stracture import AbstractStructure
from dptb.structure.structure import BaseStruct
from ase import Atoms



class LeadStruct(BaseStruct):
    def __init__(self, atom, format, cutoff, proj_atom_anglr_m, proj_atom_neles, onsitemode:str='none', time_symm=True, lead_options={}, pbc=[False, False, False]):
        self.pbc = pbc.copy()
        self.pbc[2] = True
        self.lead_options = lead_options
        super(LeadStruct, self).__init__(atom, format, cutoff, proj_atom_anglr_m, proj_atom_neles, onsitemode, time_symm)
        
    def update_struct(self, atom, format, onsitemode:str='none'):
        self.init_description()
        self.onsitemode = onsitemode
        self.read_struct(atom, format=format)
        
        # get the cell, elements and positions and construct the lead lattice
        cell = np.array(self.struct.cell)[:2]
        lead_id = [int(x) for x in self.lead_options["id"].split("-")]
        natom = lead_id[1] - lead_id[0]
        R_vec = self.struct[int(natom/2):].positions - self.struct[:int(natom/2)].positions
        assert np.abs(R_vec[0] - R_vec[-1]).sum() < 1e-5
        R_vec = R_vec.mean(axis=0) * 2
        cell = np.concatenate([cell, R_vec.reshape(1,-1)])
        elements = self.struct.get_chemical_symbols()
        positions = self.struct.positions
        symbols = self.struct.symbols
        self.struct = Atoms(str(symbols), positions=positions, cell=cell, pbc=self.pbc)
        self.struct.set_chemical_symbols(elements)


        self.atom_symbols = np.array(self.struct.get_chemical_symbols(), dtype=str)
        self.atom_numbers = np.array(self.struct.get_atomic_numbers(), dtype=int)
        self.atomtype = get_uniq_symbol(atomsymbols=self.atom_symbols)
        self.projection()
        self.proj_atom_symbols = self.projected_struct.get_chemical_symbols()
        self.proj_atom_numbers = self.projected_struct.get_atomic_numbers()
        self.proj_atom_neles_per = np.array([self.proj_atom_neles[ii] for ii in self.proj_atom_symbols])
        self.proj_atom_to_atom_id = np.array(list(range(len(self.atom_symbols))))[self.projatoms]
        self.atom_to_proj_atom_id = np.array(list(accumulate([int(i) for i in self.projatoms]))) - 1
        self.proj_atomtype = get_uniq_symbol(atomsymbols=self.proj_atom_symbols)
        self.get_bond(cutoff=self.cutoff,time_symm=self.time_symm)
        self.if_env_ready = False
        self.if_onsitenv_ready = False

        self.IndMap.update(proj_atom_anglr_m=self.proj_atom_anglr_m)
        self.bond_index_map, self.bond_num_hops = self.IndMap.Bond_Ind_Mapings()
        self.onsite_strain_index_map, self.onsite_strain_num, self.onsite_index_map, self.onsite_num = self.IndMap.Onsite_Ind_Mapings(onsitemode, atomtype=self.atomtype)

    def get_bond(self, cutoff=None, time_symm=True):
        self.projected_struct.pbc[2] = False
        super().get_bond(cutoff, time_symm)
        self.projected_struct.pbc[2] = True
        
        return self.__bonds__, self.__bonds_onsite__
    
    def get_env(self, env_cutoff=None, sorted='iatom-jatom'):
        return super().get_env(env_cutoff, sorted)
    
    def projection(self):
        out = super().projection()
        self.projected_struct.pbc = self.pbc
        return out