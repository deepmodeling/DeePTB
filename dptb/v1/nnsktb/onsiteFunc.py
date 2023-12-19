from unittest import main
from xml.etree.ElementTree import tostring
import torch as th
from dptb.utils.constants import atomic_num_dict_r
from dptb.nnsktb.onsiteDB import onsite_energy_database
from dptb.nnsktb.formula import SKFormula
from dptb.utils.index_mapping import Index_Mapings
from dptb.nnsktb.onsite_formula import onsiteFormula
from dptb.nnsktb.skintTypes import all_onsite_ene_types

import logging

# define the function for output all the onsites Es for given i.
log = logging.getLogger(__name__)

def loadOnsite(onsite_map: dict, unit="Hartree"):
    """ load the onsite energies from the database, according to the onsite_map:dict
    This function only need to run once before calculation/ training.

    Parameters:
    -----------
        onsite_map: dict, has two possible format.
            -1. {'N': {'2s': [0], '2p': [1]}, 'B': {'2s': [0], '2p': [1]}}
            -2. {'N': {'2s': [0], '2p': [1,2,3]}, 'B': {'2s': [0], '2p': [1,2,3]}}
    
    Returns:
    --------
        onsite energy: dict, the format follows the input onsite_map, e.g.:
            -1. {'N':tensor[es,ep], 'B': tensor[es,ep]}
            -2. {'N':tensor[es,ep1,ep2,ep3], 'B': tensor[es,ep1,ep2,ep3]}

    """

    atoms_types = list(onsite_map.keys())
    onsite_db = {}
    for ia in atoms_types:
        assert ia in onsite_energy_database.keys(), f'{ia} is not in the onsite_energy_database. \n see the onsite_energy_database in dptb.nnsktb.onsiteDB.py.'
        orb_energies = onsite_energy_database[ia]
        indeces = sum(list(onsite_map[ia].values()),[])
        onsite_db[ia] = th.zeros(len(indeces))
        for isk in onsite_map[ia].keys():
            assert isk in orb_energies.keys(), f'{isk} is not in the onsite_energy_database for {ia} atom. \n see the onsite_energy_database in dptb.nnsktb.onsiteDB.py.'
            if unit == "Hartree":
                factor = 1.
            elif unit == "eV":
                factor = 13.605662285137 * 2
            elif unit == "Ry":
                factor = 2.
            else:
                log.error("The unit name is not correct !")
                raise ValueError
            onsite_db[ia][onsite_map[ia][isk]] = orb_energies[isk] * factor

    return onsite_db

def onsiteFunc(batch_bonds_onsite, onsite_db: dict, nn_onsiteE: dict=None):
# this function is not used anymore.
    batch_onsiteEs = {}
    for kf in list(batch_bonds_onsite.keys()):  # kf is the index of frame number.
        bonds_onsite = batch_bonds_onsite[kf][:,1:]
        ia_list = map(lambda x: atomic_num_dict_r[int(x)], bonds_onsite[:,0]) # itype
        if nn_onsiteE is not None:
            onsiteEs = []
            for x in ia_list:
                onsite = nn_onsiteE[x].clone()
                onsite[:len(onsite_db[x])] += onsite_db[x]
                onsiteEs.append(onsite)
        else:
            onsiteEs = map(lambda x: onsite_db[x], ia_list)
        batch_onsiteEs[kf] = list(onsiteEs)

    return batch_onsiteEs

class orbitalEs(onsiteFormula):
    """ This calss is to get the onsite energies for given bonds_onsite.
     
    """
    def __init__(self, proj_atom_anglr_m,  atomtype=None, functype='none',unit='Hartree',**kwargs) -> None:
        super().__init__(functype)
        IndMap = Index_Mapings()
        IndMap.update(proj_atom_anglr_m=proj_atom_anglr_m)
        onsite_strain_index_map, onsite_strain_num, onsite_index_map, onsite_num = \
                IndMap.Onsite_Ind_Mapings(onsitemode=functype, atomtype=atomtype)
        assert functype != 'strain', 'The onsite mode strain is not from this modula.'
        self.onsite_db =  loadOnsite(onsite_index_map, unit= unit)
        _, _, self.onsite_index_dict  = all_onsite_ene_types(onsite_index_map)

        if functype == 'NRL':
            self.onsite_func_cutoff = kwargs.get('onsite_func_cutoff')
            self.onsite_func_decay_w = kwargs.get('onsite_func_decay_w')
            self.onsite_func_lambda = kwargs.get('onsite_func_lambda')

    def get_onsiteEs(self,batch_bonds_onsite, onsite_env: dict=None, nn_onsite_paras: dict=None, **kwargs):
        """
        Parameters:
        -----------
            batch_bonds_onsite: list
                e.g.:  dict(f: [[f, 7, 0, 7, 0, 0, 0, 0],
                                [f, 5, 1, 5, 1, 0, 0, 0]])
            onsite_db: dict from function loadOnsite
                e.g.: {'N':tensor[es,ep], 'B': tensor[es,ep]}
        
        Return:
        ------
        batch_onsiteEs:
            dict. 
            e.g.: {f: [tensor[es,ep], tensor[es,ep]]}
        """
        batch_onsiteEs = {}
        for kf in list(batch_bonds_onsite.keys()):  # kf is the index of frame number.
            bonds_onsite = batch_bonds_onsite[kf][:,1:]
            # ia_list = map(lambda x: atomic_num_dict_r[int(x)], bonds_onsite[:,0]) # itype
            ia_list = map(lambda x: [atomic_num_dict_r[int(x[0])],int(x[1])], bonds_onsite[:,0:2]) # [itype,i_index]

            if self.functype == 'none':
                onsiteEs = map(lambda x: self.onsite_db[x[0]], ia_list)
            
            elif self.functype in ['uniform','split']:
                onsiteEs = []
                for x in ia_list:
                    onsiteEs.append(self.skEs(xtype=x[0], onsite_db= self.onsite_db, nn_onsite_paras=nn_onsite_paras))
            elif self.functype == 'NRL':
                onsiteEs = []
                for x in ia_list:
                    ia = x[0]
                    paraArray = th.stack([nn_onsite_paras[isk] for isk in self.onsite_index_dict[f'{ia}']])

                    xind=x[1]
                    x_env_indlist = onsite_env[kf][:,2] == xind
                    x_onsite_envs = onsite_env[kf][x_env_indlist,8] # r_jis

                    paras = {'x_onsite_envs':x_onsite_envs, 
                             'nn_onsite_paras':paraArray,
                             'rcut':self.onsite_func_cutoff,
                             'w':self.onsite_func_decay_w,
                             'lda':self.onsite_func_lambda
                            }
                    onsiteEs.append(self.skEs(**paras))
            else:
                raise ValueError(f'Invalid mode: {self.functype}')
            
            batch_onsiteEs[kf] = list(onsiteEs)

        return batch_onsiteEs
    

if __name__ == '__main__':
    onsite = loadOnsite({'N': {'2s': [0], '2p': [1,2,3]}, 'B': {'2s': [0], '2p': [1,2,3]}})
    print(len(onsite['N']))