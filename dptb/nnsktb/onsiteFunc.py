from unittest import main
from xml.etree.ElementTree import tostring
import torch as th
from dptb.utils.constants import atomic_num_dict_r
from dptb.nnsktb.onsiteDB import onsite_energy_database
from dptb.nnsktb.formula import SKFormula
from dptb.nnsktb.onsite_formula import onsiteFormula
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
    def __init__(self, functype='none') -> None:
        super().__init__(functype)

    def get_onsiteEs(self,batch_bonds_onsite, onsite_db: dict, nn_onsiteE: dict=None, **kwargs):
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
            ia_list = map(lambda x: atomic_num_dict_r[int(x)], bonds_onsite[:,0]) # itype
            if self.functype == 'none':
                onsiteEs = map(lambda x: onsite_db[x], ia_list)
            else:
                onsiteEs = []
                for x in ia_list:
                    onsiteEs.append(self.skEs(xtype=x, onsite_db=onsite_db, nn_onsiteE=nn_onsiteE, **kwargs))
                
            batch_onsiteEs[kf] = list(onsiteEs)
        return batch_onsiteEs
    

if __name__ == '__main__':
    onsite = loadOnsite({'N': {'2s': [0], '2p': [1,2,3]}, 'B': {'2s': [0], '2p': [1,2,3]}})
    print(len(onsite['N']))