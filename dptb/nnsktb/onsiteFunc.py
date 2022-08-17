import torch as th
from dptb.utils.constants import atomic_num_dict_r
from dptb.nnsktb.onsiteDB import onsite_energy_database

# define the function for output all the onsites Es for given i.

def loadOnsite(onsite_map: dict):
    """ load the onsite energies from the database, according to the onsite_map:dict
    This function only need to run once before calculation/ training.

    Parameters:
    -----------
        onsite_map: dict
            for example: {'N': {'2s': [0], '2p': [1]}, 'B': {'2s': [0], '2p': [1]}}
    """

    atoms_types = list(onsite_map.keys())
    onsite_db = {}
    for ia in atoms_types:
        assert ia in onsite_energy_database.keys(), f'{ia} is not in the onsite_energy_database. \n see the onsite_energy_database in dptb.nnsktb.onsiteDB.py.'
        orb_energies = onsite_energy_database[ia]
        onsite_db[ia] = th.zeros(len(onsite_map[ia]))
        for isk in onsite_map[ia].keys():
            assert isk in orb_energies.keys(), f'{isk} is not in the onsite_energy_database for {ia} atom. \n see the onsite_energy_database in dptb.nnsktb.onsiteDB.py.'
            onsite_db[ia][onsite_map[ia][isk]] = orb_energies[isk]

    return onsite_db

def onsiteFunc(batch_bonds_onsite, onsite_db):
    """ This function is to get the onsite energies for given bonds_onsite.

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
        dict. {f: [tensor[es,ep], tensor[es,ep]]}
    """
    batch_onsiteEs = {}

    for kf in list(batch_bonds_onsite.keys()):
        bonds_onsite = batch_bonds_onsite[kf][:,1:]
        ia_list = map(lambda x: atomic_num_dict_r[int(x)], bonds_onsite[:,0])
        onsiteEs = map(lambda x: onsite_db[x], ia_list)
        batch_onsiteEs.update({kf:list(onsiteEs)})

    return batch_onsiteEs