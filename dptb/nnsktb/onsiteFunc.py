import torch as th
from dptb.utils.constants import atomic_num_dict_r
import os
# define the function for output all the onsites Es for given i.

onsite_path = ""

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
        # ToDo: define load_onstie_atom
        # call: database_data = load_onstie_atom()
        onsite_file = th.load(f=os.path.join(onsite_path, ia+".pth"))
        onsite_db[ia] = th.zeros(len(onsite_map[ia]))
        for isk in onsite_map[ia].keys():
            onsite_db[ia][onsite_map[ia][isk]] = onsite_file[ia+"-"+isk]["e"]

    return onsite_db

def onsiteFunc(bonds_onsite, onsite_db):
    """ This function is to get the onsite energies for given bonds_onsite.

    Parameters:
    -----------
        bonds_onsite: list
            e.g.:  np.array([[7, 0, 7, 0, 0, 0, 0],
                             [5, 1, 5, 1, 0, 0, 0]])
        onsite_db: dict from function loadOnsite
            e.g.: {'N':tensor[es,ep], 'B': tensor[es,ep]}
    """
    onsite_energies = []
    for i in range(len(bonds_onsite)):
        ia = atomic_num_dict_r(int(bonds_onsite[i][0]))
        onsite_energies.append(onsite_db[ia])

    return onsite_energies