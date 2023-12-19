import torch as th
from dptb.utils.constants import atomic_num_dict_r
from dptb.nnsktb.socDB import soc_strength_database
import logging
log = logging.getLogger(__name__)

# define the function for output all the onsites Es for given i.

def loadSoc(soc_map: dict):
    """ load the onsite energies from the database, according to the onsite_map:dict
    This function only need to run once before calculation/ training.

    Parameters:
    -----------
        soc_map: dict, only support the onsite mode tobe uniform and strain.
            {'N': {'2s': [0], '2p': [1]}, 'B': {'2s': [0], '2p': [1]}}    
    Returns:
    --------
        soc_db: dict, the format follows the input onsite_map, e.g.:
            {'N':tensor[lamba_s,lambda_p], 'B': tensor[lambda_s, lambda_p]}

    """

    atoms_types = list(soc_map.keys())
    soc_db = {}
    for ia in atoms_types:
        assert ia in soc_strength_database.keys(), f'{ia} is not in the onsite_energy_database. \n see the onsite_energy_database in dptb.nnsktb.onsiteDB.py.'
        soc_energies = soc_strength_database[ia]
        indeces = sum(list(soc_map[ia].values()),[])
        soc_db[ia] = th.zeros(len(indeces))
        for isk in soc_map[ia].keys():
            assert isk in soc_energies.keys(), f'{isk} is not in the onsite_energy_database for {ia} atom. \n see the onsite_energy_database in dptb.nnsktb.onsiteDB.py.'
            soc_db[ia][soc_map[ia][isk]] = soc_energies[isk]

    return soc_db

def socFunc(batch_bonds_onsite, soc_db: dict, nn_soc: dict=None):
    """ This function is to get the onsite energies for given bonds_onsite.

    Parameters:
    -----------
        batch_bonds_onsite: list
            e.g.:  dict(f: [[f, 7, 0, 7, 0, 0, 0, 0],
                            [f, 5, 1, 5, 1, 0, 0, 0]])
        onsite_db: dict from function loadOnsite
            e.g.: {'N':tensor[lamba_s, lambda_p], 'B': tensor[lambda_s,lambda_p]}
    
    Return:
    ------
    batch_onsiteEs:
        dict. 
        e.g.: {f: [tensor[es,ep], tensor[es,ep]]} or {f: [tensor[es,ep1,ep2,ep3], tensor[es,ep1,ep2,ep3]]}.
    """
    batch_socs = {}
    # TODO: change this part back to the original one, see the qgonsite branch.
    for kf in list(batch_bonds_onsite.keys()):
        bonds_onsite = batch_bonds_onsite[kf][:,1:]
        ia_list = map(lambda x: atomic_num_dict_r[int(x)], bonds_onsite[:,0]) # itype
        if nn_soc is not None:
            socs = []
            for x in ia_list:
                soc_tmp_value = nn_soc[x]
                if (soc_tmp_value <0).any():
                    log.warning(f'nn_soc[{x}] is negative, use the use its absoulte value.')
                    soc_tmp_value = soc_tmp_value.abs()

                soc = soc_tmp_value
                soc[:len(soc_db[x])] += soc_db[x]
                socs.append(soc)
        else:
            socs = map(lambda x: soc_db[x], ia_list)
        batch_socs[kf] = list(socs)

    return batch_socs

if __name__ == '__main__':
    onsite = loadSoc({'N': {'2s': [0], '2p': [1,2,3]}, 'B': {'2s': [0], '2p': [1,2,3]}})
    print(len(onsite['N']))