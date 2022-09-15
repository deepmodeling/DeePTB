from xml.etree.ElementTree import tostring
import torch as th
from dptb.utils.constants import atomic_num_dict_r
from dptb.nnsktb.onsiteDB import onsite_energy_database

# define the function for output all the onsites Es for given i.

def loadOnsite(onsite_map: dict):
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
            onsite_db[ia][onsite_map[ia][isk]] = orb_energies[isk]

    return onsite_db

def onsiteFunc(batch_bonds_onsite, onsite_db: dict, nn_onsiteE: dict=None):
    """ This function is to get the onsite energies for given bonds_onsite.

    Parameters:
    -----------
        batch_bonds_onsite: list
            e.g.:  dict(f: [[f, 7, 0, 7, 0, 0, 0, 0],
                             [f, 5, 1, 5, 1, 0, 0, 0]])
        onsite_db: dict from function loadOnsite
            e.g.: {'N':tensor[es,ep], 'B': tensor[es,ep]} or {'N':tensor[es,ep1,ep2,ep3], 'B': tensor[es,ep1,ep2,ep3]}
    
    Return:
    ------
    batch_onsiteEs:
        dict. 
        e.g.: {f: [tensor[es,ep], tensor[es,ep]]} or {f: [tensor[es,ep1,ep2,ep3], tensor[es,ep1,ep2,ep3]]}.
    """
    batch_onsiteEs = {}

    for kf in list(batch_bonds_onsite.keys()):
        bonds_onsite = batch_bonds_onsite[kf][:,1:]
        ia_list = map(lambda x: atomic_num_dict_r[int(x)], bonds_onsite[:,0]) # itype
        if nn_onsiteE is not None:
            onsiteEs = map(lambda x: onsite_db[x] + nn_onsiteE[x], ia_list)
        else:
            onsiteEs = map(lambda x: onsite_db[x], ia_list)
        batch_onsiteEs[kf] = list(onsiteEs)

    return batch_onsiteEs

def crt_onsiteFunc(batch_bonds_onsite, batch_env, onsite_db: dict, nn_onsiteE: dict=None, onsite_map: dict=None):
    """ This function is to get the onsite energies for given bonds_onsite.

    Parameters:
    -----------
        batch_bonds_onsite: list
            e.g.:  dict(f: [[f, 7, 0, 7, 0, 0, 0, 0],
                             [f, 5, 1, 5, 1, 0, 0, 0]])
        onsite_db: dict from function loadOnsite
            e.g.: {'N':tensor[es,ep], 'B': tensor[es,ep]} or {'N':tensor[es,ep1,ep2,ep3], 'B': tensor[es,ep1,ep2,ep3]}
        batch_env: dict
            e.g. {env_type:(f,i,itype,s(r),rx,ry,rz)}
    
    Return:
    ------
    batch_onsiteEs:
        dict. 
        e.g.: {f: [tensor[es,ep], tensor[es,ep]]} or {f: [tensor[es,ep1,ep2,ep3], tensor[es,ep1,ep2,ep3]]}.
    """
    batch_onsiteEs = {}
    batched_nei = {}
    # rearranged by f-i
    for item in batch_env:
        name = str(int(item[0]))+'-'+str(int(item[1]))
        if batched_nei.get(name) is None:
            batched_nei[name] = [item]
        else:
            batched_nei[name].append(item)

    for kf in list(batch_bonds_onsite.keys()):
        bonds_onsite = batch_bonds_onsite[kf][:,1:]
        iatype_list = map(lambda x: atomic_num_dict_r[int(x)], bonds_onsite[:,0])
        if nn_onsiteE is not None:
            onsiteEs = map(lambda x: onsite_db[x] + nn_onsiteE[x], iatype_list)
        else:
            onsiteEs = map(lambda x: onsite_db[x], iatype_list)
        batch_onsiteEs[kf] = list(onsiteEs)

        # neighbour correction
        for ia in range(batch_bonds_onsite[kf].shape[0]):
            aid = bonds_onsite[ia, 1]
            atype = bonds_onsite[ia, 0]
            env_key = str(int(kf))+'-'+str(int(aid))
            nei = th.stack(batched_nei[env_key])


    return batch_onsiteEs


def onsite_p(R, V1, V2_sigma_r, V2_pi_r):
    '''
        R: [n_nei, 3]
        V1: scalar
        V_2_sigma_r: [n_nei]
        V2_pi_r: [n_nei]
    '''
    l = R / R.norm(dim=1).unsqueeze(1) # [n_nei, 3]
    ml = l.unsqueeze(2) @ l.unsqueeze(1) # [n_nei, 3, 3]
    out = (V2_sigma_r - V2_pi_r).reshape(-1, 1, 1) * ml # [n_nei, 3, 3]
    out = out.sum(dim=0)

    out = th.diag(V1+V2_sigma_r.sum()) + out # [3,3]

    return out

def onsite_s(V1, V2_sigma_r):
    '''
        V1: scalar
        V_2_sigma_r: [n_nei]
    '''
    out = V1+V2_sigma_r.sum()

    return out


