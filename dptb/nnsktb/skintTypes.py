import re
import numpy as np
import torch
from dptb.utils.constants import anglrMId, SKBondType
from dptb.utils.constants import atomic_num_dict


def all_skint_types(bond_index_map):
    """ This function is to get all the possible sk integral types by given the bond_index_map.
    
    Parameters:
    ----------
    bond_index_map: 
        dict, output of Index_Mapings.Bond_Ind_Mapings()
        e.g.: {'N-N': {'2s-2s': [0]}, 'N-B': {'2s-2p': [0]}, 'B-N': {'2p-2s': [0]}, 'B-B': {'2p-2p': [0, 1]}}
                for proj_atom_anglr_m={'N':['2s'],'B':['2p']}.
    Returns:
    --------
    all_skint_types_dict: dict
        All the possible sk integral types. and maps to the reduced one.
        key is the all the possible sk integral types.
        value is the reduced one.
        e.g.:
            {'N-N-2s-2s-0': 'N-N-2s-2s-0',
             'N-B-2s-2p-0': 'N-B-2s-2p-0',
             'B-N-2p-2s-0': 'N-B-2s-2p-0',
             'B-B-2p-2p-0': 'B-B-2p-2p-0',
             'B-B-2p-2p-1': 'B-B-2p-2p-1'}

    reducted_skint_types: list
        The independent/reduced sk integral types.
        e.g.: ['N-N-2s-2s-0', 'N-B-2s-2p-0', 'B-B-2p-2p-0', 'B-B-2p-2p-1']

    sk_bond_ind_dict: dict
        the skbond type in the format in bond_index_map. 
        e.g.:
            {'N-N': ['N-N-2s-2s-0'],
             'N-B': ['N-B-2s-2p-0'],
             'B-N': ['N-B-2s-2p-0'],
             'B-B': ['B-B-2p-2p-0', 'B-B-2p-2p-1']}

    """


    reducted_skint_types = []
    all_skint_types_dict = {}
    for ibm in bond_index_map:
        # atom symbol. e.g.: 'N', 'B'
        ia, ja = ibm.split('-')  
        # atomic number. e.g.: 7, 5
        iaid, jaid = atomic_num_dict[ia], atomic_num_dict[ja] 
        if iaid == jaid: 
            #for the same atom type, exchange iorb and jorb, they are the same.
            for isk in bond_index_map[ibm].keys():  
                # orb-bond type: e.g. '2s-2s', or 's-p' or '2s-s*' ...
                iorb, jorb = isk.split('-') 
                 # len(bond_index_map[ibm][isk]) is number of bonds. for s-s 1, for p-p 2, ...
                for iisk in range(len(bond_index_map[ibm][isk])): 
                    # iskint_type: e.g.: 'N-N-s-s-0', 'N-B-2p-2p-0' or  'N-B-p-p-1', etc. 
                    iskint_type = f'{ia}-{ja}-{iorb}-{jorb}-{iisk}'
                    # iskint_type_ex exchange i-j orb in iskint_type.
                    iskint_type_ex = f'{ia}-{ja}-{jorb}-{iorb}-{iisk}'   
                    if iskint_type_ex in reducted_skint_types:
                        all_skint_types_dict[iskint_type] = iskint_type_ex
                    else:
                        assert iskint_type not in reducted_skint_types, f'{iskint_type} is already in reducted_skint_types.'
                        reducted_skint_types.append(iskint_type)
                        all_skint_types_dict[iskint_type] = iskint_type
        
        elif iaid > jaid:
            # note: here we set the order of reduced iaid > jaid.
            # for iaid < jaid, we use the one with iaid > jaid and exchange iorb and jorb.
            # for different atom type, exchange the pair (ia, iorb) with (ja, jorb), they are the same.
            for isk in bond_index_map[ibm].keys():
                iorb, jorb = isk.split('-')
                for iisk in range(len(bond_index_map[ibm][isk])):
                    iskint_type = f'{ia}-{ja}-{iorb}-{jorb}-{iisk}'
                    reducted_skint_types.append(iskint_type)
                    all_skint_types_dict[iskint_type] = iskint_type
        else:
            for isk in bond_index_map[ibm].keys():
                iorb, jorb = isk.split('-')
                for iisk in range(len(bond_index_map[ibm][isk])):
                    iskint_type_ex = f'{ja}-{ia}-{jorb}-{iorb}-{iisk}'
                    iskint_type = f'{ia}-{ja}-{iorb}-{jorb}-{iisk}'
                    all_skint_types_dict[iskint_type] = iskint_type_ex

    for ii in np.unique(list(all_skint_types_dict.values())):
        assert ii in reducted_skint_types
    
    # arrange as bond_index_map
    sk_bond_ind_dict = {}

    for ibm  in bond_index_map:
        uniq_bond_indeces = np.unique(np.concatenate(list(bond_index_map[ibm].values())))
        num_indepd_sks = len(uniq_bond_indeces)

        sk_bond_ind_dict[ibm]  = ['']*num_indepd_sks
        ia, ja = ibm.split('-')  

        for isk in bond_index_map[ibm].keys():  
            iorb, jorb = isk.split('-') 
            for iisk in range(len(bond_index_map[ibm][isk])): 
                iskint_type = f'{ia}-{ja}-{iorb}-{jorb}-{iisk}'
                sk_bond_ind_dict[ibm][ bond_index_map[ibm][isk][iisk] ] = all_skint_types_dict[iskint_type]

    return all_skint_types_dict, reducted_skint_types, sk_bond_ind_dict


def NRL_skint_type_constants(reducted_skint_types):
    '''The function `NRL_skint_type_constants` calculates a dictionary of skin type constants based on a
    list of reduced skin types.

    Parameters
    ----------
    reducted_skint_types: list
        A list of reduced skin types. e.g.: ['N-N-2s-2s-0', 'N-B-2s-2p-0', 'B-B-2p-2p-0', 'B-B-2p-2p-1']

    Returns
    -------
    sk_para_delta: dict
        A dictionary of skin type constants. e.g.: {'N-N-2s-2s-0': tensor[1.0], 'N-B-2s-2p-0': tensor[0.0], 'B-B-2p-2p-0': tensor[1.0], 'B-B-2p-2p-1': tensor[1.0]}
    
    '''
    delta_AlAl= torch.zeros(len(reducted_skint_types),1)
    for i in range(len(reducted_skint_types)):
        itype = reducted_skint_types[i]
        if itype.split('-')[0] == itype.split('-')[1] and itype.split('-')[2] == itype.split('-')[3] :
            delta_AlAl[i] = 1.0
        else:
            delta_AlAl[i] = 0.0
    sk_para_delta = dict(zip(reducted_skint_types, delta_AlAl))
    return sk_para_delta


def all_onsite_intgrl_types(onsite_intgrl_index_map):
    """ This function is to get all the possible sk like onsite integra types by given the onsite_intgrl_index_map.
    
    Parameters
    ----------
    onsite_intgrl_index_map:
           {'N-N': {'2s-2s': [0], '2s-2p': [1], '2p-2s': [1], '2p-2p': [2, 3]},
            'N-B': {'2s-2s': [0], '2s-2p': [1], '2p-2s': [1], '2p-2p': [2, 3]},
            'B-N': {'2s-2s': [0]},
            'B-B': {'2s-2s': [0]}}
        This is quite like the bond_index_map for sk hopping.
    Output
    ------ 
    all_onsite_int_types_dict:
        All the possible sk-like onsite integral types. and maps to the reduced one.
        key is the all the possible sk-like onsite integral integral types. value is the reduced one.
        {'N-N-2s-2s-0': 'N-N-2s-2s-0',
         'N-N-2s-2p-0': 'N-N-2s-2p-0',
         'N-N-2p-2s-0': 'N-N-2s-2p-0',
         'N-N-2p-2p-0': 'N-N-2p-2p-0',
         'N-N-2p-2p-1': 'N-N-2p-2p-1',
         'N-B-2s-2s-0': 'N-B-2s-2s-0',
         'N-B-2s-2p-0': 'N-B-2s-2p-0',
         'N-B-2p-2s-0': 'N-B-2s-2p-0',
         'N-B-2p-2p-0': 'N-B-2p-2p-0',
         'N-B-2p-2p-1': 'N-B-2p-2p-1',
         'B-N-2s-2s-0': 'B-N-2s-2s-0',
         'B-B-2s-2s-0': 'B-B-2s-2s-0'}

    reducted_onsite_int_types:
        The independent/reduced sk-like onsite integral types.
        ['N-N-2s-2s-0',
         'N-N-2s-2p-0',
         'N-N-2p-2p-0',
         'N-N-2p-2p-1',
         'N-B-2s-2s-0',
         'N-B-2s-2p-0',
         'N-B-2p-2p-0',
         'N-B-2p-2p-1',
         'B-N-2s-2s-0',
         'B-B-2s-2s-0']

    sk_onsite_ind_dict:
        the sk like onsite integral type in the format in onsite_intgrl_index_map. 
        {'N-N': ['N-N-2s-2s-0', 'N-N-2s-2p-0', 'N-N-2p-2p-0', 'N-N-2p-2p-1'],
         'N-B': ['N-B-2s-2s-0', 'N-B-2s-2p-0', 'N-B-2p-2p-0', 'N-B-2p-2p-1'],
         'B-N': ['B-N-2s-2s-0'],
         'B-B': ['B-B-2s-2s-0']}

    """
    reducted_onsite_int_types = []
    all_onsite_int_types_dict = {}
    for ibm in onsite_intgrl_index_map:
        ia, ja = ibm.split('-')  
        iaid, jaid = atomic_num_dict[ia], atomic_num_dict[ja] 
        for isk in onsite_intgrl_index_map[ibm].keys():
            iorb, jorb = isk.split('-') 
            for iisk in range(len(onsite_intgrl_index_map[ibm][isk])): 
                iskint_type = f'{ia}-{ja}-{iorb}-{jorb}-{iisk}'
                iskint_type_ex = f'{ia}-{ja}-{jorb}-{iorb}-{iisk}'
                if iskint_type_ex in reducted_onsite_int_types:
                    all_onsite_int_types_dict[iskint_type] = iskint_type_ex
                else:
                    reducted_onsite_int_types.append(iskint_type)
                    all_onsite_int_types_dict[iskint_type] = iskint_type
    for ii in np.unique(list(all_onsite_int_types_dict.values())):
        assert ii in reducted_onsite_int_types
    
    sk_onsite_ind_dict = {}
    for ibm  in onsite_intgrl_index_map:
        uniq_onst_indeces = np.unique(np.concatenate(list(onsite_intgrl_index_map[ibm].values())))
        num_indepd_sks = len(uniq_onst_indeces)
        
        sk_onsite_ind_dict[ibm] = [''] * num_indepd_sks
        ia, ja = ibm.split('-')  
        
        for isk in onsite_intgrl_index_map[ibm].keys():  
            iorb, jorb = isk.split('-') 
            for iisk in range(len(onsite_intgrl_index_map[ibm][isk])): 
                iskint_type = f'{ia}-{ja}-{iorb}-{jorb}-{iisk}'
                sk_onsite_ind_dict[ibm][ onsite_intgrl_index_map[ibm][isk][iisk] ] = all_onsite_int_types_dict[iskint_type]
    
    return all_onsite_int_types_dict, reducted_onsite_int_types, sk_onsite_ind_dict


def all_onsite_ene_types(onsite_index_map):
    ''' This function is to get all the possible Onsite Eergies types by given the onsite_index_map.

    Parameters
    ----------
    onsite_index_map:
        a dictionary that maps the site index to the onsite energy indeces.
        e.g.: {'N': {'2s': [0], '2p': [1]}, 'B': {'2s': [0]}}

    Output
    ------
    all_onsiteE_types_dict: dict
        All the possible sk integral types. and maps to the reduced one. 
        key is the all the possible sk integral types. value is the reduced one.
        For onsite E there is no reduction. so all the types are independent. For the input parameters above, 
        {'N-2s-0': 'N-2s-0', 
         'N-2p-0': 'N-2p-0', 
         'B-2s-0': 'B-2s-0'}
    
    reduced_onsiteE_types: list:
        The independent/reduced onsite Energy types. for the above input parameters,
         ['N-2s-0', 'N-2p-0', 'B-2s-0'],
    
    onsiteE_ind_dict: dict
        the onsite Energy type in the format in onsite_index_map. for the above input parameters,
        {'N': ['N-2s-0', 'N-2p-0'], 'B': ['B-2s-0']})

    '''
    all_onsiteE_types_dict = {}
    reduced_onsiteE_types = []
    for isite in onsite_index_map.keys():
        ia = (isite)
        for isk in onsite_index_map[isite]:
            iorb = isk
            for iisk in range(len(onsite_index_map[isite][isk])):
                onsiteE_type = f'{ia}-{iorb}-{iisk}'
                reduced_onsiteE_types.append(onsiteE_type)
                all_onsiteE_types_dict[onsiteE_type] = onsiteE_type
    
    for ii in np.unique(list(all_onsiteE_types_dict.values())):
        assert ii in reduced_onsiteE_types
    
    onsiteE_ind_dict = {}
    for isite in onsite_index_map:
        uniq_onsiteE_indeces = np.unique(np.concatenate(list(onsite_index_map[isite].values())))
        num_indepd_Es = len(uniq_onsiteE_indeces)

        onsiteE_ind_dict[isite] = [''] * num_indepd_Es

        ia  = isite
        for isk in onsite_index_map[isite]:
            iorb = isk
            for iisk in range(len(onsite_index_map[isite][isk])):
                onsiteE_type =  f'{ia}-{iorb}-{iisk}'
                onsiteE_ind_dict[isite][onsite_index_map[isite][isk][iisk]] = all_onsiteE_types_dict[onsiteE_type]
    
    return all_onsiteE_types_dict, reduced_onsiteE_types, onsiteE_ind_dict


