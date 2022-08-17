import re
import numpy as np
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
                        reducted_skint_types.append(iskint_type)
                        all_skint_types_dict[iskint_type] = iskint_type

        elif iaid > jaid:
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


