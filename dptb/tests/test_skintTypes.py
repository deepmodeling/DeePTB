from email.quoprimime import body_check
from dptb.nnsktb.skintTypes import all_skint_types
from dptb.utils.tools import Index_Mapings

def test_skintTypes():
    envtype = ['N','B']
    bondtype = ['N','B']
    proj_atom_anglr_m={'N':['2s','2p'],'B':['2p']}
    indmap = Index_Mapings(envtype=envtype, bondtype=bondtype, proj_atom_anglr_m=proj_atom_anglr_m)
    bond_map, bond_num = indmap.Bond_Ind_Mapings()
    all_skint_types_dict, reducted_skint_types, sk_bond_ind_dict = all_skint_types(bond_map)


    all_skint_types_check= {'N-N-2s-2s-0': 'N-N-2s-2s-0',
                            'N-N-2s-2p-0': 'N-N-2s-2p-0',
                            'N-N-2p-2s-0': 'N-N-2s-2p-0',
                            'N-N-2p-2p-0': 'N-N-2p-2p-0',
                            'N-N-2p-2p-1': 'N-N-2p-2p-1',
                            'N-B-2s-2p-0': 'N-B-2s-2p-0',
                            'N-B-2p-2p-0': 'N-B-2p-2p-0',
                            'N-B-2p-2p-1': 'N-B-2p-2p-1',
                            'B-N-2p-2s-0': 'N-B-2s-2p-0',
                            'B-N-2p-2p-0': 'N-B-2p-2p-0',
                            'B-N-2p-2p-1': 'N-B-2p-2p-1',
                            'B-B-2p-2p-0': 'B-B-2p-2p-0',
                            'B-B-2p-2p-1': 'B-B-2p-2p-1'}

    reducted_skint_types_check = ['N-N-2s-2s-0',
                                  'N-N-2s-2p-0',
                                  'N-N-2p-2p-0',
                                  'N-N-2p-2p-1',
                                  'N-B-2s-2p-0',
                                  'N-B-2p-2p-0',
                                  'N-B-2p-2p-1',
                                  'B-B-2p-2p-0',
                                  'B-B-2p-2p-1']

    sk_bond_ind_check = {'N-N': ['N-N-2s-2s-0', 'N-N-2s-2p-0', 'N-N-2p-2p-0', 'N-N-2p-2p-1'],
                         'N-B': ['N-B-2s-2p-0', 'N-B-2p-2p-0', 'N-B-2p-2p-1'],
                         'B-N': ['N-B-2s-2p-0', 'N-B-2p-2p-0', 'N-B-2p-2p-1'],
                         'B-B': ['B-B-2p-2p-0', 'B-B-2p-2p-1']}

    assert isinstance(all_skint_types_dict, dict)
    assert isinstance(reducted_skint_types, list)
    assert isinstance(sk_bond_ind_dict, dict)

    assert all_skint_types_dict == all_skint_types_check
    assert reducted_skint_types == reducted_skint_types_check
    assert sk_bond_ind_dict == sk_bond_ind_check
    

    uniq_sktype = set(all_skint_types_dict.values())
    assert len(uniq_sktype) == len(reducted_skint_types)
    for ia in uniq_sktype:
        assert ia in reducted_skint_types
    
    assert list(sk_bond_ind_dict.keys()) == (['N-N', 'N-B', 'B-N', 'B-B'])
    assert bond_map.keys() == sk_bond_ind_dict.keys()

    for ibt in sk_bond_ind_dict.keys():
        for isk in  bond_map[ibt].keys():
            index = bond_map[ibt][isk]
            for ii in range(len(index)):
                skbondname = f'{ibt}-{isk}-{ii}'
                assert sk_bond_ind_dict[ibt][index[ii]] == all_skint_types_dict[skbondname]

    