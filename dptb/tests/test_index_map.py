from dptb.utils.tools import Index_Mapings
import pytest

class TestIndexMap:
    envtype = ['N','B']
    bondtype = ['N','B']
    proj_atom_anglr_m={'N':['2s','2p'],'B':['2s','2p']}
    indmap = Index_Mapings(proj_atom_anglr_m=proj_atom_anglr_m)

    def test_bond_mapings(self):
        bond_map, bond_num = self.indmap.Bond_Ind_Mapings()
        assert bond_map == {'N-N': {'2s-2s': [0], '2s-2p': [1], '2p-2s': [1], '2p-2p': [2, 3]},
                            'N-B': {'2s-2s': [0], '2s-2p': [1], '2p-2s': [2], '2p-2p': [3, 4]},
                            'B-N': {'2s-2s': [0], '2s-2p': [2], '2p-2s': [1], '2p-2p': [3, 4]},
                            'B-B': {'2s-2s': [0], '2s-2p': [1], '2p-2s': [1], '2p-2p': [2, 3]}}

        assert bond_num == {'N-N': 4, 'N-B': 5, 'B-N': 5, 'B-B': 4}

    def test_onsite_mapings(self):
        onsite_map, onsite_num = self.indmap.Onsite_Ind_Mapings()
        
        assert onsite_map == {'N': {'2s': [0], '2p': [1]}, 'B': {'2s': [0], '2p': [1]}}
        assert onsite_num == {'N': 2, 'B': 2}


    def test_onsite_split_mapings(self):
        onsite_map, onsite_num = self.indmap.Onsite_Ind_Mapings_OrbSplit()
        
        assert onsite_map == {'N': {'2s': [0], '2p': [1, 2, 3]}, 'B': {'2s': [0], '2p': [1, 2, 3]}}
        assert onsite_num == {'N': 4, 'B': 4}