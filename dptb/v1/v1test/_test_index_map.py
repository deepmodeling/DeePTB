from dptb.utils.index_mapping import Index_Mapings
import pytest

class TestIndexMap:
    envtype = ['N','B']
    bondtype = ['N','B']
    proj_atom_anglr_m={'N':['2s','2p'],'B':['3s','3p']}
    indmap = Index_Mapings(proj_atom_anglr_m=proj_atom_anglr_m)

    def test_default_init (self):
        indmap2 = Index_Mapings()
        indmap2.update(proj_atom_anglr_m=self.proj_atom_anglr_m)
        assert indmap2.AnglrMID == self.indmap.AnglrMID
        assert indmap2.bondtype == self.indmap.bondtype
        assert indmap2.ProjAnglrM == self.indmap.ProjAnglrM

    def test_bond_mapings(self):
        bond_map, bond_num = self.indmap.Bond_Ind_Mapings()
        assert bond_map == {'N-N': {'2s-2s': [0], '2s-2p': [1], '2p-2s': [1], '2p-2p': [2, 3]},
                            'N-B': {'2s-3s': [0], '2s-3p': [1], '2p-3s': [2], '2p-3p': [3, 4]},
                            'B-N': {'3s-2s': [0], '3s-2p': [2], '3p-2s': [1], '3p-2p': [3, 4]},
                            'B-B': {'3s-3s': [0], '3s-3p': [1], '3p-3s': [1], '3p-3p': [2, 3]}}

        assert bond_num == {'N-N': 4, 'N-B': 5, 'B-N': 5, 'B-B': 4}

    def test_onsite_mapings(self):
        _, _, onsite_map, onsite_num = self.indmap.Onsite_Ind_Mapings(onsitemode="uniform")
        
        assert onsite_map == {'N': {'2s': [0], '2p': [1]}, 'B': {'3s': [0], '3p': [1]}}
        assert onsite_num == {'N': 2, 'B': 2}

        _, _, onsite_map, onsite_num = self.indmap.Onsite_Ind_Mapings(onsitemode="none")
        assert onsite_map == {'N': {'2s': [0], '2p': [1]}, 'B': {'3s': [0], '3p': [1]}}
        assert onsite_num == {'N': 2, 'B': 2}

    def test_onsite_split_mapings(self):
        _, _, onsite_map, onsite_num = self.indmap.Onsite_Ind_Mapings(onsitemode="split")
        
        assert onsite_map ==  {'N': {'2s': [0], '2p': [1, 2, 3]}, 'B': {'3s': [0], '3p': [1, 2, 3]}}
        assert onsite_num == {'N': 4, 'B': 4}

    def test_onsite_strain_mapings(self):
        with pytest.raises(AssertionError) as exception_info:
            self.indmap.Onsite_Ind_Mapings(onsitemode="strain")

        onsite_strain_index_map, onsite_strain_num, onsite_index_map, onsite_num = self.indmap.Onsite_Ind_Mapings(onsitemode="strain",atomtype=['N','B'])
        assert onsite_strain_index_map == {'N-N': {'2s-2s': [0], '2s-2p': [1], '2p-2s': [1], '2p-2p': [2, 3]},
                                           'N-B': {'2s-2s': [0], '2s-2p': [1], '2p-2s': [1], '2p-2p': [2, 3]},
                                           'B-N': {'3s-3s': [0], '3s-3p': [1], '3p-3s': [1], '3p-3p': [2, 3]},
                                           'B-B': {'3s-3s': [0], '3s-3p': [1], '3p-3s': [1], '3p-3p': [2, 3]}}
        assert onsite_strain_num == {'N-N': 4, 'N-B': 4, 'B-N': 4, 'B-B': 4}
        assert onsite_index_map == {'N': {'2s': [0], '2p': [1]}, 'B': {'3s': [0], '3p': [1]}}
        assert onsite_num == {'N': 2, 'B': 2}

        _, _, onsite_index_map_unifrom, onsite_num_uniform = self.indmap.Onsite_Ind_Mapings(onsitemode="uniform")

        assert onsite_index_map_unifrom == onsite_index_map
        assert onsite_num_uniform == onsite_num

    def test_onsite_nrl_mappings(self):
        # since for now nrl use the same as uniform and none.
        _, _, onsite_map, onsite_num = self.indmap.Onsite_Ind_Mapings(onsitemode="NRL")
        
        assert onsite_map == {'N': {'2s': [0], '2p': [1]}, 'B': {'3s': [0], '3p': [1]}}
        assert onsite_num == {'N': 2, 'B': 2}