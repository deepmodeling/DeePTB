import pytest
import torch as th
import numpy as np
from dptb.nnsktb.socFunc import loadSoc, socFunc

class TestOnsiteUniform:
    batch_bonds_onsite= {0: np.array([[0, 7, 0, 7, 0, 0, 0, 0],
                                      [0, 5, 1, 5, 1, 0, 0, 0]])}
    onsite_map = {'N': {'2s': [0], '2p': [1]}, 'B': {'2s': [0], '2p': [1]}}
    
    def test_loadSOC(self):
        onsitedb = loadSoc(self.onsite_map)
        assert list(onsitedb.keys()) == ['N', 'B']
        assert onsitedb['N'].shape == th.Size([2])
        assert onsitedb['B'].shape == th.Size([2])
        assert th.abs(onsitedb['N'] -th.tensor([0.0, 0.0])).sum() < 1e-6
        assert th.abs(onsitedb['B'] -th.tensor([0.0, 0.0])).sum() < 1e-6

        with pytest.raises(AssertionError):
            onsite_map = {'N1': {'2s': [0], '2p': [1]}, 'B': {'2s': [0], '2p': [1]}}
            loadSoc(onsite_map)

            onsite_map = {'N': {'2s': [0], '2p': [1], '3s': [0]}, 'B': {'2s': [0], '2p': [1]}}
            loadSoc(onsite_map)

    def test_onsiteFunc(self):
        onsitedb = loadSoc(self.onsite_map)
        onsiteSOCdict = socFunc(self.batch_bonds_onsite, onsitedb)
        result = { 0:[th.tensor([0.0, 0.0]), th.tensor([0.0, 0.0])] }
        for ionsite in onsiteSOCdict:
            for ie in range(len(onsiteSOCdict[ionsite])):
                assert (onsiteSOCdict[ionsite][ie] == result[ionsite][ie]).all()

    def test_onsiteFunc_with_NN(self):
        onsitedb = loadSoc(self.onsite_map)
        nn_onsiteSOC = {'N': th.tensor([1.0,  2.0]),
                        'B': th.tensor([1.0,  2.0])}
        onsitesSOCdict = socFunc(self.batch_bonds_onsite, onsitedb, nn_soc=nn_onsiteSOC)

        assert (th.abs(onsitesSOCdict[0][0] - th.tensor([1.0, 2.0])) < 1e-6).all()
        assert (th.abs(onsitesSOCdict[0][1] - th.tensor([1.0, 2.0])) < 1e-6).all()
