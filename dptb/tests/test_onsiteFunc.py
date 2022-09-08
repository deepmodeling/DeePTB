import pytest
import torch as th
import numpy as np
from dptb.nnsktb.onsiteFunc import loadOnsite, onsiteFunc


class TestOnsiteUniform:
    batch_bonds_onsite= {0: np.array([[0, 7, 0, 7, 0, 0, 0, 0],
                                      [0, 5, 1, 5, 1, 0, 0, 0]])}
    onsite_map = {'N': {'2s': [0], '2p': [1]}, 'B': {'2s': [0], '2p': [1]}}
    
    def test_loadOnsite(self):
        onsitedb = loadOnsite(self.onsite_map)
        assert list(onsitedb.keys()) == ['N', 'B']
        assert onsitedb['N'].shape == th.Size([2])
        assert onsitedb['B'].shape == th.Size([2])
        assert th.abs(onsitedb['N'] -th.tensor([-0.6769242 , -0.26596692])).sum() < 1e-6
        assert th.abs(onsitedb['B'] -th.tensor([-0.34482, -0.13648])).sum() < 1e-6

        with pytest.raises(AssertionError):
            onsite_map = {'N1': {'2s': [0], '2p': [1]}, 'B': {'2s': [0], '2p': [1]}}
            loadOnsite(onsite_map)

            onsite_map = {'N': {'2s': [0], '2p': [1], '3s': [0]}, 'B': {'2s': [0], '2p': [1]}}
            loadOnsite(onsite_map)

    def test_onsiteFunc(self):
        onsitedb = loadOnsite(self.onsite_map)
        onsitesEdict = onsiteFunc(self.batch_bonds_onsite, onsitedb)
        result = { 0:[th.tensor([-0.6769242 , -0.26596692]), th.tensor([-0.34482, -0.13648])] }
        for ionsite in onsitesEdict:
            for ie in range(len(onsitesEdict[ionsite])):
                assert (onsitesEdict[ionsite][ie] == result[ionsite][ie]).all()

    def test_onsiteFunc_with_NN(self):
        onsitedb = loadOnsite(self.onsite_map)
        nn_onsiteE = {'N': th.tensor([-0.0178,  0.0427]),
                      'B': th.tensor([ 0.0124, -0.0815])}
        onsitesEdict = onsiteFunc(self.batch_bonds_onsite, onsitedb, nn_onsiteE=nn_onsiteE)

        assert (th.abs(onsitesEdict[0][0] - th.tensor([-0.6947242 , -0.22326693])) < 1e-6).all()
        assert (th.abs(onsitesEdict[0][1] - th.tensor([-0.33242, -0.21798])) < 1e-6).all()


class TestOnsiteSplit:
    batch_bonds_onsite= {0: np.array([[0, 7, 0, 7, 0, 0, 0, 0],
                                      [0, 5, 1, 5, 1, 0, 0, 0]])}
    onsite_map = {'N': {'2s': [0], '2p': [1,2,3]}, 'B': {'2s': [0], '2p': [1,2,3]}}

    def test_loadOnsite(self):
        onsitedb = loadOnsite(self.onsite_map)
        assert list(onsitedb.keys()) == ['N', 'B']
        assert onsitedb['N'].shape == th.Size([4])
        assert onsitedb['B'].shape == th.Size([4])
        assert th.abs(onsitedb['N'] -th.tensor([-0.6769242 , -0.26596692, -0.26596692, -0.26596692])).sum() < 1e-6
        assert th.abs(onsitedb['B'] -th.tensor([-0.34482, -0.13648, -0.13648, -0.13648])).sum() < 1e-6

        with pytest.raises(AssertionError):
            onsite_map = {'N1': {'2s': [0], '2p': [1,2,3]}, 'B': {'2s': [0], '2p': [1,2,3]}}
            loadOnsite(onsite_map)

            onsite_map = {'N1': {'2s': [0], '2p': [1,2,3],'3s':[4]}, 'B': {'2s': [0], '2p': [1,2,3]}}
            loadOnsite(onsite_map)

    def test_onsiteFunc(self):
        onsitedb = loadOnsite(self.onsite_map)
        onsitesEdict = onsiteFunc(self.batch_bonds_onsite, onsitedb)
        result = { 0:[th.tensor([-0.6769242 , -0.26596692, -0.26596692, -0.26596692]), th.tensor([-0.34482, -0.13648, -0.13648, -0.13648])] }
        for ionsite in onsitesEdict:
            for ie in range(len(onsitesEdict[ionsite])):
                assert (onsitesEdict[ionsite][ie] == result[ionsite][ie]).all()

    
    def test_onsiteFunc_with_NN(self):
        onsitedb = loadOnsite(self.onsite_map)
        nn_onsiteE = {'N': th.tensor([0.0187, 0.0902, 0.1455,  0.1546]),
                      'B': th.tensor([0.0703, 0.2198, 0.0054, -0.0410])}
        onsitesEdict = onsiteFunc(self.batch_bonds_onsite, onsitedb, nn_onsiteE=nn_onsiteE)

        assert (th.abs(onsitesEdict[0][0] - th.tensor([-0.6582242 , -0.17576692, -0.12046692, -0.11136693])) < 1e-6).all()
        assert (th.abs(onsitesEdict[0][1] - th.tensor([-0.27451998,  0.08331999, -0.13108   , -0.17748001])) < 1e-6).all()