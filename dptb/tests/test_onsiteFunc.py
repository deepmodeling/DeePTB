import pytest
import torch as th
import torch
import numpy as np
from dptb.nnsktb.onsiteFunc import loadOnsite, onsiteFunc
from dptb.nnsktb.onsiteFunc import orbitalEs

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



class TestorbitalEs:
    envtype = ['N','B']
    bondtype = ['N','B']
    proj_atom_anglr_m={'B': ['2s'], 'N': ['2s', '2p']}
    batch_bond_onsites = {0: torch.tensor([[0., 7., 0., 7., 0., 0., 0., 0.],
         [0., 5., 1., 5., 1., 0., 0., 0.]])}
    
    def test_onsite_none(self):
        nn_onsiteE, onsite_coeffdict  = None, None
        onsitfunc = orbitalEs(proj_atom_anglr_m=self.proj_atom_anglr_m,atomtype=None, functype='none')
        
        batch_onsite_true = {0: [torch.tensor([-0.6769242287, -0.2659669220]), torch.tensor([-0.3448199928])]}

        with torch.no_grad():
            batch_onsite = onsitfunc.get_onsiteEs(batch_bonds_onsite=self.batch_bond_onsites, nn_onsite_paras=nn_onsiteE)

        assert isinstance(batch_onsite, dict)
        assert len(batch_onsite) == len(batch_onsite_true)
        assert len(batch_onsite) == len(self.batch_bond_onsites)

        for kf in batch_onsite.keys():
            assert len(batch_onsite[kf]) == len(batch_onsite_true[kf])
            assert len(batch_onsite[kf]) == len(self.batch_bond_onsites[kf])
            for i in range(len(batch_onsite[kf])):
                assert torch.allclose(batch_onsite[kf][i], batch_onsite_true[kf][i])

        
        assert isinstance(onsitfunc.onsite_db, dict)
        assert len(onsitfunc.onsite_db) == 2
        assert len(onsitfunc.onsite_db['N']) == 2
        assert len(onsitfunc.onsite_db['B']) == 1
        assert th.allclose(onsitfunc.onsite_db['N'], batch_onsite_true[0][0]) 
        assert th.allclose(onsitfunc.onsite_db['B'], batch_onsite_true[0][1]) 

        
        nn_onsiteE2 = {'N': torch.tensor([0.0019521093, 0.0031471925]), 'B': torch.tensor([0.0053026341])}

        with torch.no_grad():
            batch_onsite2 = onsitfunc.get_onsiteEs(batch_bonds_onsite=self.batch_bond_onsites, nn_onsite_paras=nn_onsiteE2)

        for kf in batch_onsite2.keys():
            assert len(batch_onsite2[kf]) == len(batch_onsite_true[kf])
            assert len(batch_onsite[kf]) == len(self.batch_bond_onsites[kf])
            for i in range(len(batch_onsite2[kf])):
                assert torch.allclose(batch_onsite2[kf][i], batch_onsite_true[kf][i])

    def test_onsite_uniform(self):
        nn_onsiteE = {'N': torch.tensor([0.0019521093, 0.0031471925]), 'B': torch.tensor([0.0053026341])}
        onsitfunc = orbitalEs(proj_atom_anglr_m=self.proj_atom_anglr_m,atomtype=None, functype='uniform')

        batch_onsite_true = {0: [torch.tensor([-0.6749721169, -0.2628197372]), torch.tensor([-0.3395173550])]}
        with torch.no_grad():
            batch_onsite = onsitfunc.get_onsiteEs(batch_bonds_onsite=self.batch_bond_onsites, nn_onsite_paras=nn_onsiteE)

        assert isinstance(batch_onsite, dict)
        assert len(batch_onsite) == len(batch_onsite_true)
        assert len(batch_onsite) == len(self.batch_bond_onsites)

        for kf in batch_onsite.keys():
            assert len(batch_onsite[kf]) == len(batch_onsite_true[kf])
            assert len(batch_onsite[kf]) == len(self.batch_bond_onsites[kf])
            for i in range(len(batch_onsite[kf])):
                assert torch.allclose(batch_onsite[kf][i], batch_onsite_true[kf][i])

    def test_onsite_nrl(self):
        batch_onsite_envs = {0: torch.tensor([[ 0.0000000000e+00,  7.0000000000e+00,  0.0000000000e+00,
                                                5.0000000000e+00,  1.0000000000e+00, -1.0000000000e+00,
                                                0.0000000000e+00,  0.0000000000e+00,  1.4456851482e+00,
                                               -8.6602538824e-01, -5.0000000000e-01,  0.0000000000e+00],
                                              [ 0.0000000000e+00,  7.0000000000e+00,  0.0000000000e+00,
                                                5.0000000000e+00,  1.0000000000e+00,  0.0000000000e+00,
                                                1.0000000000e+00,  0.0000000000e+00,  1.4456849098e+00,
                                               -5.0252534578e-08,  1.0000000000e+00,  0.0000000000e+00],
                                              [ 0.0000000000e+00,  7.0000000000e+00,  0.0000000000e+00,
                                                5.0000000000e+00,  1.0000000000e+00,  0.0000000000e+00,
                                                0.0000000000e+00,  0.0000000000e+00,  1.4456850290e+00,
                                                8.6602538824e-01, -5.0000005960e-01,  0.0000000000e+00],
                                              [ 0.0000000000e+00,  5.0000000000e+00,  1.0000000000e+00,
                                                7.0000000000e+00,  0.0000000000e+00,  0.0000000000e+00,
                                               -1.0000000000e+00,  0.0000000000e+00,  1.4456849098e+00,
                                                5.0252534578e-08, -1.0000000000e+00,  0.0000000000e+00],
                                              [ 0.0000000000e+00,  5.0000000000e+00,  1.0000000000e+00,
                                                7.0000000000e+00,  0.0000000000e+00,  0.0000000000e+00,
                                                0.0000000000e+00,  0.0000000000e+00,  1.4456850290e+00,
                                               -8.6602538824e-01,  5.0000005960e-01,  0.0000000000e+00],
                                              [ 0.0000000000e+00,  5.0000000000e+00,  1.0000000000e+00,
                                                7.0000000000e+00,  0.0000000000e+00,  1.0000000000e+00,
                                                0.0000000000e+00,  0.0000000000e+00,  1.4456851482e+00,
                                                8.6602538824e-01,  5.0000000000e-01,  0.0000000000e+00]])}
        
        nn_onsiteE = {'N-2s-0': torch.tensor([ 0.0039564464, -0.0055190362,  0.0041887821, -0.0018826023]),
                      'N-2p-0': torch.tensor([-0.0001931502, -0.0003207834,  0.0007209170, -0.0004175970]),
                      'B-2s-0': torch.tensor([-0.0040726629,  0.0048060226,  0.0017231141,  0.0074217431])}
        
        onsite_fun = orbitalEs(proj_atom_anglr_m=self.proj_atom_anglr_m, atomtype=['N','B'], functype='NRL', unit='Hartree',
                                    onsite_func_cutoff=3.0, onsite_func_decay_w=0.3, onsite_func_lambda=1.0)  

        batch_nnsk_onsiteEs = onsite_fun.get_onsiteEs(batch_bonds_onsite=self.batch_bond_onsites, onsite_env=batch_onsite_envs, nn_onsite_paras=nn_onsiteE)

        batch_nnsk_onsiteEs_true = {0: [torch.tensor([ 0.0019290929, -0.0002228779]), torch.tensor([5.6795310229e-05])]}

        assert isinstance(batch_nnsk_onsiteEs,dict)
        assert len(batch_nnsk_onsiteEs) == len(batch_nnsk_onsiteEs_true)

        for kf in batch_nnsk_onsiteEs.keys():
            assert len(batch_nnsk_onsiteEs[kf]) == len(batch_nnsk_onsiteEs_true[kf])
            assert len(batch_nnsk_onsiteEs[kf]) == len(self.batch_bond_onsites[kf])
            for i in range(len(batch_nnsk_onsiteEs[kf])):
                assert torch.allclose(batch_nnsk_onsiteEs[kf][i], batch_nnsk_onsiteEs_true[kf][i])