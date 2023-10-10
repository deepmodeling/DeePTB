import  pytest 
import torch
from dptb.nnsktb.integralFunc import SKintHops

# test for hoppings:


class TestSKintHops:
    envtype = ['N','B']
    bondtype = ['N','B']
    proj_atom_anglr_m={'B': ['2s'], 'N': ['2s', '2p']}
    batch_bonds = {0: torch.tensor([[ 0.0000000000e+00,  7.0000000000e+00,  0.0000000000e+00,
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
           8.6602538824e-01, -5.0000005960e-01,  0.0000000000e+00]])}
    
    def test_skhops_varTang96(self):
        coeffdict= {'N-N-2s-2s-0': torch.tensor([ 4.3461765745e-04, -3.9701518835e-04, -6.0277385637e-04, -6.9087851443e-05]),
                    'N-N-2s-2p-0': torch.tensor([ 2.1683995146e-04,  1.0277298134e-04, -6.2341854209e-04, 1.4911865946e-05]),
                    'N-N-2p-2p-0': torch.tensor([ 0.0008250176, -0.0005188021, -0.0002828926,  0.0006028564]),
                    'N-N-2p-2p-1': torch.tensor([ 1.0799153242e-03,  4.2950130592e-05,  1.8651155187e-05, -6.6541536944e-04]),
                    'N-B-2s-2s-0': torch.tensor([ 0.0003718118, -0.0001149630, -0.0010231513, -0.0002210326]),
                    'N-B-2p-2s-0': torch.tensor([-0.0005409718,  0.0002763696, -0.0003420392, -0.0004326820]),
                    'B-B-2s-2s-0': torch.tensor([-4.6358386498e-06, -1.4617976558e-04,  6.2484655064e-04,-8.6897460278e-04])}

        skhops  = SKintHops(proj_atom_anglr_m=self.proj_atom_anglr_m, mode='hopping', functype='varTang96')

        batch_hoppings_true = {0: [torch.tensor([ 0.0003693393, -0.0005377086]),
                                   torch.tensor([ 0.0003693393, -0.0005377086]),
                                   torch.tensor([ 0.0003693393, -0.0005377086])]}
        
        batch_hoppings = skhops.get_skhops(batch_bonds=self.batch_bonds, coeff_paras=coeffdict, rcut=3.0, w=0.3)

        assert isinstance(batch_hoppings, dict)
        assert len(batch_hoppings) == len(batch_hoppings_true)
        assert len(batch_hoppings) == len(self.batch_bonds)

        for kf in batch_hoppings.keys():
            assert len(batch_hoppings[kf]) == len(batch_hoppings_true[kf])
            assert len(batch_hoppings[kf]) == len(self.batch_bonds[kf])
            for i in range(len(batch_hoppings[kf])):
                assert torch.allclose(batch_hoppings[kf][i], batch_hoppings_true[kf][i])
    
    def test_skhops_powerlaw(self):
        coeffdict = {'N-N-2s-2s-0': torch.tensor([0.0002670568, 0.0001332831]),
                     'N-N-2s-2p-0': torch.tensor([-0.0003154497, -0.0003884580]),
                     'N-N-2p-2p-0': torch.tensor([-0.0001336335,  0.0008993127]),
                     'N-N-2p-2p-1': torch.tensor([-0.0002779329,  0.0003829031]),
                     'N-B-2s-2s-0': torch.tensor([0.0006050252, 0.0004113411]),
                     'N-B-2p-2s-0': torch.tensor([0.0002687302, 0.0007265538]),
                     'B-B-2s-2s-0': torch.tensor([-1.0157947145e-05,  3.6075818934e-04])}
        
        skhops  = SKintHops(proj_atom_anglr_m=self.proj_atom_anglr_m, mode='hopping', functype='powerlaw')

        batch_hoppings = skhops.get_skhops(batch_bonds=self.batch_bonds, coeff_paras=coeffdict, rcut=3.0, w=0.3)

        batch_hoppings_true = {0: [torch.tensor([0.0007047649, 0.0003130466]),
                                   torch.tensor([0.0007047650, 0.0003130467]),
                                   torch.tensor([0.0007047650, 0.0003130467])]}
        
        assert isinstance(batch_hoppings, dict)
        assert len(batch_hoppings) == len(batch_hoppings_true)
        assert len(batch_hoppings) == len(self.batch_bonds)

        for kf in batch_hoppings.keys():
            assert len(batch_hoppings[kf]) == len(batch_hoppings_true[kf])
            assert len(batch_hoppings[kf]) == len(self.batch_bonds[kf])
            for i in range(len(batch_hoppings[kf])):
                assert torch.allclose(batch_hoppings[kf][i], batch_hoppings_true[kf][i])
        
    def test_skhops_NRL(self):
        coeffdict = {'N-N-2s-2s-0': torch.tensor([-0.0004987070, -0.0002041683,  0.0001014816,  0.0006219005]),
                     'N-N-2s-2p-0': torch.tensor([-5.9940444771e-04,  1.9214327040e-04,  6.0049378590e-06,5.0979648950e-04]),
                     'N-N-2p-2p-0': torch.tensor([ 0.0001927754,  0.0009208557, -0.0001234336, -0.0003449220]),
                     'N-N-2p-2p-1': torch.tensor([-2.1193656721e-04,  4.3876632844e-05,  2.7689227136e-04,8.9270688477e-05]),
                     'N-B-2s-2s-0': torch.tensor([-0.0004778731,  0.0005070638,  0.0005157407,  0.0002885270]),
                     'N-B-2p-2s-0': torch.tensor([-9.8684613477e-05,  2.5365813053e-04, -7.5873947935e-04,-3.9372156607e-04]),
                     'B-B-2s-2s-0': torch.tensor([ 3.7772103678e-04,  1.5700524091e-04, -6.5438426100e-04,-9.9891236459e-05])}
        
        skhops  = SKintHops(proj_atom_anglr_m=self.proj_atom_anglr_m, mode='hopping', functype='NRL')
        batch_hoppings = skhops.get_skhops(batch_bonds=self.batch_bonds, coeff_paras=coeffdict, rcut=3.0, w=0.3)

        with  pytest.raises(AssertionError):
            skhops.get_skoverlaps(batch_bonds=self.batch_bonds, coeff_paras=coeffdict, rcut=3.0, w=0.3)


        batch_hoppings_true = {0: [torch.tensor([ 0.0007267155, -0.0007183540]),
                                   torch.tensor([ 0.0007267154, -0.0007183540]),
                                   torch.tensor([ 0.0007267154, -0.0007183540])]}
        
        assert isinstance(batch_hoppings, dict)
        assert len(batch_hoppings) == len(batch_hoppings_true)
        assert len(batch_hoppings) == len(self.batch_bonds)

        for kf in batch_hoppings.keys():
            assert len(batch_hoppings[kf]) == len(batch_hoppings_true[kf])
            assert len(batch_hoppings[kf]) == len(self.batch_bonds[kf])
            for i in range(len(batch_hoppings[kf])):
                assert torch.allclose(batch_hoppings[kf][i], batch_hoppings_true[kf][i])
        
        skhops_overlap  = SKintHops(proj_atom_anglr_m=self.proj_atom_anglr_m, mode='hopping', functype='NRL',overlap=True)
        batch_hoppings_2 = skhops_overlap.get_skhops(batch_bonds=self.batch_bonds, coeff_paras=coeffdict, rcut=3.0, w=0.3)
        
        assert isinstance(batch_hoppings_2, dict)
        assert len(batch_hoppings_2) == len(batch_hoppings_true)
        assert len(batch_hoppings_2) == len(self.batch_bonds)

        for kf in batch_hoppings_2.keys():
            assert len(batch_hoppings_2[kf]) == len(batch_hoppings_true[kf])
            assert len(batch_hoppings_2[kf]) == len(self.batch_bonds[kf])
            for i in range(len(batch_hoppings_2[kf])):
                assert torch.allclose(batch_hoppings_2[kf][i], batch_hoppings_true[kf][i])
        
        ovelap_coeff = {'N-N-2s-2s-0': torch.tensor([ 6.8039429607e-04, -2.9353532591e-04,  9.1240115580e-05,5.6466460228e-04]),
                        'N-N-2s-2p-0': torch.tensor([ 0.0001735075, -0.0001214135,  0.0007363217, -0.0003242571]),
                        'N-N-2p-2p-0': torch.tensor([ 6.4402131829e-04, -9.6975814085e-04,  5.1726761740e-05,-4.8777154007e-05]),
                        'N-N-2p-2p-1': torch.tensor([-8.8375571067e-05, -6.9440639345e-04,  1.8161005573e-04,2.6911683381e-04]),
                        'N-B-2s-2s-0': torch.tensor([-0.0002793419,  0.0005194946, -0.0003238156, -0.0003712704]),
                        'N-B-2p-2s-0': torch.tensor([ 0.0001708491,  0.0004076761,  0.0005067488, -0.0004027870]),
                        'B-B-2s-2s-0': torch.tensor([-3.0960296863e-04, -2.4765444687e-04,  1.2461096333e-07,-2.1604154608e-04])}
        
        overlap_true = {0: [torch.tensor([-0.0001616334,  0.0014338114]),
                            torch.tensor([-0.0001616333,  0.0014338112]),
                            torch.tensor([-0.0001616333,  0.0014338113])]}
        
        overlap = skhops_overlap.get_skoverlaps(batch_bonds=self.batch_bonds, coeff_paras=ovelap_coeff, rcut=3.0, w=0.3)
        assert isinstance(overlap, dict)
        assert len(overlap) == len(overlap_true)
        assert len(overlap) == len(self.batch_bonds)

        for kf in overlap.keys():
            assert len(overlap[kf]) == len(overlap_true[kf])
            assert len(overlap[kf]) == len(self.batch_bonds[kf])
            for i in range(len(overlap[kf])):
                assert torch.allclose(overlap[kf][i], overlap_true[kf][i])