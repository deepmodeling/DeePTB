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