from dptb.nnet.tb_net import TBNet
import numpy as np
import pytest
import torch

xonsite=torch.tensor([[ 0.0000e+00,  0.0000e+00,  7.0000e+00,  3.4957e-03, -5.7175e-03,
         -5.6661e-03, -2.8751e-03, -6.5396e-03,  6.3517e-03,  4.3955e-03,
          2.1346e-03,  6.4148e-03, -1.6245e-03, -5.7175e-03,  9.3863e-03,
          9.3842e-03,  4.7469e-03,  1.0706e-02, -1.0557e-02, -7.1790e-03,
         -3.5530e-03, -1.0597e-02,  2.6465e-03, -5.6661e-03,  9.3842e-03,
          9.5784e-03,  4.8140e-03,  1.0646e-02, -1.0862e-02, -7.0888e-03,
         -3.6753e-03, -1.0747e-02,  2.5983e-03, -2.8751e-03,  4.7469e-03,
          4.8140e-03,  2.4280e-03,  5.4094e-03, -5.4431e-03, -3.5995e-03,
         -1.8461e-03, -5.4079e-03,  1.3229e-03, -6.5396e-03,  1.0706e-02,
          1.0646e-02,  5.4094e-03,  1.2288e-02, -1.1941e-02, -8.2125e-03,
         -4.0432e-03, -1.2025e-02,  3.0367e-03,  6.3517e-03, -1.0557e-02,
         -1.0862e-02, -5.4431e-03, -1.1941e-02,  1.2356e-02,  7.9361e-03,
          4.1843e-03,  1.2160e-02, -2.9014e-03,  4.3955e-03, -7.1790e-03,
         -7.0888e-03, -3.5995e-03, -8.2125e-03,  7.9361e-03,  5.5309e-03,
          2.6614e-03,  8.0362e-03, -2.0456e-03,  2.1346e-03, -3.5530e-03,
         -3.6753e-03, -1.8461e-03, -4.0432e-03,  4.1843e-03,  2.6614e-03,
          1.4335e-03,  4.0997e-03, -9.7381e-04,  6.4148e-03, -1.0597e-02,
         -1.0747e-02, -5.4079e-03, -1.2025e-02,  1.2160e-02,  8.0362e-03,
          4.0997e-03,  1.2086e-02, -2.9496e-03, -1.6245e-03,  2.6465e-03,
          2.5983e-03,  1.3229e-03,  3.0367e-03, -2.9014e-03, -2.0456e-03,
         -9.7381e-04, -2.9496e-03,  7.5806e-04, -2.0222e-03,  3.3638e-03,
          3.4680e-03,  1.7375e-03,  3.8058e-03, -3.9478e-03, -2.5253e-03,
         -1.3390e-03, -3.8792e-03,  9.2294e-04, -7.9292e-03,  1.2982e-02,
          1.2893e-02,  6.5336e-03,  1.4824e-02, -1.4468e-02, -9.9682e-03,
         -4.8568e-03, -1.4591e-02,  3.6807e-03, -1.1629e-02,  1.9199e-02,
          1.9453e-02,  9.7994e-03,  2.1824e-02, -2.1998e-02, -1.4567e-02,
         -7.4302e-03, -2.1875e-02,  5.3505e-03, -5.1189e-04,  7.5392e-04,
          5.4678e-04,  3.0895e-04,  9.1796e-04, -5.2449e-04, -6.7014e-04,
         -1.5477e-04, -6.9079e-04,  2.6264e-04, -4.3428e-03,  7.1391e-03,
          7.1647e-03,  3.6239e-03,  8.1513e-03, -8.0693e-03, -5.4476e-03,
         -2.7266e-03, -8.0763e-03,  2.0075e-03, -9.2185e-05,  1.4673e-04,
          1.3404e-04,  6.8223e-05,  1.6451e-04, -1.4634e-04, -1.1807e-04,
         -4.4809e-05, -1.5756e-04,  4.3979e-05]])

def test_tbnet_onsite():
    proj_atom_type= ['N', 'B']
    atom_type= ['N', 'B']
    env_net_config= [{'n_in': 1, 'n_hidden': 4, 'n_out': 8}, {'n_in': 8, 'n_out': 16}]
    onsite_net_config= {'N': [{'n_in': 160, 'n_hidden': 10, 'n_out': 20}, {'n_in': 20, 'n_hidden': 40, 'n_out': 2}],
                        'B': [{'n_in': 160, 'n_hidden': 10, 'n_out': 20}, {'n_in': 20, 'n_hidden': 40, 'n_out': 2}]}
    bond_net_config={'N-N': [{'n_in': 161, 'n_hidden': 10, 'n_out': 20}, {'n_in': 20, 'n_hidden': 40, 'n_out': 4}],
                     'N-B': [{'n_in': 161, 'n_hidden': 10, 'n_out': 20}, {'n_in': 20, 'n_hidden': 40, 'n_out': 5}],
                     'B-B': [{'n_in': 161, 'n_hidden': 10, 'n_out': 20}, {'n_in': 20, 'n_hidden': 40, 'n_out': 4}]}
    onsite_net_activation='tanh'
    env_net_activation='tanh'
    bond_net_activation='tanh'
    onsite_net_type ='ffn'
    env_net_type= 'res'
    bond_net_type='ffn'
    if_batch_normalized = False
    device= 'cpu'
    dtype= torch.float32

    tbnet = TBNet(proj_atom_type,
                 atom_type,
                 env_net_config,
                 onsite_net_config,
                 bond_net_config,
                 onsite_net_activation,
                 env_net_activation,
                 bond_net_activation,
                 onsite_net_type,
                 env_net_type,
                 bond_net_type,
                 if_batch_normalized,
                 device,
                 dtype)

    onsite = tbnet(xonsite,flag='N',mode='onsite')
    assert onsite.shape == (1,5)
    assert (onsite[0,:3].detach().numpy().astype(int) == np.array([0,0,7])).all()
    
