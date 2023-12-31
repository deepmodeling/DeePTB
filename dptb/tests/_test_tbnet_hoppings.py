from dptb.nnet.tb_net import TBNet
import numpy as np
import pytest
import torch

xbond = torch.tensor([[ 0.0000e+00,  7.0000e+00,  0.0000e+00,  5.0000e+00,  1.0000e+00,
         -2.0000e+00,  0.0000e+00,  0.0000e+00,  3.8249e+00, -9.8198e-01,
         -1.8898e-01,  0.0000e+00,  1.4812e-02, -1.8678e-02, -2.1636e-03,
         -6.2223e-03,  1.1853e-02,  2.9717e-02,  4.5607e-03, -1.8342e-02,
         -1.1659e-02,  2.0583e-02, -1.8678e-02,  8.2229e-02,  7.5340e-03,
         -1.3779e-02,  4.2117e-04, -2.6616e-02,  7.3456e-03,  2.7705e-02,
          7.4964e-02,  2.2301e-02, -2.1636e-03,  7.5340e-03,  1.2121e-03,
         -7.8369e-04, -4.4791e-04, -3.5405e-03,  5.1915e-04,  3.2549e-03,
          6.7155e-03,  1.2122e-03, -6.2223e-03, -1.3779e-02, -7.8369e-04,
          1.0618e-02, -1.0687e-02, -1.6489e-02, -6.7489e-03,  6.0103e-03,
         -1.7293e-02, -2.6449e-02,  1.1853e-02,  4.2117e-04, -4.4791e-04,
         -1.0687e-02,  1.4032e-02,  2.6888e-02,  7.5849e-03, -1.3579e-02,
          6.6434e-03,  2.9463e-02,  2.9717e-02, -2.6616e-02, -3.5405e-03,
         -1.6489e-02,  2.6888e-02,  6.1945e-02,  1.1908e-02, -3.6209e-02,
         -1.2083e-02,  5.0343e-02,  4.5607e-03,  7.3456e-03,  5.1915e-04,
         -6.7489e-03,  7.5849e-03,  1.1908e-02,  4.8910e-03, -4.7866e-03,
          1.0103e-02,  1.7486e-02, -1.8342e-02,  2.7705e-02,  3.2549e-03,
          6.0103e-03, -1.3579e-02, -3.6209e-02, -4.7866e-03,  2.3362e-02,
          1.9041e-02, -2.1642e-02, -1.1659e-02,  7.4964e-02,  6.7155e-03,
         -1.7293e-02,  6.6434e-03, -1.2083e-02,  1.0103e-02,  1.9041e-02,
          7.1185e-02,  3.3499e-02,  2.0583e-02,  2.2301e-02,  1.2122e-03,
         -2.6449e-02,  2.9463e-02,  5.0343e-02,  1.7486e-02, -2.1642e-02,
          3.3499e-02,  6.8721e-02, -2.0976e-03, -3.8885e-02, -2.8360e-03,
          1.6210e-02, -1.2451e-02, -1.1961e-02, -9.8108e-03, -4.9485e-04,
         -4.0957e-02, -3.6885e-02, -1.6960e-02,  5.8384e-02,  5.7284e-03,
         -6.4297e-03, -4.2239e-03, -2.7368e-02,  2.7677e-03,  2.4006e-02,
          5.1275e-02,  6.7264e-03, -1.3029e-02,  1.0150e-02,  1.3870e-03,
          7.7964e-03, -1.2296e-02, -2.7448e-02, -5.6485e-03,  1.5730e-02,
          3.7070e-03, -2.3403e-02,  8.5149e-03, -2.5855e-02, -2.3801e-03,
          2.0293e-03,  2.7827e-03,  1.4283e-02, -7.7081e-04, -1.1722e-02,
         -2.2214e-02, -6.0016e-04, -6.5498e-03,  4.5110e-02,  3.9870e-03,
         -1.0776e-02,  4.0844e-03, -6.4287e-03,  5.9370e-03,  1.0981e-02,
          4.2922e-02,  2.0972e-02,  1.9499e-02,  2.1792e-03, -4.5908e-04,
         -1.8101e-02,  2.3282e-02,  4.4378e-02,  1.2653e-02, -2.2098e-02,
          1.2399e-02,  4.9682e-02]])

def test_tbnet():
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

    
    hopping = tbnet(xbond, flag='N-B', mode='hopping')
    assert hopping.shape == (1,17)
    assert (hopping[0,1:8].detach().numpy().astype(int) == np.array([ 7,  0,  5,  1, -2,  0,  0])).all()
    assert (np.abs(hopping[0,8:12].detach().numpy().astype(float) - 
            np.array([ 3.82489991, -0.98198003, -0.18898   ,  0.        ])) < 1e-6).all()


