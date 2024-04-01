from dptb.nnet.tb_net import TBNet
import numpy as np
import pytest
import torch


xenv = torch.tensor([[ 0.0000,  0.0000,  7.0000,  0.3994, -1.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  7.0000,  0.3994, -1.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  7.0000,  0.3994,  1.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  7.0000,  0.3994, -0.5000,  0.8660,  0.0000],
        [ 0.0000,  0.0000,  7.0000,  0.3994,  0.5000,  0.8660,  0.0000],
        [ 0.0000,  0.0000,  7.0000,  0.3994, -0.5000, -0.8660,  0.0000],
        [ 0.0000,  0.0000,  7.0000,  0.3994,  0.5000, -0.8660,  0.0000]])

def test_tbnet_emb():
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

    env_emb = tbnet(xenv,flag='N-N',mode='emb')
    assert env_emb.shape == (7,23)
    assert (np.abs((env_emb[:,0:7] - xenv).detach().numpy())<1e-8).all()