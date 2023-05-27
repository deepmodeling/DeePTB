from dptb.dataprocess.datareader import read_data
import numpy as np
import pytest
from dptb.utils.tools import get_env_neuron_config, get_hopping_neuron_config, get_onsite_neuron_config


def test_envnet():
    env_nnl = [4,8,16]
    env_net_config = get_env_neuron_config(env_nnl)
    assert env_net_config == [{'n_in': 1, 'n_hidden': 4, 'n_out': 8}, {'n_in': 8, 'n_out': 16}]

    env_nnl = [4,8,16,32]
    env_net_config = get_env_neuron_config(env_nnl)
    assert env_net_config == [{'n_in': 1, 'n_hidden': 4, 'n_out': 8}, {'n_in': 8, 'n_hidden': 16, 'n_out': 32}]


def test_bondnet():
    bond_num_hops = {'N-N': 4, 'N-B': 5, 'B-N': 5, 'B-B': 4}
    env_axisnn = 10
    env_nnl_out = 32
    bond_type = ['N-N', 'N-B', 'B-B']

    bond_nnl = [10,20]
    bond_net_config = get_hopping_neuron_config(bond_nnl, bond_num_hops, bond_type, env_axisnn,  env_nnl_out)
    assert bond_net_config == {'N-N': [{'n_in': 321, 'n_hidden': 10, 'n_out': 20}, {'n_in': 20, 'n_out': 4}],
                        'N-B': [{'n_in': 321, 'n_hidden': 10, 'n_out': 20}, {'n_in': 20, 'n_out': 5}],
                        'B-B': [{'n_in': 321, 'n_hidden': 10, 'n_out': 20}, {'n_in': 20, 'n_out': 4}]}
    
    bond_nnl = [10,20,40]
    bond_net_config = get_hopping_neuron_config(bond_nnl, bond_num_hops, bond_type, env_axisnn,  env_nnl_out)
    assert bond_net_config == {'N-N': [{'n_in': 321, 'n_hidden': 10, 'n_out': 20}, {'n_in': 20, 'n_hidden': 40, 'n_out': 4}],
                        'N-B': [{'n_in': 321, 'n_hidden': 10, 'n_out': 20}, {'n_in': 20, 'n_hidden': 40, 'n_out': 5}],
                        'B-B': [{'n_in': 321, 'n_hidden': 10, 'n_out': 20}, {'n_in': 20, 'n_hidden': 40, 'n_out': 4}]}

    bond_nnl = [10,20,40,80]
    bond_net_config = get_hopping_neuron_config(bond_nnl, bond_num_hops, bond_type, env_axisnn,  env_nnl_out)
    assert bond_net_config ==  {'N-N': [{'n_in': 321, 'n_hidden': 10, 'n_out': 20}, {'n_in': 20, 'n_hidden': 40, 'n_out': 80}, {'n_in': 80, 'n_out': 4}],
                         'N-B': [{'n_in': 321, 'n_hidden': 10, 'n_out': 20}, {'n_in': 20, 'n_hidden': 40, 'n_out': 80}, {'n_in': 80, 'n_out': 5}],
                         'B-B': [{'n_in': 321, 'n_hidden': 10, 'n_out': 20}, {'n_in': 20, 'n_hidden': 40, 'n_out': 80}, {'n_in': 80, 'n_out': 4}]}

def test_onsite_net():
    onsite_num = {'N': 2, 'B': 2}
    proj_atom_type = ['N', 'B']
    env_axisnn = 10
    env_nnl_out = 32

    onsite_nnl = [10,20]
    onsite_net_config = get_onsite_neuron_config(onsite_nnl, onsite_num, proj_atom_type, env_axisnn, env_nnl_out)
    assert onsite_net_config == {'N': [{'n_in': 320, 'n_hidden': 10, 'n_out': 20}, {'n_in': 20, 'n_out': 2}],
                                 'B': [{'n_in': 320, 'n_hidden': 10, 'n_out': 20}, {'n_in': 20, 'n_out': 2}]}

    onsite_nnl = [10,20,40]
    onsite_net_config = get_onsite_neuron_config(onsite_nnl, onsite_num, proj_atom_type, env_axisnn, env_nnl_out)
    assert onsite_net_config == {'N': [{'n_in': 320, 'n_hidden': 10, 'n_out': 20}, {'n_in': 20, 'n_hidden': 40, 'n_out': 2}],
                                 'B': [{'n_in': 320, 'n_hidden': 10, 'n_out': 20}, {'n_in': 20, 'n_hidden': 40, 'n_out': 2}]}     

          