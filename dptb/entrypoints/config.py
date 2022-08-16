'''
This file initialize all the required config items
'''
import torch

DEFAULT_CONFIG = {
    "device": "cpu",
    "dtype": torch.float32,
    "train_options": {
        "num_epoch": 5,
        "display_epoch": 1,
        "use_reference": True
    },
    "optimizer_options": {
        "lr": 1e-4,
        "opt_type": "Adam"
    },
    "sch_options": {
        "sch_type": "Expo",
        "gamma": 0.998
    },
    "data_options": {
        "batch_size": 1,
        "sk_file_path": " ",
        "bond_cutoff": 4,
        "env_cutoff": 3.5,
        "train_data_path": " ",
        "train_data_prefix": "set",
        "test_data_path": " ",
        "test_data_prefix": "set",
        "ref_data_path": " ",
        "ref_data_prefix": "set",
        "proj_atom_neles": {},
        "proj_atom_anglr_m": {},
        "time_symm": True,
        "band_min": 0,
        "band_max": 4,
        "ref_band_min": 0,
        "ref_band_max": 4
    },
    "model_options": {
        "axis_neuron": 10,
        "onsite_net_neuron": [128, 128, 256, 256],
        "env_net_neuron": [128, 128, 256, 256],
        "bond_net_neuron": [128, 128, 256, 256],
        "onsite_net_activation": "tanh",
        "env_net_activation": "tanh",
        "bond_net_activation": "tanh",
        "onsite_net_type": "ffn",
        "env_net_type": "res",
        "bond_net_type": "ffn",
        "if_batch_normalized": False,
        "device": "cpu"
    }
}

