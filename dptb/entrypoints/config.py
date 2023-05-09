from typing import Dict, List, Optional, Any
import json
from pathlib import Path
import os
'''
This file initialize all the required config items
'''

DEFAULT_CONFIG = {
    "init_model": {
        "path": None,
        "interpolate": False
    },
    "common_options": {
        "onsitemode": "strain",
        "onsite_cutoff": 2.6,
        "bond_cutoff": 3.5,
        "env_cutoff": 3.5,
        "atomtype": ["A", "B"],
        "proj_atom_neles": {"A": 5, "B": 3},
        "proj_atom_anglr_m": {"A": ["2s", "2p"], "B": ["2s", "2p"]}
    },
    "train_options": {
        "seed":120478,
        "num_epoch": 4000,
        "optimizer": {"lr":5e-3}
    },
    "data_options": {
        "use_reference": True,
        "train": {
            "batch_size": 1,
            "path": "./data",
            "prefix": "set"
        },
        "validation": {
            "batch_size": 1,
            "path": "./data",
            "prefix": "set"
        },
        "reference": {
            "batch_size": 1,
            "path": "./data",
            "prefix": "set"
        }
    },
    "model_options": {
        "sknetwork": {
            "sk_hop_nhidden": 20,
            "sk_onsite_nhidden": 20
        },
        "skfunction": {
            "sk_cutoff": 3.5,
            "sk_decay_w": 0.3
        }
    }
}


FULL_CONFIG={
    "init_model": {
        "path": None,
        "interpolate": False
    },
    "common_options": {
        "onsitemode": "strain",
        "onsite_cutoff": 2.6,
        "bond_cutoff": 3.5,
        "env_cutoff": 3.5,
        "atomtype": [
            "A",
            "B"
        ],
        "proj_atom_neles": {
            "A": 5,
            "B": 3
        },
        "proj_atom_anglr_m": {
            "A": [
                "2s",
                "2p"
            ],
            "B": [
                "2s",
                "2p"
            ]
        },
        "device": "cpu",
        "dtype": "float32",
        "sk_file_path": "./",
        "time_symm": True,
        "soc": False,
        "unit": "Hartree"
    },
    "train_options": {
        "seed": 120478,
        "num_epoch": 4000,
        "optimizer": {
            "lr": 0.005,
            "type": "Adam",
            "betas": [
                0.9,
                0.999
            ],
            "eps": 1e-08,
            "weight_decay": 0,
            "amsgrad": False
        },
        "lr_scheduler": {
            "type": "Exp",
            "gamma": 0.999
        },
        "save_freq": 10,
        "validation_freq": 10,
        "display_freq": 1
    },
    "data_options": {
        "use_reference": True,
        "train": {
            "batch_size": 1,
            "path": "./data",
            "prefix": "set"
        },
        "validation": {
            "batch_size": 1,
            "path": "./data",
            "prefix": "set"
        },
        "reference": {
            "batch_size": 1,
            "path": "./data",
            "prefix": "set"
        }
    },
    "model_options": {
        "sknetwork": {
            "sk_hop_nhidden": 20,
            "sk_onsite_nhidden": 20,
            "sk_soc_nhidden": None
        },
        "skfunction": {
            "sk_cutoff": 3.5,
            "sk_decay_w": 0.3,
            "skformula": "varTang96"
        },
        "dptb": {
            "soc_env": False,
            "axis_neuron": 10,
            "onsite_net_neuron": [
                128,
                128,
                256,
                256
            ],
            "soc_net_neuron": [
                128,
                128,
                256,
                256
            ],
            "env_net_neuron": [
                128,
                128,
                256,
                256
            ],
            "bond_net_neuron": [
                128,
                128,
                256,
                256
            ],
            "onsite_net_activation": "tanh",
            "env_net_activation": "tanh",
            "bond_net_activation": "tanh",
            "soc_net_activation": "tanh",
            "onsite_net_type": "res",
            "env_net_type": "res",
            "bond_net_type": "res",
            "soc_net_type": "res",
            "if_batch_normalized": False
        }
    },
    "loss_options": {
        "losstype": "eigs_l2dsf",
        "sortstrength": [
            0.01,
            0.01
        ],
        "nkratio": None
    }
}

def config(
        PATH: str,
        full_config: bool,
        log_level: int,
        log_path: Optional[str],
        **kwargs
):
    if not PATH.endswith(".json"):
        PATH = os.path.join(PATH, "input_templete.json")
    with open(PATH, "w") as fp:
        if full_config:
            json.dump(FULL_CONFIG, fp, indent=4)
        else:
            json.dump(DEFAULT_CONFIG, fp, indent=4)
