from .build import build_model
from .deeptb import NNENV, MIX
from .nnsk import NNSK
from .dftbsk import DFTBSK
from .hamiltonian import E3Hamiltonian, SKHamiltonian
from .hr2hk import HR2HK
from .energy import Eigenvalues, Eigh

__all__ = [
    build_model,
    E3Hamiltonian,
    SKHamiltonian,
    HR2HK,
    Eigenvalues,
    Eigh,
    NNENV,
    NNSK,
    MIX,
    DFTBSK,
]
"""

nn module is the model class for the graph neural network model, which is the core of the deeptb package.
It provide two interfaces which is used to interact with other module:
1. The build_model method, which is used to construct a model based on the model_options, common_options and run_options.
    - the model options decides the structure of the model, such as the number of layers, the activation function, the number of neurons in each layer, etc.
    - the common options contains some common parameters, such as the dtype, device, and the basis, which also related to the model
    - the run options decide how to initialize the model. Whether it is from scratch, init from checkpoint, freeze or not, or whether to deploy it.
   The build model method will return a model class and a config dict.

2. A config dict of the model, which contains the essential information of the model to be initialized again.

The build model method should composed of the following steps:
1. process the configs from user input and the config from the checkpoint (if any).
2. construct the model based on the configs.
3. process the config dict for the output dict.

The deeptb model can be constructed by the following steps (which have been conpacted in deeptb.py):
1. choose the way to construct edge and atom embedding, either a descriptor, GNN or both. 
    - in: data with env and edge vectors, and atomic numbers
    - out: data with edge and atom embedding
    - user view: this can be defined as a tag in model_options
2. constructing the prediction layer, which named as sktb layer or e3tb layer, it is either a linear layer or a neural network
    - in: data with edge and atom embedding
    - out: data with the e3 irreducible matrix element, or the sk parameters
3. constructing hamiltonian model, either a SKTB or E3TB
    - in: data with properties/parameters predicted
    - out data with SK/E3 hamiltonian

model_options = {
    "embedding": {
        "mode":"se2/gnn/se3...",
        # mode specific
        # se2
        "env_cutoff": 3.5,
        "rs": float,
        "rc": float,
        "n_axis": int,
        "radial_embedding": {
            "neurons": [int],
            "activation": str,
            "if_batch_normalized": bool
        }
        # gnn
        # se3
    },
    "prediction": {
        "mode": "linear/nn",
        # linear
        # nn
        "neurons": [int],
        "activation": str,
        "if_batch_normalized": bool,
        "hamiltonian" = {
            "method": "sktb/e3tb",
            "rmax": 3.5,
            "precision": float,                 # use to check if rmax is large enough
            "soc": bool,
            "overlap": bool,
            # sktb
            # e3tb
        },
    },
    "nnsk":{
        "hopping_function": {
            "formula": "varTang96/powerlaw/NRL",
            ...
        },
        "onsite_function": {
            "formula": "strain/uniform/NRL",
            # strain
            "strain_cutoff": float,
            # NRL
            "cutoff": float,
            "decay_w": float,
            "lambda": float
        }
    }
}
"""

common_options = {
    "basis": {
        "B": "2s2p1d",
        "N": "2s2p1d",
    },
    "device": "cpu",
    "dtype": "float32",
    "r_max": 2.0,
    "er_max": 4.0,
    "oer_max": 6.0,
}


data_options = {
    "train": {

    }
}


dptb_model_options = {
    "embedding": {
        "method": "se2",
        "rs": 2.0, 
        "rc": 7.0,
        "n_axis": 10,
        "radial_embedding": {
            "neurons": [128,128,20],
            "activation": "tanh",
            "if_batch_normalized": False,
        },
    },
    "prediction":{
        "method": "nn",
        "neurons": [256,256,256],
        "activation": "tanh",
        "if_batch_normalized": False,
        "quantities": ["hamiltonian"],
        "hamiltonian":{
            "method": "e3tb",
            "precision": 1e-5,
            "overlap": False,
        },
    },
    "nnsk": {
        "onsite": {"method": "strain", "rs":6.0, "w":0.1},
        "hopping": {"method": "powerlaw", "rs":3.2, "w": 0.15},
        "overlap": False
    }
}