from .build import build_model

__all__ = [

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