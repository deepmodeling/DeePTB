# from ._base import AtomicLinear

# __all__ = [

# ]
"""
The model can be constructed by the following steps:
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
4. choose the loss target, and its metric, it can be MSE, MAE, etc.

 
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
            # e3tb
        },
    },
}
"""