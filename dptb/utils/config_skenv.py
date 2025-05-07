TrainFullConfigSKEnv= {
    "common_options": {
        "basis": {
            "A": ["2s","2p"],
            "B": ["3s","3p"]
        },
        "device": "cpu",
        "dtype": "float32",
        "overlap": False,
        "seed": 3982377700
    },
    "train_options": {
        "num_epoch": 2,
        "batch_size": 1,
        "optimizer": {
            "lr": 0.001,
            "type": "Adam"
        },
        "lr_scheduler": {
            "type": "exp",
            "gamma": 0.999
        },
        "loss_options": {
            "train": {
                "method": "eigvals",
                "diff_on": False,
                "eout_weight": 0.001,
                "diff_weight": 0.01
            }
        },
        "save_freq": 1,
        "validation_freq": 10,
        "display_freq": 100,
        "ref_batch_size": 1,
        "val_batch_size": 1,
        "max_ckpt":4
    },
    "model_options": {
        "embedding": {
            "method": "se2",
            "rs": 2.5,
            "rc": 5.0,
            "radial_net": {
                "neurons": [10,20,30],
                "activation": "tanh",
                "if_batch_normalized": False
            },
            "n_axis": None
        },
        "prediction": {
            "method": "sktb",
            "neurons": [16,16,16],
            "activation": "tanh",
            "if_batch_normalized": False
        },
        "nnsk": {
            "onsite": {
                "method": "strain",
                "rs": 2.5,
                "w": 0.3
            },
            "hopping": {
                "method": "powerlaw",
                "rs": 2.6,
                "w": 0.35
            },
            "freeze": True,
            "std": 0.01,
            "push": False
        }
    },
    "data_options": {
        "train": {
            "root": "path/to/dataset",
            "prefix": "prexfix_for_dataset",
            "get_eigenvalues": True,
            "type": "DefaultDataset",
            "get_Hamiltonian": False
        },
        "validation": {
            "root": "path/to/dataset",
            "prefix": "prexfix_for_dataset",
            "get_eigenvalues": True,
            "type": "DefaultDataset",
            "get_Hamiltonian": False
        },
        "reference":{
            "root": "path/to/dataset",
            "prefix": "prexfix_for_dataset",
            "get_eigenvalues": True,
            "type": "DefaultDataset",
            "get_Hamiltonian": False
        }
    }
}



TestFullConfigSKEnv= {
    "common_options": {
        "basis": {
            "A": ["2s","2p"],
            "B": ["3s","3p"]
        },
        "device": "cpu",
        "dtype": "float32",
        "overlap": False,
        "seed": 3982377700
    },
    "test_options": {
        "batch_size": 1,
        "display_freq": 1,
        "loss_options": {
            "train": {
                "method": "eigvals",
                "diff_on": False,
                "eout_weight": 0.01,
                "diff_weight": 0.01
            }
        }
    },
    "data_options": {
        "test": {
            "root": "path/to/dataset",
            "prefix": "prexfix_for_dataset",
            "get_eigenvalues": True,
            "type": "DefaultDataset",
            "get_Hamiltonian": False
        }
    }
}

