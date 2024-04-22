TrainFullConfigSK={
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
            "lr": 0.01,
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
            "type": "exp",
            "gamma": 0.999
        },
        "loss_options": {
            "train": {
                "method": "eigvals",
                "diff_on": False,
                "eout_weight": 0.01,
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
            "freeze": False,
            "std": 0.01,
            "push": False or {"w_thr": 0.0,"period": 1,"rs_thr": 0.0, "ovp_thr": 0.0, "rc_thr": 0.0},
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





TestFullConfigSK={
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











