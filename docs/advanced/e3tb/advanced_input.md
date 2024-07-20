# More on Input Parameters
In `common_options`, the user should define the some global param like:
```JSON
"common_options": {
            "basis": {
                "C": "2s2p1d",
                "N": "2s2p1d"
            },
            "device": "cuda",
            "dtype": "float32",
            "overlap": true,
            "seed": 42
    }
```
- `basis` sould align with the basis used to perform LCAO DFT calculations. The `"2s2p1d"` here indicates 2x`s` orbital, 2x`p`orbital and one `d` orbital. The 
- `seed` controls the global random seed of all related package. `dtype` can be chosen between `float32` and `float64`, but the former are accurate enough in most cases. If you have multiple card, the 
- `device` can be setted as `cuda:0`, `cuda:1` and so on, where the number is the device id. 
- `overlap` controls the fitting of overlap matrix. The user should provide overlap in the dataset when configuring the data_options if `overlap` is setted as True.

In `train_options`, a common parameter looks like this:
```JSON
"train_options": {
    "num_epoch": 500,
    "batch_size": 1,
    "optimizer": {
        "lr": 0.05,
        "type": "Adam"
    },
    "lr_scheduler": {
        "type": "rop",
        "factor": 0.8,
        "patience": 6,
        "min_lr": 1e-6
    },
    "loss_options":{
        "train": {"method": "hamil_abs", "onsite_shift": false},
        "validation": {"method": "hamil_abs", "onsite_shift": false}
    },
    "save_freq": 10,
    "validation_freq": 10,
    "display_freq": 10
}
```
For `lr_scheduler`, please ensure the `patience` x `num_samples` / `batch_size` ranged between 2000 to 6000.

When the dataset contraining multiple elements, and you are fitting the Hamiltonian, it is suggested to open a tag in loss_options for better performance. Since most DFT software would allow for a uniform shift when computing the electrostatic potentials, therefore, bringing extra degree of freedom. The `onsite_shift` tag allow such freedom and make the model generalizable to all sort of elements combinations:
```JSON
"loss_options":{
        "train": {"method": "hamil_abs", "onsite_shift": true},
        "validation": {"method": "hamil_abs", "onsite_shift" : true}
    }
```

In `model_options`, we support two type of e3 group equivariant embedding methods: Strictly Localized Equivariant Message-passing or `slem`, and Localized Equivariant Message-passing or `lem`. Former one ensure strict localization by truncate the propagation of distant eighbours' information, therefore are suitable for bulk systems where the electron localization is enhanced by scattering effect. `lem` method, on the other hand, contrained such localization design inherently by incooperating a learnable decaying functions describing the dependency across distance.

The model options for slem and lem are the same, here is an short example:
```JSON
"model_options": {
    "embedding": {
        "method": "slem", # or lem
        "r_max": {"Mo":7.0, "S":7.0, "W": 8.0},
        "irreps_hidden": "64x0e+32x1o+32x2e+32x3o+32x4e+16x5o+8x6e+4x7o+4x8e",
        "n_layers": 4,
        "env_embed_multiplicity": 10,
        "avg_num_neighbors": 51,
        "latent_dim": 64,
    },
    "prediction":{
        "method": "e3tb", # need to be set as e3tb here
        "neurons": [32, 32]
    }
}
```
Here, `method` indicates the e3 descripor employed. 

`r_max` can be a float or int number, or a dict with atom species specific float/int number, which indicates their cutoff envolope function, used to decaying the distant atom's effect smoothly. We highly suggest the user go to the DFT calculation files, and check the orbital's radial cutoff information to figure out how large this value should be.

`irreps_hidden`: Very important! This parameter decide mostly the representation capacity of the model, along with the model size and consumption of GPU memory. This parameter indicates the irreps of hidden equiraviant space, the definition here follows the  for example, `64x0e` states `64` ireducible representation with `l=0` and `even` parity. For each basis setting, we provide a tool to generate least essential `irreps_hidden`, we also highly suggest the user to add at least 3 times of the number of essential irreps to enhance to representation capacity.

```IPYTHON
In [5]: from dptb.data import OrbitalMapper

In [6]: idp = OrbitalMapper(basis={"Si": "2s2p1d"})

In [7]: idp.get_irreps_ess()
Out[7]: 7x0e+6x1o+6x2e+2x3o+1x4e
```

`n_layers`: indicates the number of layers of the networks.

`env_embed_multiplicity`: dicide the irreps number when initilizating the edge and node features.

`avg_num_neighbors`: the averaged number of neighbors in the system given the cutoff radius setted as `r_max`. It is recommended to do a statistics of the system you are modeling, but just pick up a number ranged from 50 to 100 is also okay.

`latent_dim`: The scalar channel's dimension of the system. 32/64/128 is good enough.

For params in prediction, there is not much to be changed. The setting is pretty good.