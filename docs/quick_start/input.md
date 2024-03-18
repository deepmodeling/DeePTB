# Brief Introduction to Inputs and Commands

The following files are the central input files for DeePTB. Before executing the program, please make sure these files are prepared and stored in the working directory. Here we give some simple descriptions XXX. For more details, users should consult the Advanced session.

## Inputs
### Data
The dataset of one structure is recommended to be prepared in the following format:
```
data/
-- set.x
-- -- eigenvalues.npy  # numpy array of fixed shape [nframes, nkpoints, nbands]
-- -- kpoints.npy      # numpy array of fixed shape [nkpoints, 3]
-- -- xdat.traj        # ase trajectory file with nframes
-- -- info.json        # defining the parameters used in building AtomicData graph data
```
One should prepare the **atomic structures** and **electronic band structures**. The **atomic structures** data is in ASE trajectory binary format, where each structure is stored using an **Atom** class defined in ASE package. The provided trajectory file must have suffix `.traj` and the length of the trajectory is nframes.
> Instead of loading structures from a single binary `.traj` file, three seperate textfiles for **atomic structures** can also be provided: `atomic_numbers.dat`, `cell.dat` and `positions.dat`. The length unit used in `cell.dat` and `positions.dat` (if cartesian coordinates) is Angstrom.

The **band structures** data contains the kpoints list and eigenvalues in the binary format of npy. The shape of kpoints data is fixed as **[nkpoints,3]** and eigenvalues is fixed as **[nframes,nkpoints,nbands]**. The `nframes` here must be the same as in **atomic structures** files.

### Info

In **DeePTB**, the **atomic structures** and **band structures** data are stored in AtomicData graph structure. `info.json` defines the key parameters used in building AtomicData graph dataset, which looks like:
```bash
{
    "nframes": 1,
    "natoms": 2,
    "pos_type": "ase",
    "AtomicData_options": {
        "r_max": 5.0,
        "er_max": 5.0,
        "oer_max": 2.5,
        "pbc": true
    },
    "bandinfo": {
        "band_min": 0,
        "band_max": 6,
        "emin": null,
        "emax": null
    }
}
```
`nframes` is the length of the trajectory, as we defined in the previous section. `natoms` is the number of atoms in each of the structures in the trajectory. `pos_type` defines the input format of the **atomic structures**, which is set to `ase` if  ASE `.traj` file is provided, and `cart` or `frac` if cartesian / fractional coordinate in `positions.dat` file provided.

In the `AtomicData_options` section, the key arguments in defining graph structure is provided. `r_max` is the maximum cutoff in building neighbour list for each atom, and `pbc` assigns the PBC condition in cell. `er_max` and `oer_max` is the environment cutoff used in `dptb` model. All inputs are in Angstrom.

We can get the bond cutoff by `DeePTB`'s bond analysis function, using:
```bash
dptb bond <structure path> [[-c] <cutoff>] [[-acc] <accuracy>]
```

`Bandinfo` defines the settings of the training objective of each structure, which enables flexible training objectives for various structures with different atom numbers and atom types.
> **note:** The `0` energy point is located at the lowest energy eigenvalues of the data files, to generalize bandstructure data computed by different DFT packages.


### Input.json
**DeePTB** provides input config templates for quick setup. User can run:
```bash
dptb config <generated input config path> [-full]
```
The template config file will be generated at the path `./input.json`.
For the full document about the input parameters, we refer to the detail [document](https://deeptb.readthedocs.io/en/latest). For now, we only need to consider a few vital parameters that can set the training:

`common_options` provides vital information for building **DeePTB** models and their training. 

```json
"common_options": {
            "basis": {
                "C": ["2s", "2p"],
                "N": ["2s", "2p", "d*"]
            },
            "device": "cpu",
            "dtype": "float32",
            "seed": 42
    }
```

`train_options` section is used to spicify the training procedure.

```json
"train_options": {
    "num_epoch": 500,
    "batch_size": 1,
    "optimizer": {
        "lr": 0.05,
        "type": "Adam"
    },
    "lr_scheduler": {
        "type": "exp",
        "gamma": 0.999
    },
    "loss_options":{
        "train": {"method": "eigvals"}
    },
    "save_freq": 10,
    "validation_freq": 10,
    "display_freq": 10
}
```

`model_options` section is the key section that provides information to build **DeePTB** models. For a Slater-Kohster TB parameteriation model, only the `nnsk` section is provided. The example of a `nnsk` model:

```json
"model_options": {
    "nnsk": {
        "onsite": {"method": "strain", "rs":2.6, "w": 0.3},
        "hopping": {"method": "powerlaw", "rs":2.6, "w": 0.3},
        "freeze": false
    }
}
```

For a environment-dependent **DeePTB** model, the `embedding`, `prediction` and `nnsk` sections are required as in this example:

```json
    "model_options": {
        "embedding":{
            "method": "se2", "rs": 2.5, "rc": 5.0,
            "radial_net": {
                "neurons": [10,20,30]
            }
        },
        "prediction":{
            "method": "sktb",
            "neurons": [16,16,16]
        },
        "nnsk": {
            "onsite": {"method": "strain", "rs":2.5 ,"w":0.3},
            "hopping": {"method": "powerlaw", "rs":5.0, "w": 0.1},
            "freeze": true
        }
    }
```

`data_options` assigns the datasets used in training.

```json
"data_options": {
    "train": {
        "root": "./data/",
        "prefix": "kpathmd100",
        "get_eigenvalues": true
    },
    "test": {
        "root": "./data/",
        "prefix": "kpathmd100",
        "get_eigenvalues": true
    }
}
```

## Commands
### Training
When data and input config file is prepared, we are ready to train the model:
```bash
dptb train <input config> [[-o] <output directory>] [[-i|-r] <nnsk checkpoint path>]
```
