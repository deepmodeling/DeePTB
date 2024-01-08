# Brief Introduction to Inputs and Commands

The following files are the central input files for DeePTB. Before executing the program, please make sure these files are prepared and stored in the working directory. Here we give some simple descriptions XXX. For more details, users should consult the Advanced session.

## Inputs
### Data
The dataset of one structure is recommended to be prepared in the following format:
```
data/
-- set.x
-- -- eigs.npy         # numpy array of shape [num_frame, num_kpoint, num_band]
-- -- kpoints.npy      # numpy array of shape [num_kpoint, 3]
-- -- xdat.traj        # ase trajectory file with num_frame
-- -- bandinfo.json    # defining the training objective of this bandstructure
```
One should prepare the **atomic structures** and **electronic band structures**. The **atomic structures** data is in ASE trajectory binary format, where each structure is stored using an **Atom** class defined in ASE package. The **band structures** data contains the kpoints list and eigenvalues in the binary format of npy. The shape of kpoints data is **[num_kpoint,3]** and eigenvalues is **[num_frame,nk,nbands]**. nsnaps is the number of snapshots, nk is the number of kpoints and nbands is the number of bands.

### Bandinfo

`bandinfo.json` defines the settings of the training objective of each structure, basicly you can have specific settings for different structures, which enables flexible training objectives for various structures with different atom numbers and atom types.

The **bandinfo.json** file looks like this:
```bash
{
    "band_min": 0,
    "band_max": 4,
    "emin": null, # minimum of fitting energy window
    "emax": null, # maximum of fitting energy window
    "weight": [1] # optional, indicating the weight for each band separately
}
```
**note:** The `0` energy point is located at the lowest energy eigenvalues of the data files, to generalize bandstructure data computed by different DFT packages.


### Input.json
**DeePTB** provides input config templates for quick setup. User can run:
```bash
dptb config <generated input config path> [-full]
```
The template config file will be generated at the path `./input.json`.
For the full document about the input parameters, we refer to the detail [document](https://deeptb.readthedocs.io/en/latest). For now, we only need to consider a few vital parameters that can set the training:

```json
"common_options": {
    "onsitemode": "none",
    "bond_cutoff": 3.2,
    "atomtype": ["A","B"],
    "proj_atom_anglr_m": {
        "A": ["2s","2p"],
        "B": ["2s","2p"]
    }
}
```
We can get the bond cutoff by `DeePTB`'s bond analysis function, using:
```bash
dptb bond <structure path> [[-c] <cutoff>] [[-acc] <accuracy>]
```

```json
"model_options": {
    "skfunction": {
        "sk_cutoff": 3.5,
        "sk_decay_w": 0.3,
    }
}
```

```json
"data_options": {
    "use_reference": true,
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
}
```

## Commands
### Training
When data and input config file is prepared, we are ready to train the model.
To train a neural network parameterized Slater-Koster Tight-Binding model (**nnsk**) with Gradient-Based Optimization method, we can run:
```bash
dptb train -sk <input config> [[-o] <output directory>] [[-i|-r] <nnsk checkpoint path>]
```
<!--
For training a environmentally dependent Tight-Binding model (**dptb**), we can run:
```bash
dptb train <input config> [[-o] <output directory>] [[-i|-r] <dptb checkpoint path>]
```
-->
For training an environmental dependent Tight-Binding model (**dptb**), the suggested procedure is first to train a **nnsk** model, and use environment dependent neural network as a correction with the command **"-crt"** as proposed in our paper: xxx:
```bash
dptb train <input config> -crt <nnsk checkpoint path> [[-i|-r] <dptb checkpoint path>] [[-o] <output directory>]
```

### Testing
After the model is converged, the testing function can be used to do the model test or compute the eigenvalues for other analyses. 

Test config is just attained by a little modification of the train config. 
Delete the `train_options` since it is not useful when testing the model. And we delete all lines contained in `data_options`, and add the `test` dataset config:
```json
"test": {
    "batch_size": 1,  
    "path": "./data", # dataset path
    "prefix": "set"   # prefix of the data folder
}
```
if test **nnsk** model, we can run:
```bash 
dptb test -sk <test config> -i <nnsk checkpoint path> [[-o] <output directory>]
```
if test **dptb** model, we can run:
```bash
dptb test <test config> -crt <nnsk checkpoint path> -i <dptb checkpoint path> [[-o] <output directory>]
```

### Processing
**DeePTB** integrates multiple post-processing functionalities in `dptb run` command, including:
- band structure plotting
- density of states plotting
- fermi surface plotting
- slater-koster parameter transcription

Please see the template config file in `examples/hBN/run/`, and the running command is:
```bash
dptb run [-sk] <run config> [[-o] <output directory>] -i <nnsk/dptb checkpoint path> [[-crt] <nnsk checkpoint path>]
```

For detailed documents, please see our [Document page](https://deeptb.readthedocs.io/en/latest).
