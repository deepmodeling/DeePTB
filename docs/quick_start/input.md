# Brief Intro. to Inputs and Commands

The following files are the central input files for DeePTB. Before executing the program, please make sure these files are prepared and stored in the working directory. Here we give some simple descriptions, for more details, users should consult the Advanced session.

## Inputs
### Data
The dataset files contrains both the **atomic structure** and the **training label** information. 

The atomic structure should be prepared as a ASE trajectory binary file, where each structure is stored using an **Atom** class defined in ASE package. The provided trajectory file must have suffix `.traj` and the length of the trajectory is `nframes`. For labels, we currently support `eigenvalues`, `Hamiltonian`, `density matrix` and `overlap matrix`. 


For training a **DeePTB-SK** model, we need to prepare the `eigenvalues` label, which contrains the `eigenvalues.npy` and `kpoints.npy`. A typical dataset of **DeePTB-SK** task looks like:

```
data/
-- set.x
-- -- eigenvalues.npy  # numpy array of fixed shape [nframes, nkpoints, nbands]
-- -- kpoints.npy      # numpy array of fixed shape [nkpoints, 3]
-- -- xdat.traj        # ase trajectory file with nframes
-- -- info.json        # defining the parameters used in building AtomicData graph data
```

> We also support another format to provide structure information, instead of loading structures from a single binary `.traj` file. In this way, three seperate textfiles for **atomic structures** need to be provided: `atomic_numbers.dat`, `pbc.dat`, `cell.dat` and `positions.dat`. The length unit used in `cell.dat` and `positions.dat` (if cartesian coordinates) is Angstrom.

The **band structures** data includes the kpoints list and eigenvalues in the binary format of `.npy`. The shape of kpoints data is fixed as **[nkpoints,3]** and eigenvalues is fixed as **[nframes,nkpoints,nbands]**. The `nframes` here must be the same as in **atomic structures** files.

> **Important:** The eigenvalues.npy should not contain bands that contributed by the core electrons, which is not setted as the TB orbitals in model setting.

For typical **DeePTB-E3** task, we need to prepare the Hamiltonian/density matrix along with overlap matrix as labels. They are arranged as hdf5 binary format, and named as `hamiltonians.h5`/`density_matrices.h5` and `overlaps.h5` respectively. A typical dataset of **DeePTB-E3** looks like:

```
data/
-- set.x
-- -- positions.dat     # a text file with nframe x natom row and 3 col
-- -- pbc.dat           # a text file of three bool variables
-- -- cell.dat          # a text file with nframe x 3 row and 3 col, or 3 rol and 3 col.
-- -- atomic_numbers.dat    # a text file with nframe x natom row and 1 col
-- -- hamiltonian.h5    # a hdf5 dataset file with group named "0", "1", ..., "nframe". Each group contains a dict of {"i_j_Rx_Ry_Rz": numpy.ndarray} 
-- -- overlaps.h5       # a hdf5 dataset file with group named "0", "1", ..., "nframe". Each group contains a dict of {"i_j_Rx_Ry_Rz": numpy.ndarray} 
-- -- info.json
```

### Data settings: info.json

In **DeePTB**, the **atomic structures** and **band structures** data are stored in AtomicData graph structure. `info.json` defines the key parameters used in building AtomicData graph dataset, which looks like:
```bash
{
    "nframes": 1,
    "pos_type": "ase/cart/frac"
}
```
`nframes` is the length of the trajectory, as we defined in the previous section. `pos_type` defines the input format of the **atomic structures**, which is set to `ase` if  ASE `.traj` file is provided, and `cart` or `frac` if cartesian / fractional coordinate in `positions.dat` file provided.

<!--In the `AtomicData_options` section, the key arguments in defining graph structure is provided. `r_max` is the maximum cutoff in building neighbour list for each atom. `er_max` and `oer_max` are optional value for additional environmental dependence TB parameterization in **DeePTB-SK** mode, such as strain correction and `nnenv`. All cutoff variables have the unit of Angstrom.
For **DeePTB-SK**, We can get the recommended `r_max` value by `DeePTB`'s bond analysis function, using:
```bash
dptb bond <structure path> [[-c] <cutoff>] [[-acc] <accuracy>]
```

For **DeePTB-E3**, we suggest the user align the `r_max` value to the LCAO basis's cutoff radius used in DFT calculation.
-->
For **DeePTB-SK** model, we should also specify the parameters in `info.json` that controls the fitting eigenvalues:
```JSON
{
    "nframes": 1,
    "pos_type": "ase/cart/frac",
    "bandinfo": {
        "band_min": 0,
        "band_max": 6,
        "emin": null, # optional
        "emax": null # optional
    }
}
```

`bandinfo` defines the fitting target. The `emin` and `emax` defines the fitting energy window of the band, while the `band_min` and `band_max` select which band are targeted.
> **note:** The `0` energy point is located at the lowest energy eigenvalues from the band of band_min. The band_min should be aligned to the same index for  eigenvalues from DFT ans DeePTB. 


### Train config: input.json
**DeePTB** provides input config templates for quick setup. User can run:
```bash
dptb config -tr [[-e3] <e3tb>] [[-sk] <sktb>] [[-skenv] <sktbenv>] PATH
```
The template config file will be generated at the `PATH`.
We provide several template for different mode of deeptb, please run `dptb config -h` to checkout.

In general, the `input.json` file contains following parts:

#### common_options: 

provides vital information to build a **DeePTB** models. 
    
- For **DeePTB-SK** mode. The example of `common_options` is:
```json
    "common_options": {
                "basis": {
                    "C": ["2s", "2p"],
                    "N": ["2s", "2p", "d*"]
                },
                "device": "cpu",
                "overlap": false,
                "dtype": "float32",
                "seed": 42
        }
``` 
- For **DeePTB-E3** mode, the basis with similar format is different using string instead a list. For `C` and `N` with DZP basis, the `basis` should be defined as：
```json
    "basis": {
    "C": "2s2p1d",
    "N": "2s2p1d"
    }
```
#### train_options
spicify the training procedure.

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
The `loss_options` section is used to specify the loss function used in training. The `method` key is used to specify the loss function, which can be `eigvals`, `hamil_abs`, and `hamil_blas`. The `eigvals` loss is used for **DeePTB-SK** model, while `hamil_abs` and `hamil_blas` are used for **DeePTB-E3** model. Here `Adam` optimizer is always preferred for better convergence speed. While the `lr_scheduler` are recommended to use "rop", as:
```json
    "lr_scheduler": {
            "type": "rop",
            "factor": 0.8,
            "patience": 50,
            "min_lr": 1e-6
    }
```

More details about rop is available at: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
    


#### model_options
key setting to build **DeePTB** models. 

For **DeePTB-SK** model without env correction, only the `nnsk` section is needed. The example of a `nnsk` model is:

```json
    "model_options": {
        "nnsk": {
            "onsite": {"method": "uniform"},
            "hopping": {"method": "powerlaw", "rs":2.6, "w": 0.3},
            "freeze": false,
            "push": false
        }
    }
```

Different method of `onsite` and `hopping` have their specific parameters requirements, please checkout our full parameter lists.

For **DeePTB-SK** model with environment dependency, the `embedding`, `prediction` and `nnsk` sections are required as in this example:

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
            "onsite": {"method": "uniform"},
            "hopping": {"method": "powerlaw", "rs":5.0, "w": 0.1},
            "freeze": true
        }
    }
```

For **DeePTB-E3** model, only `embedding` and `prediction` is required, as:
```json
    "model_options": {
        "embedding": {
            "method": "slem/lem", # s in slem stands for strict localization
            "r_max": {
                "C": 7.0,
                "N": 7.0
            },
            "irreps_hidden": "32x0e+32x1o+16x2e+8x3o+8x4e+4x5o",
            "n_layers": 3,
            "n_radial_basis": 18,
            "env_embed_multiplicity": 10,
            "avg_num_neighbors": 51,
            "latent_dim": 64,
            "latent_channels": [
                32
            ],
            "tp_radial_emb": true,
            "tp_radial_channels": [
                32
            ],
            "PolynomialCutoff_p": 6,
            "cutoff_type": "polynomial",
            "res_update": true,
            "res_update_ratios": 0.5,
            "res_update_ratios_learnable": false
        },
        "prediction":{
            "method": "e3tb",
            "scales_trainable":false,
            "shifts_trainable":false,
            "neurons": [64,64] # optional, required when overlap in common_options is True
        }
    }
```

#### data_options
assigns the datasets used in training.

```json
    "data_options": {
        "train": {
            "type": "DefaultDataset", # optional, default "DefaultDataset"
            "root": "./data/",
            "prefix": "kpathmd100",
            "get_Hamiltonian": false, # optional, default false
            "get_eigenvalues": true, # optional, default false
            "get_overlap": false, # optional, default false
            "get_DM": false # optional, default false
        },
        "validation": {
            "type": "DefaultDataset",
            "root": "./data/",
            "prefix": "kpathmd100",
            "get_Hamiltonian": false,
            "get_eigenvalues": true,
            "get_overlap": false,
            "get_DM": false
        }
    }
```

## Commands
### Training
When data and input config file is prepared, we are ready to train the model:
```bash
dptb train <input config> [[-o] <output directory>] [[-i|-r] <deeptb checkpoint path>]
```
