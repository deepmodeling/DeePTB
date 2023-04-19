# **DeePTB**
Electronic, atomic, and spin are the three degrees of freedom involved in material simulation, corresponding to the three main levels of material properties. The cornerstone of accurately simulating material properties is describing the most fundamental interactions in a system based on the principles of quantum mechanics. Traditional simulation methods, such as DFT methods, provide relatively accurate quantum mechanical descriptions of materials' electrons, atoms, and spins, making them indispensable tools in modern scientific research. However, the high computational demand of traditional DFT methods remains a challenge for large-scale and long-term material simulations, even with the rapid development of supercomputers today.

The development of artificial intelligence (AI) offers a new approach to solve traditional problems. DeePTB aims to create a universal and applicable representation method for the electronic interaction Hamiltonian at the electronic degree of freedom level. It is a machine learning method that accurately and efficiently describes the electronic system's interaction and Hamiltonian, playing a crucial role in fundamentally accelerating the material's electronic-related properties.

The DeePTB method uses a neural network to predictively represent the tight-binding model (TB) model, which enables the model to have first-principles accuracy while maintaining high computational efficiency. It provides an efficient way to obtain the electronic Hamiltonian for large or complex systems, enabling rapid modeling for simulating material electronic properties and revolutionizing the method for simulating electronic properties.

- [**DeePTB**](#DeePTB)
  - [**dependency**:](#dependency)
  - [**installation**](#installation)
  - [**usage**:](#usage)
    - [1. **Traing TB Hamiltonian.**](#1-traing-tb-hamiltonian)
      - [1.1 **Data prepare**:](#11-data-prepare)
      - [1.2 **input**](#12-input)
      - [1.3 **Training**](#13-training)
      - [1.4 **Ploting**](#14-ploting)


## **dependency**:
- python >= 3.8
- pytest = ">=7.2.0"
- numpy
- scipy
- spglib
- matplotlib
- ase
- torch >= 1.13.0
- pyyaml
- future
- dargs

## **installation**
1. install torch following the instruction in : [PyTorch: Get Started](https://pytorch.org/get-started/locally)

2. located to the repository root

3. running ```pip install .```


## **usage**:
Will be added gradually

### 1. **Traing TB Hamiltonian.**

#### 1.1 **Data prepare**:
        
To train the TB model, one should supply the atomic structures and electronic structures.  The **atomic structures** data are in the format of ASE traj binary format, where each structure are stored using an **Atom** class defined  in ASE package. The **electronic structures** data contains the kpoints list and eigenvalues are in the binary format of npy. The shape of kpoints data is **[num_kpoint,3]** and eigenvalues is **[num_frame,nk,nbands]**. nsnaps is the number of snapshots, nk is the number of kpoints and nbands is the number of bands. In the  example. we proved a script to transfer the txt data file into binary file for training.

The dataset of one structure is recommended to formulate as following format:
```
data/
-- set.x
-- -- eigs.npy         # numpy array of shape [num_frame, num_kpoint, num_band]
-- -- kpoints.npy      # numpy array of shape [num_kpoint, 3]
-- -- xdat.traj        # ase trajectory file with num_frame
-- -- bandinfo.json    # defining the training objective of this bandstructure
```

The bandinfo defines the settings of the training objective of each structure, basicly you can have specific settings for different structure, which allow training across structures across diffrent atom number and atom type.

The **bandinfo.json** file looks like:
```json
{
    "band_min": 0,
    "band_max": 4,
    "gap_penalty": false,
    "fermi_band": 3,
    "loss_gap_eta": 0.1,
    "emin": null,
    "emax": null,
    "weight": [1]
}
```

 **Note**: the electronic structures data for training calculated on the irreducible Brillouin zone are high recommended. Besides, the electronic structures data can be obtained from any DFT package.
    
#### 1.2 **input**

We explain the input file with the hBN example.

```json
{
    "init_model": {
        "path": null,
        "interpolate": true
    },
    "common_options": {
        "onsitemode": "split",
        "onsite_cutoff": 3.6,
        "bond_cutoff": 3.5,
        "env_cutoff": 3.5,
        "atomtype": [
            "N",
            "B"
        ],
        "proj_atom_neles": {
            "N": 5,
            "B": 3
        },
        "proj_atom_anglr_m": {
            "N": [
                "2s",
                "2p"
            ],
            "B": [
                "2s",
                "2p"
            ]
        }
    },
    "train_options": {
        "seed":120478,
        "num_epoch": 1000,
        "optimizer": {"lr":1e-2}
    },
    "data_options": {
        "use_reference": true,
        "train": {
            "batch_size": 1,
            "path": "./dptb/tests/data/hBN/data",
            "prefix": "set"
        },
        "validation": {
            "batch_size": 1,
            "path": "./dptb/tests/data/hBN/data",
            "prefix": "set"
        },
        "reference": {
            "batch_size": 1,
            "path": "./dptb/tests/data/hBN/data",
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
```

#### 1.3 **Training**
See the example hBN.