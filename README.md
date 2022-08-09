# **DeepNEGF**
Deep learning on quantum transport simulations with NEGF method.

- [**DeepNEGF**](#deepnegf)
  - [**dependency**:](#dependency)
  - [**installation**](#installation)
  - [**usage**:](#usage)
    - [1. **Traing TB Hamiltonian.**](#1-traing-tb-hamiltonian)
      - [1.1 **Data prepare**:](#11-data-prepare)
      - [1.2 **input**](#12-input)
      - [1.3 **Training**](#13-training)
    - [2. **the NEGF calculations.**](#2-the-negf-calculations)
      - [2.1. **Device structure**](#21-device-structure)
      - [2.2. **input**](#22-input)
    - [3. **Transport calculations.**](#3-transport-calculations)


## **dependency**:
- python >= 3.8
- numpy
- scipy
- spglib
- matplotlib
- ase
- torch
- torchsort==0.1.7


## **installation**
1. install torch following the instruction in : [PyTorch: Get Started](https://pytorch.org/get-started/locally)


2.  ```python setup.py install```


## **usage**:
Will be added gradually

### 1. **Traing TB Hamiltonian.**

#### 1.1 **Data prepare**:
        
To train the TB model, one should supply the atomic structures and electronic structures.  The **atomic structures** data are in the format of ASE traj binary format, where each structure are stored using an **Atom** class defined  in ASE package. The **electronic structures** data contains the kpoints list and eigenvalues are in the binary format of npy. The shape of kpoints data is [nk,3] and eigenvalues is [nsnaps,nk,nbands]. nsnaps is the number of snapshots, nk is the number of kpoints and nbands is the number of bands. In the  example. we proved a script to transfer the txt data file into binary file for training.

 **Note**: the electronic structures data for training calculated on the irreducible Brillouin zone are high recommended. Besides, the electronic structures data can be obtained from any DFT package.
    
#### 1.2 **input**

We explain the input file with the hBN example.

```json
    {
        "_comment":"general paras.",
        "AtomType" : ["N","B"],
        "ProjAtomType" : ["N","B"],
        "ProjAnglrM" : {"N":["s","p"],"B":["s","p"]},
        "ValElec" : {"N":5,"B":3},
        "EnvCutOff" : 3.5,
        "NumEnv" : [8,8],
        "CutOff" : 4,
        "SKFilePath" :	"./slakos",
        "Separator" : "-" ,
        "Suffix" : ".skf",
    
        "_comment" : "NN paras",
        "Task" : "NNTB",
        "Envnet" : [10,20,40],
        "Envout" : 10,
        "Bondnet" : [100,100,100],
        "onsite_net" : [100,100,100],
        "active_func": "tanh",
        "train_data_path" : "../data",
        "prefix"  : "set",
        "valddir" : "../data/set.0",
        "withref" : false,
        "refdir"  : "none",
        "ref_ratio": 0.5,
        "xdatfile" : "xdat.traj",
        "eigfile"  : "eigs.npy",
        "kpfile"   : "kpoints.npy",
        "num_epoch" : 200,
        "batch_size": 1,
        "valid_size": 1,
        "start_learning_rate": 0.0001,
        "decay_rate":0.99,
        "decay_step":2,
        "savemodel":true,
        "save_epoch":4,
        "save_checkpoint":"./checkpoint.pl",
        "display_epoch": 1,
        "read_checkpoint": "./checkpoint.pl",
    	"use_E_win":false,
    	"energy_max":8,
    	"energy_min":-20,
    	"use_I_win":true,
    	"band_max":4,
    	"band_min":0,
        "sort_strength":[0.01,0.01],
        "corr_strength":[1, 1]
    }

 ```
- **AtomType**: the atoms types in the structure.
- **ProjAtomType**: the atoms types where the local orbitals are used to construct TB.
- **ProjAnglrM**: local orbitals type on the ProjAtomType.
- **ValElec**: num of valence electrons in the selected energy window. this is used to calculated Fermi energy. be carefully, this doesn'tmean the total valence electrons, only the valence electrons in the selected energy/band window.
- **EnvCutOff**: the cut-off for local environment.
- **NumEnv**: maximum of number of atoms inside the cut-off. 
- **CutOff**: the cut-off for the maximum distance where hopping is finite.
- **SKFilePath**: the path for slater koster files.
- **Separator**: Separator for the name of slater koster files.
- **Suffix**:  Suffix for slater koster files.
- **Envnet**: neural network for environment embeding.
- **Envout**: output size of local environment.
- **Bondnet**: neural network for hoppings.
- **onsite_net**: neural network for on-site energy.
- **train_data_path**: the path for training data.
- **valddir**: validation data path.
- **withref**: train TB model with a reference or not. if true, one should set the **refdir** for the path of ref data and **ref_ratio**for the ratio in loss of reference data.
- **xdatfile**, **eigfile**, **kpfile** : the data file name.
- **use_E_win**: use energy to define the selected band window. if true, one should set the **energy_min** and **energy_max**.
- **use_I_win**: use band index to define the selected band window. if true, one should set the **band_min** and **band_max**.
- **sort_strength**: [start_value,final_value] soft sort strength, in the training the process, the strength exponentially changes fromstart_value to final_value.
- **corr_strength**: only take effect when correction_mode=2.  by default correction_mode=1.


#### 1.3 **Training**
See the example.
,




### 2. **the NEGF calculations.**

#### 2.1. **Device structure**

#### 2.2. **input**

### 3. **Transport calculations.**

---

