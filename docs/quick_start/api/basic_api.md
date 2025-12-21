# Basic API
Instead of commandline, we provide many function interfaced that enable you to build and use the DeePTB model, data module. Here are some quick examples:

## build the DeePTB model:
We can quickly build a deeptb model using `build_model` function in `dptb.nn`:
```Python
from dptb.nn import build_model

model = build_model(checkpoint="Your trained checkpoint path")
```

or, we can also build a model from sratch, using a setted model_options and common_options:
```Python
model = build_model(
    model_options=Dict[Your model options], 
    common_options=Dict[Your common options]
    )
```

## get a atomicdata
The data to represent a atomic structure are named AtomicData in DeePTB (following the convension in nequip). We can get a atomicdata class from ase Atoms class using `AtomicData.from_ase()`, or simply by inserting the necessary variables using `AtomicData.from_points()`:
```Python
from dptb.data import AtomicData
atomicdata = AtomicData.from_ase(
    atoms=ase.Atoms[Your ase Atoms structure class],
    r_max=float[some cutoff radius],
    er_max: Optional[float], # work in sk mode, ignore in e3
    oer_max: Optional[float] # work in sk mode, ignore in e3
)

atomicdata.from_points(
    pos=pos, # positions of atoms
    r_max=r_max, # your bond cutoff radius
    atomic_numbers=atomic_numbers, # your atomic numbers of the structure
)
```

In addition, we can also get the data from well-structured dataset files, using `dptb.data.build_dataset`:
```Python
from dptb.data import build_dataset

dataset = build_dataset(
    root: str, # your dataset root dir
    prefix: str, # your prefix of datafile folders
    get_eigenvalues: bool, # whether to get eigenvalues when build the dataset
    basis: dict # the basis defined in the common_options
)

# Then, you can get the atomicdata from built dataset as:
atomicdata = dataset[0] # dataset contains a list of atomicdata, you can get any of then by simply indexing the dataset
```

## make prediction

### SKTB

The prediction can be simply performed once we have a AtomicData class and a model:
```Python
atomicdata = model(AtomicData.to_AtomicDataDict(atomicdata))
```
Here we call `AtomicData.to_AtomicDataDict` to transcript the AtomicData class into dict where its key are strings and values are torch.Tensor. After calling the `model()`, the atomicdata output will contrains the parameters infered on this atomic structure using the deeptb model, and can be further use to compute the Hamiltonian:

```Python
from dptb.nn import SKHamiltonian

skham = SKHamiltonian(basis=model.idp.basis)
```

The SKTB hamiltonian can be constructed from the infered atomicdata:
```Python
atomicdata = skham(atomicdata)
```

### E3TB
For E3 model, the hamiltonian and overlap can be predict at one shot, by:
```Python
atomicdata = model(AtomicData.to_AtomicDataDict(atomicdata))
```
Then the "edge_features", "node_features", "edge_overlap", "node_overlap" would be the Hamiltonian and Overlap features.


After this step, we have attained the Hamiltonian and/or Overlap stored in atomicdata class. The hamiltonian is arranged as the edge(bond)-wide hopping block and node(atom)-wise onsite block, which is then reshaped as a 1-D array. We doing this to save memory and help to perform batchlized operations. You can simply recover this format to a conventional `H(R)` with $(i,j,R)->H_{ij}(R)$ block with a transcript function:

```Python
from dptb.data import feature_to_block

H_block = feature_to_block(data=atomicdata, idp=model.idp)
S_block = feature_to_block(data=atomicdata, idp=model.idp, overlap=True)
```
H_block and S_block is a dictionary of python.

## compute properties
The DeePTB package provide many tools to compute physical quantities from the predicted `H(R)`, here we make a list of the functions and their importing method.
```Python
from dptb.nn import HR2HK, Eigenvalues
from dptb.postprocess import Band

HR2HR # converting the atomicdata with HR to Hamiltonian in K space
Eigenvalues # compute eigenvalues with given kpoint and HR
Band # compute and ploting band structure with a given deeptb model, structure and k-path information
```