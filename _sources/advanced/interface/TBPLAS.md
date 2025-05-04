# TBPLAS
TBPLaS (Tight-Binding Package for Large-scale Simulation) is a package for building and solving tight-binding models, with emphasis on handling large systems. Thanks to the utilization of tight-binding propagation method (TBPM), sparse matrices, Cython/FORTRAN extensions, and hybrid OpenMP+MPI parallelization, TBPLaS is capable of solving models with billions of orbitals on computers with moderate hardware.

The DeePTB model support the interface to transcript the predicted hamiltonian to TBPLaS format. This allow the user to utlize vast amount of quantities computing tools supported by TBPLaS. Benefitting from both the abinitio accuracy of DeePTB and the scalability of TBPLaS, you can compute electronic quantities of extremely large system with DFT level accuracy. Here is a illustration of computing a million atoms GaN system:

<div align=center>
<img src="https://raw.githubusercontent.com/deepmodeling/DeePTB/main/docs/img/million_atoms.png" width = "80%" height = "50%" alt="Silicon FS velocity arrow" align=center />
</div>

## How to use the DeePTB-TBPLaS interfaces
```Python
from dptb.data import AtomicData
from dptb.nn import build_model
from ase.io import read


AtomicData_options= {
        "r_max": 5.0,
        "er_max": 5.0,
        "oer_max": 2.5,
        "pbc": True
    }

# using from_ase method to generate an AtomicData class for silicon
dataset = AtomicData.from_ase(
    atoms=read("DeePTB/examples/silicon/data/silicon.vasp"),
    **AtomicData_options
    )

# loading trained DeePTB model
model = build_model(
    "DeePTB/examples/silicon/ref_ckpts/dptb/checkpoint/mix.ep50.pth", 
    )


model.eval()

```
Then, we need to import the TBPLaS interface class, and performed the transcription:
```Python
from dptb.postprocess.totbplas import TBPLaS

tbplas = TBPLaS(model=model, device="cpu")
cell = tbplas.get_cell(
    data=dataset, 
    e_fermi=-7.72461
)

```

The cell class now is the PrimitiveCell class of TBPLaS, which support computing various physical quantities. We refer to the [document](http://www.tbplas.net/tutorial/prim_cell/index.html) of TBPLaS for the usage.