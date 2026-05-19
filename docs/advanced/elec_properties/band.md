# Band structure

We can compute the bandstructure from trained model with simply a few line of code or command. Here is an example:

## Band api
Import the necessary package
```Python
from dptb.nn.build import build_model
from dptb.postprocess.import Band
```

build the model and band class
```Python
model = build_model("your model path")
band = Band(model=model, device=model.device)
```

Then, we can call `band.get_bands()` to compute the band structure, which takes a few variables as input:
```Python
band.get_bands(
    data: Union[AtomicData, ase.Atoms, str], 
    kpath_kwargs: dict, 
    AtomicData_options: dict={}
):
```
Here `data` can be an AtomicData class(see quick_start/basic_API for usage), an ase.Atoms class, or any atomic structure files format path that is readable by `ase.io.read`.

If using ase.Atoms class or string of structure file path, we need to set the AtomicData_options, which is a dict contrains the basis definition of the atomic graph. a template looks like:
```json
AtomicData_options = {
    "r_max": 5.0,
    "er_max": 5.0, # optional, needed if using neural network env correction
    "oer_max": 5.0 # optional, needed if using onsite strain correction
    "pbc": true # optional, default true
}
```

Now what is needed is the `kpath_kwargs`, which defines the kpoint setting in plotting the bandstructure. We support different types of kpath definition, includes `ase/abacus/vasp/array`. Here we introduce the `ase` and `abacus` mode here:
```Python
# ase
kpath_kwargs = {
    "kline_type" = "ase",
    "kpath" = "GMKG",
    "nkpoint" = 300,
}

# abacus
kpath_kwargs = {
    "kline_type" = "abacus",
    "kpath" = [
            [0, 0, 0, 30],
            [0.5, 0, 0, 30],
            [0.3333333, 0.3333333, 0, 30],
            [0, 0, 0, 1]
        ],
    "klabels" = ["G", "M", "K", "G"]
}
```

## Ill-conditioned overlap handling

For non-orthogonal models, DeePTB solves the generalized eigenvalue problem
`H(k)c = E S(k)c`. This requires the overlap matrix `S(k)` to be positive
definite. Large or redundant local basis sets can make `S(k)` nearly singular,
especially in band/DOS post-processing. In that case the default Cholesky
solver fails fast, which is still the recommended behavior during training.

For inference or post-processing, you can explicitly enable a canonical
orthogonalization fallback by setting `ill_threshold`. DeePTB then removes
overlap modes with eigenvalues below this threshold and solves the projected
problem. The removed bands are padded with `ill_pad_value` to keep the usual
dense band array shape, and a valid-eigenvalue mask is stored in the returned
data under `eigenvalue_valid_mask`.

Example for the band JSON used by `dptb run`:

```json
{
    "task_options": {
        "task": "band",
        "kline_type": "abacus",
        "kpath": [
            [0.0, 0.0, 0.0, 30],
            [0.5, 0.0, 0.0, 30],
            [0.0, 0.0, 0.0, 1]
        ],
        "klabels": ["G", "X", "G"],
        "override_overlap": "./overlaps.h5",
        "eig_solver": "torch",
        "ill_threshold": 5e-4,
        "ill_pad_value": 10000.0
    }
}
```

The same option works with the Python band API:

```python
from dptb.nn.build import build_model
from dptb.postprocess.bandstructure.band import Band

model = build_model(checkpoint="model.pth")
band = Band(model=model, device=model.device)

kpath_kwargs = {
    "kline_type": "abacus",
    "kpath": [[0.0, 0.0, 0.0, 30], [0.5, 0.0, 0.0, 1]],
    "klabels": ["G", "X"],
    "override_overlap": "overlaps.h5",
    "eig_solver": "torch",
    "ill_threshold": 5e-4,
}

band_status = band.get_bands(
    data="structure.vasp",
    kpath_kwargs=kpath_kwargs,
)
eigenvalues = band_status["eigenvalues"]
```

Or through the unified `TBSystem` interface:

```python
from dptb.postprocess.unified.system import TBSystem

system = TBSystem(
    data="structure.vasp",
    calculator="model.pth",
    override_overlap="overlaps.h5",
)

bands = system.get_bands(
    kpath_config={
        "method": "abacus",
        "kpath": [[0.0, 0.0, 0.0, 30], [0.5, 0.0, 0.0, 1]],
        "klabels": ["G", "X"],
    },
    ill_threshold=5e-4,
)
```

Use this option only when the ill-conditioned overlap comes from near-linear
dependencies in a non-orthogonal basis. If many modes are projected out, the
basis, overlap source, or model should be checked instead of increasing the
threshold blindly.
