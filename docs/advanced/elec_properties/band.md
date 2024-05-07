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