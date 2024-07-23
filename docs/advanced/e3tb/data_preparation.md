# Data Preparation
We suggest the user use a data parsing tool [dftio](https://github.com/floatingCatty/dftio) to directly convert the output data from DFT calculation into readable datasets. Our implementation supports the parsed dataset format of `dftio`. Users can just clone the `dftio` repository and run `pip install .` in its root directory. Then one can use the following parsing command for the parallel data processing directly from the DFT output:
```bash
usage: dftio parse [-h] [-ll {DEBUG,3,INFO,2,WARNING,1,ERROR,0}] [-lp LOG_PATH] [-m MODE] [-n NUM_WORKERS] [-r ROOT] [-p PREFIX] [-o OUTROOT] [-f FORMAT] [-ham] [-ovp] [-dm] [-eig]

optional arguments:
  -h, --help            show this help message and exit
  -ll {DEBUG,3,INFO,2,WARNING,1,ERROR,0}, --log-level {DEBUG,3,INFO,2,WARNING,1,ERROR,0}
                        set verbosity level by string or number, 0=ERROR, 1=WARNING, 2=INFO and 3=DEBUG (default: INFO)
  -lp LOG_PATH, --log-path LOG_PATH
                        set log file to log messages to disk, if not specified, the logs will only be output to console (default: None)
  -m MODE, --mode MODE  The name of the DFT software. (default: abacus)
  -n NUM_WORKERS, --num_workers NUM_WORKERS
                        The number of workers used to parse the dataset. (For n>1, we use the multiprocessing to accelerate io.) (default: 1)
  -r ROOT, --root ROOT  The root directory of the DFT files. (default: ./)
  -p PREFIX, --prefix PREFIX
                        The prefix of the DFT files under root. (default: frame)
  -o OUTROOT, --outroot OUTROOT
                        The output root directory. (default: ./)
  -f FORMAT, --format FORMAT
                        The output root directory. (default: dat)
  -ham, --hamiltonian   Whether to parse the Hamiltonian matrix. (default: False)
  -ovp, --overlap       Whether to parse the Overlap matrix (default: False)
  -dm, --density_matrix
                        Whether to parse the Density matrix (default: False)
  -eig, --eigenvalue    Whether to parse the kpoints and eigenvalues (default: False)
```

After parsing, the user need to write a info.json file and put it in the dataset. For default dataset type, the `info.json` looks like:

```JSON
{
    "nframes": 1,
    "pos_type": "cart",
    "AtomicData_options": {
        "r_max": 7.0,
        "pbc": true
    }
}

```
Here `pos_type` can be `cart`, `dirc` or `ase`. For `dftio` output dataset, we use `cart` by default. The `r_max`, in principle, should align with the orbital cutoff in the DFT calculation. For a single element, the `r_max` should be a float number, indicating the largest bond distance included. When the system has multiple atoms, the `r_max` can also be a dict of atomic species-specific number like `{A: 7.0, B: 8.0}`. Then the largest bond `A-A` would be 7 and `A-B` be (7+8)/2=7.5, and `B-B` would be 8. `pbc` can be a bool variable, indicating the open or close of the periodic boundary conditions of the model. It can also be a list of three bool elements like `[true, true, false]`, which means we can set the periodicity of each direction independently.

For LMDB type Dataset, the info.json is much simpler, which looks like this:
```JSON
{
    "r_max": 7.0
}
```
Where other information has been stored in the dataset. LMDB dataset is designed for handeling very large data that cannot be fit into the memory directly.

Then you can set the `data_options` in the input parameters to point directly to the prepared dataset, like:
```JSON
"data_options": {
        "train": {
            "root": "./data",
            "prefix": "Si64",
            "get_Hamiltonian": true,
            "get_overlap": true
        }
    }
```

If you are using a python script, the dataset can be build with the same parameters using `build_datasets`:
```Python
from dptb.data import build_dataset

dataset = build_dataset(
    root="your dataset root",
    type="DefaultDataset",
    prefix="frame",
    get_overlap=True,
    get_Hamiltonian=True,
    basis={"Si":"2s2p1d"}
    )
```
