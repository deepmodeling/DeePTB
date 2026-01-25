# DeePTB Pardiso Backend - Modular Architecture

This directory contains the modular Julia backend for high-performance eigenvalue calculations using Intel MKL Pardiso.

## Directory Structure

```
dptb/postprocess/pardiso/
├── io/
│   └── io.jl                # Unified I/O (structure + Hamiltonian)
├── solvers/
│   ├── pardiso_solver.jl    # Pardiso eigenvalue solver
│   └── dense_solver.jl      # Dense LAPACK solver (fallback)
├── tasks/
│   ├── band_calculation.jl  # Band structure calculation
│   └── dos_calculation.jl   # Density of states calculation
├── utils/
│   ├── hamiltonian.jl       # H(R) -> H(k) transformation
│   └── kpoints.jl           # K-point generation utilities
├── main.jl                  # Main entry point
├── sparse_calc_npy_print.jl # Legacy monolithic script (for reference)
└── README.md                # This file
```

## Key Improvements

### 1. Modular Design
- **Separation of concerns**: I/O, solving, and tasks are separate modules
- **Reusability**: Solver can be used for band, DOS, optical calculations
- **Testability**: Each module can be unit tested independently

### 2. Optimized Data Format
- **JSON structure file**: Replaces 4 text files with single JSON
- **Pre-computed data**: `site_norbits` and `norbits` computed by Python
- **No parsing needed**: Julia directly uses pre-computed values
- **Backward compatibility**: Falls back to legacy `.dat` files if `structure.json` is missing

### 3. Better Maintainability
- **Clear interfaces**: Each module has well-defined inputs/outputs
- **Documentation**: Docstrings for all public functions
- **Error handling**: Better error messages and logging

## Usage

### Basic Usage

```bash
julia main.jl --input_dir ./pardiso_input --output_dir ./results --config ./band.json
```

### Command Line Options

**Julia Backend (`main.jl`):**
- `--input_dir, -i`: Directory containing exported data (default: `./input_data`)
- `--output_dir, -o`: Output directory for results (default: `./results`)
- `--config`: Configuration JSON file (default: `./band.json`)
- `--ill_project`: Enable ill-conditioned projection (default: `true`)
- `--ill_threshold`: Threshold for ill-conditioning (default: `5e-4`)

**Python CLI (`dptb pdso`):**
- `-INPUT`: Configuration JSON file (required)
- `-i, --init_model`: Model checkpoint path (for export mode)
- `-stu, --structure`: Structure file path (for export mode)
- `-d, --data_dir`: Pre-exported data directory (for run-only mode)
- `-o, --output_dir`: Output directory (default: `./`)
- `--ill_project`: Enable ill-conditioned projection (default: `True`)
- `--ill_threshold`: Ill-conditioning threshold (default: `5e-4`)

### Python Integration

```python
from dptb.postprocess.unified.system import TBSystem

# Initialize system
tbsys = TBSystem(data="structure.vasp", calculator="model.pth")

# Export for Julia (Standard JSON format)
tbsys.to_pardiso(output_dir="pardiso_input")

# Or use CLI integration
from dptb.entrypoints.pdso import pdso
pdso(
    INPUT="band.json",
    init_model="model.pth",
    structure="structure.vasp",
    output_dir="./output"
)

# Load results
import numpy as np
band_data = np.load("output/results/bandstructure.npy", allow_pickle=True).item()
```

## Module Documentation

### io/io.jl

**Functions:**
- `load_structure(input_dir; spinful=false)`: Load structure (JSON or legacy .dat)
- `load_structure_json(input_dir)`: Load from `structure.json`
- `load_structure_dat(input_dir, spinful)`: Load from legacy `.dat` files
- `load_matrix_hdf5(filename)`: Load HDF5 matrix blocks

**Returns:**
- Dictionary with keys: `cell`, `positions`, `site_norbits`, `norbits`, `symbols`, `natoms`, `spinful`, `basis`

**Features:**
- **Automatic format detection**: Tries JSON first, falls back to `.dat`
- **Spin handling**: Correctly accounts for spin degeneracy in orbital counts
- **Robust parsing**: Handles both modern and legacy data formats

### solvers/pardiso_solver.jl

**Functions:**
- `construct_linear_map(H, S)`: Create linear map for shift-invert
- `solve_eigen_at_k(H_k, S_k, fermi_level, num_band, ...)`: Solve eigenvalue problem

**Features:**
- Shift-invert technique for better convergence
- Ill-conditioned state projection
- Automatic memory cleanup

### tasks/band_calculation.jl

**Functions:**
- `run_band_calculation(config, H_R, S_R, structure, ...)`: Main band calculation
- `parse_kpath_abacus(kpath_config, lat, labels)`: Parse k-path
- `save_bandstructure_npy(...)`: Export to NPY format

**Outputs:**
- `EIGENVAL`: VASP-format eigenvalues
- `bandstructure.npy`: NumPy format for Python visualization

## Configuration File Format

```json
{
  "task_options": {
    "task": "band",
    "kline_type": "abacus",
    "kpath": [
      [0.0, 0.0, 0.0, 100],
      [0.0, 0.0, 0.5, 1]
    ],
    "klabels": ["G", "Z"],
    "E_fermi": -9.03841,
    "emin": -2,
    "emax": 2
  },
  "num_band": 30,
  "max_iter": 400,
  "out_wfc": "false",
  "isspinful": "false"
}
```

## Performance Tips

1. **Caching**: Sparse matrices are cached in `sparse_matrix.jld` for faster subsequent runs
2. **Ill-conditioning**: Enable `--ill_project` for systems with near-singular overlap matrices
3. **Convergence**: Increase `max_iter` if eigenvalues don't converge
4. **Memory**: For very large systems (>10000 orbitals), consider reducing `num_band`

## Comparison: Old vs New

| Aspect | Old (sparse_calc_npy_print.jl) | New (Modular) |
|--------|--------------------------------|---------------|
| **Lines of code** | ~636 lines | ~400 lines (split across modules) |
| **Structure** | Monolithic | Modular |
| **Testability** | Difficult | Easy (unit tests per module) |
| **Reusability** | Low | High (solver reusable) |
| **Data format** | 4 text files + 2 H5 | 1 JSON + 2 H5 |
| **Parsing** | Complex (50+ lines) | Simple (5 lines) |
| **Maintainability** | Low | High |

## Future Enhancements

1. **DOS calculation**: Add `tasks/dos_calculation.jl`
2. **Optical properties**: Add `tasks/optical_calculation.jl`
3. **PyJulia integration**: Direct Python-Julia calls (no file I/O)
4. **Parallel k-points**: Distribute k-point calculations
5. **GPU support**: Add cuSOLVER backend

## Dependencies

Required Julia packages:
```julia
using Pkg
Pkg.add(["JSON", "HDF5", "ArgParse", "Pardiso", "Arpack", "LinearMaps", "JLD", "SparseArrays"])
```

## Troubleshooting

**Issue**: `structure.json not found`
- **Solution**: Run `tbsys.to_pardiso()` first to export data, or ensure legacy `.dat` files are present

**Issue**: Eigenvalues don't converge
- **Solution**: Increase `max_iter` or adjust `E_fermi` closer to actual Fermi level

**Issue**: Ill-conditioned overlap matrix
- **Solution**: Enable `--ill_project` and adjust `--ill_threshold`

**Issue**: Dimension mismatch for spinful systems
- **Solution**: Ensure `isspinful` is correctly set in config file

## License

Same as DeePTB main package.
