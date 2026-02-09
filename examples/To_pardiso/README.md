# DeePTB Pardiso Integration - Example Directory

This directory contains complete examples for using DeePTB's Pardiso backend for high-performance band structure calculations.

## Files

### Data Files
- `min.vasp`: Example structure file (C84 fullerene)
- `nnsk.iter_ovp0.000.pth`: Pre-trained DeePTB model
- `band_new.json`: Configuration file for band structure calculation

### Notebooks
- **`pardiso_tutorial.ipynb`** ⭐: **Complete tutorial** covering:
  - Python API usage (`to_pardiso()`)
  - CLI integration (`dptb pdso`)
  - Manual Julia execution
  - Result visualization
  - Backward compatibility

- `dptb_to_Pardiso_new.ipynb`: Legacy notebook (kept for reference)
- `dptb_to_Pardiso.ipynb`: Original notebook (deprecated)

### Scripts
- `test_pardiso_new.py`: Automated test script for the export workflow

## Quick Start

### Option 1: Using CLI (Recommended)

```bash
dptb pdso \
  band_new.json \
  -i nnsk.iter_ovp0.000.pth \
  -stu min.vasp \
  -o ./output
```

With custom solver parameters:
```bash
dptb pdso \
  band_new.json \
  -i nnsk.iter_ovp0.000.pth \
  -stu min.vasp \
  -o ./output \
  --ill_project false \
  --ill_threshold 1e-3
```

Results will be in `./output/results/`.

### Option 2: Using Python API

```python
from dptb.postprocess.unified.system import TBSystem

# Initialize
tbsys = TBSystem(data="min.vasp", calculator="nnsk.iter_ovp0.000.pth")

# Export
tbsys.to_pardiso(output_dir="pardiso_data")

# Run Julia backend
import subprocess
subprocess.run([
    "julia", "../../dptb/postprocess/pardiso/main.jl",
    "--input_dir", "pardiso_data",
    "--output_dir", "results",
    "--config", "band_new.json"
])
```

### Option 3: Interactive Tutorial

Open `pardiso_tutorial.ipynb` in Jupyter for a complete walkthrough.

## Configuration File Format

`band_new.json` example:

```json
{
  "task_options": {
    "task": "band",
    "kline_type": "abacus",
    "kpath": [
      [0.0, 0.0, 0.0, 100],
      [0.0, 0.0, 0.5, 1]
    ],
    "klabels": ["Γ", "Z"],
    "E_fermi": -9.03841,
    "emin": -2,
    "emax": 2
  },
  "num_band": 30,
  "max_iter": 400,
  "isspinful": "false"
}
```

## Output Files

After running the workflow, you'll get:

### Export Directory (`pardiso_data/`)
- `structure.json`: Structure and basis information
- `predicted_hamiltonians.h5`: Hamiltonian matrix blocks
- `predicted_overlaps.h5`: Overlap matrix blocks

### Results Directory (`results/`)
- `bandstructure.npy`: Band structure data (NumPy format)
- `EIGENVAL`: Eigenvalues in VASP format
- `bands.dat`: Text format band data

## Advanced Usage

### Backward Compatibility

For legacy workflows, use `to_pardiso_debug()` to export `.dat` files:

```python
tbsys.to_pardiso_debug(output_dir="legacy_data")
```

The Julia backend will automatically detect and load these files if `structure.json` is missing.

### Performance Tuning

Edit `band_new.json`:
- `num_band`: Number of bands to calculate (default: 30)
- `max_iter`: Maximum iterations for eigenvalue solver (default: 400)
- `emin`, `emax`: Energy window for band structure

For ill-conditioned systems, run Julia with:
```bash
julia main.jl --input_dir data --output_dir results --config band.json --ill_project true
```

## Troubleshooting

**Issue**: `structure.json not found`
- **Solution**: Run `tbsys.to_pardiso()` first

**Issue**: Eigenvalues don't converge
- **Solution**: Increase `max_iter` in config or adjust `E_fermi`

**Issue**: Dimension mismatch for spinful systems
- **Solution**: Ensure `isspinful` matches your model (check `hasattr(model, 'soc_param')`)

## References

- [DeePTB Documentation](https://deeptb.readthedocs.io/)
- [Pardiso Backend README](../../dptb/postprocess/pardiso/README.md)
- [Unit Tests](../../dptb/tests/test_to_pardiso.py)
