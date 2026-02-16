# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeePTB is a Python package that uses deep learning to accelerate *ab initio* electronic structure simulations. It provides versatile, accurate, and efficient simulations for materials and phenomena using two main approaches:

1. **DeePTB-SK**: Deep learning-based local environment-dependent Slater-Koster tight-binding
2. **DeePTB-E3**: E3-equivariant neural networks for representing quantum operators

## Development Commands

### Installation

The project uses `uv` for dependency management. Install with:

```bash
# CPU version (default)
./install.sh
# or
uv sync

# GPU version (specify CUDA version)
./install.sh cu121  # for CUDA 12.1
# or
uv sync --find-links https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

Supported CUDA versions: cu118, cu121, cu124

### Testing

```bash
# Run all tests
uv run pytest ./dptb/tests/

# Run specific test file
uv run pytest ./dptb/tests/test_band.py

# Run specific test function
uv run pytest ./dptb/tests/test_band.py::test_function_name

# Run tests with verbose output
uv run pytest ./dptb/tests/ -v

# Run tests with specific markers (if defined)
uv run pytest ./dptb/tests/ -m marker_name
```

### Running DeePTB

All commands use `uv run dptb <subcommand>`:

```bash
# Generate config templates
uv run dptb config PATH [--train] [--test] [--e3tb] [--sktb]

# Train a model
uv run dptb train INPUT [-i INIT_MODEL] [-r RESTART] [-o OUTPUT]

# Test a model
uv run dptb test INPUT -i INIT_MODEL [-o OUTPUT]

# Run TB model for postprocessing
uv run dptb run INPUT -i INIT_MODEL [-stu STRUCTURE] [-o OUTPUT]

# Bond distance analysis
uv run dptb bond STRUCTURE [-c CUTOFF] [-acc ACCURACY]

# Data preprocessing
uv run dptb data INPUT [--parse] [--split] [--collect]

# Convert formats
uv run dptb n2j INPUT --nrl_file NRL_FILE [-o OUTDIR]  # NRL to json
uv run dptb p2j -i PTH_FILE [-o OUTDIR]                # PTH to json

# Collect SK parameters
uv run dptb cskf -d DIR_PATH [-o OUTPUT]

# Convert SK files to nn-sk model
uv run dptb skf2nn INPUT -i INIT_MODEL [-o OUTPUT]

# Generate empirical SK parameters
uv run dptb esk INPUT [-o OUTPUT] [-m poly2|poly4] [--soc]

# Export to external formats
uv run dptb export INPUT -i INIT_MODEL [-stu STRUCTURE] [-f wannier90|pythtb] [-o OUTPUT]
```

## Architecture Overview

### Model Types

DeePTB supports four model modes, determined by `model_options` configuration:

1. **NNSK Mode**: Neural network Slater-Koster model
   - Set only `model_options.nnsk`
   - Class: `dptb.nn.nnsk.NNSK`

2. **DFTBSK Mode**: DFTB-based Slater-Koster model
   - Set only `model_options.dftbsk`
   - Class: `dptb.nn.dftbsk.DFTBSK`
   - Base models available: `poly2`, `poly4`

3. **NNENV Mode**: Neural network environment-dependent model (DeePTB-E3)
   - Set both `model_options.embedding` and `model_options.prediction`
   - No `nnsk` or `dftbsk` options
   - Class: `dptb.nn.deeptb.NNENV`
   - Supports `e3tb` prediction method with E3-equivariant networks

4. **MIX Mode**: Mixed model combining SK with environment-dependent corrections
   - Set `model_options.nnsk` or `model_options.dftbsk` AND both `embedding` and `prediction`
   - Prediction method must be `sktb`
   - Embedding method must be `se2`
   - Class: `dptb.nn.deeptb.MIX`

Model selection logic is in `dptb/nn/build.py:build_model()`.

### Core Module Structure

- **`dptb/entrypoints/`**: CLI command implementations
  - `main.py`: Argument parsing and command routing
  - `train.py`, `test.py`, `run.py`: Training, testing, and inference workflows
  - `config.py`: Config template generation
  - `data.py`: Data preprocessing workflows

- **`dptb/nn/`**: Neural network models and layers
  - `base.py`: Base model class with common functionality
  - `deeptb.py`: NNENV and MIX model implementations
  - `nnsk.py`: NNSK model implementation
  - `dftbsk.py`: DFTBSK model implementation
  - `build.py`: Model factory and initialization logic
  - `hamiltonian.py`: Hamiltonian construction
  - `hr2hk.py`: Real-space to k-space transformation
  - `embedding/`: Graph embedding networks (e3baseline, se2, mpnn)
  - `sktb/`: Slater-Koster tight-binding modules (hopping, onsite, SOC)
  - `dftb/`: DFTB-specific parameterization

- **`dptb/data/`**: Data loading and processing
  - `AtomicData.py`: Core data structure for atomic systems
  - `build.py`: Dataset factory (`dataset_from_config`)
  - `transforms.py`: Data transformations (TypeMapper, OrbitalMapper)
  - `dataset/`: Dataset implementations (DefaultDataset, HDF5Dataset, LMDBDataset, DeePHE3Dataset)
  - `interfaces/`: Parsers for external formats (ABACUS, OpenMX, SIESTA, etc.)

- **`dptb/nnops/`**: Training and testing operations
  - `trainer.py`, `base_trainer.py`: Training loop implementations
  - `tester.py`, `base_tester.py`: Testing/evaluation implementations
  - `loss.py`: Loss functions for different properties (energy, eigenvalues, Hamiltonian)

- **`dptb/postprocess/`**: Post-processing and analysis
  - `elec_struc_cal.py`: Electronic structure calculations
  - `interfaces.py`: Export interfaces (Wannier90, PythTB, TB2J)
  - `bandstructure/`: Band structure plotting and analysis
  - `unified/`: Unified postprocessing interface
  - `tbtrans_init.py`: TBtrans initialization

- **`dptb/utils/`**: Utility functions
  - Configuration validation and normalization
  - Logging and tools

### Data Pipeline

1. **Data Parsing**: External DFT outputs → standardized format
   - Parsers in `dptb/data/interfaces/` for ABACUS, OpenMX, SIESTA, etc.
   - Command: `uv run dptb data INPUT --parse`

2. **Dataset Creation**: Standardized data → PyTorch datasets
   - Factory: `dptb/data/build.py:dataset_from_config()`
   - Supports HDF5, LMDB, and default formats
   - Applies transforms (TypeMapper, OrbitalMapper)

3. **Training**: Dataset → trained model
   - Entry: `dptb/entrypoints/train.py`
   - Trainer: `dptb/nnops/trainer.py`
   - Model building: `dptb/nn/build.py:build_model()`

4. **Inference**: Trained model + structure → predictions
   - Entry: `dptb/entrypoints/run.py`
   - Postprocessing: `dptb/postprocess/`

### Configuration System

DeePTB uses JSON/YAML configuration files with three main sections:

1. **`common_options`**: Shared settings (basis, device, dtype, etc.)
2. **`model_options`**: Model architecture (nnsk/dftbsk/embedding/prediction)
3. **`data_options`**: Dataset configuration (train/validation sets)

Generate templates with: `uv run dptb config PATH --train --e3tb` (or `--sktb`)

Configuration validation is in `dptb/utils/config_check.py` and `dptb/utils/argcheck.py`.

## Key Concepts

### Basis Sets and Orbitals

- Basis sets defined in `common_options.basis`
- Format: `{"element": ["orbital1", "orbital2", ...]}` (e.g., `{"C": ["2s", "2p"]}`)
- OrbitalMapper in `dptb/data/transforms.py` handles orbital indexing

### Cutoff Radii

- Multiple cutoff types: `r_max` (bond), `er_max` (environment), `oer_max` (overlap environment)
- Can be global or per-bond-type
- Collection logic in `dptb/utils/argcheck.py:collect_cutoffs()`

### Hamiltonian Construction

- Real-space Hamiltonian (H(R)) built from bond features
- k-space Hamiltonian (H(k)) via Fourier transform in `dptb/nn/hr2hk.py`
- Block structure handling in `dptb/nn/hamiltonian.py`

### Spin-Orbit Coupling (SOC)

- SOC support in SK models via `dptb/nn/sktb/soc.py`
- Enable with `--soc` flag in `esk` command
- Requires appropriate basis with spin channels

## Testing Notes

- Tests use `pytest` with `pytest-order` for execution ordering
- Test data in `dptb/tests/data/`
- Some tests may require optional dependencies (3Dfermi, tbtrans_init, pybinding)
- CI runs tests in Docker container (see `.github/workflows/unit_test.yml`)

## Common Workflows

### Training a New Model

1. Prepare DFT data and parse: `uv run dptb data config.json --parse`
2. Generate training config: `uv run dptb config input.json --train --e3tb`
3. Edit config with data paths and model settings
4. Train: `uv run dptb train input.json -o output_dir/`

### Fine-tuning from Checkpoint

```bash
uv run dptb train input.json -i checkpoint.pth -o output_dir/
```

### Running Inference

```bash
uv run dptb run run_config.json -i model.pth -stu structure.vasp -o results/
```

### Converting Models

```bash
# PTH to JSON (for portability)
uv run dptb p2j -i model.pth -o output_dir/

# Export to Wannier90 format
uv run dptb export export_config.json -i model.pth -stu structure.vasp -f wannier90
```

## Documentation

- Online docs: https://deeptb.readthedocs.io
- Papers:
  - DeePTB-SK: [Nat Commun 15, 6772 (2024)](https://doi.org/10.1038/s41467-024-51006-4)
  - DeePTB-E3: [ICLR 2025 Spotlight](https://openreview.net/forum?id=kpq3IIjUD3)
