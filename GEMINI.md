# GEMINI.md

## Project Overview

This project, DeePTB, is a Python package that uses deep learning to accelerate *ab initio* electronic structure simulations. It provides a command-line interface to train, test, and run deep learning-based tight-binding models. The project is built using Python, PyTorch, and other scientific computing libraries. It uses `poetry` for dependency management.

The project has two main components:
1.  **DeePTB-SK**: deep learning based local environment dependent Slater-Koster TB.
2.  **DeePTB-E3**: E3-equivariant neural networks for representing quantum operators.


## Building and Running

### Installation

The project uses `poetry` for dependency management. To install the dependencies, you can follow the instructions in the `README.md`. A virtual environment is recommended.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/deepmodeling/DeePTB.git
    cd DeePTB
    ```

2.  **Create and activate a virtual environment (e.g., with conda):**
    ```bash
    conda create -n dptb_venv python=3.10
    conda activate dptb_venv
    ```

3.  **Install dependencies:**
    ```bash
    pip install "torch>=2.0.0,<=2.5.0"
    python docs/auto_install_torch_scatter.py
    pip install .
    ```

### Running the application

The project provides a command-line interface through the `dptb` command. Here are the available subcommands:

*   `dptb config`: Generate a config template.
*   `dptb bond`: Bond distance analysis.
*   `dptb train`: Train a model.
*   `dptb test`: Test a model.
*   `dptb run`: Run a TB model.
*   `dptb n2j`: Convert NRL file to json.
*   `dptb p2j`: Convert PTH ckpt to json.
*   `dptb data`: Preprocess data.
*   `dptb cskf`: Collect SK parameters from sk files.
*   `dptb skf2nn`: Convert sk files to nn-sk TB model.
*   `dptb esk`: Generate initial empirical SK parameters.

For more details on each command, you can use the `-h` or `--help` flag, for example `dptb train -h`.

### Testing

To run the tests, use the following command:

```bash
pytest ./dptb/tests/
```

## Development Conventions

*   The project uses `poetry` for dependency management.
*   The code is structured into a main `dptb` package with subpackages for different functionalities like `data`, `nn`, `negf`, etc.
*   Entry points for the command-line interface are located in the `dptb/entrypoints` directory.
*   Tests are located in the `dptb/tests` directory.

## Directory Structure

```
├───dptb/                  # Main source code
│   ├───__main__.py        # Main entry point for the CLI
│   ├───entrypoints/       # Subcommands for the CLI
│   ├───data/              # Data loading and processing
│   ├───nn/                # Neural network models
│   ├───negf/              # NEGF calculations
│   └───tests/             # Unit tests
├───docs/                  # Documentation
├───examples/              # Example usage
├───pyproject.toml         # Project metadata and dependencies
├───README.md              # Project overview
└───ut.sh                  # Test script
```
