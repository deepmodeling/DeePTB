# Installation Guide

This guide will help you install DeePTB, a Python package that utilizes deep learning to construct electronic tight-binding Hamiltonians.

## Prerequisites

Before installing DeePTB, ensure you have the following prerequisites:
  - Git
  - Python 3.10 to 3.13.
  - UV, used by `install.sh` as the fast installer frontend.
  - For GPU installs, an NVIDIA driver compatible with the selected CUDA runtime.
  - ifermi (optional, for 3D fermi-surface plotting).
  - TBPLaS (optional).

## Installation Methods



### Standalone install from source

Use this path when you want to run DeePTB directly after cloning this
repository. The installer creates a local `.venv` under the DeePTB repository.

1. **Install UV**:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. **Clone DeePTB and navigate to the root directory**:
    ```bash
    git clone https://github.com/deepmodeling/DeePTB.git
    cd DeePTB
    ```

3. **Install DeePTB with the tested installer**:

    CPU-only:
    ```bash
    ./install.sh cpu
    ```

    Auto-select CPU/GPU:
    ```bash
    ./install.sh
    ```

    Force a GPU wheel path:
    ```bash
    nvidia-smi
    ./install.sh gpu
    ./install.sh cu128
    ./install.sh cu130
    ```

    The installer creates `.venv` and installs a tested PyTorch / PyG /
    `torch-scatter` binary-wheel combination. It refuses unsupported Python or
    CUDA/backend combinations instead of falling back to source builds, and it
    includes the test dependencies needed for installation validation.

4. **Activate the standalone environment**:
    ```bash
    source .venv/bin/activate
    dptb --help
    ```

5. **Validate the installation**:
    DeePTB is under active development, so new installations should run the test
    suite once before production use.

    ```bash
    python -m pytest ./dptb/tests/
    ```

    For a faster local check while iterating:

    ```bash
    python -m pytest ./dptb/tests/ -m "not slow"
    ```

6. **Install optional extras**:
    ```bash
    ./install.sh auto --extra 3Dfermi
    ./install.sh auto --extra tbtrans_init
    ./install.sh auto --extra pybinding
    ```

### Library install in an existing environment

Use this path when another project imports DeePTB as a library, or when you
already manage the Python environment yourself.

1. Install the PyTorch build required by your project.
2. Install a matching `torch-scatter` binary wheel for the current PyTorch
   version and CPU/CUDA backend. If you are working from a DeePTB source
   checkout, the helper can inspect the current environment and print the
   matching PyG wheel command:
   ```bash
   python docs/auto_install_torch_scatter.py --dry-run
   python docs/auto_install_torch_scatter.py
   ```
3. Install DeePTB from the current source checkout:
   ```bash
   pip install .
   ```
   Use `pip install -e .` instead for an editable developer install.

Published package installs, such as `pip install dptb`, were not part of this
compatibility test pass; prefer a source checkout until that path is tested.

Do not rely on a source build of `torch-scatter` unless you intentionally
maintain the compiler and CUDA build environment. For direct DeePTB use on a
new machine, prefer the standalone `install.sh` path above.

### Additional Tips

- Keep your DeePTB installation up-to-date by pulling the latest changes from the repository and re-installing.
- If you encounter any issues during installation, consult the [DeePTB documentation](https://deeptb.readthedocs.io/en/latest/) or seek help from the community.

## Contributing

We welcome contributions to DeePTB. If you are interested in contributing, please read our [contributing guidelines](https://deeptb.readthedocs.io/en/latest/community/contribution_guide.html).

## License

DeePTB is open-source software released under the [LGPL-3.0](https://github.com/deepmodeling/DeePTB/blob/main/LICENSE) provided in the repository.
