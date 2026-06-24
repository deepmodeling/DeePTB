# Installation Guide

This guide will help you install DeePTB, a Python package that utilizes deep learning to construct electronic tight-binding Hamiltonians.

## Prerequisites

Before installing DeePTB, ensure you have the following prerequisites:
  - Git
  - Python 3.10 to 3.13.
  - UV, the recommended installer frontend.
  - For GPU installs, an NVIDIA driver compatible with the selected CUDA runtime.
  - ifermi (optional, for 3D fermi-surface plotting).
  - TBPLaS (optional).

## Installation Methods



### From Source

Highly recommended to install DeePTB from source to get the latest features and bug fixes.
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
    CUDA/backend combinations instead of falling back to source builds.

4. **Install optional extras**:
    ```bash
    ./install.sh auto --extra 3Dfermi
    ./install.sh auto --extra tbtrans_init
    ./install.sh auto --extra pybinding
    ```

### From PyPi

For new machines, source installation with `install.sh` is recommended. PyPI
installation is suitable only when a compatible PyTorch and `torch-scatter`
binary wheel are already installed.

1. Install PyTorch and `torch-scatter` matching your CPU/GPU backend.
2. Install DeePTB:
   ```bash
   pip install dptb
   ```

### Additional Tips

- Keep your DeePTB installation up-to-date by pulling the latest changes from the repository and re-installing.
- If you encounter any issues during installation, consult the [DeePTB documentation](https://deeptb.readthedocs.io/en/latest/) or seek help from the community.

## Contributing

We welcome contributions to DeePTB. If you are interested in contributing, please read our [contributing guidelines](https://deeptb.readthedocs.io/en/latest/community/contribution_guide.html).

## License

DeePTB is open-source software released under the [LGPL-3.0](https://github.com/deepmodeling/DeePTB/blob/main/LICENSE) provided in the repository.
