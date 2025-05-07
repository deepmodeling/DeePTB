# Installation Guide

This guide will help you install DeePTB, a Python package that utilizes deep learning to construct electronic tight-binding Hamiltonians.

## Prerequisites

Before installing DeePTB, ensure you have the following prerequisites:
  - Git
  - Python 3.9 to 3.12.
  - Torch 2.0.0 to 2.5.1 ([PyTorch Installation](https://pytorch.org/get-started/locally)).
  - ifermi (optional, for 3D fermi-surface plotting).
  - TBPLaS (optional).

## Installation Methods



### From Source
  
Highly recommended to install DeePTB from source to get the latest features and bug fixes.
1. **Setup Python environment**:
    
    Using conda (recommended, python >=3.9, <=3.12 ), e.g.,
    ```bash
    conda create -n dptb_venv python=3.10
    conda activate dptb_venv
    ```
    or using venv (make sure python >=3.9,<=3.12)
    ```bash
    python -m venv dptb_venv
    source dptb_venv/bin/activate
    ```
2. **Clone DeePTB and  Navigate to the root directory**:
    ```bash
    git clone https://github.com/deepmodeling/DeePTB.git
    cd DeePTB
    ```
3. **Install `torch`**:
    ```bash
    pip install "torch>=2.0.0,<=2.5.0"
    ```
4. **Install `torch-scatter`** (two ways):
    - **Recommended**: Install torch and torch-scatter using the following commands:
        ```bash
         python docs/auto_install_torch_scatter.py
        ```
    - **Manual**: Install torch and torch-scatter manually:
        ```bash
        pip install torch-scatter -f https://data.pyg.org/whl/torch-${version}+${CUDA}.html
        ```
        where `${version}` is the version of torch, e.g., 2.5.0, and `${CUDA}` is the CUDA version, e.g., cpu, cu118, cu121, cu124. See [torch_scatter doc](https://github.com/rusty1s/pytorch_scatter) for more details.   

5. **Install DeePTB**:   
    ```bash
    pip install .
    ```
    
### From PyPi

1. Install PyTorch first by following the instructions on [PyTorch: Get Started](https://pytorch.org/get-started/locally).
2. Install DeePTB using pip:
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

