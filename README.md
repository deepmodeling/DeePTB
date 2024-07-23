# DeePTB

[![DeepModeling](https://img.shields.io/badge/DeepModeling-Incubating_Project-blue)](https://github.com/deepmodeling)
[![Build](https://github.com/deepmodeling/DeePTB/actions/workflows/image.yml/badge.svg)](https://github.com/deepmodeling/DeePTB/actions/workflows/image.yml)
[![Test](https://github.com/deepmodeling/DeePTB/actions/workflows/unit_test.yml/badge.svg)](https://github.com/deepmodeling/DeePTB/actions/workflows/unit_test.yml)

## About DeePTB

**DeePTB** is an innovative Python package that employs deep learning to construct electronic tight-binding (TB) Hamiltonians with a minimal basis. It is designed to:

- Efficiently predict TB Hamiltonians for large, unseen structures based on training with smaller ones.
- Enable simulations of large systems under structural perturbations, finite temperature simulations integrating molecular dynamics (MD) for comprehensive atomic and electronic behavior.
- Support customizable Slater-Koster parameterization with neural network incorporation for local environmental corrections. 
- Operate independently of the choice of bases and exchange-correlation functionals, offering flexibility and adaptability.
- Handle systems with strong spin-orbit coupling (SOC) effects.

**DeePTB** is a versatile tool adaptable for a wide range of materials and phenomena, providing accurate and efficient simulations. See more details in our DeePTB paper: [arXiv:2307.04638](http://arxiv.org/abs/2307.04638)

## Installation

Installing **DeePTB** is straightforward. We recommend using a virtual environment for dependency management.

### Requirements
- Cmake 3.17 or later.
- Python 3.8 or later.
- Torch 1.13.0 or later ([PyTorch Installation](https://pytorch.org/get-started/locally)).
- ifermi (optional, for 3D fermi-surface plotting).

### Installation Steps

#### Using PyPi
1. Ensure you have Python 3.8 or later and Torch installed.
2. Install DeePTB with pip:
   ```bash
   pip install dptb
   ```

#### From Source
1. Clone the repository:
   ```bash
   git clone https://github.com/deepmodeling/DeePTB.git
   ```
2. Navigate to the root directory and install DeePTB:
   ```bash
   cd DeePTB
   pip install .
   ```

### Install optical response module
1. Ensure you have the intel MKL library installed.

2. Clone the repository:
   ```bash
   git clone https://github.com/deepmodeling/DeePTB.git
   ```
3. Navigate to the root directory and install DeePTB:
   ```bash
   cd DeePTB
   BUILD_FORTRAN=ON USE_INTEL=ON USE_OPENMP=ON pip install .
   ```
   Note: by default, the optical response module is not installed. To install it, you need to set the `BUILD_FORTRAN=ON` flag. The `USE_INTEL=ON` flag is used to enable the Intel MKL library, and the `USE_OPENMP=ON` flag is used to enable OpenMP parallelization. Both `USE_INTEL` and `USE_OPENMP` are `ON` by default.

## Usage
For a comprehensive guide and usage tutorials, visit [our documentation website](https://deeptb.readthedocs.io/en/latest/).



## Community

**DeePTB** joins the DeepModeling community, a community devoted of AI for science, as an incubating level project. To learn more about the DeepModeling community, see the [introduction of community](https://github.com/deepmodeling/community).

## Contributing
We welcome contributions to **DeePTB**. Please refer to our [contributing guidelines](https://deeptb.readthedocs.io/en/latest/community/contribution_guide.html) for details.


## How to Cite

When utilizing the DeePTB package in your research, we request that you cite the following reference:

```text
Gu, Qiangqiang, et al. "DeePTB: A deep learning-based tight-binding approach with ab initio accuracy." arXiv preprint arXiv:2307.04638 (2023).
```

## Full Dependencies
- python = ">=3.8"
- pytest = ">=7.2.0"
- pytest-order = "1.2.0"
- numpy = "*"
- scipy = "1.9.1"
- spglib = "*"
- matplotlib = "*"
- torch = ">=1.13.0"
- ase = "*"
- pyyaml = "*"
- future = "*"
- dargs = "0.4.4"
- xitorch = "0.3.0"
- fmm3dpy = "1.0.0"
- e3nn = ">=0.5.1"
- torch-runstats = "0.2.0"
- torch_scatter = "2.1.2"
- torch_geometric = ">=2.4.0"
- opt-einsum = "3.3.0"
- h5py = "3.7.0"
- lmdb = "1.4.1"

