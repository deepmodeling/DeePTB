# Installation Guide

This guide will help you install DeePTB, a Python package that utilizes deep learning to construct electronic tight-binding Hamiltonians.

## Prerequisites

Before installing DeePTB, ensure you have the following prerequisites:

- Python 3.8 or later.
- PyTorch 1.13.0 or later.
- Git (to clone the repository).

### Optional
- ifermi (for 3D fermi-surface plotting).

## Virtual Environment (Recommended)

We recommend using a virtual environment to manage dependencies. Create one with the following command:

```bash
python -m venv /path/to/new/virtual/environment
```

Activate the virtual environment:

- On macOS and Linux:
  ```bash
  source /path/to/new/virtual/environment/bin/activate
  ```
- On Windows:
  ```bash
  \path\to\new\virtual\environment\Scripts\activate
  ```

For more details on virtual environments, see the [Python documentation](https://docs.python.org/3/tutorial/venv.html).

## Installation Methods

### From PyPi

1. Install PyTorch first by following the instructions on [PyTorch: Get Started](https://pytorch.org/get-started/locally).
2. Install DeePTB using pip:
   ```bash
   pip install dptb
   ```

### From Source

1. Clone the DeePTB repository:
   ```bash
   git clone https://github.com/deepmodeling/DeePTB.git
   ```
2. Change to the repository directory:
   ```bash
   cd DeePTB
   ```
3. Install DeePTB and its dependencies:
   ```bash
   pip install .
   ```


### Additional Tips

- Keep your DeePTB installation up-to-date by pulling the latest changes from the repository and re-installing.
- If you encounter any issues during installation, consult the [DeePTB documentation](https://deeptb.readthedocs.io/en/latest/) or seek help from the community.

## Contributing

We welcome contributions to DeePTB. If you are interested in contributing, please read our [contributing guidelines](https://deeptb.readthedocs.io/en/latest/community/contribution_guide.html).

## License

DeePTB is open-source software released under the [LGPL-3.0](https://github.com/deepmodeling/DeePTB/blob/main/LICENSE) provided in the repository.

