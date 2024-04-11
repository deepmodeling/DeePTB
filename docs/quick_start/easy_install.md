# Installation

This guide helps you install DeePTB with basic features. We recommend building DeePTB using **virtual environment** to avoid dependency issues by

```python
python -m venv /path/to/new/virtual/environment
``` 
See more details for virtual environment [here](https://docs.python.org/3/tutorial/venv.html).

## From Source
If you are installing from source, you will need:

- Python 3.8 or later
- torch 1.13.0 or later, following the instruction on [PyTorch: Get Started](https://pytorch.org/get-started/locally) if GPU support is required, otherwise this can be installed with the building procedure.
- ifermi (optional, install only when 3D fermi-surface plotting is needed.)

First clone or download the source code from the website.
```bash
git clone https://github.com/deepmodeling/DeePTB.git
```
Then, locate in the repository root and simply running 
```bash
cd path/deeptb
pip install .
```

## From Pypi

DeePTB is available on PyPi and Conda. 
First make sure you have python 3.8 or later installed. 
Then make sure you have installed torch

If so, you can install DeePTB by running
```bash
pip install dptb
```
