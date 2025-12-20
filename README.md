<p align="center">
    <img src="docs/deeptb-logo-new.png" alt="DeePTB Logo" style="width: 80vw; height: auto;" />
</p>
<!-- <h1 align="center">DeePTB</h1> -->

<!--# DeePTB -->
<p align="center">
 <a href="https://github.com/deepmodeling"><img src="https://img.shields.io/badge/DeepModeling-Incubating_Project-blue" alt="DeepModeling"></a>
 <a href="https://github.com/deepmodeling/DeePTB/actions/workflows/image.yml"><img src="https://github.com/deepmodeling/DeePTB/actions/workflows/image.yml/badge.svg" alt="Build"></a>
 <a href="https://github.com/deepmodeling/DeePTB/actions/workflows/unit_test.yml"><img src="https://github.com/deepmodeling/DeePTB/actions/workflows/unit_test.yml/badge.svg" alt="Test"></a>
 <a href="https://pypi.org/project/dptb/"><img src="https://img.shields.io/pypi/v/dptb.svg" alt="PyPI version"></a>
 <a href="https://github.com/deepmodeling/DeePTB/blob/main/LICENSE"><img src="https://img.shields.io/github/license/deepmodeling/DeePTB.svg" alt="License"></a>
</p>

<!--
[![DeepModeling](https://img.shields.io/badge/DeepModeling-Incubating_Project-blue)](https://github.com/deepmodeling)
[![Build](https://github.com/deepmodeling/DeePTB/actions/workflows/image.yml/badge.svg)](https://github.com/deepmodeling/DeePTB/actions/workflows/image.yml)
[![Test](https://github.com/deepmodeling/DeePTB/actions/workflows/unit_test.yml/badge.svg)](https://github.com/deepmodeling/DeePTB/actions/workflows/unit_test.yml)
-->

## 🚀 About DeePTB
DeePTB is an innovative Python package that uses deep learning to accelerate *ab initio* electronic structure simulations. It offers versatile, accurate, and efficient simulations for a wide range of materials and phenomena. Trained on small systems, DeePTB can predict electronic structures of large systems, handle structural perturbations, and integrate with molecular dynamics for finite temperature simulations, providing comprehensive insights into atomic and electronic behavior.

- **Key Features**
DeePTB contains two main components: 
  1. **DeePTB-SK**: deep learning based local environment dependent Slater-Koster TB.
      - Customizable Slater-Koster parameterization with neural network corrections for . 
      - Flexible basis and exchange-correlation functional choices.
      - Handle systems with strong spin-orbit coupling (SOC) effects.

  2. **DeePTB-E3**: E3-equivariant neural networks for representing quantum operators.
      - Construct DFT Hamiltonians/density and overlap matrices under full LCAO basis.
      - Utilize (**S**trictly) **L**ocalized **E**quivariant **M**essage-passing (**(S)LEM**) model for high data-efficiency and accuracy.
      - Employs SO(2) convolution for efficient handling of higher-order orbitals in LCAO basis.


For more details, see our papers:
- [DeePTB-SK: Nat Commun 15, 6772 (2024)](https://doi.org/10.1038/s41467-024-51006-4)
- [DeePTB-E3: ICLR 2025 Spotlight](https://openreview.net/forum?id=kpq3IIjUD3)


## 📚 Documentation

- **Online documentation**
  
    For a comprehensive guide and usage tutorials, visit [Documentation website](https://deeptb.readthedocs.io/en/latest/).

- **Contributing**

    We welcome contributions to DeePTB. Please refer to our [contributing guidelines](https://deeptb.readthedocs.io/en/latest/community/contribution_guide.html) for details.



## 🛠️ Installation

Installing **DeePTB** is straightforward with UV, a fast Python package manager.

- **Requirements**
  - Git
  - Python 3.9 to 3.12 (UV can auto-install if needed)
  - PyTorch 2.0.0 to 2.5.1 (auto-installed by UV)

- **From Source** (Recommended)
  
  1. **Install UV** (if not already installed):
     ```bash
     # On macOS and Linux
     curl -LsSf https://astral.sh/uv/install.sh | sh
     
     # Or using pip
     pip install uv
     
     # On Windows (PowerShell)
     powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```

  2. **Clone DeePTB**:
     ```bash
     git clone https://github.com/deepmodeling/DeePTB.git
     cd DeePTB
     ```

  3. **Install DeePTB with all dependencies**:
     
     **CPU version (default)**:
     ```bash
     uv sync
     # Or use the convenience script
     ./install.sh
     ```
     
     **GPU version** (specify CUDA version via command line, no file editing needed):
     ```bash
     # Check your CUDA version first
     nvidia-smi  # Look for CUDA Version
     
     # Install with your CUDA version (examples):
     uv sync --find-links https://data.pyg.org/whl/torch-2.5.0+cu118.html  # CUDA 11.8
     uv sync --find-links https://data.pyg.org/whl/torch-2.5.0+cu121.html  # CUDA 12.1
     uv sync --find-links https://data.pyg.org/whl/torch-2.5.0+cu124.html  # CUDA 12.4
     
     # Or use the convenience script
     ./install.sh cu121  # for CUDA 12.1
     ```
     
     This single command will:
     - Automatically create a virtual environment (`.venv`)
     - Install PyTorch (>=2.0.0, <=2.5.1) with the specified variant (CPU/GPU)
     - Install torch_scatter from PyTorch Geometric index
     - Install all other dependencies
     
  4. **Install optional dependencies** (if needed):
     ```bash
     # For 3D Fermi surface plotting
     uv sync --extra 3Dfermi
     
     # For TBtrans initialization
     uv sync --extra tbtrans_init
     
     # For pybinding support
     uv sync --extra pybinding
     
     # Install all optional dependencies
     uv sync --all-extras
     ```

  5. **Run DeePTB**:
     ```bash
     # UV automatically activates the environment when using 'uv run'
     uv run dptb --help
     
     # Or activate the environment manually
     source .venv/bin/activate  # On Unix/macOS
     .venv\Scripts\activate     # On Windows
     dptb --help
     ```

- **GPU Support** (Optional)
  
  GPU installation is now built into the main installation step above! Simply use:
  ```bash
  # Check CUDA version
  nvidia-smi
  
  # Install with command line (recommended - no file editing!)
  uv sync --find-links https://data.pyg.org/whl/torch-2.5.0+cu121.html
  
  # Or use convenience script
  ./install.sh cu121
  ```
  
  See step 3 above for all available CUDA versions.

- **Easy Installation** (PyPI)
  
  > [!WARNING]
  > PyPI installation requires manual torch_scatter installation first, as torch_scatter is not available on PyPI.
  
  **For CPU**:
  ```bash
  # 1. Install torch_scatter first
  pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
  
  # 2. Install DeePTB
  pip install dptb
  ```
  
  **For GPU** (example with CUDA 12.1):
  ```bash
  # 1. Install torch with CUDA support
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  
  # 2. Install torch_scatter matching your CUDA version
  pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
  
  # 3. Install DeePTB
  pip install dptb
  ```
  
  > [!TIP]
  > For easier installation with automatic GPU/CPU detection, use the **From Source** method above instead.

## Test code 

To ensure the code is correctly installed, please run the unit tests first:
```bash
uv run pytest ./dptb/tests/
```
Be careful if not all tests pass!

## 🤝 How to Cite

The following references are required to be cited when using DeePTB. Specifically:

- **For DeePTB-SK:**

    Q. Gu, Z. Zhouyin, S. K. Pandey, P. Zhang, L. Zhang, and W. E, Deep Learning Tight-Binding Approach for Large-Scale Electronic Simulations at Finite Temperatures with Ab Initio Accuracy, Nat Commun 15, 6772 (2024).
  
- **For DeePTB-E3:**
  
    Z. Zhouyin, Z. Gan, S. K. Pandey, L. Zhang, and Q. Gu, Learning Local Equivariant Representations for Quantum Operators, In The 13th International Conference on Learning Representations (ICLR) 2025. 
