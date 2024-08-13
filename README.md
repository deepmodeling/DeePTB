# DeePTB

[![DeepModeling](https://img.shields.io/badge/DeepModeling-Incubating_Project-blue)](https://github.com/deepmodeling)
[![Build](https://github.com/deepmodeling/DeePTB/actions/workflows/image.yml/badge.svg)](https://github.com/deepmodeling/DeePTB/actions/workflows/image.yml)
[![Test](https://github.com/deepmodeling/DeePTB/actions/workflows/unit_test.yml/badge.svg)](https://github.com/deepmodeling/DeePTB/actions/workflows/unit_test.yml)

## About DeePTB
DeePTB is an innovative Python package that uses deep learning to accelerate *ab initio* electronic structure simulations. It offers versatile, accurate, and efficient simulations for a wide range of materials and phenomena. Trained on small systems, DeePTB can predict electronic structures of large systems, handle structural perturbations, and integrate with molecular dynamics for finite temperature simulations, providing comprehensive insights into atomic and electronic behavior.


#### Key Features:
DeePTB contains two main components: 
1. **DeePTB-SK**: deep learning based local environment dependent Slater-Koster TB.
   - Customizable Slater-Koster parameterization with neural network corrections for . 
   - Flexible basis and exchange-correlation functional choices.
   - Handle systems with strong spin-orbit coupling (SOC) effects.

1. **DeePTB-E3**: E3-equivariant neural networks for representing quantum operators.
   - Construct DFT Hamiltonians/density and overlap matrices under full LCAO basis.
   - Utilize (**S**trictly) **L**ocalized **E**quivariant **M**essage-passing (**(S)LEM**) model for high data-efficiency and accuracy.
   - Employs SO(2) convolution for efficient handling of higher-order orbitals in LCAO basis.


For more details, see our papers:
- [DeePTB-SK: Nat Commun 15, 6772 (2024)](https://doi.org/10.1038/s41467-024-51006-4)
- [DeePTB-E3: arXiv:2407.06053](https://arxiv.org/pdf/2407.06053)



## Installation

Installing **DeePTB** is straightforward. We recommend using a virtual environment for dependency management.

### Requirements
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

## Usage
For a comprehensive guide and usage tutorials, visit [our documentation website](https://deeptb.readthedocs.io/en/latest/).



## Community

**DeePTB** joins the DeepModeling community, a community devoted of AI for science, as an incubating level project. To learn more about the DeepModeling community, see the [introduction of community](https://github.com/deepmodeling/community).

## Contributing
We welcome contributions to **DeePTB**. Please refer to our [contributing guidelines](https://deeptb.readthedocs.io/en/latest/community/contribution_guide.html) for details.


## How to Cite

The following references are required to be cited when using DeePTB. Specifically:

- **For DeePTB-SK:**
    [1] Q. Gu, Z. Zhouyin, S. K. Pandey, P. Zhang, L. Zhang, and W. E, Deep Learning Tight-Binding Approach for Large-Scale Electronic Simulations at Finite Temperatures with Ab Initio Accuracy, Nat Commun 15, 6772 (2024).
    ```latex
    @article{guDeep2024,
      title = {Deep Learning Tight-Binding Approach for Large-Scale Electronic Simulations at Finite Temperatures with  Ab Initio Accuracy},
      author = {Gu, Qiangqiang and Zhouyin, Zhanghao and Pandey, Shishir Kumar and Zhang, Peng and Zhang, Linfeng and E,    Weinan},
      year = {2024},
      month = aug,
      journal = {Nature Communications},
      volume = {15},
      number = {1},
      pages = {6772},
      publisher = {Nature Publishing Group},
      issn = {2041-1723},
      doi = {10.1038/s41467-024-51006-4},
      copyright = {2024 The Author(s)},
      keywords = {Computational methods,Electronic properties and materials,Electronic structure}
    }
    ```
- **For DeePTB-E3:**
    [2] Z. Zhouyin, Z. Gan, S. K. Pandey, L. Zhang, and Q. Gu, Learning Local Equivariant Representations for Quantum Operators, arXiv:2407.06053.
    

    ```latex
    @misc{zhouyinLearning2024,
      title = {Learning Local Equivariant Representations for Quantum Operators},
      author = {Zhouyin, Zhanghao and Gan, Zixi and Pandey, Shishir Kumar and Zhang, Linfeng and Gu, Qiangqiang},
      year = {2024},
      month = jul,
      number = {arXiv:2407.06053},
      eprint = {2407.06053},
      primaryclass = {cond-mat, physics:quant-ph},
      publisher = {arXiv},
      doi = {10.48550/arXiv.2407.06053},
      archiveprefix = {arXiv},
      keywords = {Computer Science - Machine Learning,Condensed Matter - Materials Science,Quantum Physics},
    }
    ```