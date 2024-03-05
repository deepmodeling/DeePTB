# DeePTB

[![DeepModeling](https://img.shields.io/badge/DeepModeling-Incubating_Project-blue)](https://github.com/deepmodeling)
[![Build](https://github.com/deepmodeling/DeePTB/actions/workflows/image.yml/badge.svg)](https://github.com/deepmodeling/DeePTB/actions/workflows/image.yml)
[![Test](https://github.com/deepmodeling/DeePTB/actions/workflows/unit_test.yml/badge.svg)](https://github.com/deepmodeling/DeePTB/actions/workflows/unit_test.yml)

## About DeePTB

**DeePTB** is a Python package that adopts the deep learning method to construct electronic tight-binding (TB) Hamiltonians using a minimal basis.
Trained on smaller structures, **DeePTB** can efficiently predict TB Hamiltonians for large-size unseen structures. This feature enables efficient simulations of large-size systems under structural perturbations. Furthermore, DeePTB offers the ability to perform efficient and accurate finite temperature simulations, incorporating both atomic and electronic behavior through the integration of molecular dynamics (MD). Another significant advantage is that  **DeePTB** is independent of the choice of various bases (PW or LCAO) and the exchange-correlation (XC) functionals (LDA, GGA, and even HSE) used in preparing the training labels. In addition, **DeePTB** can handle systems with strong spin-orbit coupling (SOC) effects.
These capabilities make **DeePTB** adaptable to various research scenarios, extending its applicability to a wide range of materials and phenomena and offering a powerful and versatile tool for accurate and efficient simulations.


See more details in our DeePTB paper: [arXiv:2307.04638](http://arxiv.org/abs/2307.04638)

<!--
## Key Features:
- Slater-Koster parameterization with customizable radial dependence.
- Orthogonal basis with the customizable number of basis and bond neighbors.
- Incorporation of local environmental corrections by neural networks.
- Gradient-based fitting algorithm based on autograd implementation.
- Flexibility on bases and XC functionals used in preparing the training labels.
- Ability to handle systems with  SOC effects.
- Finite temperature simulations through integration with MD.
-->

**DeePTB** joins the DeepModeling community, a community devoted of AI for science, as an incubating level project. To learn more about the DeepModeling community, see the [introduction of community](https://github.com/deepmodeling/community).

## Online Documentation
For detailed documentation, please refer to [our documentation website](https://deeptb.readthedocs.io/en/latest/).

