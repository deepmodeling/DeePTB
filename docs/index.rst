.. DeePTB documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=================================================
DeePTB Documentation
=================================================

.. **DeePTB** is a Python package that adopts the deep learning method to construct electronic tight-binding (TB) Hamiltonians using a minimal basis.
.. With a neural network environmental correction scheme, **DeePTB** can efficiently predict TB Hamiltonians for large-size unseen structures with *ab initio* accuracy after training with *ab initio* eigenvalues from smaller sizes. 
.. This feature enables efficient simulations of large-size systems under structural perturbations such as strain, which is crucial for semiconductor band gap engineering. Furthermore, DeePTB offers the ability to perform efficient and accurate finite temperature simulations, incorporating both atomic and electronic behaviour through the integration of molecular dynamics (MD). Another significant advantage is that using eigenvalues as the training labels makes DeePTB much more flexible and independent of the choice of various bases (PW or LCAO) and the exchange-correlation (XC) functionals (LDA, GGA and even HSE) used in preparing the training labels. In addition, **DeePTB** can handle systems with strong spin-orbit coupling (SOC) effects.
.. These capabilities make **DeePTB** adaptable to various research scenarios, extending its applicability to a wide range of materials and phenomena and offering a powerful and versatile tool for accurate and efficient simulations.
**DeePTB** is an innovative Python package that employs deep learning to construct electronic Hamiltonians using a minimal basis Slater-Koster TB(**SKTB**), and full LCAO basis using E3-equivariant neural networks (**E3TB**). It is designed to:

- Efficiently predict TB/LCAO Hamiltonians for large, unseen structures based on training with smaller ones.
- Enable simulations of large systems under structural perturbations, finite temperature simulations integrating molecular dynamics (MD) for comprehensive atomic and electronic behavior.

For **SKTB**:
- Support customizable Slater-Koster parameterization with neural network incorporation for local environmental corrections. 
- Operate independently of the choice of bases and exchange-correlation functionals, offering flexibility and adaptability.
- Handle systems with strong spin-orbit coupling (SOC) effects.

For **E3TB**:
- Support constructing DFT Hamiltonians/density and overlap matrices under full LCAO basis.
- Utilize strictly local and semi-local E3-equivariant neural networks to achieve high data-efficiency and accuracy.
- Speed up via SO(2)convolution to support LCAO basis containing f and g orbitals.

**DeePTB** is a versatile tool adaptable for a wide range of materials and phenomena, providing accurate and efficient simulations. See more details in our DeePTB paper: [SKTB: arXiv:2307.04638](http://arxiv.org/abs/2307.04638), [E3TB: arXiv:2407.06053](https://arxiv.org/pdf/2407.06053)

.. toctree::
   :maxdepth: 2
   :caption: Quick Start
   

   quick_start/easy_install
   quick_start/input
   quick_start/hands_on/index
   quick_start/basic_api


.. toctree::
   :maxdepth: 2
   :caption: INPUT TAG
   
   input_params/index


.. toctree::
   :maxdepth: 2
   :caption: Advanced
   
   advanced/sktb/index
   advanced/elec_properties/index
   advanced/interface/index

.. toctree::
   :maxdepth: 2
   :caption: Citing DeePTB

   CITATIONS

.. toctree::
   :maxdepth: 2
   :caption: Developing Team

   DevelopingTeam

.. toctree::
   :maxdepth: 2
   :caption: Community

   community/contribution_guide
   CONTRIBUTING

