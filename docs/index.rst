.. DeePTB documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=================================================
DeePTB Documentation
=================================================

**DeePTB** is a Python package that adopts the deep learning method to construct electronic tight-binding (TB) Hamiltonians using a minimal basis.
With a neural network environmental correction scheme, **DeePTB** can efficiently predict TB Hamiltonians for large-size unseen structures with *ab initio* accuracy after training with *ab initio* eigenvalues from smaller sizes. 
This feature enables efficient simulations of large-size systems under structural perturbations such as strain, which is crucial for semiconductor band gap engineering. Furthermore, DeePTB offers the ability to perform efficient and accurate finite temperature simulations, incorporating both atomic and electronic behaviour through the integration of molecular dynamics (MD). Another significant advantage is that using eigenvalues as the training labels makes DeePTB much more flexible and independent of the choice of various bases (PW or LCAO) and the exchange-correlation (XC) functionals (LDA, GGA and even HSE) used in preparing the training labels. In addition, **DeePTB** can handle systems with strong spin-orbit coupling (SOC) effects.
These capabilities make **DeePTB** adaptable to various research scenarios, extending its applicability to a wide range of materials and phenomena and offering a powerful and versatile tool for accurate and efficient simulations.

.. toctree::
   :maxdepth: 2
   :caption: Quick Start

   quick_start/easy_install
   quick_start/input
   quick_start/hands_on
   quick_start/basic_api


.. toctree::
   :maxdepth: 2
   :caption: INPUT TAG
   
   input_params/index


.. toctree::
   :maxdepth: 2
   :caption: Advanced

   advanced/dptb_env
   advanced/soc
   advanced/nrl_tb
   advanced/dftb
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

.. toctree::
   :glob:
   :titlesonly:

   community/faq
