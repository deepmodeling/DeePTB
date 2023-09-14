.. ABACUS documentation master file, created by
   sphinx-quickstart on Fri Mar 11 10:42:27 2022.
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
   quick_start/hands_on
   quick_start/input

.. toctree::
   :maxdepth: 2
   :caption: Advanced

   advanced/install
   advanced/scf/index
   advanced/pp_orb
   advanced/opt
   advanced/md
   advanced/acceleration/index
   advanced/elec_properties/index
   advanced/interface/index
   advanced/input_files/index

.. toctree::
   :maxdepth: 2
   :caption: Citing ABACUS

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
