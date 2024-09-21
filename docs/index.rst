.. DeePTB documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=================================================
DeePTB Documentation
=================================================

DeePTB is an innovative Python package that uses deep learning to accelerate *ab initio* electronic structure simulations. It offers versatile, accurate, and efficient simulations for a wide range of materials and phenomena. Trained on small systems, DeePTB can predict electronic structures of large systems, handle structural perturbations, and integrate with molecular dynamics for finite temperature simulations, providing comprehensive insights into atomic and electronic behavior.

--------------
Key Features:
--------------

DeePTB contains two main components: 

1. **DeePTB-SK**: deep learning based local environment dependent Slater-Koster TB.

   - Customizable Slater-Koster parameterization with neural network corrections.
   - Flexible basis and exchange-correlation functional choices.
   - Handle systems with strong spin-orbit coupling (SOC) effects.

2. **DeePTB-E3**: E3-equivariant neural networks for representing quantum operators.

   - Construct DFT Hamiltonians/density and overlap matrices under full LCAO basis.
   - Utilize (**S**\ trictly) **L**\ ocalized **E**\ quivariant **M**\ essage-passing (**(S)LEM**) model for high data-efficiency and accuracy.
   - Employs SO(2) convolution for efficient handling of higher-order orbitals in LCAO basis.


For more details, see our papers:

* `DeePTB-SK: Nat Commun 15, 6772 (2024) <https://doi.org/10.1038/s41467-024-51006-4>`_
* `DeePTB-E3: arXiv:2407.06053 <https://arxiv.org/pdf/2407.06053>`_

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
   advanced/e3tb/index
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

