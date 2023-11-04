"""This file refactor the SK and E3 Rotation in dptb/hamiltonian/transform_se3.py], it will take input of AtomicDataDict.Type
    perform rotation from irreducible matrix element / sk parameters in EDGE/NODE FEATURE, and output the atomwise/ pairwise hamiltonian
    as the new EDGE/NODE FEATURE. The rotation should also be a GNN module and speed uptable by JIT. The HR2HK should also be included here.
    """