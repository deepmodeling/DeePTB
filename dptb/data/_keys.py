"""Keys for dictionaries/AtomicData objects.

This is a seperate module to compensate for a TorchScript bug that can only recognize constants when they are accessed as attributes of an imported module.
"""

import sys
from typing import List

if sys.version_info[1] >= 8:
    from typing import Final
else:
    from typing_extensions import Final

# == Define allowed keys as constants ==
# The positions of the atoms in the system
POSITIONS_KEY: Final[str] = "pos"
# The [2, n_edge] index tensor giving center -> neighbor relations
EDGE_INDEX_KEY: Final[str] = "edge_index"
# The [2, n_env] index tensor giving center -> neighbor relations
ENV_INDEX_KEY: Final[str] = "env_index"
# The [2, n_onsitenv] index tensor giving center -> neighbor relations
ONSITENV_INDEX_KEY: Final[str] = "onsitenv_index"
# A [n_edge, 3] tensor of how many periodic cells each env crosses in each cell vector
ENV_CELL_SHIFT_KEY: Final[str] = "env_cell_shift"
# A [n_edge, 3] tensor of how many periodic cells each edge crosses in each cell vector
EDGE_CELL_SHIFT_KEY: Final[str] = "edge_cell_shift"
# [n_batch, 3, 3] or [3, 3] tensor where rows are the cell vectors
ONSITENV_CELL_SHIFT_KEY: Final[str] = "onsitenv_cell_shift"
# [n_batch, 3, 3] or [3, 3] tensor where rows are the cell vectors
CELL_KEY: Final[str] = "cell"
# [n_kpoints, 3] or [n_batch, nkpoints, 3] tensor
KPOINT_KEY = "kpoint"

HAMILTONIAN_KEY = "hamiltonian"

OVERLAP_KEY = "overlap"
# [n_batch, 3] bool tensor
PBC_KEY: Final[str] = "pbc"
# [n_atom, 1] long tensor
ATOMIC_NUMBERS_KEY: Final[str] = "atomic_numbers"
# [n_atom, 1] long tensor
ATOM_TYPE_KEY: Final[str] = "atom_types"
# [n_batch, n_kpoint, n_orb]
ENERGY_EIGENVALUE_KEY: Final[str] = "eigenvalue"
EIGENVECTOR_KEY: Final[str] = "eigenvector"

# [n_batch, 2]
ENERGY_WINDOWS_KEY = "ewindow"
BAND_WINDOW_KEY = "bwindow"

BASIC_STRUCTURE_KEYS: Final[List[str]] = [
    POSITIONS_KEY,
    EDGE_INDEX_KEY,
    EDGE_CELL_SHIFT_KEY,
    CELL_KEY,
    PBC_KEY,
    ATOM_TYPE_KEY,
    ATOMIC_NUMBERS_KEY,
]

# A [n_edge, 3] tensor of displacement vectors associated to edges
EDGE_VECTORS_KEY: Final[str] = "edge_vectors"
# A [n_edge, 3] tensor of displacement vectors associated to envs
ENV_VECTORS_KEY: Final[str] = "env_vectors"
# A [n_edge, 3] tensor of displacement vectors associated to onsitenvs
ONSITENV_VECTORS_KEY: Final[str] = "onsitenv_vectors"
# A [n_edge] tensor of the lengths of EDGE_VECTORS
EDGE_LENGTH_KEY: Final[str] = "edge_lengths"
# A [n_edge] tensor of the lengths of ENV_VECTORS
ENV_LENGTH_KEY: Final[str] = "env_lengths"
# A [n_edge] tensor of the lengths of ONSITENV_VECTORS
ONSITENV_LENGTH_KEY: Final[str] = "onsitenv_lengths"
# [n_edge, dim] (possibly equivariant) attributes of each edge
EDGE_ATTRS_KEY: Final[str] = "edge_attrs"
ENV_ATTRS_KEY: Final[str] = "env_attrs"
ONSITENV_ATTRS_KEY: Final[str] = "onsitenv_attrs"
# [n_edge, dim] invariant embedding of the edges
EDGE_EMBEDDING_KEY: Final[str] = "edge_embedding"
ENV_EMBEDDING_KEY: Final[str] = "env_embedding"
ONSITENV_EMBEDDING_KEY: Final[str] = "onsitenv_embedding"
EDGE_FEATURES_KEY: Final[str] = "edge_features"
ENV_FEATURES_KEY: Final[str] = "env_features"
ONSITENV_FEATURES_KEY: Final[str] = "onsitenv_features"
# [n_edge, 1] invariant of the radial cutoff envelope for each edge, allows reuse of cutoff envelopes
EDGE_CUTOFF_KEY: Final[str] = "edge_cutoff"
# [n_edge, 1] invariant of the radial cutoff envelope for each env edge, allows reuse of cutoff envelopes
ENV_CUTOFF_KEY: Final[str] = "env_cutoff"
# [n_edge, 1] invariant of the radial cutoff envelope for each onsitenv edge, allows reuse of cutoff envelopes
ONSITENV_CUTOFF_KEY: Final[str] = "onsitenv_cutoff"
# edge energy as in Allegro
EDGE_ENERGY_KEY: Final[str] = "edge_energy"
EDGE_OVERLAP_KEY: Final[str] = "edge_overlap"
NODE_OVERLAP_KEY: Final[str] = "node_overlap"
EDGE_HAMILTONIAN_KEY: Final[str] = "edge_hamiltonian"
NODE_HAMILTONIAN_KEY: Final[str] = "node_hamiltonian"

NODE_FEATURES_KEY: Final[str] = "node_features"
NODE_ATTRS_KEY: Final[str] = "node_attrs"
EDGE_TYPE_KEY: Final[str] = "edge_type"

# SOC keys
NODE_SOC_KEY: Final[str] = "node_soc"
NODE_SOC_SWITCH_KEY: Final[str] = "node_soc_switch"

PER_ATOM_ENERGY_KEY: Final[str] = "atomic_energy"
TOTAL_ENERGY_KEY: Final[str] = "total_energy"
FORCE_KEY: Final[str] = "forces"
PARTIAL_FORCE_KEY: Final[str] = "partial_forces"
STRESS_KEY: Final[str] = "stress"
VIRIAL_KEY: Final[str] = "virial"

ALL_ENERGY_KEYS: Final[List[str]] = [
    EDGE_ENERGY_KEY,
    PER_ATOM_ENERGY_KEY,
    TOTAL_ENERGY_KEY,
    FORCE_KEY,
    PARTIAL_FORCE_KEY,
    STRESS_KEY,
    VIRIAL_KEY,
]

BATCH_KEY: Final[str] = "batch"
BATCH_PTR_KEY: Final[str] = "ptr"

# Make a list of allowed keys
ALLOWED_KEYS: List[str] = [
    getattr(sys.modules[__name__], k)
    for k in sys.modules[__name__].__dict__.keys()
    if k.endswith("_KEY")
]
