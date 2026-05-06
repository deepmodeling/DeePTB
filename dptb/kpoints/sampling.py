"""High-level k-point sampling entry points."""

import logging
from typing import List, Optional, Tuple

import numpy as np
from ase.atoms import Atoms

from dptb.kpoints.geometry import get_symm_ops
from dptb.kpoints.mesh import monkhorst_pack_sampling
from dptb.kpoints.reduction import reduce

log = logging.getLogger(__name__)


def sample(
    structure: Atoms,
    scheme: str = "mp",
    meshgrid: Optional[List[int]] = None,
    meshspacing: Optional[List[float]] = None,
    gamma_centered: bool = True,
    rotational_symmetry: bool = True,
    time_inversion_symmetry: bool = True,
    symm_prec: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample and optionally symmetry-reduce k-points for a crystal structure."""
    if not isinstance(structure, Atoms):
        raise TypeError("structure must be an ase.Atoms instance.")
    if scheme != "mp":
        raise ValueError("Only scheme='mp' is supported.")
    if meshgrid is not None:
        if not isinstance(meshgrid, (list, tuple)) or len(meshgrid) != 3:
            raise ValueError("meshgrid must be a list or tuple of 3 positive integers.")
    if meshspacing is not None:
        if not isinstance(meshspacing, (list, tuple)) or len(meshspacing) != 3:
            raise ValueError("meshspacing must be a list or tuple of 3 positive values.")
    if not np.isfinite(symm_prec) or symm_prec <= 0:
        raise ValueError("symm_prec must be positive and finite.")
    if meshgrid is not None and meshspacing is not None:
        log.warning("Both meshgrid and meshspacing are provided. meshspacing will take precedence.")

    nk1, nk2, nk3 = 1, 1, 1
    if meshgrid is not None and meshspacing is None:
        nk1, nk2, nk3 = meshgrid

    symm_ops = get_symm_ops(structure) if rotational_symmetry else [np.eye(3)]
    k, wk = reduce(
        monkhorst_pack_sampling(
            nk1,
            nk2,
            nk3,
            structure.get_cell(),
            meshspacing,
            gamma_centered=gamma_centered,
            direct=True,
        ),
        symm_op=symm_ops,
        time_inversion_symmetry=time_inversion_symmetry,
        symm_prec=symm_prec,
    )
    return k, wk / np.sum(wk)
