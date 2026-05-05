import logging
from typing import List, Tuple

import numpy as np
from ase.atoms import Atoms

log = logging.getLogger(__name__)


def is_integer(num):
    return abs(round(num) - num) < 1e-10


def calculate_reciprocal_vectors(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate reciprocal lattice vectors."""
    vol = np.dot(a, np.cross(b, c))
    assert np.abs(vol) > 1e-10
    rmat = np.vstack([x.reshape((1, 3)) for x in [a, b, c]])
    gmat = np.linalg.solve(rmat, np.eye(3)).T * 2 * np.pi
    return gmat[0], gmat[1], gmat[2]


def get_symm_ops(atoms: Atoms) -> List[np.ndarray]:
    """Get rotational symmetry operations for an atomic structure."""
    from spglib import get_symmetry as _spglib_get_symmetry

    latt = atoms.get_cell()
    coords = atoms.get_positions()
    type_map = atoms.get_atomic_numbers()
    return [op for op in _spglib_get_symmetry((latt, coords, type_map))["rotations"]]


def rot_revlatt_2D(rev_latt, index=[0, 1]):  # 0, x; 1,y, 2,z
    """Transform reciprocal lattice vectors to a 2D-oriented coordinate system."""
    rev_latt = np.asarray(rev_latt)
    if rev_latt.shape != (3, 3):
        log.error("Error! rev_latt must be a 3x3 array!")
        raise ValueError

    index_left = [0, 1, 2]
    for i in index:
        index_left.remove(i)

    vec1 = np.array(rev_latt[index[0]]).reshape(-1)
    vec2 = np.array(rev_latt[index[1]]).reshape(-1)

    avec1 = vec1 / np.linalg.norm(vec1)
    avec3 = np.cross(avec1, vec2) / np.linalg.norm(np.cross(avec1, vec2))
    avec2 = np.cross(avec3, avec1)
    if np.dot(np.cross(avec1, avec2), avec3) < 0:
        avec3 = -avec3

    newcorr = np.zeros((3, 3))
    newcorr[index[0]] = avec1
    newcorr[index[1]] = avec2
    newcorr[index_left[0]] = avec3

    rev_latt_new = rev_latt @ np.linalg.inv(newcorr)
    return rev_latt_new, newcorr
