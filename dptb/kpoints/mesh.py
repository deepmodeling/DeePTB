import logging
from itertools import product as itprod
from typing import List, Optional, Union

import ase
import numpy as np

from dptb.kpoints.geometry import calculate_reciprocal_vectors, is_integer

log = logging.getLogger(__name__)


def mp(
    nk1: int,
    nk2: int,
    nk3: int,
    b1: Optional[np.ndarray] = None,
    b2: Optional[np.ndarray] = None,
    b3: Optional[np.ndarray] = None,
    gamma_centered: bool = True,
    direct: bool = True,
):
    """Monkhorst-Pack sampling, including Gamma-centered and shifted grids."""
    assert is_integer(nk1) and is_integer(nk2) and is_integer(nk3)
    assert nk1 > 0 and nk2 > 0 and nk3 > 0

    if gamma_centered:
        def k_1d(n):
            k = np.arange(n) / n
            k[k >= 0.5] -= 1
            return k
    else:
        k_1d = lambda n: (np.arange(n) + 0.5) / n - 0.5

    k_taud = np.array(list(itprod(*[k_1d(n) for n in [nk1, nk2, nk3]])))
    k_taud = k_taud.reshape(-1, 3)
    assert len(k_taud) == nk1 * nk2 * nk3

    if direct:
        return k_taud

    assert all(x is not None for x in [b1, b2, b3])
    return np.tensordot(k_taud, np.array([b1, b2, b3]), axes=(1, 0))


def build_kmeshgrid(b1, b2, b3, kspac: Union[int, float, List[float]]) -> List[int]:
    """Build a k-mesh grid from reciprocal vectors and target k-spacing."""
    norms = [np.linalg.norm(x) for x in [b1, b2, b3]]
    kspac = [kspac] * 3 if isinstance(kspac, (int, float)) else kspac
    assert len(norms) == len(kspac), f"kspacing should be a list of 3 floats: {kspac}"
    norms = [int(norm / kspac) for norm, kspac in zip(norms, kspac)]
    return list(map(lambda x: max(1, x + 1), norms))


def monkhorst_pack_sampling(
    nk1: int = 1,
    nk2: int = 1,
    nk3: int = 1,
    cell: Optional[np.ndarray] = None,
    kspac: Optional[Union[int, float, List[float]]] = None,
    gamma_centered: bool = True,
    direct: bool = True,
) -> np.ndarray:
    """Sample k-points by explicit mesh or by target k-spacing."""
    b1, b2, b3 = None, None, None

    if kspac is not None:
        cell = np.asarray(cell)
        assert cell.shape in [(3, 3), (9,)]
        cell = cell.reshape((3, 3))
        b1, b2, b3 = calculate_reciprocal_vectors(cell[0], cell[1], cell[2])
        nk1, nk2, nk3 = build_kmeshgrid(b1, b2, b3, kspac)

    return mp(nk1, nk2, nk3, b1, b2, b3, gamma_centered=gamma_centered, direct=direct)


def monkhorst_pack(meshgrid=[1, 1, 1]):
    """Generate shifted Monkhorst-Pack k-points in [-0.5, 0.5)."""
    if len(meshgrid) != 3 or not (np.array(meshgrid, dtype=int) > 0).all():
        log.error("Error! meshgrid must be a list of 3 positive integers!")
        raise ValueError
    return mp(*meshgrid, gamma_centered=False)


def gamma_center(meshgrid=[1, 1, 1]):
    """Generate Gamma-centered k-points in [-0.5, 0.5)."""
    if len(meshgrid) != 3 or not (np.array(meshgrid, dtype=int) > 0).all():
        log.error("Error! meshgrid must be a list of 3 positive integers!")
        raise ValueError
    return mp(*meshgrid, gamma_centered=True)


def kmesh_sampling(meshgrid=[1, 1, 1], is_gamma_center=True):
    """Generate a Gamma-centered or shifted Monkhorst-Pack mesh."""
    return gamma_center(meshgrid) if is_gamma_center else monkhorst_pack(meshgrid)


def time_symmetry_reduce(meshgrid=[1, 1, 1], is_gamma_center=True):
    """Generate a mesh and reduce it by time-reversal symmetry."""
    from dptb.kpoints.reduction import reduce_time_inversion

    kpoints = kmesh_sampling(meshgrid, is_gamma_center=is_gamma_center)
    reduced, weights = reduce_time_inversion(kpoints)
    weights = weights / len(kpoints)
    assert abs(weights.sum() - 1.0) < 1e-5, "The sum of weight is not 1.0"
    return reduced, weights


def kmesh_sampling_negf(meshgrid=[1, 1, 1], is_gamma_center=True, is_time_reversal=True):
    """Generate k-points for NEGF calculations."""
    if is_time_reversal:
        return time_symmetry_reduce(meshgrid, is_gamma_center=is_gamma_center)

    kpoints = kmesh_sampling(meshgrid, is_gamma_center=is_gamma_center)
    wk = np.ones(len(kpoints)) / len(kpoints)
    return kpoints, wk


def kmesh_fs(meshgrid=[1, 1, 1]):
    """Generate endpoint-inclusive k-points for Fermi-surface calculations."""
    nx, ny, nz = meshgrid
    lx, ly, lz = np.linspace(0, 1, nx), np.linspace(0, 1, ny), np.linspace(0, 1, nz)
    xx, yy, zz = np.meshgrid(lx, ly, lz, indexing="ij")
    kgrids = np.array([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]).T
    return (lx, ly, lz), kgrids


def kgrid_spacing(structase, kspacing: float, sampling="MP"):
    """Generate k-points from a target k-spacing."""
    assert isinstance(structase, ase.Atoms)
    rev_latt = 2 * np.pi * np.linalg.inv(np.array(structase.cell)).T
    meshgrid = np.maximum(1, np.floor(np.linalg.norm(rev_latt, axis=1) / kspacing).astype(int) + 1)

    if sampling == "MP":
        kpoints = monkhorst_pack(meshgrid)
    elif sampling == "Gamma":
        kpoints = gamma_center(meshgrid)
    else:
        log.error("Error! sampling must be either 'MP' or 'Gamma'!")
        raise ValueError

    return kpoints
