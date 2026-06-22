'''
Utils for calculating Fermi levels and occupation numbers with various smearing methods.

Usage
-----
```python
ef, occ, eband, eband_free = calculate_fermi_level(ekb, wk, ne, sigma, method=m)

nspin, nk, nb = occ.shape
```

calculate the fermi energy with given:
ekb : np.ndarray
    the energy of each k-point, shape (nspin, nk, nbnd)
wk : np.ndarray
    the weight of each k-point, shape (nk,)
ne : int
    the number of electrons
sigma : float
    the smearing width
method : str
    the smearing method, can be:
    - Gaussian: alias includes 'gau', 'gaussian'
    - Fermi-Dirac: alias includes 'fd', 'fermi-dirac', 'f-d'
    - Methfessel-Paxton: alias includes 'mp', 'methfessel-paxton', 'm-p'
    - Marzari-Vanderbilt: alias includes 'mv', 'marzari-vanderbilt', 'cold', 'm-v'

Example
-------
>>> import numpy as np
>>> from dptb.utils.occupy import calculate_fermi_level
>>> # Simple usage with Gaussian smearing
>>> ekb = np.linspace(-5, 5, 50).reshape(1, 1, 50)  # (nspin, nk, nb)
>>> wk = np.array([1.0])  # k-point weight
>>> ne = 20  # number of electrons
>>> sigma = 0.1  # smearing width (eV)
>>> ef, occ, eband, eband_free = calculate_fermi_level(ekb, wk, ne, sigma)
>>> print(f"Fermi energy: {ef:.3f} eV")
>>> print(f"Total electrons: {np.sum(occ):.1f}")
'''

import unittest
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import brentq, minimize
import logging
log = logging.getLogger(__name__)
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'san-serif'
# plt.rcParams['font.size'] = 16

def fgau(x):
    '''distribution function of Gaussian smearing, in which the x = (e - ef) / sigma
    is the smearing-normalized energy'''
    # the integration on 
    # deltagau(x) = 1 / np.sqrt(np.pi) * np.exp(-x**2)
    # results in 1/2 * (1 + erf(x))
    # but here the x is reversed for physical reason
    return 1/2 * (1 + erf(-x))

def sgau(x):
    '''entropy contribution of Gaussian smearing'''
    return 1/2/np.sqrt(np.pi) * np.exp(-x**2)

def ffd(x):
    '''distribution function of Fermi-Dirac smearing, in which the x = (e - ef) / sigma
    is the smearing-normalized energy'''
    # the integration on
    # deltafd(x) = 1 / (2*np.cosh(x) + 2)
    # results in - 1 / (1 + np.exp(x))
    # but here the x is reversed, and normalize to be in the range (0, 1)
    #
    # Use masking to avoid overflow/underflow in exp(x):
    # - For x < -40: exp(x) ~ 0, so 1/(1+exp(x)) ~ 1
    # - For x > 40: exp(x) ~ inf, so 1/(1+exp(x)) ~ 0
    # - For x in between: use expm1 for numerical stability
    x = np.asarray(x)
    mask_min = x < -40.0
    mask_max = x > 40.0
    mask_in_limit = ~(mask_min | mask_max)
    out = np.zeros_like(x, dtype=np.float64)
    out[mask_min] = 1.0
    out[mask_max] = 0.0
    out[mask_in_limit] = 1.0 / (np.expm1(x[mask_in_limit]) + 2.0)
    return out


def dffd(x):
    '''Derivative of Fermi-Dirac distribution with respect to Fermi energy.

    Computes df/dEf = (1/sigma) * f * (1 - f) = (1/sigma) * exp(x) / (1 + exp(x))^2
    where x = (E - ef) / sigma.

    Note: This returns the derivative with respect to the normalized variable x,
    so caller should divide by sigma to get df/dEf.

    Parameters
    ----------
    x : array_like
        Normalized energy (E - ef) / sigma.

    Returns
    -------
    np.ndarray
        Derivative df/dx at each energy, same shape as x.
        Multiply by (1/sigma) to get df/dEf.

    Notes
    -----
    For numerical stability:
    - When |x| > 40: f ~ 0 or 1, so df/dx ~ 0
    - In between: use stable formula exp(x) / (1 + exp(x))^2

    Reference: DFTBplus src/dftbp/dftb/etemp.F90:derivElectronCount
    '''
    x = np.asarray(x)
    mask_extreme = np.abs(x) > 40.0
    mask_in_limit = ~mask_extreme

    out = np.zeros_like(x, dtype=np.float64)

    # For values in safe range, compute exp(x) / (1 + exp(x))^2
    # This is equivalent to f * (1 - f)
    x_valid = x[mask_in_limit]
    exp_x = np.exp(x_valid)
    out[mask_in_limit] = exp_x / ((1.0 + exp_x) ** 2)

    return out

def sfd(x):
    '''entropy contribution of Fermi-Dirac smearing

    S = -f*ln(f) - (1-f)*ln(1-f)

    When f -> 0 or f -> 1, the entropy -> 0 (since x*ln(x) -> 0 as x -> 0)
    Use masking to avoid log(0) and overflow in exp(x).
    '''
    x = np.asarray(x)
    mask_min = x < -40.0  # f ~ 1, entropy ~ 0
    mask_max = x > 40.0   # f ~ 0, entropy ~ 0
    mask_in_limit = ~(mask_min | mask_max)
    out = np.zeros_like(x, dtype=np.float64)
    # Only compute for values in the valid range
    x_valid = x[mask_in_limit]
    f_valid = 1.0 / (np.expm1(x_valid) + 2.0)  # stable ffd
    f1_valid = 1.0 - f_valid
    out[mask_in_limit] = -f_valid * np.log(f_valid) - f1_valid * np.log(f1_valid)
    return out

def fmp1(x):
    '''distribution function of Methfessel-Paxton smearing order 1, in which the x = (e - ef) / sigma
    is the smearing-normalized energy'''
    return 1/2 * (1 - erf(x)) - 1/2/np.sqrt(np.pi) * x * np.exp(-x**2)

def smp1(x):
    '''entropy contribution of Methfessel-Paxton smearing order 1

    > it eliminates the quadratic (main contribution) and the cubic terms 
    > in the free energy (linear and second orders in the entropy)

    Please see PHYSICAL REVIEW B 107, 195122 (2023) for details.'''
    return 0

def fcold(x):
    '''distribution function of Marzari-Vanderbilt smearing, in which the x = (e - ef) / sigma
    is the smearing-normalized energy'''
    f_analytical = lambda e:  1/np.sqrt(2*np.pi) * np.exp(-1/4 * (np.sqrt(2) - 2*e)**2) \
                            - 1/2 * erf(1/np.sqrt(2) - e)
    f = lambda x: f_analytical(x) - f_analytical(-np.inf)
    return 1 - f(x)

def scold(x):
    '''entropy contribution of Marzari-Vanderbilt smearing'''
    return 0

# Module-level dictionaries to avoid repeated creation
_FOCCUPY_METHODS = {'gaussian': fgau, 'fd': ffd, 'mp': fmp1, 'mv': fcold}
_SOCCUPY_METHODS = {'gaussian': sgau, 'fd': sfd, 'mp': smp1, 'mv': scold}
_OCCUPY_ALIASES = {
    'gaussian': ['gau'],
    'fd': ['fermi-dirac', 'f-d'],
    'mp': ['methfessel-paxton', 'm-p'],
    'mv': ['marzari-vanderbilt', 'cold', 'm-v'],
}
_OCCUPY_METHOD_LOOKUP = {alias: method for method, aliases in _OCCUPY_ALIASES.items() for alias in aliases + [method]}
_SUPPORTED_OCCUPY_METHODS = ', '.join(sorted(_OCCUPY_METHOD_LOOKUP))


def _normalize_occupy_method(method: str) -> str:
    if not isinstance(method, str):
        raise TypeError("method must be a string.")
    normalized = _OCCUPY_METHOD_LOOKUP.get(method.lower())
    if normalized is None:
        raise ValueError(f"Unsupported smearing method '{method}'. Supported methods and aliases: {_SUPPORTED_OCCUPY_METHODS}.")
    return normalized


def foccupy(e, ef, sigma, method):
    """Compute occupation function. Works on arrays of any shape."""
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    return _FOCCUPY_METHODS[method]((e - ef) / sigma)

def soccupy(e, ef, sigma, method):
    """Compute entropy function. Works on arrays of any shape."""
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    return _SOCCUPY_METHODS[method]((e - ef) / sigma)

# Vectorized function to calculate the number of electrons with given fermi energy
# This operates on the entire eigs array at once instead of looping
def fne_vectorized(eigs, ef, sigma, method, spindeg, wk_broadcast):
    '''Calculate total number of electrons with given fermi energy (vectorized).'''
    # foccupy works on arrays of any shape, returns same shape
    occ = foccupy(eigs, ef, sigma, method)  # shape: (nspin, nk, nb)
    # Sum over all dimensions with k-point weights
    return spindeg * np.sum(occ * wk_broadcast)


def calculate_fermi_level(eigs: np.ndarray,
                          wk: np.ndarray,
                          ne: int,
                          sigma: float,
                          method: str = 'gaussian',
                          with_eband: bool = True) -> Tuple[float, np.ndarray, float, float]:
    '''
    calculate the fermi energy and occupation number
    
    Parameters
    ----------
    eigs : np.ndarray
        the eigenvalues solved from HF/KS equation, indexed by (ispin, ik, ib) -> float,
        in which ispin is the spin index, ik is the k-point index, and ib is the band index.
    wk : np.ndarray
        the weight of kpoints, indexed by (ik) -> float,
        in which ik is the k-point index.
    ne : int
        the number of electrons.
    sigma : float
        the smearing width.
    method : str, optional
        the smearing method, by default 'gaussian'. All supported smearing methods are:
        - Gaussian: alias includes 'gau', 'gaussian'
        - Fermi-Dirac: alias includes 'fd', 'fermi-dirac', 'f-d'
        - Methfessel-Paxton: alias includes 'mp', 'methfessel-paxton', 'm-p'
        - Marzari-Vanderbilt: alias includes 'mv', 'marzari-vanderbilt', 'cold', 'm-v'
    with_eband : bool, optional
        whether to calculate the free energy, by default True
    
    Returns
    -------
    ef : float
        the fermi energy.
    occ : np.ndarray
        the occupation number, indexed by (ispin, ik, ib) -> float,
        in which ispin is the spin index, ik is the k-point index, and ib is the band index.
    eband : float or None
        the band energy E_band = sum_{n,k,s} w_k * f_{nks} * e_{nks}, if with_eband is True, otherwise None.
    eband_free : float or None
        the band free energy E_band - sigma * S (E - TS), if with_eband is True, otherwise None.

    Examples
    --------
    >>> import numpy as np
    >>> from dptb.utils.occupy import calculate_fermi_level

    # Simple example with a single k-point and 100 energy bands
    >>> ekb = np.linspace(-4, 4, 100).reshape(1, 1, -1)  # shape: (nspin=1, nk=1, nb=100)
    >>> wk = np.array([1.0])  # k-point weight
    >>> ne = 10  # total number of electrons
    >>> sigma = 0.1  # smearing width in eV

    # Calculate Fermi level with Gaussian smearing (default)
    >>> ef, occ, eband, eband_free = calculate_fermi_level(ekb, wk, ne, sigma)
    >>> print(f"Fermi energy: {ef:.3f} eV")
    >>> print(f"Total electrons: {np.sum(occ):.1f}")

    # Use different smearing methods
    >>> ef_fd, occ_fd, eband_fd, eband_free_fd = calculate_fermi_level(ekb, wk, ne, sigma, method='fermi-dirac')
    >>> ef_mp, occ_mp, eband_mp, eband_free_mp = calculate_fermi_level(ekb, wk, ne, sigma, method='methfessel-paxton')

    # Multiple k-points example
    >>> nk = 8  # number of k-points
    >>> ekb_multi = np.random.uniform(-4, 4, (1, nk, 20))  # 1 spin, 8 k-points, 20 bands
    >>> wk_multi = np.ones(nk) / nk  # uniform k-point weights
    >>> ef_multi, occ_multi, eband_multi, eband_free_multi = calculate_fermi_level(ekb_multi, wk_multi, 40, 0.05)

    # Spin-polarized calculation (nspin=2)
    >>> ekb_sp = np.random.uniform(-3, 3, (2, 4, 15))  # 2 spins, 4 k-points, 15 bands
    >>> wk_sp = np.ones(4) / 4  # k-point weights
    >>> ef_sp, occ_sp, eband_sp, eband_free_sp = calculate_fermi_level(ekb_sp, wk_sp, 30, 0.1, method='mv')
    '''
    if not isinstance(eigs, np.ndarray):
        raise TypeError("eigs must be a numpy.ndarray.")
    if eigs.ndim != 3:
        raise ValueError("eigs must have shape (nspin, nk, nb).")
    if not np.all(np.isfinite(eigs)):
        raise ValueError("eigs must contain only finite values.")
    nspin, nk, nb = eigs.shape
    if nspin not in [1, 2]:
        raise ValueError("eigs first dimension nspin must be 1 or 2.")
    if nk <= 0 or nb <= 0:
        raise ValueError(f'Invalid ekb data: nk={nk}, nb={nb}')

    if not isinstance(wk, np.ndarray):
        raise TypeError("wk must be a numpy.ndarray.")
    if wk.ndim != 1:
        raise ValueError("wk must be a 1D array.")
    if len(wk) != nk:
        raise ValueError("wk length must match the k-point dimension of eigs.")
    if not np.all(np.isfinite(wk)):
        raise ValueError("wk must contain only finite values.")

    if ne < 0:
        raise ValueError("ne must be non-negative.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite.")

    method = _normalize_occupy_method(method)
    
    # the spin-degeneracy
    spindeg = 2 if nspin == 1 else 1

    # the method mask: for mp and mv, use gaussian smearing to get an initial guess first
    # please see the PHYSICAL REVIEW B 107, 195122 (2023) for details.
    fmmask = lambda m: 'gaussian' if m in ['mp', 'mv'] else m

    # Precompute wk with proper shape for broadcasting: (1, nk, 1) for (nspin, nk, nb)
    wk_broadcast = wk.reshape(1, -1, 1)

    # Firstly, solve the fermi energy using brent method
    def ferr(ef):
        '''Calculate the error in total number of electrons.'''
        return fne_vectorized(eigs, ef, sigma, fmmask(method), spindeg, wk_broadcast) - ne
    
    # handle with corner cases: zero electrons and all electrons
    if abs(ne) < 1e-10:
        # no electrons, the fermi level is... (NOT HOMO/HOCO)
        occ = np.zeros_like(eigs)
        if with_eband:
            return None, occ, 0.0, 0.0
        return None, occ, None, None
    
    if abs(spindeg * np.sum(np.ones_like(eigs) * wk_broadcast) - ne) < 1e-10:
        # all electrons
        ef = np.max(eigs)
    else:
        # common case
        ef_brent = brentq(ferr, a=np.min(eigs), b=np.max(eigs), xtol=1e-10)

        # check ef_brent validity
        ne_brent = fne_vectorized(eigs, ef_brent, sigma, fmmask(method), spindeg, wk_broadcast)
        if abs(ne_brent - ne) > 0.01:
            log.warning(f'Brent method may not have converged properly: '
                        f'ne_brent={ne_brent}, target ne={ne}. '
                        f'Falling back to bisection method.')

            # solve the fermi energy using bisection (more robust than brent for multi-minima cases)
            # brent can get trapped in local minima for systems with outlier eigenvalues (e.g., C dimer)
            nextafter = np.nextafter
            min_Ef, max_Ef = np.min(eigs), np.max(eigs)
            # Extend boundaries slightly to ensure Fermi level can be found
            drange = sigma * np.sqrt(50)  # ~7 sigma range
            min_Ef = min_Ef - drange
            max_Ef = max_Ef + drange
            ef = (min_Ef + max_Ef) * 0.5
            # Bisection loop
            while nextafter(min_Ef, max_Ef) < max_Ef:
                q_cal = fne_vectorized(eigs, ef, sigma, 
                                    fmmask(method), spindeg, 
                                    wk_broadcast)
                if q_cal >= ne:
                    max_Ef = ef
                else:
                    min_Ef = ef
                ef = (min_Ef + max_Ef) * 0.5
        else:
            ef = ef_brent

    # refine the fermi energy for mp and mv cases in which the root of fermi energy is not unique
    if method in ['mp', 'mv']:
        def ferr_(ef):
            return np.abs(fne_vectorized(eigs, ef, sigma, method, spindeg, wk_broadcast) - ne)
        ef = minimize(ferr_, x0=ef).x.item()  # BFGS

    # With fermi energy, calculate the occupations and eband (vectorized)
    # foccupy already works on full arrays, no need to loop
    occ = spindeg * foccupy(eigs, ef, sigma, method)  # shape: (nspin, nk, nb)

    eband = None
    eband_free = None
    if with_eband:
        # E_band = sum_{n,k,s} w_k * f_{nks} * e_{nks}
        # E_band_free = E_band - sigma * S
        # k-point weights applied consistently to both energy and entropy terms
        # Vectorized: soccupy works on full array
        entropy = spindeg * soccupy(eigs, ef, sigma, method)  # shape: (nspin, nk, nb)
        eband = np.sum(eigs * occ * wk_broadcast)
        eband_free = eband - sigma * np.sum(entropy * wk_broadcast)

    return ef, occ, eband, eband_free
