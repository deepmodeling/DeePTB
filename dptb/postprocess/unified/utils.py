import numpy as np
import logging
from dptb.utils.constants import Boltzmann, eV2J

log = logging.getLogger(__name__)

def fermi_dirac_smearing(E, kT=0.025852, mu=0.0):
    """
    Fermi-Dirac smearing function.
    """
    x = (E - mu) / kT
    mask_min = x < -40.0  # 40 results e16 precision
    mask_max = x > 40.0
    mask_in_limit = ~(mask_min | mask_max)
    out = np.zeros_like(x)
    out[mask_min] = 1.0
    out[mask_max] = 0.0
    out[mask_in_limit] = 1.0 / (np.expm1(x[mask_in_limit]) + 2.0)
    return out

def gaussian_smearing(E, sigma=0.025852, mu=0.0):
    """
    Gaussian smearing function.
    """
    from scipy.special import erfc
    x = (mu - E) / sigma
    return 0.5 * erfc(-1 * x)

def calculate_fermi_level(eigenvalues: np.ndarray, total_electrons: float, spindeg: int = 2,
                          weights: np.ndarray = None, q_tol: float = 1e-10,
                          smearing_method: str = 'FD', temperature: float = 300):
    """
    Calculates the Fermi energy using iteration algorithm (Bisection method).

    Parameters
    ----------
    eigenvalues : np.ndarray
        The eigenvalues of the system. Shape (Nk, Nb) or (Nb,).
    total_electrons : float
        The total number of electrons in the system.
    spindeg : int, optional
        Spin degeneracy factor (typically 2 for spin-degenerate systems, 1 for SOC/spin-polarized). Default is 2.
    weights : np.ndarray, optional
        Weights assigned to each k-point. If None, equal weights are assumed.
    q_tol : float, optional
        Tolerance level for charge convergence. Default is 1e-10.
    smearing_method : str, optional
        Smearing method: 'FD' (Fermi-Dirac) or 'Gaussian'. Default is 'FD'.
    temperature : float, optional
        Temperature in Kelvin for smearing. Default is 300 K.

    Returns
    -------
    float
        The calculated Fermi energy (Ef).
    """

    nextafter = np.nextafter
    
    # Adjust total electrons for spin degeneracy
    target_electrons = total_electrons / spindeg
    
    log.info(f"Calculating Fermi energy. Target electrons per spin channel: {target_electrons}")

    # calculate boundaries
    min_Ef, max_Ef = eigenvalues.min(), eigenvalues.max()
    kT = Boltzmann / eV2J * temperature
    
    # Expand search range to ensure Ef is within bounds even if it's in the band gap
    drange = kT * np.sqrt(-np.log(q_tol * 1e-2))
    min_Ef = min_Ef - drange
    max_Ef = max_Ef + drange

    Ef = (min_Ef + max_Ef) * 0.5

    if weights is None:
        # Assuming flattened eigenvalues or handled by broadcasting if logic allows, 
        # but typically weights correspond to K-points.
        # If eigenvalues is (Nk, Nb), weights should be (Nk,).
        # We need to broadcast weights to (Nk, Nb) or flatten everything.
        
        # Let's handle generic shaping: flatten everything for summation
        weights = np.ones(eigenvalues.shape[0]) / eigenvalues.shape[0]
        # log.info('Weights not provided, using equal weights for 1st dimension (ktools).')

    # Ensure weights match eigenvalues shape for broadcasting
    # If eigenvalues is (Nk, Nb) and weights is (Nk,), reshape weights to (Nk, 1)
    if eigenvalues.ndim == 2 and weights.ndim == 1:
        if eigenvalues.shape[0] == weights.shape[0]:
            weights = weights.reshape(-1, 1)
    
    icounter = 0
    # Use a safe max iteration count to prevent infinite loops in pathological cases
    MAX_ITER = 200 
    
    while icounter < MAX_ITER:
        icounter += 1
        
        # Determine smearing width parameter
        sigma_val = kT # For FD, it's kT. For Gaussian, passing kT as sigma is common convention here.
        
        if smearing_method == 'FD':
            occupation = fermi_dirac_smearing(eigenvalues, kT=sigma_val, mu=Ef)
        elif smearing_method == 'Gaussian':
            occupation = gaussian_smearing(eigenvalues, sigma=sigma_val, mu=Ef)
        else:
            raise ValueError(f'Unknown smearing method: {smearing_method}')

        # q_cal is weighted sum of occupation
        # weights broadcasted against occupation
        q_cal = (weights * occupation).sum()

        if abs(q_cal - target_electrons) < q_tol:
            log.info(f'Fermi energy converged after {icounter} iterations. Ef = {Ef:.6f} eV')
            return Ef
        
        # Check convergence limit of bisection interval
        if nextafter(min_Ef, max_Ef) >= max_Ef:
             break

        if q_cal >= target_electrons:
            max_Ef = Ef
        else:
            min_Ef = Ef
        Ef = (min_Ef + max_Ef) * 0.5

    log.warning(f'Fermi level bisection did not converge under tolerance {q_tol} after {icounter} iterations.')
    log.info(f'q_cal: {q_cal * spindeg}, total_nel: {total_electrons}, diff: {abs(q_cal - target_electrons) * spindeg}')
    
    return Ef
