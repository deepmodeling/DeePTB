"""
SCC (Self-Consistent Charge) Mixers for DFTB-SCC calculations.

This module implements various charge mixing algorithms for accelerating
convergence in self-consistent charge (SCC) iterations. The implementations
follow the DFTBplus mixer module design.

Available mixers:
- SimpleMixer: Linear/simple mixing
- AndersonMixer: Anderson/Pulay mixing with history
- BroydenMixer: Modified Broyden mixing (Johnson, PRB 38, 12807 (1988))
- DIISMixer: Direct Inversion in the Iterative Subspace

References:
    - D.D. Johnson, PRB 38, 12807 (1988) - Broyden mixing
    - P. Pulay, Chem. Phys. Lett. 73, 393 (1980) - DIIS
    - D.G. Anderson, J. ACM 12, 547 (1965) - Anderson mixing
    - Kovalenko et al., J. Comput. Chem., 20: 928-936 (1999) - gDIIS modification
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
import logging

log = logging.getLogger(__name__)


class SCCMixer(ABC):
    """
    Abstract base class for SCC charge mixers.

    This class defines the interface for charge mixing algorithms used in
    self-consistent charge (SCC) iterations. Subclasses must implement the
    `reset` and `mix` methods.

    The mixer takes the current input charge and the charge difference
    (output - input) and returns a mixed charge for the next iteration.

    Attributes
    ----------
    n_elem : int
        Number of elements in the charge vector being mixed.
    """

    def __init__(self):
        """Initialize the base mixer."""
        self.n_elem = 0

    @abstractmethod
    def reset(self, n_elem: int) -> None:
        """
        Reset the mixer for a new SCC cycle.

        This method should be called at the beginning of each new SCC
        calculation to clear any stored history and prepare for new iterations.

        Parameters
        ----------
        n_elem : int
            Number of elements in the charge vectors to mix.
        """
        pass

    @abstractmethod
    def mix(self, q_inp: np.ndarray, q_diff: np.ndarray) -> np.ndarray:
        """
        Perform charge mixing.

        Parameters
        ----------
        q_inp : np.ndarray
            Current input charge vector. Shape: (n_elem,)
        q_diff : np.ndarray
            Charge difference vector (output - input). Shape: (n_elem,)

        Returns
        -------
        np.ndarray
            Mixed charge vector for next iteration. Shape: (n_elem,)
        """
        pass


class SimpleMixer(SCCMixer):
    """
    Simple linear charge mixer.

    Performs linear mixing of the form:
        q_new = q_inp + mix_param * q_diff

    This is equivalent to:
        q_new = (1 - mix_param) * q_inp + mix_param * q_out

    where q_out = q_inp + q_diff.

    Parameters
    ----------
    mix_param : float
        Mixing parameter (0 < mix_param <= 1). Smaller values provide more
        stable but slower convergence. Default is 0.2.

    Attributes
    ----------
    mix_param : float
        The mixing parameter used in linear mixing.

    Examples
    --------
    >>> mixer = SimpleMixer(mix_param=0.3)
    >>> mixer.reset(n_elem=10)
    >>> q_new = mixer.mix(q_inp, q_diff)
    """

    def __init__(self, mix_param: float = 0.2):
        """
        Initialize the simple mixer.

        Parameters
        ----------
        mix_param : float, optional
            Mixing parameter. Default is 0.2.
        """
        super().__init__()
        if not 0 < mix_param <= 1:
            raise ValueError("mix_param must be in range (0, 1]")
        self.mix_param = mix_param

    def reset(self, n_elem: int) -> None:
        """
        Reset the mixer for a new SCC cycle.

        Parameters
        ----------
        n_elem : int
            Number of elements in the charge vectors.
        """
        if n_elem <= 0:
            raise ValueError("n_elem must be positive")
        self.n_elem = n_elem

    def mix(self, q_inp: np.ndarray, q_diff: np.ndarray) -> np.ndarray:
        """
        Perform simple linear mixing.

        Parameters
        ----------
        q_inp : np.ndarray
            Current input charge vector.
        q_diff : np.ndarray
            Charge difference vector.

        Returns
        -------
        np.ndarray
            Mixed charge vector: q_inp + mix_param * q_diff
        """
        assert len(q_inp) == len(q_diff), "Input vectors must have same length"
        return q_inp + self.mix_param * q_diff


class AndersonMixer(SCCMixer):
    """
    Anderson/Pulay mixing for charge self-consistency.

    The Anderson mixing builds weighted averages over previous input charges
    and charge differences. The weights are determined by solving a system of
    linear equations that minimizes the error in a least-squares sense.

    Parameters
    ----------
    n_generations : int
        Number of previous iterations to store (including current). Must be >= 2.
        Default is 6.
    mix_param : float
        Mixing parameter for general iterations. Default is 0.2.
    init_mix_param : float
        Mixing parameter for the first n_generations-1 iterations. Default is 0.01.
    omega0 : float
        Symmetry breaking parameter. Diagonal elements of the linear system
        are multiplied by (1 + omega0^2). Set to <= 0 to disable. Default is 0.01.

    Attributes
    ----------
    n_generations : int
        Maximum number of stored vector pairs.
    mix_param : float
        General mixing parameter.
    init_mix_param : float
        Initial mixing parameter.
    omega0 : float
        Symmetry breaking parameter.

    References
    ----------
    D.G. Anderson, "Iterative Procedures for Nonlinear Integral Equations",
    J. ACM 12, 547 (1965).

    Examples
    --------
    >>> mixer = AndersonMixer(n_generations=8, mix_param=0.3)
    >>> mixer.reset(n_elem=100)
    >>> for i in range(max_iter):
    ...     q_new = mixer.mix(q_inp, q_diff)
    """

    def __init__(self,
                 n_generations: int = 6,
                 mix_param: float = 0.2,
                 init_mix_param: float = 0.01,
                 omega0: float = 0.01):
        """
        Initialize the Anderson mixer.

        Parameters
        ----------
        n_generations : int, optional
            Number of generations to consider. Default is 6.
        mix_param : float, optional
            Mixing parameter. Default is 0.2.
        init_mix_param : float, optional
            Initial mixing parameter. Default is 0.01.
        omega0 : float, optional
            Symmetry breaking parameter. Default is 0.01.
        """
        super().__init__()
        if n_generations < 2:
            raise ValueError("n_generations must be >= 2")
        if not 0 < mix_param <= 1:
            raise ValueError("mix_param must be in range (0, 1]")
        if not 0 < init_mix_param <= 1:
            raise ValueError("init_mix_param must be in range (0, 1]")

        self.n_generations = n_generations
        self.m_prev_vector = n_generations - 1  # Max stored vectors
        self.mix_param = mix_param
        self.init_mix_param = init_mix_param
        self.omega0 = omega0
        self.break_sym = omega0 > 0

        # Storage arrays (initialized in reset)
        self.n_prev_vector = -1
        self.indx = None
        self.prev_q_input = None
        self.prev_q_diff = None

    def reset(self, n_elem: int) -> None:
        """
        Reset the mixer for a new SCC cycle.

        Parameters
        ----------
        n_elem : int
            Number of elements in the charge vectors.
        """
        if n_elem <= 0:
            raise ValueError("n_elem must be positive")

        self.n_elem = n_elem
        self.n_prev_vector = -1

        # Allocate storage arrays
        self.prev_q_input = np.zeros((n_elem, self.m_prev_vector))
        self.prev_q_diff = np.zeros((n_elem, self.m_prev_vector))

        # Create reverse index array for LIFO access
        self.indx = np.array([self.m_prev_vector - i for i in range(self.m_prev_vector)])

    def mix(self, q_inp: np.ndarray, q_diff: np.ndarray) -> np.ndarray:
        """
        Perform Anderson mixing.

        Parameters
        ----------
        q_inp : np.ndarray
            Current input charge vector.
        q_diff : np.ndarray
            Charge difference vector.

        Returns
        -------
        np.ndarray
            Mixed charge vector.
        """
        assert len(q_inp) == self.n_elem, f"Expected {self.n_elem} elements, got {len(q_inp)}"
        assert len(q_diff) == self.n_elem, f"Expected {self.n_elem} elements, got {len(q_diff)}"

        # Update iteration counter
        if self.n_prev_vector < self.m_prev_vector:
            self.n_prev_vector += 1
            mix_param = self.init_mix_param
        else:
            mix_param = self.mix_param

        # First iteration: store vectors and return simple mixed vector
        if self.n_prev_vector == 0:
            self._store_vectors(q_inp, q_diff)
            return q_inp + self.init_mix_param * q_diff

        # Calculate Anderson averages
        q_inp_middle, q_diff_middle = self._calc_anderson_averages(q_inp, q_diff)

        # Store vectors before overwriting
        self._store_vectors(q_inp, q_diff)

        # Mix averaged input charge and averaged charge difference
        return q_inp_middle + mix_param * q_diff_middle

    def _store_vectors(self, q_inp: np.ndarray, q_diff: np.ndarray) -> None:
        """Store vector pair in limited LIFO storage."""
        # Rotate indices
        tmp = self.indx[-1]
        self.indx[1:] = self.indx[:-1]
        self.indx[0] = tmp

        # Store vectors at the new index
        idx = self.indx[0] - 1  # Convert to 0-based index
        self.prev_q_input[:, idx] = q_inp
        self.prev_q_diff[:, idx] = q_diff

    def _calc_anderson_averages(self, q_inp: np.ndarray, q_diff: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Anderson averages for input charges and charge differences.

        Parameters
        ----------
        q_inp : np.ndarray
            Current input charge.
        q_diff : np.ndarray
            Current charge difference.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (q_inp_middle, q_diff_middle) - Averaged input and difference vectors.
        """
        n = self.n_prev_vector

        # Build the system of linear equations
        # a(i,j) = <F(m)|F(m)-F(m-i)> - <F(m-j)|F(m)-F(m-i)>
        # b(i)   = <F(m)|F(m)-F(m-i)>
        aa = np.zeros((n, n))
        bb = np.zeros((n, 1))

        for ii in range(n):
            idx_i = self.indx[ii] - 1  # 0-based index
            tmp = q_diff - self.prev_q_diff[:, idx_i]
            tmp2 = np.dot(q_diff, tmp)
            bb[ii, 0] = tmp2

            for jj in range(n):
                idx_j = self.indx[jj] - 1
                aa[ii, jj] = tmp2 - np.dot(self.prev_q_diff[:, idx_j], tmp)

        # Apply symmetry breaking if enabled
        if self.break_sym:
            factor = 1.0 + self.omega0 ** 2
            for ii in range(n):
                aa[ii, ii] *= factor

        # Solve system of linear equations
        try:
            coeffs = np.linalg.solve(aa, bb)
        except np.linalg.LinAlgError:
            # Fall back to simple mixing if system is singular
            log.warning("Anderson mixer: singular matrix, falling back to simple mixing")
            return q_inp, q_diff

        # Build averages with calculated coefficients
        q_diff_middle = np.zeros(self.n_elem)
        q_inp_middle = np.zeros(self.n_elem)

        coeff_sum = coeffs.sum()
        for ii in range(n):
            idx = self.indx[ii] - 1
            q_diff_middle += coeffs[ii, 0] * self.prev_q_diff[:, idx]
            q_inp_middle += coeffs[ii, 0] * self.prev_q_input[:, idx]

        q_diff_middle += (1.0 - coeff_sum) * q_diff
        q_inp_middle += (1.0 - coeff_sum) * q_inp

        return q_inp_middle, q_diff_middle


class BroydenMixer(SCCMixer):
    """
    Modified Broyden mixing for charge self-consistency.

    Implements the modified Broyden method as described by Johnson (PRB 38, 12807, 1988).
    This method builds an approximate inverse Jacobian from the history of
    charge vectors and their differences.

    Parameters
    ----------
    max_iter : int
        Maximum number of SCC iterations (determines storage size). Default is 100.
    mix_param : float
        Mixing parameter (alpha). Default is 0.2.
    omega0 : float
        Weight for Jacobi matrix differences. Default is 0.01.
    min_weight : float
        Minimum allowed weight. Default is 1.0.
    max_weight : float
        Maximum allowed weight. Default is 1e5.
    weight_fac : float
        Numerator for weight calculation (weight_fac / ||q_diff||). Default is 0.01.

    References
    ----------
    D.D. Johnson, "Modified Broyden's method for accelerating convergence
    in self-consistent calculations", PRB 38, 12807 (1988).

    Examples
    --------
    >>> mixer = BroydenMixer(max_iter=200, mix_param=0.3)
    >>> mixer.reset(n_elem=100)
    >>> for i in range(max_iter):
    ...     q_new = mixer.mix(q_inp, q_diff)
    """

    def __init__(self,
                 max_iter: int = 100,
                 mix_param: float = 0.2,
                 omega0: float = 0.01,
                 min_weight: float = 1.0,
                 max_weight: float = 1e5,
                 weight_fac: float = 0.01):
        """
        Initialize the Broyden mixer.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations. Default is 100.
        mix_param : float, optional
            Mixing parameter. Default is 0.2.
        omega0 : float, optional
            Jacobi matrix weight. Default is 0.01.
        min_weight : float, optional
            Minimum weight. Default is 1.0.
        max_weight : float, optional
            Maximum weight. Default is 1e5.
        weight_fac : float, optional
            Weight factor. Default is 0.01.
        """
        super().__init__()
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if mix_param <= 0:
            raise ValueError("mix_param must be positive")
        if omega0 <= 0:
            raise ValueError("omega0 must be positive")

        self.max_iter = max_iter
        self.alpha = mix_param
        self.omega0 = omega0
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.weight_fac = weight_fac

        # State variables (initialized in reset)
        self.i_iter = 0
        self.ww = None
        self.q_diff_last = None
        self.q_inp_last = None
        self.aa = None
        self.dF = None
        self.uu = None

    def reset(self, n_elem: int) -> None:
        """
        Reset the mixer for a new SCC cycle.

        Parameters
        ----------
        n_elem : int
            Number of elements in the charge vectors.
        """
        if n_elem <= 0:
            raise ValueError("n_elem must be positive")

        self.n_elem = n_elem
        self.i_iter = 0

        # Allocate storage arrays
        m = self.max_iter - 1
        self.ww = np.zeros(m)
        self.q_inp_last = np.zeros(n_elem)
        self.q_diff_last = np.zeros(n_elem)
        self.aa = np.zeros((m, m))
        self.dF = np.zeros((n_elem, m))
        self.uu = np.zeros((n_elem, m))

    def mix(self, q_inp: np.ndarray, q_diff: np.ndarray) -> np.ndarray:
        """
        Perform modified Broyden mixing.

        Parameters
        ----------
        q_inp : np.ndarray
            Current input charge vector.
        q_diff : np.ndarray
            Charge difference vector.

        Returns
        -------
        np.ndarray
            Mixed charge vector.

        Raises
        ------
        RuntimeError
            If maximum number of iterations is exceeded.
        """
        self.i_iter += 1
        if self.i_iter > self.max_iter:
            raise RuntimeError("Broyden mixer: Maximum number of iterations exceeded")

        return self._modified_broyden_mixing(q_inp, q_diff)

    def _modified_broyden_mixing(self, q_inp: np.ndarray, q_diff: np.ndarray) -> np.ndarray:
        """
        Perform the actual modified Broyden mixing.

        Parameters
        ----------
        q_inp : np.ndarray
            Current input charge.
        q_diff : np.ndarray
            Charge difference.

        Returns
        -------
        np.ndarray
            Mixed charge vector.
        """
        nn = self.i_iter
        nn_1 = nn - 1

        # First iteration: simple mix and storage
        if nn == 1:
            self.q_inp_last[:] = q_inp
            self.q_diff_last[:] = q_diff
            return q_inp + self.alpha * q_diff

        # Create weight factor for current iteration
        norm_q_diff = np.sqrt(np.dot(q_diff, q_diff))
        if norm_q_diff > self.weight_fac / self.max_weight:
            self.ww[nn_1 - 1] = self.weight_fac / norm_q_diff
        else:
            self.ww[nn_1 - 1] = self.max_weight
        if self.ww[nn_1 - 1] < self.min_weight:
            self.ww[nn_1 - 1] = self.min_weight

        # Build |DF(m-1)> normalized
        dF_uu = q_diff - self.q_diff_last
        inv_norm = np.sqrt(np.dot(dF_uu, dF_uu))
        inv_norm = max(inv_norm, np.finfo(float).eps)
        inv_norm = 1.0 / inv_norm
        dF_uu = inv_norm * dF_uu

        # Build a, beta, c matrices
        for ii in range(nn - 2):
            self.aa[ii, nn_1 - 1] = np.dot(self.dF[:, ii], dF_uu)
            self.aa[nn_1 - 1, ii] = self.aa[ii, nn_1 - 1]
        self.aa[nn_1 - 1, nn_1 - 1] = 1.0

        # Calculate c coefficients
        cc = np.zeros(nn_1)
        for ii in range(nn - 2):
            cc[ii] = self.ww[ii] * np.dot(self.dF[:, ii], q_diff)
        cc[nn_1 - 1] = self.ww[nn_1 - 1] * np.dot(dF_uu, q_diff)

        # Build beta matrix
        beta = np.zeros((nn_1, nn_1))
        for ii in range(nn_1):
            for jj in range(nn_1):
                beta[jj, ii] = self.ww[jj] * self.ww[ii] * self.aa[jj, ii]
            beta[ii, ii] += self.omega0 ** 2

        # Solve system of linear equations
        try:
            cc = np.linalg.solve(beta, cc)
        except np.linalg.LinAlgError:
            log.warning("Broyden mixer: singular matrix, falling back to simple mixing")
            return q_inp + self.alpha * q_diff

        # Store |dF(m-1)>
        self.dF[:, nn_1 - 1] = dF_uu

        # Create |u(m-1)>
        dF_uu = self.alpha * dF_uu + inv_norm * (q_inp - self.q_inp_last)

        # Save charge vectors before overwriting
        self.q_inp_last[:] = q_inp
        self.q_diff_last[:] = q_diff

        # Build new vector
        q_result = q_inp + self.alpha * q_diff
        for ii in range(nn - 2):
            q_result -= self.ww[ii] * cc[ii] * self.uu[:, ii]
        q_result -= self.ww[nn_1 - 1] * cc[nn_1 - 1] * dF_uu

        # Save |u(m-1)>
        self.uu[:, nn_1 - 1] = dF_uu

        return q_result


class DIISMixer(SCCMixer):
    """
    Direct Inversion in the Iterative Subspace (DIIS) mixer.

    DIIS minimizes the residue of the error by building a weighted combination
    of previous input charges. The weights are determined by solving a
    constrained least-squares problem.

    Parameters
    ----------
    n_generations : int
        Number of previous iterations to store (including current). Must be >= 2.
        Default is 6.
    init_mix_param : float
        Mixing parameter for the first n_generations-1 iterations. Default is 0.2.
    from_start : bool
        If True, use DIIS from iteration 2. If False, wait until n_generations
        vectors are accumulated. Default is True.
    alpha : float
        If > 0, fraction of extrapolated downhill direction to include (gDIIS).
        Default is 0.0 (standard DIIS).

    References
    ----------
    P. Pulay, "Convergence acceleration of iterative sequences. The case of
    SCF iteration", Chem. Phys. Lett. 73, 393 (1980).

    Kovalenko et al., "Self-consistent description of a metal-water interface
    by the Kohn-Sham density functional theory and the three-dimensional
    reference interaction site model", J. Comput. Chem. 20, 928-936 (1999).

    Examples
    --------
    >>> mixer = DIISMixer(n_generations=8, init_mix_param=0.3)
    >>> mixer.reset(n_elem=100)
    >>> for i in range(max_iter):
    ...     q_new = mixer.mix(q_inp, q_diff)
    """

    def __init__(self,
                 n_generations: int = 6,
                 init_mix_param: float = 0.2,
                 from_start: bool = True,
                 alpha: float = 0.0):
        """
        Initialize the DIIS mixer.

        Parameters
        ----------
        n_generations : int, optional
            Number of generations. Default is 6.
        init_mix_param : float, optional
            Initial mixing parameter. Default is 0.2.
        from_start : bool, optional
            Use DIIS from start. Default is True.
        alpha : float, optional
            gDIIS gradient fraction. Default is 0.0.
        """
        super().__init__()
        if n_generations < 2:
            raise ValueError("n_generations must be >= 2")
        if not 0 < init_mix_param <= 1:
            raise ValueError("init_mix_param must be in range (0, 1]")

        self.n_generations = n_generations
        self.m_prev_vector = n_generations
        self.init_mix_param = init_mix_param
        self.from_start = from_start
        self.alpha = alpha
        self.use_gradient = alpha > 0

        # State variables (initialized in reset)
        self.i_prev_vector = 0
        self.indx = 0
        self.prev_q_input = None
        self.prev_q_diff = None
        self.delta_r = None

    def reset(self, n_elem: int) -> None:
        """
        Reset the mixer for a new SCC cycle.

        Parameters
        ----------
        n_elem : int
            Number of elements in the charge vectors.
        """
        if n_elem <= 0:
            raise ValueError("n_elem must be positive")

        self.n_elem = n_elem
        self.i_prev_vector = 0
        self.indx = 0

        # Allocate storage arrays
        self.prev_q_input = np.zeros((n_elem, self.m_prev_vector))
        self.prev_q_diff = np.zeros((n_elem, self.m_prev_vector))

        if self.use_gradient:
            self.delta_r = np.zeros(n_elem)
        else:
            self.delta_r = None

    def mix(self, q_inp: np.ndarray, q_diff: np.ndarray) -> np.ndarray:
        """
        Perform DIIS mixing.

        Parameters
        ----------
        q_inp : np.ndarray
            Current input charge vector.
        q_diff : np.ndarray
            Charge difference vector.

        Returns
        -------
        np.ndarray
            Mixed charge vector.
        """
        assert len(q_inp) == self.n_elem, f"Expected {self.n_elem} elements, got {len(q_inp)}"
        assert len(q_diff) == self.n_elem, f"Expected {self.n_elem} elements, got {len(q_diff)}"

        # Update counter
        if self.i_prev_vector < self.m_prev_vector:
            self.i_prev_vector += 1

        # Store vectors
        self._store_vectors(q_inp, q_diff)

        # Determine if DIIS should be applied
        apply_diis = self.from_start or (self.i_prev_vector == self.m_prev_vector)

        if apply_diis:
            q_result = self._apply_diis(q_inp, q_diff)
        else:
            q_result = q_inp.copy()

        # For early iterations, also apply simple mixing
        if self.i_prev_vector < self.m_prev_vector:
            q_result = q_result + self.init_mix_param * q_diff

        return q_result

    def _store_vectors(self, q_inp: np.ndarray, q_diff: np.ndarray) -> None:
        """Store vector pair using circular buffer."""
        self.indx = self.indx % self.m_prev_vector
        self.prev_q_input[:, self.indx] = q_inp
        self.prev_q_diff[:, self.indx] = q_diff
        self.indx += 1

    def _apply_diis(self, q_inp: np.ndarray, q_diff: np.ndarray) -> np.ndarray:
        """
        Apply DIIS extrapolation.

        Parameters
        ----------
        q_inp : np.ndarray
            Current input charge.
        q_diff : np.ndarray
            Current charge difference.

        Returns
        -------
        np.ndarray
            DIIS extrapolated charge.
        """
        n = self.i_prev_vector

        # Build the DIIS matrix
        # aa(i,j) = <F_i|F_j> where F is the error/difference vector
        # We solve: min ||sum_i c_i F_i||^2 subject to sum_i c_i = 1
        # Using Lagrange multiplier, this becomes:
        # | A   -1 | | c |   | 0  |
        # | -1   0 | | λ | = | -1 |

        aa = np.zeros((n + 1, n + 1))
        bb = np.zeros((n + 1, 1))

        for ii in range(n):
            for jj in range(n):
                aa[ii, jj] = np.dot(self.prev_q_diff[:, ii], self.prev_q_diff[:, jj])

        # Lagrange multiplier constraints
        aa[n, :n] = -1.0
        aa[:n, n] = -1.0
        bb[n, 0] = -1.0

        # Solve DIIS system
        try:
            coeffs = np.linalg.solve(aa, bb)
        except np.linalg.LinAlgError:
            log.warning("DIIS mixer: singular matrix, falling back to input")
            return q_inp.copy()

        # Build extrapolated charge
        # q_new = sum_i c_i (q_inp_i + q_diff_i)
        q_result = np.zeros(self.n_elem)
        for ii in range(n):
            q_result += coeffs[ii, 0] * (self.prev_q_input[:, ii] + self.prev_q_diff[:, ii])

        # Apply gDIIS gradient modification if enabled
        if self.use_gradient and self.delta_r is not None:
            # Check if old DIIS gradient points in same direction as current
            if np.abs(np.dot(self.delta_r, q_diff)) > 0:
                self.alpha = 1.5 * self.alpha  # Mix in more gradient
            else:
                self.alpha = 0.5 * self.alpha  # Mix in less gradient

            # Build DIIS estimated gradient
            self.delta_r[:] = 0.0
            for ii in range(n):
                self.delta_r += coeffs[ii, 0] * self.prev_q_diff[:, ii]

            # Add fraction down the gradient
            q_result -= self.alpha * self.delta_r

        return q_result


def get_mixer(mixer_type: str, **kwargs) -> SCCMixer:
    """
    Factory function to create a mixer instance.

    Parameters
    ----------
    mixer_type : str
        Type of mixer: 'simple', 'anderson', 'broyden', or 'diis'.
    **kwargs
        Keyword arguments passed to the mixer constructor.

    Returns
    -------
    SCCMixer
        An instance of the requested mixer type.

    Raises
    ------
    ValueError
        If mixer_type is not recognized.

    Examples
    --------
    >>> mixer = get_mixer('anderson', n_generations=8, mix_param=0.3)
    >>> mixer = get_mixer('broyden', max_iter=200)
    >>> mixer = get_mixer('simple', mix_param=0.2)
    """
    mixer_type = mixer_type.lower()

    mixer_classes = {
        'simple': SimpleMixer,
        'linear': SimpleMixer,
        'anderson': AndersonMixer,
        'pulay': AndersonMixer,
        'broyden': BroydenMixer,
        'diis': DIISMixer,
    }

    if mixer_type not in mixer_classes:
        valid_types = list(set(mixer_classes.keys()))
        raise ValueError(f"Unknown mixer type '{mixer_type}'. Valid types: {valid_types}")

    return mixer_classes[mixer_type](**kwargs)
