"""
DFTB+-compatible 1D interpolation for Slater-Koster integrals.

This module provides the main interpolation class that combines Neville's
polynomial interpolation with smooth poly5ToZero extrapolation, matching
the behavior of DFTB+ exactly.

Reference: DFTB+ source file src/dftbp/dftb/slakoeqgrid.F90
"""

import torch
from typing import Optional

from .poly_interp import NevillePolyInterp
from .poly5_decay import Poly5ToZero

# DFTB+ constants (in Bohr, will be converted to Angstrom when needed)
DFTBPLUS_DIST_FUDGE = 1.0          # Bohr - modern method extrapolation zone
DFTBPLUS_DELTA_R = 1e-5            # Bohr - step for numerical derivatives
DFTBPLUS_N_INTER = 8               # Number of interpolation points
DFTBPLUS_N_RIGHT_INTER = 4         # Points to prefer on the right

# Conversion factor
BOHR_TO_ANG = 0.529177210903


class DFTBPlusInterp1D:
    """
    DFTB+-compatible 1D interpolation for SK integrals.

    This class provides interpolation that produces identical results to DFTB+
    when reading the same SKF files. It implements:

    1. **Grid region**: 8-point polynomial interpolation using Neville's algorithm
    2. **Decay region**: Smooth poly5ToZero extrapolation with C² continuity
    3. **Beyond cutoff**: Zero values

    Parameters
    ----------
    x : torch.Tensor
        Grid points with shape [n_grid]. Must be uniformly spaced.
        Units should be Angstrom (matching the DFTB+ interpolation convention).
    y : torch.Tensor, optional
        Values at grid points with shape [n_channels, n_grid].
        If provided at init, will be used for all interpolation calls.
    dist_fudge : float, optional
        Extrapolation zone size in Bohr. Default is 1.0 (DFTB+ modern method).
    n_points : int, optional
        Number of points for polynomial interpolation. Default is 8.
    delta_r : float, optional
        Step size for numerical derivatives in Bohr. Default is 1e-5.

    Attributes
    ----------
    x : torch.Tensor
        Grid points.
    x_min : float
        Minimum grid value.
    x_max : float
        Maximum grid value.
    x_cutoff : float
        Effective cutoff distance (x_max + dist_fudge_ang).
    grid_spacing : float
        Uniform grid spacing.

    Notes
    -----
    Distance regions (following DFTB+ slakoeqgrid.F90):

    1. `xq < x_min`: Zero (below grid - physically no interaction)
    2. `x_min <= xq <= x_max`: Polynomial interpolation
    3. `x_max < xq <= x_cutoff`: poly5ToZero decay
    4. `xq > x_cutoff`: Zero (beyond cutoff)

    Examples
    --------
    >>> x = torch.linspace(0.5, 10.0, 100)  # Grid in Angstrom
    >>> y = torch.randn(4, 100)  # 4 SK integral channels
    >>> interp = DFTBPlusInterp1D(x, dist_fudge=1.0)
    >>> xq = torch.tensor([1.0, 5.0, 9.5, 11.0])
    >>> result = interp(xq, y)  # [4, 4]
    """

    def __init__(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        dist_fudge: float = DFTBPLUS_DIST_FUDGE,
        n_points: int = DFTBPLUS_N_INTER,
        delta_r: float = DFTBPLUS_DELTA_R,
    ):
        self.x = x
        self._y = y
        self._y_is_given = y is not None

        # Store parameters
        self.dist_fudge_bohr = dist_fudge
        self.dist_fudge_ang = dist_fudge * BOHR_TO_ANG
        self.n_points = n_points
        self.delta_r_bohr = delta_r
        self.delta_r_ang = delta_r * BOHR_TO_ANG

        # Grid properties
        self.x_min = x.min().item()
        self.x_max = x.max().item()
        n_grid = x.shape[0]
        self.grid_spacing = (self.x_max - self.x_min) / (n_grid - 1) if n_grid > 1 else 1.0
        self.n_grid = n_grid

        # Effective cutoff
        self.x_cutoff = self.x_max + self.dist_fudge_ang

        # Initialize interpolators
        self.poly_interp = NevillePolyInterp(n_points=n_points, n_right=n_points // 2)
        self.poly5_decay = Poly5ToZero()

    def __call__(
        self,
        xq: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Interpolate at query points.

        Parameters
        ----------
        xq : torch.Tensor
            Query points with shape [n_query].
        y : torch.Tensor, optional
            Values at grid points [n_channels, n_grid].
            If y was provided at init, this is ignored.

        Returns
        -------
        torch.Tensor
            Interpolated values with shape [n_channels, n_query].
        """
        if self._y_is_given:
            y = self._y
        elif y is None:
            raise ValueError("y must be provided either at init or call time")

        x = self.x.to(dtype=y.dtype, device=y.device)
        xq = xq.to(dtype=y.dtype, device=y.device)

        n_channels = y.shape[0]
        n_query = xq.shape[0]

        # Initialize output
        result = torch.zeros(n_channels, n_query, dtype=y.dtype, device=y.device)

        # Create masks for different regions
        # DFTB+ boundary conditions (slakoeqgrid.F90):
        # - Grid region: ind < nGrid, i.e., rr < nGrid*dist (exclusive of last grid point)
        # - Decay region: rr >= nGrid*dist AND rr < rMax (inclusive at boundary, exclusive at cutoff)
        # - Zero region: rr >= rMax (inclusive at cutoff)

        mask_grid = (xq >= self.x_min) & (xq < self.x_max)  # exclusive at x_max
        mask_decay = (xq >= self.x_max) & (xq < self.x_cutoff)  # inclusive at x_max, exclusive at cutoff


        # Region 1: Below grid -> zero (already initialized)

        # Region 2: Grid interpolation
        if mask_grid.any():
            xq_grid = xq[mask_grid]
            result[:, mask_grid] = self.poly_interp(x, y, xq_grid)

        # Region 3: Decay zone
        if mask_decay.any():
            xq_decay = xq[mask_decay]
            result[:, mask_decay] = self._compute_decay(xq_decay, y, x)

        # Region 4: Above cutoff -> zero (already initialized)

        return result

    def _compute_decay(
        self,
        xq: torch.Tensor,
        y: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute values in the decay zone using poly5ToZero.

        This method matches DFTB+ behavior exactly:
        - y1 (center value) uses the actual table value at nGrid, NOT interpolated
        - y0, y2 (values at x_max ± delta) are interpolated
        - Derivatives are computed from these three values

        Reference: DFTB+ slakoeqgrid.F90 (SlakoEqGrid_interNew_)

        Parameters
        ----------
        xq : torch.Tensor
            Query points in decay zone [n_decay].
        y : torch.Tensor
            Values at grid points [n_channels, n_grid].

        Returns
        -------
        torch.Tensor
            Decay values [n_channels, n_decay].
        """
        # DFTB+ behavior: use actual table value at last grid point (not interpolated)
        # y1 = skTab(nGrid, ii)
        f = y[:, -1]  # [n_channels] - actual table value at x_max

        # Interpolate at x_max - delta and x_max + delta for derivative computation
        # y0 = polyInterUniform(xa, yb, xa(8) - deltaR)
        # y2 = polyInterUniform(xa, yb, xa(8) + deltaR)
        xq_deriv = torch.tensor(
            [self.x_max - self.delta_r_ang, self.x_max + self.delta_r_ang],
            dtype=x.dtype, device=x.device
        )
        vals_deriv = self.poly_interp(x, y, xq_deriv)  # [n_channels, 2]
        f_minus = vals_deriv[:, 0]  # y0
        f_plus = vals_deriv[:, 1]   # y2

        # Compute derivatives in the original xq-coordinate (DFTB+ convention)
        # y1p = (y2 - y0) / (2 * deltaR)
        # y1pp = (y2 + y0 - 2*y1) / deltaR²
        fp_xq = (f_plus - f_minus) / (2.0 * self.delta_r_ang)
        fpp_xq = (f_plus + f_minus - 2.0 * f) / (self.delta_r_ang * self.delta_r_ang)

        # Distance from cutoff (x_cutoff) to query points
        # In DFTB+ convention: decay starts at x_max, ends at x_cutoff
        # We parameterize as distance from cutoff (which is zero at x_cutoff)
        x_from_cutoff = self.x_cutoff - xq  # [n_decay], positive values

        # Distance from cutoff to boundary (x_max)
        x_boundary_from_cutoff = self.x_cutoff - self.x_max  # = dist_fudge_ang

        # Coordinate transformation for derivatives:
        # In DFTB+: xx = rr - rMax, dx = -distFudge (both negative in decay zone)
        # In dptb: x = x_cutoff - xq (positive), x_boundary = dist_fudge (positive)
        #
        # The coordinate transformation is: x = x_cutoff - xq
        # Therefore: dx/dxq = -1
        # So: df/dx = df/dxq * dxq/dx = fp_xq * (-1) = -fp_xq
        # And: d²f/dx² = d/dx(df/dx) = d(-fp_xq)/dx = -d(fp_xq)/dxq * dxq/dx
        #            = -fpp_xq * (-1) = fpp_xq
        #
        # This matches DFTB+ where dx1 = y0p * (-distFudge) effectively negates fp.
        fp = -fp_xq  # Negate first derivative for coordinate transformation
        fpp = fpp_xq  # Second derivative unchanged (double negation)

        # Evaluate poly5ToZero
        result = self.poly5_decay(
            x=x_from_cutoff,
            x_boundary=x_boundary_from_cutoff,
            f=f,
            fp=fp,
            fpp=fpp
        )

        return result

    def get_cutoff(self) -> float:
        """
        Get the effective cutoff distance.

        Returns
        -------
        float
            Cutoff distance in Angstrom (x_max + dist_fudge).
        """
        return self.x_cutoff

    def interp_single(
        self,
        xq_val: float,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpolate at a single query point.

        This is a convenience method for single-point queries.

        Parameters
        ----------
        xq_val : float
            Single query point value.
        y : torch.Tensor
            Values at grid points [n_channels, n_grid].

        Returns
        -------
        torch.Tensor
            Interpolated values [n_channels].
        """
        xq = torch.tensor([xq_val], dtype=self.x.dtype, device=self.x.device)
        return self(xq, y).squeeze(1)

    def getparamnames(self):
        """Return parameter names for gradient tracking."""
        if self._y_is_given:
            return ["x", "_y"]
        return ["x"]


def create_dftbplus_interp(
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    dist_fudge: float = DFTBPLUS_DIST_FUDGE,
) -> DFTBPlusInterp1D:
    """
    Factory function to create a DFTB+-compatible interpolator.

    Parameters
    ----------
    x : torch.Tensor
        Grid points [n_grid].
    y : torch.Tensor, optional
        Values at grid points [n_channels, n_grid].
    dist_fudge : float, optional
        Extrapolation zone size in Bohr. Default is 1.0.

    Returns
    -------
    DFTBPlusInterp1D
        Configured interpolator instance.
    """
    return DFTBPlusInterp1D(x=x, y=y, dist_fudge=dist_fudge)
