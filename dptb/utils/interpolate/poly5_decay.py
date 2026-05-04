"""
5th-order polynomial decay to zero with C² continuity.

This module implements the poly5ToZero function used by DFTB+ for smooth
extrapolation of SK integrals beyond the tabulated grid range. The polynomial
ensures that the function smoothly decays to zero while maintaining continuous
first and second derivatives.

Reference: DFTB+ source file src/dftbp/math/interpolation.F90
"""

import torch
from typing import Tuple


class Poly5ToZero:
    """
    5th-order polynomial decay to zero with C² continuity.

    This class implements the smooth extrapolation used by DFTB+ beyond
    the SK integral grid. The polynomial is constructed to satisfy:

    At the boundary (x = x_boundary):
        p(x_boundary) = f
        p'(x_boundary) = fp
        p''(x_boundary) = fpp

    At the cutoff (x = 0, after coordinate shift):
        p(0) = 0
        p'(0) = 0
        p''(0) = 0

    This ensures C² continuity at both ends.

    Notes
    -----
    The DFTB+ implementation uses a coordinate system where:
    - x = 0 is the cutoff (beyond which everything is zero)
    - x = dx (negative in DFTB+ convention) is the boundary

    In our implementation, we use:
    - x = distance from cutoff (positive)
    - x_boundary = distance from cutoff to boundary (positive)

    References
    ----------
    DFTB+ source: src/dftbp/math/interpolation.F90 (poly5ToZero)
    """

    def __init__(self):
        pass

    def __call__(
        self,
        x: torch.Tensor,
        x_boundary: float,
        f: torch.Tensor,
        fp: torch.Tensor,
        fpp: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate the 5th-order polynomial decay.

        Parameters
        ----------
        x : torch.Tensor
            Query points (distance from cutoff) [n_query].
            Should be in range [0, x_boundary].
        x_boundary : float
            Distance from cutoff to boundary (where f, fp, fpp are defined).
        f : torch.Tensor
            Function value at boundary [n_channels].
        fp : torch.Tensor
            First derivative at boundary [n_channels].
        fpp : torch.Tensor
            Second derivative at boundary [n_channels].

        Returns
        -------
        torch.Tensor
            Polynomial values at query points [n_channels, n_query].
        """
        # Compute polynomial coefficients
        # Following DFTB+ poly5ToZero exactly:
        # dx1 = y0p * dx
        # dx2 = y0pp * dx * dx
        # dd =  10 * y0 - 4 * dx1 + 0.5 * dx2
        # ee = -15 * y0 + 7 * dx1 - 1.0 * dx2
        # ff =   6 * y0 - 3 * dx1 + 0.5 * dx2

        dx = x_boundary
        dx1 = fp * dx                    # [n_channels]
        dx2 = fpp * dx * dx              # [n_channels]

        dd = 10.0 * f - 4.0 * dx1 + 0.5 * dx2   # [n_channels]
        ee = -15.0 * f + 7.0 * dx1 - 1.0 * dx2  # [n_channels]
        ff = 6.0 * f - 3.0 * dx1 + 0.5 * dx2    # [n_channels]

        # Normalize query points
        assert dx > 1e-12, "x_boundary must be positive and non-zero."
        inv_dx = 1.0 / dx
        xr = x * inv_dx  # [n_query]

        # Evaluate polynomial: p(x) = ((ff*xr + ee)*xr + dd) * xr^3
        # Broadcast: [n_channels, 1] * [1, n_query] -> [n_channels, n_query]
        xr = xr.unsqueeze(0)  # [1, n_query]
        dd = dd.unsqueeze(1)  # [n_channels, 1]
        ee = ee.unsqueeze(1)  # [n_channels, 1]
        ff = ff.unsqueeze(1)  # [n_channels, 1]

        result = ((ff * xr + ee) * xr + dd) * xr * xr * xr

        return result

    def compute_coefficients(
        self,
        x_boundary: float,
        f: torch.Tensor,
        fp: torch.Tensor,
        fpp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the polynomial coefficients.

        Parameters
        ----------
        x_boundary : float
            Distance from cutoff to boundary.
        f : torch.Tensor
            Function value at boundary [n_channels].
        fp : torch.Tensor
            First derivative at boundary [n_channels].
        fpp : torch.Tensor
            Second derivative at boundary [n_channels].

        Returns
        -------
        dd, ee, ff : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Polynomial coefficients, each with shape [n_channels].

        Notes
        -----
        The polynomial is: p(xr) = (dd + ee*xr + ff*xr²) * xr³
        where xr = x / x_boundary is the normalized coordinate.
        """
        dx = x_boundary
        dx1 = fp * dx
        dx2 = fpp * dx * dx

        dd = 10.0 * f - 4.0 * dx1 + 0.5 * dx2
        ee = -15.0 * f + 7.0 * dx1 - 1.0 * dx2
        ff = 6.0 * f - 3.0 * dx1 + 0.5 * dx2

        return dd, ee, ff

    def eval_with_derivatives(
        self,
        x: torch.Tensor,
        x_boundary: float,
        f: torch.Tensor,
        fp: torch.Tensor,
        fpp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate polynomial and its derivatives.

        Parameters
        ----------
        x : torch.Tensor
            Query points [n_query].
        x_boundary : float
            Distance from cutoff to boundary.
        f, fp, fpp : torch.Tensor
            Function value and derivatives at boundary [n_channels].

        Returns
        -------
        p, pp, ppp : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Polynomial value, first and second derivatives at query points.
            Each has shape [n_channels, n_query].
        """
        dd, ee, ff = self.compute_coefficients(x_boundary, f, fp, fpp)

        inv_dx = 1.0 / x_boundary
        xr = x * inv_dx  # [n_query]

        # Broadcast
        xr = xr.unsqueeze(0)
        dd = dd.unsqueeze(1)
        ee = ee.unsqueeze(1)
        ff = ff.unsqueeze(1)

        # p(xr) = (dd + ee*xr + ff*xr²) * xr³
        # Let q(xr) = dd + ee*xr + ff*xr²
        # p(xr) = q(xr) * xr³

        q = dd + ee * xr + ff * xr * xr
        p = q * xr * xr * xr

        # p'(xr) = q'(xr)*xr³ + q(xr)*3*xr²
        # q'(xr) = ee + 2*ff*xr
        qp = ee + 2.0 * ff * xr
        pp_xr = qp * xr * xr * xr + q * 3.0 * xr * xr
        pp = pp_xr * inv_dx  # Chain rule: dp/dx = dp/dxr * dxr/dx

        # p''(xr) = q''(xr)*xr³ + 2*q'(xr)*3*xr² + q(xr)*6*xr
        # q''(xr) = 2*ff
        qpp = 2.0 * ff
        ppp_xr = qpp * xr * xr * xr + 2.0 * qp * 3.0 * xr * xr + q * 6.0 * xr
        ppp = ppp_xr * inv_dx * inv_dx

        return p, pp, ppp
