"""
Neville's polynomial interpolation algorithm.

This module implements the polynomial interpolation used by DFTB+ for
Slater-Koster integral tables. The algorithm is based on Neville's method
which is numerically stable and efficient for moderate numbers of points.

Reference: DFTB+ source file src/dftbp/math/interpolation.F90
"""

import torch
import math
from typing import Tuple


class NevillePolyInterp:
    """
    N-point polynomial interpolation using Neville's algorithm.

    This class implements the same interpolation algorithm used by DFTB+
    for SK integral tables. Given n points, it constructs a polynomial
    of degree n-1 passing through all points.

    Parameters
    ----------
    n_points : int, optional
        Number of interpolation points. Default is 8 (DFTB+ modern method).
    n_right : int, optional
        Number of points to prefer on the right side of query point.
        Default is 4 (DFTB+ modern method).

    Attributes
    ----------
    n_points : int
        Number of interpolation points used.
    n_right : int
        Number of points preferred on the right side.

    Notes
    -----
    The algorithm complexity is O(n²) per query point, where n is n_points.
    For batched queries, the algorithm is vectorized over channels but
    loops over query points for numerical stability.

    References
    ----------
    - Numerical Recipes: The Art of Scientific Computing
    - DFTB+ source: src/dftbp/math/interpolation.F90 (polyInterUniform)
    """

    def __init__(self, n_points: int = 8, n_right: int = 4):
        if not isinstance(n_points, int) or n_points < 2:
            raise ValueError("n_points must be an integer greater than or equal to 2.")
        if not isinstance(n_right, int) or n_right < 0 or n_right > n_points:
            raise ValueError("n_right must be an integer between 0 and n_points.")
        self.n_points = n_points
        self.n_right = n_right

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        xq: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpolate y values at query points xq.

        Parameters
        ----------
        x : torch.Tensor
            Grid points with shape [n_grid]. Must be uniformly spaced.
        y : torch.Tensor
            Values at grid points with shape [n_channels, n_grid].
        xq : torch.Tensor
            Query points with shape [n_query].

        Returns
        -------
        torch.Tensor
            Interpolated values with shape [n_channels, n_query].

        Notes
        -----
        The grid x must be uniformly spaced for the algorithm to work correctly.
        Query points outside the grid range will still be interpolated
        (extrapolation), but results may be inaccurate.
        """
        if x.ndim != 1:
            raise ValueError("x must be a 1D tensor.")
        if xq.ndim != 1:
            raise ValueError("xq must be a 1D tensor.")
        if y.ndim != 2:
            raise ValueError("y must be a 2D tensor with shape [n_channels, n_grid].")
        n_grid = x.shape[0]
        n_channels = y.shape[0]
        n_query = xq.shape[0]
        if y.shape[1] != n_grid:
            raise ValueError("y.shape[1] must match x.shape[0].")
        if n_grid < self.n_points:
            raise ValueError("x must contain at least n_points grid points.")
        if not torch.isfinite(x).all():
            raise ValueError("x must contain only finite values.")
        dx = x[1:] - x[:-1]
        if not (dx > 0).all():
            raise ValueError("x must be strictly increasing.")
        if not torch.allclose(dx, dx[0].expand_as(dx), rtol=1e-4, atol=1e-12):
            raise ValueError("x must be uniformly spaced.")

        # Grid spacing (assumes uniform grid)
        grid_spacing = (x[-1] - x[0]) / (n_grid - 1)
        x_min = x[0]

        # Output tensor
        result = torch.zeros(n_channels, n_query, dtype=y.dtype, device=y.device)

        # Process each query point
        for iq in range(n_query):
            xq_val = xq[iq]

            # Find grid index (use floor to match DFTB+ exactly)
            # DFTB+: ind = floor(rr / dist)
            ind = math.floor(((xq_val - x_min) / grid_spacing).item())

            # Select interpolation points centered around query
            # iLast = min(n_grid, ind + n_right)
            # iLast = max(iLast, n_points)
            i_last = min(n_grid, ind + self.n_right)
            i_last = max(i_last, self.n_points)
            i_first = i_last - self.n_points

            # Ensure indices are valid
            if i_first < 0:
                i_first = 0
                i_last = min(self.n_points, n_grid)

            # Extract local x and y values
            x_local = x[i_first:i_last]
            y_local = y[:, i_first:i_last]

            # Interpolate using Neville's algorithm
            result[:, iq] = self._neville_interp_vectorized(x_local, y_local, xq_val)

        return result

    def _neville_interp_vectorized(
        self,
        x_local: torch.Tensor,
        y_local: torch.Tensor,
        xq_val: torch.Tensor
    ) -> torch.Tensor:
        """
        Neville's algorithm vectorized over channels.

        Parameters
        ----------
        x_local : torch.Tensor
            Local x values [n_points].
        y_local : torch.Tensor
            Local y values [n_channels, n_points].
        xq_val : float
            Single query point value.

        Returns
        -------
        torch.Tensor
            Interpolated values [n_channels].
        """
        _, nn = y_local.shape

        # Initialize C and D tableaux
        cc = y_local.clone()  # [n_channels, nn]
        dd = y_local.clone()  # [n_channels, nn]

        # Find closest point for initial estimate
        # Matches DFTB+ polyInterUniform exactly (src/dftbp/math/interpolation.F90)
        # iCl = ceiling((xx-xp(1))/abs(xp(2)-xp(1)))  [Fortran, 1-indexed]
        if nn > 1:
            dx = (x_local[-1] - x_local[0]) / (nn - 1)
            i_cl = int(torch.ceil((xq_val - x_local[0]) / dx).item())
            i_cl = max(1, min(i_cl, nn))  # Clamp to valid 1-indexed range [1, nn]
        else:
            i_cl = 1

        # Initial estimate from closest point (convert to 0-indexed for array access)
        yy = y_local[:, i_cl - 1].clone()  # [n_channels]
        # Shift column pointer (matches Fortran: iCl = iCl - 1)
        # i_cl now represents the column in Neville tableau (1-indexed semantics)
        i_cl = i_cl - 1

        # Neville's algorithm iteration
        # Matches DFTB+ exactly: do mm = 1, nn - 1
        for mm in range(1, nn):
            # Inner loop: do ii = 1, nn - mm (Fortran 1-indexed)
            # In Python 0-indexed: ii goes from 0 to nn-mm-1
            for ii in range(nn - mm):
                # Compute divided difference
                # Fortran: rTmp = (cc(ii+1) - dd(ii)) / (xp(ii) - xp(ii+mm))
                # Python 0-indexed: cc[ii+1] corresponds to Fortran cc(ii+2), but
                # since Fortran ii starts at 1, Fortran cc(ii+1) = Python cc[ii]
                # Wait, this is getting confusing. Let me just match the array access:
                # Fortran ii=1: cc(2)-dd(1) -> Python ii=0: cc[1]-dd[0] ✓
                x_diff = x_local[ii] - x_local[ii + mm]
                if torch.abs(x_diff) <= torch.finfo(x_local.dtype).eps:
                    raise ValueError("Interpolation grid contains duplicate or indistinguishable local points.")
                r_tmp = (cc[:, ii + 1] - dd[:, ii]) / x_diff

                cc[:, ii] = (x_local[ii] - xq_val) * r_tmp
                dd[:, ii] = (x_local[ii + mm] - xq_val) * r_tmp

            # Select correction from C or D tableau
            # Fortran: if (2 * iCl < nn - mm) then dyy = cc(iCl + 1)
            #          else dyy = dd(iCl); iCl = iCl - 1
            # i_cl is in 1-indexed semantics (matches Fortran iCl after the -1 shift)
            # So cc(iCl + 1) in Fortran = cc[:, i_cl] in Python (since i_cl+1-1 = i_cl)
            # And dd(iCl) in Fortran = dd[:, i_cl - 1] in Python
            if 2 * i_cl < nn - mm:
                dyy = cc[:, i_cl]  # Fortran: cc(iCl + 1)
            else:
                dyy = dd[:, i_cl - 1]  # Fortran: dd(iCl)
                i_cl = i_cl - 1

            yy = yy + dyy

        return yy

    def interp_with_derivatives(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        xq: float,
        delta_r: float = 1e-5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Interpolate and compute numerical derivatives at a single point.

        Parameters
        ----------
        x : torch.Tensor
            Grid points [n_grid].
        y : torch.Tensor
            Values at grid points [n_channels, n_grid].
        xq : float
            Query point.
        delta_r : float, optional
            Step size for numerical derivatives. Default is 1e-5.

        Returns
        -------
        f : torch.Tensor
            Function value at xq [n_channels].
        fp : torch.Tensor
            First derivative at xq [n_channels].
        fpp : torch.Tensor
            Second derivative at xq [n_channels].
        """
        if not torch.isfinite(torch.tensor(delta_r)) or delta_r <= 0:
            raise ValueError("delta_r must be positive and finite.")
        original_dtype = y.dtype
        if y.dtype == torch.float32:
            x_eval = x.to(torch.float64)
            y_eval = y.to(torch.float64)
            xq_dtype = torch.float64
        else:
            x_eval = x
            y_eval = y
            xq_dtype = x.dtype

        # Evaluate at xq - delta, xq, xq + delta
        xq_tensor = torch.tensor([xq - delta_r, xq, xq + delta_r],
                                  dtype=xq_dtype, device=x.device)

        vals = self(x_eval, y_eval, xq_tensor)  # [n_channels, 3]

        f_minus = vals[:, 0]
        f_val = vals[:, 1]
        f_plus = vals[:, 2]

        # Numerical derivatives
        fp = (f_plus - f_minus) / (2.0 * delta_r)
        fpp = (f_plus + f_minus - 2.0 * f_val) / (delta_r * delta_r)

        return f_val.to(original_dtype), fp.to(original_dtype), fpp.to(original_dtype)
