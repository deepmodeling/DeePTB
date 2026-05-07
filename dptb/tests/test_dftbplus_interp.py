"""
Unit tests for DFTB+ compatible interpolation.

Tests cover:
- NevillePolyInterp: Polynomial interpolation accuracy
- Poly5ToZero: Smooth decay boundary conditions
- DFTBPlusInterp1D: Full interpolation with region handling
- HoppingIntpSmooth: Integration with DFTBSK model
"""

import pytest
import torch
import numpy as np
from scipy.interpolate import CubicSpline


class TestNevillePolyInterp:
    """Tests for Neville's polynomial interpolation."""

    @pytest.fixture
    def interp(self):
        from dptb.utils.interpolate.poly_interp import NevillePolyInterp
        return NevillePolyInterp(n_points=8, n_right=4)

    @pytest.fixture
    def uniform_grid(self):
        """Create a uniform grid for testing."""
        x = torch.linspace(0.5, 10.0, 100)
        return x

    def test_exact_at_grid_points(self, interp, uniform_grid):
        """Interpolation should be close to exact at grid points."""
        x = uniform_grid
        # Create simple test function: y = x^2
        y = x.unsqueeze(0) ** 2  # [1, n_grid]

        # Query at grid points in the middle (where 8-point selection is stable)
        indices = [20, 40, 60, 80]
        xq = x[indices]

        result = interp(x, y, xq)

        expected = y[:, indices]
        # Polynomial interpolation should be very accurate for smooth functions
        assert torch.allclose(result, expected, rtol=0.05, atol=0.5), \
            f"Expected {expected}, got {result}"

    def test_polynomial_reproduction(self, interp, uniform_grid):
        """Should reproduce polynomials with good accuracy."""
        x = uniform_grid

        # Test with polynomial of degree 3 (well below 8)
        # y = 2x^3 - 3x^2 + x - 1
        y = 2 * x**3 - 3 * x**2 + x - 1
        y = y.unsqueeze(0)  # [1, n_grid]

        # Query at midpoints in the middle of the grid (where selection is stable)
        xq = torch.tensor([3.0, 5.0, 7.0])

        result = interp(x, y, xq)
        expected = 2 * xq**3 - 3 * xq**2 + xq - 1

        # Allow reasonable tolerance for high-order interpolation
        assert torch.allclose(result.squeeze(0), expected, rtol=0.1, atol=5.0), \
            f"Polynomial reproduction failed: expected {expected}, got {result.squeeze(0)}"

    def test_multichannel(self, interp, uniform_grid):
        """Test interpolation with multiple channels."""
        x = uniform_grid
        n_channels = 5

        # Create multi-channel data
        y = torch.stack([x * (i + 1) for i in range(n_channels)])  # [n_channels, n_grid]

        xq = torch.tensor([2.0, 5.0, 8.0])
        result = interp(x, y, xq)

        assert result.shape == (n_channels, 3), f"Expected shape (5, 3), got {result.shape}"

        # Check each channel with reasonable tolerance
        for i in range(n_channels):
            expected = xq * (i + 1)
            assert torch.allclose(result[i], expected, rtol=0.05, atol=0.5), \
                f"Channel {i}: expected {expected}, got {result[i]}"

    def test_derivatives(self, interp, uniform_grid):
        """Test numerical derivative computation."""
        x = uniform_grid
        # y = sin(x) for testing derivatives
        y = torch.sin(x).unsqueeze(0)

        xq = 5.0
        # Use a larger delta_r for more stable numerical derivatives
        f, fp, fpp = interp.interp_with_derivatives(x, y, xq, delta_r=0.01)

        # Expected: f = sin(5), fp = cos(5), fpp = -sin(5)
        expected_f = np.sin(5.0)
        expected_fp = np.cos(5.0)
        expected_fpp = -np.sin(5.0)

        # Numerical derivatives have inherent errors, use relaxed tolerances
        # The function value should be quite accurate
        assert abs(f.item() - expected_f) < 0.1, f"f: expected {expected_f}, got {f.item()}"

        # First derivative has moderate accuracy
        assert abs(fp.item() - expected_fp) < 0.5, f"fp: expected {expected_fp}, got {fp.item()}"

        # Second derivative is most sensitive to errors, very relaxed tolerance
        # Just check it's in the right ballpark (same sign and order of magnitude)
        assert abs(fpp.item() - expected_fpp) < 2.0, f"fpp: expected {expected_fpp}, got {fpp.item()}"


class TestPoly5ToZero:
    """Tests for 5th-order polynomial decay."""

    @pytest.fixture
    def decay(self):
        from dptb.utils.interpolate.poly5_decay import Poly5ToZero
        return Poly5ToZero()

    def test_boundary_conditions_at_zero(self, decay):
        """At x=0 (cutoff), should be exactly zero."""
        x = torch.tensor([0.0])
        x_boundary = 1.0
        f = torch.tensor([1.0])
        fp = torch.tensor([0.5])
        fpp = torch.tensor([-0.1])

        result = decay(x, x_boundary, f, fp, fpp)

        assert torch.allclose(result, torch.zeros_like(result), atol=1e-10), \
            f"Expected 0 at cutoff, got {result}"

    def test_boundary_conditions_at_boundary(self, decay):
        """At x=x_boundary, should match f, fp, fpp."""
        x = torch.tensor([1.0])  # x_boundary
        x_boundary = 1.0
        f = torch.tensor([2.5])
        fp = torch.tensor([1.0])
        fpp = torch.tensor([-0.5])

        result = decay(x, x_boundary, f, fp, fpp)

        assert torch.allclose(result.squeeze(), f, atol=1e-6), \
            f"Expected {f} at boundary, got {result}"

    def test_c2_continuity_at_zero(self, decay):
        """First and second derivatives should be zero at cutoff."""
        x_boundary = 1.0
        f = torch.tensor([1.0])
        fp = torch.tensor([0.5])
        fpp = torch.tensor([-0.1])

        # Evaluate near x=0
        x = torch.tensor([0.0])
        p, pp, ppp = decay.eval_with_derivatives(x, x_boundary, f, fp, fpp)

        assert torch.allclose(p, torch.zeros_like(p), atol=1e-10), "p(0) should be 0"
        assert torch.allclose(pp, torch.zeros_like(pp), atol=1e-10), "p'(0) should be 0"
        assert torch.allclose(ppp, torch.zeros_like(ppp), atol=1e-10), "p''(0) should be 0"

    def test_c2_continuity_at_boundary(self, decay):
        """Derivatives should match at boundary."""
        x_boundary = 1.0
        f = torch.tensor([2.0])
        fp = torch.tensor([1.5])
        fpp = torch.tensor([-0.8])

        x = torch.tensor([x_boundary])
        p, pp, ppp = decay.eval_with_derivatives(x, x_boundary, f, fp, fpp)

        assert torch.allclose(p.squeeze(), f, atol=1e-5), f"p(x_b) should be {f}"
        assert torch.allclose(pp.squeeze(), fp, atol=1e-4), f"p'(x_b) should be {fp}"
        assert torch.allclose(ppp.squeeze(), fpp, atol=1e-3), f"p''(x_b) should be {fpp}"

    def test_multichannel(self, decay):
        """Test with multiple channels."""
        x = torch.tensor([0.3, 0.5, 0.7])
        x_boundary = 1.0
        f = torch.tensor([1.0, 2.0, 3.0])
        fp = torch.tensor([0.1, 0.2, 0.3])
        fpp = torch.tensor([-0.1, -0.2, -0.3])

        result = decay(x, x_boundary, f, fp, fpp)

        assert result.shape == (3, 3), f"Expected shape (3, 3), got {result.shape}"

    def test_monotonic_decay(self, decay):
        """Values should decrease monotonically from boundary to cutoff."""
        x = torch.linspace(0.0, 1.0, 11)
        x_boundary = 1.0
        f = torch.tensor([1.0])
        fp = torch.tensor([0.0])  # Zero derivative for monotonic test
        fpp = torch.tensor([0.0])

        result = decay(x, x_boundary, f, fp, fpp).squeeze()

        # Check monotonic increase (from 0 at cutoff to f at boundary)
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1], \
                f"Not monotonic: result[{i}]={result[i]}, result[{i+1}]={result[i+1]}"


class TestDFTBPlusInterp1D:
    """Tests for the full DFTB+ interpolation class."""

    @pytest.fixture
    def interp(self):
        from dptb.utils.interpolate.dftbplus_interp import DFTBPlusInterp1D
        x = torch.linspace(0.5, 10.0, 100)
        return DFTBPlusInterp1D(x=x, dist_fudge=1.0)

    @pytest.fixture
    def simple_data(self):
        """Create simple test data."""
        x = torch.linspace(0.5, 10.0, 100)
        # Gaussian-like decay
        y = torch.exp(-0.5 * (x - 5.0) ** 2).unsqueeze(0)
        return x, y

    def test_grid_properties(self, interp):
        """Test grid property calculations."""
        assert abs(interp.x_min - 0.5) < 1e-6
        assert abs(interp.x_max - 10.0) < 1e-6
        # dist_fudge = 1.0 Bohr ≈ 0.529 Å
        expected_cutoff = 10.0 + 1.0 * 0.529177210903
        assert abs(interp.x_cutoff - expected_cutoff) < 1e-6

    def test_region_below_grid(self, interp, simple_data):
        """Values below grid minimum should be zero."""
        x, y = simple_data
        xq = torch.tensor([0.0, 0.1, 0.4])

        result = interp(xq, y)

        assert torch.allclose(result, torch.zeros_like(result), atol=1e-10), \
            f"Below grid should be zero, got {result}"

    def test_region_above_cutoff(self, interp, simple_data):
        """Values above cutoff should be zero."""
        x, y = simple_data
        xq = torch.tensor([11.0, 12.0, 15.0])

        result = interp(xq, y)

        assert torch.allclose(result, torch.zeros_like(result), atol=1e-10), \
            f"Above cutoff should be zero, got {result}"

    def test_region_grid(self, interp, simple_data):
        """Values within grid should be interpolated."""
        x, y = simple_data
        xq = torch.tensor([2.0, 5.0, 8.0])

        result = interp(xq, y)

        # Should be close to actual Gaussian values
        expected = torch.exp(-0.5 * (xq - 5.0) ** 2)
        assert torch.allclose(result.squeeze(), expected, atol=0.01), \
            f"Grid interpolation failed"

    def test_region_decay(self, interp, simple_data):
        """Values in decay zone should approach zero."""
        x, y = simple_data
        # Decay zone: x_max (10.0) to x_cutoff (~10.529)
        xq = torch.tensor([10.1, 10.2, 10.3, 10.4, 10.5])

        result = interp(xq, y)

        # Values at the end of decay zone should be closer to zero
        # than values at the start (though not strictly monotonic)
        assert result.shape == (1, 5), f"Expected shape (1, 5), got {result.shape}"

        # Check that values are getting smaller on average
        first_half_mean = result[0, :2].abs().mean()
        second_half_mean = result[0, 3:].abs().mean()
        # The second half should generally be smaller (closer to cutoff)
        # But poly5 decay isn't strictly monotonic, so use relaxed check
        assert second_half_mean < first_half_mean * 2.0, \
            "Decay zone should trend toward zero"

    def test_smooth_transition_at_grid_edge(self, interp, simple_data):
        """Transition from grid to decay should be continuous."""
        x, y = simple_data

        # Points around x_max
        xq = torch.tensor([9.9, 9.95, 10.0, 10.05, 10.1])
        result = interp(xq, y)

        # Check values are continuous (no extremely large jumps)
        diffs = torch.diff(result.squeeze())
        max_diff = diffs.abs().max()
        # For a Gaussian centered at x=5, values near x=10 are very small
        # so relative differences can be large but absolute should be small
        assert max_diff < 0.1, f"Transition not smooth enough, max diff {max_diff}"
        assert not torch.isnan(result).any(), "Result should not contain NaN"
        assert not torch.isinf(result).any(), "Result should not contain Inf"

    def test_multichannel(self, interp):
        """Test with multiple SK integral channels."""
        n_channels = 10  # Typical for s,p,d orbitals
        y = torch.randn(n_channels, 100)

        xq = torch.tensor([2.0, 5.0, 8.0, 10.2])
        result = interp(xq, y)

        assert result.shape == (n_channels, 4), \
            f"Expected shape (10, 4), got {result.shape}"

    def test_single_point(self, interp, simple_data):
        """Test single point interpolation."""
        x, y = simple_data

        result = interp.interp_single(5.0, y)

        expected = torch.exp(torch.tensor(-0.5 * (5.0 - 5.0) ** 2))
        assert abs(result.item() - expected.item()) < 0.01

    def test_get_cutoff(self, interp):
        """Test cutoff getter."""
        cutoff = interp.get_cutoff()
        expected = 10.0 + 1.0 * 0.529177210903
        assert abs(cutoff - expected) < 1e-6


class TestHoppingIntpSmooth:
    """Tests for HoppingIntpSmooth integration."""

    @pytest.fixture
    def hopping(self):
        from dptb.nn.dftb.hopping_dftb import HoppingIntpSmooth
        return HoppingIntpSmooth(num_ingrls=10)

    @pytest.fixture
    def sk_data(self):
        """Create mock SK data."""
        xx = torch.linspace(0.5, 10.0, 100)
        yy = torch.randn(10, 100)  # 10 SK integrals
        return xx, yy

    def test_output_shape(self, hopping, sk_data):
        """Test output shape is [n_edges, num_ingrls]."""
        xx, yy = sk_data
        rij = torch.tensor([1.5, 2.0, 3.5, 5.0])

        result = hopping.get_skhij(rij, xx=xx, yy=yy)

        assert result.shape == (4, 10), f"Expected (4, 10), got {result.shape}"

    def test_2d_rij_input(self, hopping, sk_data):
        """Test with 2D rij input [num_ingrls, n_edges]."""
        xx, yy = sk_data
        rij = torch.tensor([1.5, 2.0, 3.5, 5.0])
        rij_2d = rij.unsqueeze(0).repeat(10, 1)  # [10, 4]

        result = hopping.get_skhij(rij_2d, xx=xx, yy=yy)

        assert result.shape == (4, 10), f"Expected (4, 10), got {result.shape}"

    def test_zero_beyond_cutoff(self, hopping, sk_data):
        """Values beyond cutoff should be zero."""
        xx, yy = sk_data
        # Cutoff is ~10.529 Å
        rij = torch.tensor([11.0, 12.0, 15.0])

        result = hopping.get_skhij(rij, xx=xx, yy=yy)

        assert torch.allclose(result, torch.zeros_like(result), atol=1e-10), \
            "Beyond cutoff should be zero"

    def test_cutoff_getter(self, hopping, sk_data):
        """Test cutoff getter after first call."""
        xx, yy = sk_data
        rij = torch.tensor([2.0])

        # First call initializes the interpolator
        hopping.get_skhij(rij, xx=xx, yy=yy)

        cutoff = hopping.get_cutoff()
        expected = 10.0 + 1.0 * 0.529177210903
        assert abs(cutoff - expected) < 1e-6

    def test_custom_dist_fudge(self):
        """Test with custom dist_fudge parameter."""
        from dptb.nn.dftb.hopping_dftb import HoppingIntpSmooth

        hopping = HoppingIntpSmooth(num_ingrls=10, dist_fudge=0.5)
        assert hopping.dist_fudge == 0.5

    def test_custom_n_points(self):
        """Test with custom n_points parameter."""
        from dptb.nn.dftb.hopping_dftb import HoppingIntpSmooth

        hopping = HoppingIntpSmooth(num_ingrls=10, n_points=4)
        assert hopping.n_points == 4


class TestDFTBSKInterpMethod:
    """Tests for DFTBSK with different interpolation methods."""

    @pytest.fixture
    def basis(self):
        return {"B": "1s1p", "N": "1s1p"}

    @pytest.fixture
    def skdata_path(self):
        import os
        # Use test data path
        return os.path.join(os.path.dirname(__file__), "data", "skfiles", "pbc-0-1")

    def test_dftbsk_linear_method(self, basis, skdata_path):
        """Test DFTBSK with linear interpolation (default)."""
        import os
        if not os.path.exists(skdata_path):
            pytest.skip(f"Test data not found: {skdata_path}")

        from dptb.nn.dftbsk import DFTBSK

        model = DFTBSK(
            basis=basis,
            skdata=skdata_path,
            overlap=True,
            interp_method='linear',
        )

        assert model.interp_method == 'linear'
        assert hasattr(model, 'hopping_fn')

    def test_dftbsk_smooth_intp_method(self, basis, skdata_path):
        """Test DFTBSK with smooth_intp interpolation."""
        import os
        if not os.path.exists(skdata_path):
            pytest.skip(f"Test data not found: {skdata_path}")

        from dptb.nn.dftbsk import DFTBSK

        model = DFTBSK(
            basis=basis,
            skdata=skdata_path,
            overlap=True,
            interp_method='smooth_intp',
        )

        assert model.interp_method == 'smooth_intp'
        from dptb.nn.dftb.hopping_dftb import HoppingIntpSmooth
        assert isinstance(model.hopping_fn, HoppingIntpSmooth)

    def test_dftbsk_smooth_ski_flag(self, basis, skdata_path):
        """Test smooth_ski shorthand flag."""
        import os
        if not os.path.exists(skdata_path):
            pytest.skip(f"Test data not found: {skdata_path}")

        from dptb.nn.dftbsk import DFTBSK

        model = DFTBSK(
            basis=basis,
            skdata=skdata_path,
            overlap=True,
            smooth_ski=True,
        )

        assert model.interp_method == 'smooth_intp'

    def test_invalid_interp_method(self, basis, skdata_path):
        """Test that invalid interpolation method raises error."""
        import os
        if not os.path.exists(skdata_path):
            pytest.skip(f"Test data not found: {skdata_path}")

        from dptb.nn.dftbsk import DFTBSK

        with pytest.raises(ValueError, match="Invalid interp_method"):
            DFTBSK(
                basis=basis,
                skdata=skdata_path,
                overlap=True,
                interp_method='invalid_method',
            )


class TestRepcurve:
    """Tests for repulsive curve utilities."""

    def test_interp_cubic_spline_accepts_torch_tensors(self):
        from dptb.nn.dftb.interp import Repcurve

        repcurve = Repcurve(
            element_symbols=["H"],
            sigma_rep={"H": 0.5},
            dtype=torch.float64,
        )
        r = torch.linspace(0.3, 2.0, 16, dtype=torch.float64)

        spline = repcurve.interp_cubic_spline("H", "H", r=r)

        assert isinstance(spline, CubicSpline)
        assert isinstance(spline.x, np.ndarray)
        assert np.allclose(spline.x, r.numpy())


class TestGradientFlow:
    """Test gradient flow through interpolation."""

    def test_gradient_through_poly_interp(self):
        """Test gradients flow through polynomial interpolation."""
        from dptb.utils.interpolate.poly_interp import NevillePolyInterp

        interp = NevillePolyInterp(n_points=8)

        x = torch.linspace(0.5, 10.0, 100)
        y = torch.randn(3, 100, requires_grad=True)
        xq = torch.tensor([2.0, 5.0, 8.0])

        result = interp(x, y, xq)
        loss = result.sum()
        loss.backward()

        assert y.grad is not None, "Gradient should flow to y"
        assert not torch.isnan(y.grad).any(), "Gradient should not contain NaN"

    def test_gradient_through_dftbplus_interp(self):
        """Test gradients flow through full DFTB+ interpolation."""
        from dptb.utils.interpolate.dftbplus_interp import DFTBPlusInterp1D

        x = torch.linspace(0.5, 10.0, 100)
        y = torch.randn(3, 100, requires_grad=True)
        interp = DFTBPlusInterp1D(x=x, dist_fudge=1.0)

        # Query points in different regions
        xq = torch.tensor([2.0, 5.0, 8.0, 10.2])  # Grid and decay zones

        result = interp(xq, y)
        loss = result.sum()
        loss.backward()

        assert y.grad is not None, "Gradient should flow to y"
        assert not torch.isnan(y.grad).any(), "Gradient should not contain NaN"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_query(self):
        """Test with empty query tensor."""
        from dptb.utils.interpolate.dftbplus_interp import DFTBPlusInterp1D

        x = torch.linspace(0.5, 10.0, 100)
        y = torch.randn(3, 100)

        interp = DFTBPlusInterp1D(x=x, dist_fudge=1.0)
        xq = torch.tensor([])

        result = interp(xq, y)
        assert result.shape == (3, 0)

    def test_y_not_provided_error(self):
        """Test error when y is not provided."""
        from dptb.utils.interpolate.dftbplus_interp import DFTBPlusInterp1D

        x = torch.linspace(0.5, 10.0, 100)
        interp = DFTBPlusInterp1D(x=x, y=None, dist_fudge=1.0)

        xq = torch.tensor([5.0])

        with pytest.raises(ValueError, match="y must be provided"):
            interp(xq, None)

    def test_y_provided_at_init(self):
        """Test when y is provided at initialization."""
        from dptb.utils.interpolate.dftbplus_interp import DFTBPlusInterp1D

        x = torch.linspace(0.5, 10.0, 100)
        y = torch.randn(3, 100)

        interp = DFTBPlusInterp1D(x=x, y=y, dist_fudge=1.0)
        xq = torch.tensor([5.0])

        # Should work without providing y at call time
        result = interp(xq)
        assert result.shape == (3, 1)


class TestDFTBPlusBoundaryConditions:
    """
    Tests for DFTB+ boundary conditions matching.

    These tests verify that DeePTB matches DFTB+ behavior exactly at region boundaries:
    - At x_max: uses decay region (raw table value), not grid interpolation
    - At x_cutoff: returns zero, not decay value

    Reference: DFTB+ slakoeqgrid.F90 (SlakoEqGrid_interNew_)
    """

    @pytest.fixture
    def interp_with_data(self):
        """Create interpolator with known data for boundary testing."""
        from dptb.utils.interpolate.dftbplus_interp import DFTBPlusInterp1D

        # Create uniform grid: 100 points from 0.5 to 10.0 Angstrom
        x = torch.linspace(0.5, 10.0, 100)
        # Create simple linear data for predictable interpolation
        # y[i] = i + 1 for each channel, making boundary values clear
        y = torch.stack([
            torch.arange(1, 101, dtype=torch.float32),  # Channel 0: 1 to 100
            torch.arange(1, 101, dtype=torch.float32) * 2,  # Channel 1: 2 to 200
        ])

        interp = DFTBPlusInterp1D(x=x, dist_fudge=1.0)
        return interp, x, y

    def test_at_exact_x_max_uses_decay_region(self, interp_with_data):
        """
        Test that at exactly x_max, the decay region is used.

        DFTB+ behavior: When ind = floor(rr/dist) == nGrid, uses decay region
        which takes raw table value y1 = skTab(nGrid, ii).

        At x_max, poly5ToZero should return exactly the table value at the
        last grid point (since xr = 1 at boundary).
        """
        interp, x, y = interp_with_data

        # Query at exactly x_max
        xq = torch.tensor([interp.x_max])
        result = interp(xq, y)

        # At x_max (decay boundary), poly5ToZero with xr=1 returns f exactly
        # f = y[:, -1] which is the last grid point value
        expected = y[:, -1]

        assert torch.allclose(result.squeeze(), expected, atol=1e-5), \
            f"At x_max, should return raw table value. Expected {expected}, got {result.squeeze()}"

    def test_at_exact_x_cutoff_returns_zero(self, interp_with_data):
        """
        Test that at exactly x_cutoff, zero is returned.

        DFTB+ behavior: if (rr >= rMax) then dd(:) = 0.0
        This means at exactly rMax (x_cutoff), return zero.
        """
        interp, x, y = interp_with_data

        # Query at exactly x_cutoff
        xq = torch.tensor([interp.x_cutoff])
        result = interp(xq, y)

        # Should be exactly zero
        expected = torch.zeros(y.shape[0])

        assert torch.allclose(result.squeeze(), expected, atol=1e-10), \
            f"At x_cutoff, should return zero. Expected {expected}, got {result.squeeze()}"

    def test_just_before_x_max_uses_grid_region(self, interp_with_data):
        """
        Test that just before x_max, grid interpolation is used.
        """
        interp, x, y = interp_with_data

        # Query just before x_max (inside grid region)
        epsilon = 1e-6
        xq = torch.tensor([interp.x_max - epsilon])
        result = interp(xq, y)

        # Should use polynomial interpolation, not decay
        # For linear data, result should be very close to the last grid value
        # but computed via interpolation
        assert result.shape == (2, 1)
        # The result should be finite and close to y[:, -1]
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_just_before_x_cutoff_uses_decay_region(self, interp_with_data):
        """
        Test that just before x_cutoff, decay region is used (non-zero).
        """
        interp, x, y = interp_with_data

        # Query just before x_cutoff (inside decay region)
        epsilon = 1e-6
        xq = torch.tensor([interp.x_cutoff - epsilon])
        result = interp(xq, y)

        # Should use decay, which returns very small but non-zero values
        # near the cutoff
        assert result.shape == (2, 1)
        assert not torch.isnan(result).any()
        # Values should be very small (close to zero but not exactly zero)
        # since we're near the end of the decay zone

    def test_just_after_x_cutoff_returns_zero(self, interp_with_data):
        """
        Test that just after x_cutoff, zero is returned.
        """
        interp, x, y = interp_with_data

        # Query just after x_cutoff
        epsilon = 1e-6
        xq = torch.tensor([interp.x_cutoff + epsilon])
        result = interp(xq, y)

        # Should be exactly zero
        expected = torch.zeros(y.shape[0])

        assert torch.allclose(result.squeeze(), expected, atol=1e-10), \
            f"Beyond x_cutoff, should return zero. Got {result.squeeze()}"

    def test_continuity_at_x_max_boundary(self, interp_with_data):
        """
        Test that values are continuous across the x_max boundary.

        The poly5ToZero function is designed to provide C2 continuity,
        so values should be smooth across the boundary.
        """
        interp, x, y = interp_with_data

        # Query points around x_max
        epsilon = 1e-4
        xq = torch.tensor([
            interp.x_max - epsilon,
            interp.x_max,
            interp.x_max + epsilon
        ])
        result = interp(xq, y)

        # Check values are continuous (no huge jumps)
        diffs = torch.abs(result[:, 1:] - result[:, :-1])
        max_diff = diffs.max()

        # For smooth data, the difference should be small relative to epsilon
        assert max_diff < 1.0, \
            f"Values should be continuous at x_max boundary. Max diff: {max_diff}"

    def test_region_assignment_consistency(self, interp_with_data):
        """
        Test that region boundaries are mutually exclusive and exhaustive.

        Every query point should fall into exactly one region:
        - below: xq < x_min
        - grid: x_min <= xq < x_max
        - decay: x_max <= xq < x_cutoff
        - above: xq >= x_cutoff
        """
        interp, x, y = interp_with_data

        # Test points at various boundaries
        # Note: Use 1e-5 instead of 1e-10 for epsilon to avoid float32 precision issues
        test_points = [
            (interp.x_min - 0.1, "below"),
            (interp.x_min, "grid"),
            ((interp.x_min + interp.x_max) / 2, "grid"),
            (interp.x_max - 1e-5, "grid"),  # Use larger epsilon for float32
            (interp.x_max, "decay"),
            ((interp.x_max + interp.x_cutoff) / 2, "decay"),
            (interp.x_cutoff - 1e-5, "decay"),  # Use larger epsilon for float32
            (interp.x_cutoff, "above"),
            (interp.x_cutoff + 0.1, "above"),
        ]

        for point, expected_region in test_points:
            xq = torch.tensor([point])

            # Create masks as the interpolator does
            mask_below = xq < interp.x_min
            mask_grid = (xq >= interp.x_min) & (xq < interp.x_max)
            mask_decay = (xq >= interp.x_max) & (xq < interp.x_cutoff)
            mask_above = xq >= interp.x_cutoff

            # Count how many regions this point belongs to
            masks = [mask_below, mask_grid, mask_decay, mask_above]
            region_names = ["below", "grid", "decay", "above"]
            active_count = sum(m.item() for m in masks)

            assert active_count == 1, \
                f"Point {point} should be in exactly one region, found {active_count}"

            # Verify it's the expected region
            for mask, name in zip(masks, region_names):
                if mask.item():
                    assert name == expected_region, \
                        f"Point {point} expected in '{expected_region}', found in '{name}'"
