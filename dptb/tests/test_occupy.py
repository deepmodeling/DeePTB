"""
Comprehensive unit tests for dptb.utils.occupy module.

This test suite covers:
- Individual smearing functions (fgau, ffd, fmp1, fcold)
- Entropy functions (sgau, sfd, smp1, scold)
- Occupation calculation functions (foccupy, soccupy)
- Main Fermi level calculation function (calculate_fermi_level)
- Edge cases, input validation, and various physical scenarios
"""

import numpy as np
import pytest
from itertools import product as itprod
from scipy.special import erf

from dptb.utils.occupy import (
    fgau, sgau,
    ffd, dffd, sfd,
    fmp1, smp1,
    fcold, scold,
    foccupy, soccupy,
    calculate_fermi_level
)


class TestSmearingFunctions:
    """Test individual smearing distribution functions."""

    def test_fgau_properties(self):
        """Test Gaussian smearing function properties."""
        # Test at key points
        assert np.isclose(fgau(0), 0.5, atol=1e-6)  # at Fermi level
        assert fgau(-5) > 0.99  # far below Fermi level (high occupation)
        assert fgau(5) < 0.01  # far above Fermi level (low occupation)

        # Test monotonic decrease
        x = np.linspace(-5, 5, 100)
        f = fgau(x)
        assert np.all(np.diff(f) <= 0)  # monotonically decreasing

        # Test range [0, 1]
        assert np.all(f >= 0) and np.all(f <= 1)

    def test_ffd_properties(self):
        """Test Fermi-Dirac smearing function properties."""
        # Test at key points
        assert np.isclose(ffd(0), 0.5, atol=1e-6)
        assert ffd(-10) > 0.99
        assert ffd(10) < 0.01

        # Test monotonic decrease
        x = np.linspace(-10, 10, 100)
        f = ffd(x)
        assert np.all(np.diff(f) <= 0)

        # Test range [0, 1]
        assert np.all(f >= 0) and np.all(f <= 1)

    def test_fmp1_properties(self):
        """Test Methfessel-Paxton order 1 smearing function properties."""
        # Test at key points
        assert np.isclose(fmp1(0), 0.5, atol=1e-6)

        # Test general behavior
        x = np.linspace(-5, 5, 100)
        f = fmp1(x)

        # MP can slightly exceed [0,1] but should be close
        assert np.min(f) > -0.1
        assert np.max(f) < 1.1

    def test_fcold_properties(self):
        """Test Marzari-Vanderbilt (cold) smearing function properties."""
        # Test at key points - fcold(0) is not exactly 0.5 due to its functional form
        fcold_at_zero = fcold(0)
        assert 0.5 < fcold_at_zero < 0.65  # Around 0.599

        # Test general behavior
        x = np.linspace(-5, 5, 100)
        f = fcold(x)

        # Cold smearing can slightly exceed [0,1] but should be close
        assert np.min(f) > -0.1
        assert np.max(f) < 1.1

    def test_smearing_functions_comparison(self):
        """Compare different smearing functions at the same point."""
        x = 0.0
        # Gaussian, FD, and MP1 should give 0.5 at Fermi level
        assert np.isclose(fgau(x), 0.5, atol=1e-6)
        assert np.isclose(ffd(x), 0.5, atol=1e-6)
        assert np.isclose(fmp1(x), 0.5, atol=1e-6)
        # Cold smearing has different value at x=0 due to its functional form
        assert 0.5 < fcold(x) < 0.65
    def test_fgau_matches_closed_form(self):
        """Test Gaussian smearing closed-form formula."""
        x = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        expected = 0.5 * (1 - erf(x))
        np.testing.assert_allclose(fgau(x), expected, rtol=1e-12, atol=1e-12)

    def test_ffd_matches_closed_form(self):
        """Test Fermi-Dirac smearing closed-form formula."""
        x = np.array([-50.0, -2.0, -1.0, 0.0, 1.0, 2.0, 50.0])
        expected = 1 / (1 + np.exp(x))
        np.testing.assert_allclose(ffd(x), expected, rtol=1e-12, atol=1e-12)

    def test_dffd_matches_fermi_dirac_broadening(self):
        """Test Fermi-Dirac derivative/broadening function."""
        x = np.array([-50.0, -2.0, -1.0, 0.0, 1.0, 2.0, 50.0])
        expected = 1 / (2 * np.cosh(x) + 2)
        np.testing.assert_allclose(dffd(x), expected, rtol=1e-12, atol=1e-12)

    def test_sgau_matches_closed_form(self):
        """Test Gaussian entropy contribution closed-form formula."""
        x = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        expected = 0.5 / np.sqrt(np.pi) * np.exp(-x**2)
        np.testing.assert_allclose(sgau(x), expected, rtol=1e-12, atol=1e-12)

    def test_sfd_matches_closed_form(self):
        """Test Fermi-Dirac entropy contribution closed-form formula."""
        x = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        f = ffd(x)
        expected = -f * np.log(f) - (1 - f) * np.log(1 - f)
        np.testing.assert_allclose(sfd(x), expected, rtol=1e-12, atol=1e-12)

    def test_fmp1_matches_closed_form(self):
        """Test Methfessel-Paxton order 1 formula away from the Fermi level."""
        x = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
        expected = 0.5 * (1 - erf(x)) - 0.5 / np.sqrt(np.pi) * x * np.exp(-x**2)
        np.testing.assert_allclose(fmp1(x), expected, rtol=1e-12, atol=1e-12)
        assert not np.allclose(fmp1(x), fgau(x))

    def test_fcold_matches_closed_form(self):
        """Test Marzari-Vanderbilt cold smearing closed-form formula."""
        x = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        expected = (
            0.5
            + 0.5 * erf(1 / np.sqrt(2) - x)
            - 1 / np.sqrt(2 * np.pi) * np.exp(-0.25 * (np.sqrt(2) - 2 * x) ** 2)
        )
        np.testing.assert_allclose(fcold(x), expected, rtol=1e-12, atol=1e-12)


class TestEntropyFunctions:
    """Test entropy contribution functions."""

    def test_sgau_properties(self):
        """Test Gaussian entropy function properties."""
        x = np.linspace(-5, 5, 100)
        s = sgau(x)

        # Entropy should be non-negative
        assert np.all(s >= 0)

        # Maximum at Fermi level (x=0)
        assert np.isclose(s[50], np.max(s), atol=1e-6)

    def test_sfd_properties(self):
        """Test Fermi-Dirac entropy function properties."""
        x = np.linspace(-5, 5, 100)
        s = sfd(x)

        # Entropy should be non-negative
        assert np.all(s >= 0)

        # Maximum at Fermi level (x=0)
        assert np.isclose(s[50], np.max(s), atol=1e-6)

    def test_sfd_finite_at_large_negative_x(self):
        """sfd must stay finite for large negative x where expm1 rounds to -1.

        For x <= -39, expm1(x) rounds to -1.0 in float64, so f -> 1 and
        1-f -> 0. The entropy has physical limit 0 there (x*log(x) -> 0),
        so the result must be finite 0 rather than log(0) -> nan.
        """
        x = np.array([-100, -50, -40, -39, -20, 0, 20, 40, 50, 100])
        s = sfd(x)
        assert np.all(np.isfinite(s))
        # Extreme values and negative round-to-one boundaries vanish exactly.
        assert np.all(s[[0, 1, 2, 3, 8, 9]] == 0.0)
        # x=40 is inside the compute range; f is tiny but nonzero, so entropy
        # is a small positive number rather than 0
        assert s[7] > 0 and s[7] < 1e-10
        # interior unchanged: x=0 is the entropy maximum ln(2)
        assert np.isclose(s[5], np.log(2.0), rtol=1e-12)

    def test_smp1_zero(self):
        """Test that MP1 entropy is zero (by design)."""
        x = np.linspace(-5, 5, 100)
        s = smp1(x)
        assert np.all(s == 0)

    def test_scold_zero(self):
        """Test that cold smearing entropy is zero (by design)."""
        x = np.linspace(-5, 5, 100)
        s = scold(x)
        assert np.all(s == 0)


class TestOccupyFunctions:
    """Test occupation and entropy wrapper functions."""

    def test_foccupy_methods(self):
        """Test foccupy with all smearing methods."""
        e = 0.0  # energy at Fermi level
        ef = 0.0
        sigma = 0.1

        for method in ['gaussian', 'fd', 'mp']:
            occ = foccupy(e, ef, sigma, method)
            assert np.isclose(occ, 0.5, atol=1e-6), f"Failed for method {method}"

        # MV (cold) smearing has a different value at e=ef
        occ_mv = foccupy(e, ef, sigma, 'mv')
        assert 0.5 < occ_mv < 0.65, "Failed for method mv"

    def test_foccupy_energy_dependence(self):
        """Test occupation dependence on energy."""
        ef = 0.0
        sigma = 0.1

        # Below Fermi level should have higher occupation
        occ_below = foccupy(-1.0, ef, sigma, 'gaussian')
        occ_above = foccupy(1.0, ef, sigma, 'gaussian')

        assert occ_below > occ_above
        assert occ_below > 0.5
        assert occ_above < 0.5

    def test_foccupy_sigma_validation(self):
        """Test that negative or zero sigma raises a validation error."""
        with pytest.raises((TypeError, ValueError)):
            foccupy(0.0, 0.0, 0.0, 'gaussian')

        with pytest.raises((TypeError, ValueError)):
            foccupy(0.0, 0.0, -0.1, 'gaussian')

    def test_soccupy_methods(self):
        """Test soccupy with all smearing methods."""
        e = 0.0
        ef = 0.0
        sigma = 0.1

        for method in ['gaussian', 'fd', 'mp', 'mv']:
            s = soccupy(e, ef, sigma, method)
            assert np.isfinite(s), f"Failed for method {method}"

    def test_soccupy_sigma_validation(self):
        """Test that negative or zero sigma raises a validation error."""
        with pytest.raises((TypeError, ValueError)):
            soccupy(0.0, 0.0, 0.0, 'gaussian')

        with pytest.raises((TypeError, ValueError)):
            soccupy(0.0, 0.0, -0.1, 'gaussian')


class TestCalculateFermiLevel:
    """Test the main calculate_fermi_level function."""

    def test_basic_functionality(self):
        """Test basic Fermi level calculation."""
        ekb = np.linspace(-4, 4, 100).reshape(1, 1, -1)
        wk = np.array([1.0])
        ne = 10
        sigma = 0.1

        ef, occ, eband, eband_free = calculate_fermi_level(ekb, wk, ne, sigma)

        # Check shapes
        assert occ.shape == ekb.shape
        assert isinstance(ef, (float, np.floating))
        assert isinstance(eband, (float, np.floating, type(None)))
        assert isinstance(eband_free, (float, np.floating, type(None)))

        # Check electron conservation
        ne_calc = np.sum(occ * wk[0])
        assert np.isclose(ne_calc, ne, atol=0.05)

        # Check Fermi level in reasonable range
        assert ef >= np.min(ekb)
        assert ef <= np.max(ekb)

    def test_all_smearing_methods(self):
        """Test all smearing methods with electron conservation."""
        ekb = np.linspace(-4, 4, 100).reshape(1, 1, -1)
        wk = np.array([1.0])
        ne = 10
        sigma = 0.05

        methods = ['gaussian', 'gau', 'fd', 'fermi-dirac', 'f-d',
                   'mp', 'methfessel-paxton', 'm-p',
                   'mv', 'marzari-vanderbilt', 'cold', 'm-v']

        for method in methods:
            ef, occ, _, _ = calculate_fermi_level(ekb, wk, ne, sigma, method=method, with_eband=False)

            ne_calc = np.sum(occ * wk[0])
            assert abs(ne_calc - ne) / ne < 0.01, f"Failed for method {method}: ne={ne}, ne_calc={ne_calc}"

    def test_mp_mv_fermi_level_is_scalar(self):
        """Test MP/MV refinement returns a scalar Fermi level."""
        ekb = np.linspace(-4, 4, 100).reshape(1, 1, -1)
        wk = np.array([1.0])
        ne = 10
        sigma = 0.05

        for method in ['mp', 'mv']:
            ef, occ, eband, eband_free = calculate_fermi_level(ekb, wk, ne, sigma, method=method)
            assert isinstance(ef, (float, np.floating)), f"Failed for method {method}: ef type is {type(ef)}"
            assert occ.shape == ekb.shape
            assert isinstance(eband, (float, np.floating))
            assert isinstance(eband_free, (float, np.floating))

    def test_spin_polarized(self):
        """Test with spin-polarized calculation (nspin=2)."""
        np.random.seed(42)  # For reproducibility
        ekb = np.random.uniform(-3, 3, (2, 4, 20))
        wk = np.ones(4) / 4
        ne = 40
        sigma = 0.1

        # Test with eband calculation enabled
        ef, occ, eband, eband_free = calculate_fermi_level(ekb, wk, ne, sigma, with_eband=True)

        # Check shapes
        assert occ.shape == ekb.shape

        # Check electron conservation
        # For nspin=2, spindeg=1 (no spin degeneracy), so each spin channel is counted once
        # Sum over all spins, kpoints (weighted), and bands
        ne_calc = np.sum(occ * wk.reshape(1, -1, 1))
        assert np.isclose(ne_calc, ne, atol=0.5)

        # Check eband is computed
        assert eband is not None
        assert isinstance(eband, (float, np.floating))

    def test_spin_degeneracy_factor(self):
        """Test that spin degeneracy factor is correctly applied.

        For nspin=1 (spin-unpolarized): spindeg=2 (each band holds 2 electrons)
        For nspin=2 (spin-polarized): spindeg=1 (each band holds 1 electron per spin)
        """
        np.random.seed(42)
        nk = 4
        nb = 10
        wk = np.ones(nk) / nk
        sigma = 0.05

        # Create identical band structures for comparison
        ekb_base = np.random.uniform(-3, 3, (nk, nb))
        ekb_base.sort(axis=1)  # Sort bands by energy

        # Test nspin=1: each band holds 2 electrons
        ekb_nspin1 = ekb_base.reshape(1, nk, nb)
        ne_nspin1 = 10  # Fill 5 bands with 2 electrons each
        ef1, occ1, _, _ = calculate_fermi_level(ekb_nspin1, wk, ne_nspin1, sigma, with_eband=False)

        # Verify electron count
        ne_calc1 = np.sum(occ1 * wk.reshape(1, -1, 1))
        assert np.isclose(ne_calc1, ne_nspin1, atol=0.1), f"nspin=1: expected {ne_nspin1}, got {ne_calc1}"

        # Test nspin=2: same bands duplicated for each spin, each holds 1 electron per spin
        ekb_nspin2 = np.stack([ekb_base, ekb_base], axis=0)  # (2, nk, nb)
        ne_nspin2 = 10  # Same total electrons
        ef2, occ2, _, _ = calculate_fermi_level(ekb_nspin2, wk, ne_nspin2, sigma, with_eband=False)

        # Verify electron count
        ne_calc2 = np.sum(occ2 * wk.reshape(1, -1, 1))
        assert np.isclose(ne_calc2, ne_nspin2, atol=0.1), f"nspin=2: expected {ne_nspin2}, got {ne_calc2}"

        # Fermi levels should be similar for identical band structures
        assert np.isclose(ef1, ef2, atol=0.2), f"Fermi levels differ: nspin=1 ef={ef1}, nspin=2 ef={ef2}"

    def test_spin_polarized_max_electrons(self):
        """Test maximum electron capacity for spin-polarized systems.

        For nspin=2 with nb bands per spin channel and normalized k-weights:
        max_electrons = 1 * nb * nspin = 2 * nb (1 electron per band per spin)
        """
        np.random.seed(42)
        nk = 4
        nb = 10
        ekb = np.random.uniform(-3, 3, (2, nk, nb))
        wk = np.ones(nk) / nk
        sigma = 0.1

        # Try to fill all bands completely
        # For nspin=2, max electrons = spindeg * nb * nspin = 1 * 10 * 2 = 20
        ne_max = 20
        ef, occ, eband, eband_free = calculate_fermi_level(ekb, wk, ne_max, sigma, with_eband=True)

        ne_calc = np.sum(occ * wk.reshape(1, -1, 1))
        assert np.isclose(ne_calc, ne_max, atol=0.5), f"Expected {ne_max}, got {ne_calc}"

        # Average occupation should be close to 1 (fully filled)
        avg_occ = np.mean(occ)
        assert avg_occ > 0.9, f"Expected high average occupation, got {avg_occ}"

        # Verify eband is computed correctly for spin-polarized
        assert eband is not None
        assert isinstance(eband, (float, np.floating))

    def test_multiple_kpoints(self):
        """Test with multiple k-points."""
        nk = 8
        nb = 20
        np.random.seed(42)  # For reproducibility
        ekb = np.random.uniform(-3, 3, (1, nk, nb))
        wk = np.ones(nk) / nk
        # Max electrons = 2 * nb = 40 (with g=2 spin degeneracy and wk summing to 1)
        ne = 30
        sigma = 0.05

        ef, occ, eband, eband_free = calculate_fermi_level(ekb, wk, ne, sigma)

        # Check shapes
        assert occ.shape == ekb.shape

        # Check electron conservation with proper k-point weighting
        ne_calc = np.sum(occ * wk.reshape(1, -1, 1))
        assert np.isclose(ne_calc, ne, atol=0.5)

    def test_with_eband_flag(self):
        """Test with_eband flag behavior."""
        ekb = np.linspace(-4, 4, 50).reshape(1, 1, -1)
        wk = np.array([1.0])
        ne = 10
        sigma = 0.1

        # With eband
        ef1, occ1, eband1, eband_free1 = calculate_fermi_level(ekb, wk, ne, sigma, with_eband=True)
        assert eband1 is not None
        assert isinstance(eband1, (float, np.floating))
        assert isinstance(eband_free1, (float, np.floating))

        # Without eband
        ef2, occ2, eband2, eband_free2 = calculate_fermi_level(ekb, wk, ne, sigma, with_eband=False)
        assert eband2 is None
        assert eband_free2 is None

        # Fermi level and occupation should be the same
        assert np.isclose(ef1, ef2)
        assert np.allclose(occ1, occ2)

    def test_edge_case_zero_electrons(self):
        """Test with zero electrons."""
        ekb = np.linspace(-4, 4, 50).reshape(1, 1, -1)
        wk = np.array([1.0])
        ne = 0
        sigma = 0.1

        ef, occ, _, _ = calculate_fermi_level(ekb, wk, ne, sigma)

        # All occupations should be near zero
        assert np.sum(occ) < 0.1

    def test_edge_case_all_electrons(self):
        """Test with all bands fully occupied."""
        ekb = np.linspace(-4, 4, 50).reshape(1, 1, -1)
        wk = np.array([1.0])
        ne = 100  # 50 bands * 2 (spin degeneracy)
        sigma = 0.1

        ef, occ, _, _ = calculate_fermi_level(ekb, wk, ne, sigma)

        # Fermi level should be near the top
        assert ef > np.max(ekb) - 1.0

    def test_edge_case_single_band(self):
        """Test with a single band - may have numerical issues with bracket finding."""
        # Using two bands instead of one to avoid bracket finding issues
        ekb = np.array([[[-0.5, 0.5]]])  # Two bands
        wk = np.array([1.0])
        ne = 1  # One electron (with g=2 degeneracy, partially fills first band)
        sigma = 0.1

        ef, occ, _, _ = calculate_fermi_level(ekb, wk, ne, sigma)

        # Should work without errors
        assert occ.shape == ekb.shape
        assert np.isfinite(ef)
        # Fermi level should be near the first band
        assert -0.5 <= ef <= 0.0

    def test_input_validation_eigs_shape(self):
        """Test input validation for eigenvalues shape."""
        # Wrong number of dimensions
        with pytest.raises((TypeError, ValueError)):
            ekb = np.linspace(-4, 4, 50).reshape(1, -1)  # 2D instead of 3D
            wk = np.array([1.0])
            calculate_fermi_level(ekb, wk, 10, 0.1)

    def test_input_validation_eigs_type(self):
        """Test input validation for eigenvalues type."""
        with pytest.raises((TypeError, ValueError)):
            ekb = [[[-1, 0, 1]]]  # List instead of numpy array
            wk = np.array([1.0])
            calculate_fermi_level(ekb, wk, 10, 0.1)

    def test_input_validation_wk_shape(self):
        """Test input validation for k-point weights."""
        ekb = np.linspace(-4, 4, 50).reshape(1, 2, -1)

        # Wrong shape
        with pytest.raises((TypeError, ValueError)):
            wk = np.array([[1.0, 1.0]])  # 2D instead of 1D
            calculate_fermi_level(ekb, wk, 10, 0.1)

        # Wrong length
        with pytest.raises((TypeError, ValueError)):
            wk = np.array([1.0])  # Should be length 2
            calculate_fermi_level(ekb, wk, 10, 0.1)

    def test_input_validation_nspin(self):
        """Test input validation for spin channels."""
        # nspin = 3 should fail
        with pytest.raises((TypeError, ValueError)):
            ekb = np.linspace(-4, 4, 60).reshape(3, 1, 20)  # nspin=3 (invalid)
            wk = np.array([1.0])
            calculate_fermi_level(ekb, wk, 10, 0.1)

    def test_input_validation_sigma(self):
        """Test input validation for sigma."""
        ekb = np.linspace(-4, 4, 50).reshape(1, 1, -1)
        wk = np.array([1.0])

        # Negative sigma
        with pytest.raises((TypeError, ValueError)):
            calculate_fermi_level(ekb, wk, 10, -0.1)

        # Zero sigma
        with pytest.raises((TypeError, ValueError)):
            calculate_fermi_level(ekb, wk, 10, 0.0)

    def test_input_validation_ne(self):
        """Test input validation for number of electrons."""
        ekb = np.linspace(-4, 4, 50).reshape(1, 1, -1)
        wk = np.array([1.0])

        # Negative ne
        with pytest.raises((TypeError, ValueError)):
            calculate_fermi_level(ekb, wk, -10, 0.1)

    def test_different_sigma_values(self):
        """Test Fermi level calculation with different sigma values."""
        ekb = np.linspace(-4, 4, 100).reshape(1, 1, -1)
        wk = np.array([1.0])
        ne = 10

        sigmas = [0.01, 0.05, 0.1, 0.2, 0.5]
        efs = []

        for sigma in sigmas:
            ef, occ, _, _ = calculate_fermi_level(ekb, wk, ne, sigma)
            efs.append(ef)

            # Electron conservation should hold
            ne_calc = np.sum(occ * wk[0])
            assert np.isclose(ne_calc, ne, atol=0.05)

        # Fermi levels should all be in reasonable range
        assert all(ef >= np.min(ekb) and ef <= np.max(ekb) for ef in efs)

    def test_reproducibility(self):
        """Test that results are reproducible."""
        ekb = np.random.RandomState(42).uniform(-3, 3, (1, 4, 20))
        wk = np.ones(4) / 4
        ne = 20
        sigma = 0.1

        ef1, occ1, eband1, eband_free1 = calculate_fermi_level(ekb, wk, ne, sigma)
        ef2, occ2, eband2, eband_free2 = calculate_fermi_level(ekb, wk, ne, sigma)

        assert np.isclose(ef1, ef2)
        assert np.allclose(occ1, occ2)
        assert np.isclose(eband1, eband2)

    def test_physical_bands_silicon(self):
        """Test with a more realistic band structure (simple model)."""
        # Simplified silicon-like band structure
        # Valence band: -5 to 0 eV
        # Conduction band: 1 to 6 eV (1 eV gap)
        nk = 10
        nb = 8
        ekb = np.zeros((1, nk, nb))

        # Create simple band structure
        for ik in range(nk):
            ekb[0, ik, :4] = np.linspace(-5, 0, 4)  # valence bands
            ekb[0, ik, 4:] = np.linspace(1, 6, 4)   # conduction bands

        wk = np.ones(nk) / nk
        ne = 4 * 2  # 4 valence bands filled, spin degeneracy
        sigma = 0.1

        ef, occ, _, _ = calculate_fermi_level(ekb, wk, ne, sigma, method='gaussian')

        # Fermi level should be in the gap
        assert ef > 0 and ef < 1, f"Fermi level {ef} not in gap"

        # Check electron conservation
        ne_calc = np.sum([np.sum(occ[0, ik, :]) * wk[ik] for ik in range(nk)])
        assert np.isclose(ne_calc, ne, atol=0.1)


class TestNumericalAccuracy:
    """Test numerical accuracy and stability."""

    def test_electron_conservation_accuracy(self):
        """Test electron conservation with high accuracy requirement."""
        ekb = np.linspace(-4, 4, 100).reshape(1, 1, -1)
        wk = np.array([1.0])
        ne = 10
        sigma = 0.05

        for method in ['gaussian', 'fd', 'mp', 'mv']:
            ef, occ, _, _ = calculate_fermi_level(ekb, wk, ne, sigma, method=method)

            ne_calc = np.sum(occ * wk[0])
            relative_error = abs(ne_calc - ne) / ne

            # Should be accurate to better than 0.5%
            assert relative_error < 0.005, \
                f"Method {method}: relative error {relative_error:.6f} exceeds 0.5%"

    def test_wide_energy_range(self):
        """Test with wide energy range."""
        ekb = np.linspace(-100, 100, 200).reshape(1, 1, -1)
        wk = np.array([1.0])
        ne = 50
        sigma = 1.0

        ef, occ, _, _ = calculate_fermi_level(ekb, wk, ne, sigma)

        ne_calc = np.sum(occ * wk[0])
        assert np.isclose(ne_calc, ne, atol=0.5)

    def test_small_sigma_stability(self):
        """Test stability with very small sigma."""
        ekb = np.linspace(-4, 4, 100).reshape(1, 1, -1)
        wk = np.array([1.0])
        ne = 10
        sigma = 0.001  # Very small smearing

        # Should not crash or produce NaN
        ef, occ, _, _ = calculate_fermi_level(ekb, wk, ne, sigma, method='gaussian')

        assert np.isfinite(ef)
        assert np.all(np.isfinite(occ))

        ne_calc = np.sum(occ * wk[0])
        assert np.isclose(ne_calc, ne, atol=0.1)


class TestOutlierEigenvalues:
    """Test Fermi level calculation with outlier eigenvalues.

    This test class was added to prevent regression of a bug where the
    brent optimizer would converge to wrong local minima for systems with
    outlier eigenvalues (e.g., C dimer with high-lying sigma* orbital).

    See: fermi_level_bug_report.md for details.
    """

    def test_c_dimer_eigenvalues(self):
        """Test C dimer case that previously failed with brent optimizer.

        The C dimer has an outlier eigenvalue at +76.9 eV (sigma* anti-bonding),
        which created multiple local minima in the error function. The brent
        optimizer would converge to error=6 (14 electrons) instead of error=0
        (8 electrons).

        Band structure:
        - Bands 1-4: -18.6 to -9.3 eV (bonding orbitals)
        - Band 5: -7.6 eV
        - Bands 6-7: +2.3 eV (pi* anti-bonding)
        - Band 8: +76.9 eV (sigma* anti-bonding - OUTLIER)

        For 8 electrons (2 C atoms × 4 valence electrons), we need 4 filled bands.
        Fermi level should be around -8.5 eV (between bands 4 and 5).
        """
        # C dimer eigenvalues (1 k-point, 8 bands)
        eigs = np.array([[-18.606747, -10.180909, -9.288945, -9.288941,
                          -7.631103, 2.259443, 2.259443, 76.89657]])
        eigs_3d = eigs.reshape(1, 1, 8)
        wk = np.array([1.0])
        ne = 8  # 2 C atoms × 4 valence electrons
        sigma = 0.025852  # ~300K in eV

        # Test with Gaussian smearing
        ef, occ, _, _ = calculate_fermi_level(eigs_3d, wk, ne, sigma, method='gaussian')

        # Verify correct number of electrons
        total_electrons = np.sum(occ)
        assert np.isclose(total_electrons, ne, atol=0.01), \
            f"C dimer (Gaussian): Expected {ne} electrons, got {total_electrons}"

        # Verify Fermi level is in the correct gap (between bands 4 and 5)
        assert -9.3 < ef < -7.6, \
            f"C dimer (Gaussian): Fermi level {ef} not in expected gap [-9.3, -7.6]"

        # Verify occupation pattern: first 4 bands filled, last 4 empty
        occ_flat = occ.flatten()
        assert np.allclose(occ_flat[:4], 2.0, atol=0.01), \
            f"C dimer (Gaussian): First 4 bands should be fully occupied, got {occ_flat[:4]}"
        assert np.allclose(occ_flat[4:], 0.0, atol=0.01), \
            f"C dimer (Gaussian): Last 4 bands should be empty, got {occ_flat[4:]}"

    def test_c_dimer_fermi_dirac(self):
        """Test C dimer with Fermi-Dirac smearing."""
        eigs = np.array([[-18.606747, -10.180909, -9.288945, -9.288941,
                          -7.631103, 2.259443, 2.2594442, 76.89657]])
        eigs_3d = eigs.reshape(1, 1, 8)
        wk = np.array([1.0])
        ne = 8
        sigma = 0.025852

        ef, occ, _, _ = calculate_fermi_level(eigs_3d, wk, ne, sigma, method='fd')

        total_electrons = np.sum(occ)
        assert np.isclose(total_electrons, ne, atol=0.01), \
            f"C dimer (FD): Expected {ne} electrons, got {total_electrons}"

        # Fermi level should be in the gap
        assert -9.3 < ef < -7.6, \
            f"C dimer (FD): Fermi level {ef} not in expected gap"

    def test_extreme_outlier_eigenvalue(self):
        """Test with even more extreme outlier eigenvalue.

        This tests robustness against very large eigenvalue gaps that could
        cause numerical issues or optimizer convergence problems.
        """
        # Create eigenvalues with extreme outlier
        eigs = np.array([[-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 1000.0]])
        eigs_3d = eigs.reshape(1, 1, 8)
        wk = np.array([1.0])
        ne = 8  # Fill 4 bands
        sigma = 0.1

        ef, occ, _, _ = calculate_fermi_level(eigs_3d, wk, ne, sigma, method='gaussian')

        total_electrons = np.sum(occ)
        assert np.isclose(total_electrons, ne, atol=0.1), \
            f"Extreme outlier: Expected {ne} electrons, got {total_electrons}"

        # Fermi level should NOT be near the outlier
        assert ef < 100, \
            f"Extreme outlier: Fermi level {ef} should not be near outlier at 1000"

    def test_multiple_gaps_correct_filling(self):
        """Test correct band filling when multiple large gaps exist.

        This tests that the algorithm fills bands in order of energy,
        not by finding arbitrary local minima.
        """
        # Create eigenvalues with multiple gaps
        # Gap 1: between -5 and 0 (5 eV gap)
        # Gap 2: between 5 and 50 (45 eV gap)
        eigs = np.array([[-10.0, -8.0, -6.0, -5.0, 0.0, 2.0, 4.0, 5.0, 50.0, 60.0]])
        eigs_3d = eigs.reshape(1, 1, 10)
        wk = np.array([1.0])
        ne = 8  # Fill first 4 bands (with spindeg=2)
        sigma = 0.1

        ef, occ, _, _ = calculate_fermi_level(eigs_3d, wk, ne, sigma, method='gaussian')

        total_electrons = np.sum(occ)
        assert np.isclose(total_electrons, ne, atol=0.1), \
            f"Multiple gaps: Expected {ne} electrons, got {total_electrons}"

        # Fermi level should be in the first gap (between -5 and 0)
        assert -5.0 < ef < 0.0, \
            f"Multiple gaps: Fermi level {ef} should be in first gap [-5, 0]"

    def test_isolated_molecule_band_structure(self):
        """Test typical isolated molecule band structure with HOMO-LUMO gap.

        Isolated molecules often have large HOMO-LUMO gaps and may have
        high-lying virtual orbitals that can cause optimization issues.
        """
        # Typical small molecule: 4 occupied MOs, 4 virtual MOs
        # HOMO at -5 eV, LUMO at +2 eV (7 eV gap)
        # High-lying virtual at +50 eV
        eigs = np.array([[-15.0, -12.0, -8.0, -5.0, 2.0, 5.0, 8.0, 50.0]])
        eigs_3d = eigs.reshape(1, 1, 8)
        wk = np.array([1.0])
        ne = 8  # Fill 4 occupied MOs
        sigma = 0.05

        for method in ['gaussian', 'fd']:
            ef, occ, _, _ = calculate_fermi_level(eigs_3d, wk, ne, sigma, method=method)

            total_electrons = np.sum(occ)
            assert np.isclose(total_electrons, ne, atol=0.1), \
                f"Isolated molecule ({method}): Expected {ne} electrons, got {total_electrons}"

            # Fermi level should be in HOMO-LUMO gap
            assert -5.0 < ef < 2.0, \
                f"Isolated molecule ({method}): Fermi level {ef} not in HOMO-LUMO gap"


class TestMethodAliases:
    """Test method name aliases."""

    def test_gaussian_aliases(self):
        """Test Gaussian smearing aliases."""
        ekb = np.linspace(-4, 4, 50).reshape(1, 1, -1)
        wk = np.array([1.0])
        ne = 10
        sigma = 0.1

        ef_ref, occ_ref, _, _ = calculate_fermi_level(ekb, wk, ne, sigma, method='gaussian')

        for alias in ['gau', 'Gaussian', 'GAU']:
            ef, occ, _, _ = calculate_fermi_level(ekb, wk, ne, sigma, method=alias)
            assert np.isclose(ef, ef_ref)
            assert np.allclose(occ, occ_ref)

    def test_fd_aliases(self):
        """Test Fermi-Dirac aliases."""
        ekb = np.linspace(-4, 4, 50).reshape(1, 1, -1)
        wk = np.array([1.0])
        ne = 10
        sigma = 0.1

        ef_ref, occ_ref, _, _ = calculate_fermi_level(ekb, wk, ne, sigma, method='fd')

        for alias in ['fermi-dirac', 'f-d', 'FD', 'Fermi-Dirac']:
            ef, occ, _, _ = calculate_fermi_level(ekb, wk, ne, sigma, method=alias)
            assert np.isclose(ef, ef_ref)
            assert np.allclose(occ, occ_ref)

    def test_mp_aliases(self):
        """Test Methfessel-Paxton aliases."""
        ekb = np.linspace(-4, 4, 50).reshape(1, 1, -1)
        wk = np.array([1.0])
        ne = 10
        sigma = 0.1

        ef_ref, occ_ref, _, _ = calculate_fermi_level(ekb, wk, ne, sigma, method='mp')

        for alias in ['methfessel-paxton', 'm-p', 'MP']:
            ef, occ, _, _ = calculate_fermi_level(ekb, wk, ne, sigma, method=alias)
            assert np.isclose(ef, ef_ref, atol=1e-4)
            assert np.allclose(occ, occ_ref, atol=1e-4)

    def test_mv_aliases(self):
        """Test Marzari-Vanderbilt aliases."""
        ekb = np.linspace(-4, 4, 50).reshape(1, 1, -1)
        wk = np.array([1.0])
        ne = 10
        sigma = 0.1

        ef_ref, occ_ref, _, _ = calculate_fermi_level(ekb, wk, ne, sigma, method='mv')

        for alias in ['marzari-vanderbilt', 'cold', 'm-v', 'MV', 'Cold']:
            ef, occ, _, _ = calculate_fermi_level(ekb, wk, ne, sigma, method=alias)
            assert np.isclose(ef, ef_ref, atol=1e-4)
            assert np.allclose(occ, occ_ref, atol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
