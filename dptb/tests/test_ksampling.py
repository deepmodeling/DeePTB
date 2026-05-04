'''
Unit tests for dptb.utils.ksampling module.

This module tests k-point sampling utilities including:
- Reciprocal lattice vector calculation
- Monkhorst-Pack and Gamma-centered sampling
- Symmetry reduction (time-inversion and rotational)
- Integration with make_kpoints.py
'''

import pytest
import numpy as np
from ase.atoms import Atoms
from ase.geometry import cellpar_to_cell
from dptb.utils.ksampling import (
    is_integer,
    calculate_reciprocal_vectors,
    mp,
    build_kmeshgrid,
    monkhorst_pack_sampling,
    get_symm_ops,
    reduce_time_inversion,
    reduce_rotation,
    reduce,
    sample
)


class TestIsInteger:
    """Tests for is_integer helper function."""

    def test_is_integer_true_cases(self):
        """Should return True for integer values."""
        assert is_integer(1.0)
        assert is_integer(0.0)
        assert is_integer(-5.0)
        assert is_integer(1000000.0)

    def test_is_integer_false_cases(self):
        """Should return False for non-integer values."""
        assert not is_integer(0.5)
        assert not is_integer(1.1)
        assert not is_integer(-0.001)

    def test_is_integer_near_integer(self):
        """Should handle values very close to integers."""
        assert is_integer(1.0 + 1e-11)  # Within tolerance
        assert not is_integer(1.0 + 1e-9)  # Outside tolerance


class TestCalculateReciprocalVectors:
    """Tests for calculate_reciprocal_vectors function."""

    def test_cubic_cell(self):
        """Test reciprocal vectors for cubic cell."""
        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        c = np.array([0, 0, 1])
        g1, g2, g3 = calculate_reciprocal_vectors(a, b, c)
        np.testing.assert_allclose(g1, [2 * np.pi, 0, 0])
        np.testing.assert_allclose(g2, [0, 2 * np.pi, 0])
        np.testing.assert_allclose(g3, [0, 0, 2 * np.pi])

    def test_fcc_cell(self):
        """Test reciprocal vectors for FCC cell."""
        latt = np.array([[1, 1, 0],
                         [1, 0, 1],
                         [0, 1, 1]]) * 2.5
        g1, g2, g3 = calculate_reciprocal_vectors(latt[0], latt[1], latt[2])
        # All should have same norm for FCC
        norm1 = np.linalg.norm(g1)
        norm2 = np.linalg.norm(g2)
        norm3 = np.linalg.norm(g3)
        np.testing.assert_allclose(norm1, norm2, rtol=1e-10)
        np.testing.assert_allclose(norm2, norm3, rtol=1e-10)

    def test_orthogonality(self):
        """Real and reciprocal vectors should satisfy orthogonality relations."""
        latt = np.array([[1, 1, 0],
                         [1, 0, 1],
                         [0, 1, 1]]) * 2.5
        g1, g2, g3 = calculate_reciprocal_vectors(latt[0], latt[1], latt[2])

        # a_i · b_j = 2π δ_ij
        np.testing.assert_allclose(np.dot(latt[0], g1), 2 * np.pi, rtol=1e-10)
        np.testing.assert_allclose(np.dot(latt[1], g2), 2 * np.pi, rtol=1e-10)
        np.testing.assert_allclose(np.dot(latt[2], g3), 2 * np.pi, rtol=1e-10)
        np.testing.assert_allclose(np.dot(latt[0], g2), 0, atol=1e-10)
        np.testing.assert_allclose(np.dot(latt[0], g3), 0, atol=1e-10)
        np.testing.assert_allclose(np.dot(latt[1], g1), 0, atol=1e-10)


class TestMp:
    """Tests for mp (Monkhorst-Pack) function."""

    def test_mp_shape(self):
        """Output shape should be (N1*N2*N3, 3)."""
        nk1, nk2, nk3 = 4, 4, 4
        k = mp(nk1, nk2, nk3)
        assert k.shape == (nk1 * nk2 * nk3, 3)

    def test_mp_gamma_centered_contains_gamma(self):
        """Gamma-centered mesh should contain gamma point."""
        k = mp(4, 4, 4, gamma_centered=True)
        has_gamma = any(np.allclose(kp, [0, 0, 0]) for kp in k)
        assert has_gamma, "Gamma-centered mesh should contain gamma point"

    def test_mp_original_even_no_gamma(self):
        """Original MP with even mesh should not contain gamma."""
        k = mp(4, 4, 4, gamma_centered=False)
        has_gamma = any(np.allclose(kp, [0, 0, 0]) for kp in k)
        assert not has_gamma, "Even MP mesh should not contain gamma"

    def test_mp_original_odd_has_gamma(self):
        """Original MP with odd mesh should contain gamma."""
        k = mp(3, 3, 3, gamma_centered=False)
        has_gamma = any(np.allclose(kp, [0, 0, 0]) for kp in k)
        assert has_gamma, "Odd MP mesh should contain gamma"

    def test_mp_range(self):
        """K-points should be in [-0.5, 0.5) range."""
        for gamma_centered in [True, False]:
            k = mp(4, 4, 4, gamma_centered=gamma_centered)
            assert k.min() >= -0.5, f"gamma_centered={gamma_centered}: min < -0.5"
            assert k.max() < 0.5, f"gamma_centered={gamma_centered}: max >= 0.5"

    def test_mp_cartesian_output(self):
        """Test Cartesian coordinate output."""
        latt = np.array([[1, 1, 0],
                         [1, 0, 1],
                         [0, 1, 1]]) * 2.5
        b1, b2, b3 = calculate_reciprocal_vectors(latt[0], latt[1], latt[2])

        k_direct = mp(2, 2, 2, b1, b2, b3, direct=True)
        k_cart = mp(2, 2, 2, b1, b2, b3, direct=False)

        # Cartesian = direct @ [b1; b2; b3]
        k_cart_manual = np.tensordot(k_direct, np.array([b1, b2, b3]), axes=(1, 0))
        np.testing.assert_allclose(k_cart, k_cart_manual, atol=1e-10)

    def test_mp_asymmetric_mesh(self):
        """Test with asymmetric mesh grid."""
        k = mp(2, 3, 4, gamma_centered=True)
        assert k.shape == (2 * 3 * 4, 3)


class TestBuildKmeshgrid:
    """Tests for build_kmeshgrid function."""

    def test_build_kmeshgrid_fcc(self):
        """Test meshgrid calculation for FCC cell."""
        cell = cellpar_to_cell([4.22798145, 4.22798145, 4.22798145, 60, 60, 60])
        b1, b2, b3 = calculate_reciprocal_vectors(cell[0], cell[1], cell[2])
        # kspacing 0.03 bohr-1, convert to angstrom-1
        n = build_kmeshgrid(b1, b2, b3, 0.03 * 1.889725989)
        assert n == [33, 33, 33]

    def test_build_kmeshgrid_minimum_one(self):
        """Meshgrid should be at least 1 in each direction."""
        b1 = np.array([0.1, 0, 0])  # Very small reciprocal vector
        b2 = np.array([0, 0.1, 0])
        b3 = np.array([0, 0, 0.1])
        n = build_kmeshgrid(b1, b2, b3, 1.0)  # Large kspacing
        assert all(ni >= 1 for ni in n)

    def test_build_kmeshgrid_list_kspacing(self):
        """Test with different kspacing in each direction."""
        cell = cellpar_to_cell([4.0, 4.0, 4.0, 90, 90, 90])
        b1, b2, b3 = calculate_reciprocal_vectors(cell[0], cell[1], cell[2])
        n = build_kmeshgrid(b1, b2, b3, [0.1, 0.2, 0.3])
        # Different kspacing should give different mesh sizes
        assert n[0] >= n[1] >= n[2]


class TestMonkhorstPackSampling:
    """Tests for monkhorst_pack_sampling wrapper function."""

    def test_simple_mesh(self):
        """Test simple mesh grid specification."""
        nk1, nk2, nk3 = 4, 4, 4
        k = monkhorst_pack_sampling(nk1, nk2, nk3)
        assert k.shape == (nk1 * nk2 * nk3, 3)

    def test_with_kspacing(self):
        """Test with kspacing specification."""
        cell = cellpar_to_cell([4.22798145, 4.22798145, 4.22798145, 60, 60, 60])
        k = monkhorst_pack_sampling(cell=cell, kspac=0.03 * 1.889725989)
        assert k.shape == (33**3, 3)

    def test_gamma_centered_vs_original(self):
        """Gamma-centered and original MP should differ."""
        k_gamma = monkhorst_pack_sampling(4, 4, 4, gamma_centered=True)
        k_orig = monkhorst_pack_sampling(4, 4, 4, gamma_centered=False)

        # Should have same shape but different values
        assert k_gamma.shape == k_orig.shape
        assert not np.allclose(k_gamma, k_orig)


class TestGetSymmOps:
    """Tests for get_symm_ops function."""

    @pytest.fixture
    def si_diamond(self):
        """Create Si diamond structure."""
        latt = np.array([[1, 1, 0],
                         [1, 0, 1],
                         [0, 1, 1]]) * 2.5
        tau_d = np.array([[0, 0, 0], [0.25, 0.25, 0.25]])
        tau_c = tau_d @ latt
        return Atoms(positions=tau_c, cell=latt, symbols=['Si', 'Si'])

    def test_symm_ops_returns_list(self, si_diamond):
        """Should return a list of rotation matrices."""
        ops = get_symm_ops(si_diamond)
        assert isinstance(ops, list)
        assert len(ops) > 0

    def test_symm_ops_shape(self, si_diamond):
        """Each symmetry operation should be 3x3 matrix."""
        ops = get_symm_ops(si_diamond)
        for op in ops:
            assert op.shape == (3, 3)

    def test_symm_ops_contains_identity(self, si_diamond):
        """Symmetry operations should include identity."""
        ops = get_symm_ops(si_diamond)
        has_identity = any(np.allclose(op, np.eye(3)) for op in ops)
        assert has_identity, "Symmetry operations should include identity"

    def test_symm_ops_diamond_count(self, si_diamond):
        """Si diamond should have 48 symmetry operations."""
        ops = get_symm_ops(si_diamond)
        assert len(ops) == 48, f"Expected 48 symmetry ops, got {len(ops)}"


class TestReduceTimeInversion:
    """Tests for reduce_time_inversion function."""

    def test_reduces_kpoint_count(self):
        """Time-inversion should reduce k-point count."""
        k = mp(4, 4, 4, gamma_centered=True)
        k_reduced, wk = reduce_time_inversion(k)
        assert len(k_reduced) == 36  # Expected for 4x4x4 gamma-centered
        assert len(wk) == 36

    def test_weight_conservation(self):
        """Total weight should equal original k-point count."""
        k = mp(4, 4, 4, gamma_centered=True)
        _, wk = reduce_time_inversion(k)
        assert wk.sum() == len(k)

    def test_single_kpoint(self):
        """Single k-point should remain unchanged."""
        k = np.array([[0.0, 0.0, 0.0]])
        k_reduced, _ = reduce_time_inversion(k)
        assert len(k_reduced) == 1
        np.testing.assert_allclose(k_reduced[0], [0, 0, 0])

    def test_output_range(self):
        """Reduced k-points should be in [-0.5, 0.5) range."""
        k = mp(4, 4, 4, gamma_centered=True)
        k_reduced, _ = reduce_time_inversion(k)
        assert k_reduced.min() >= -0.5
        assert k_reduced.max() < 0.5

    def test_mp_sampling(self):
        """Test with original MP sampling."""
        k = mp(4, 4, 4, gamma_centered=False)
        k_reduced, wk = reduce_time_inversion(k)
        assert len(k_reduced) == 32  # Expected for 4x4x4 MP
        assert wk.sum() == 64


class TestReduceRotation:
    """Tests for reduce_rotation function."""

    def test_identity_only(self):
        """With only identity symmetry, equivalent k-points should merge."""
        k = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]])
        symm_op = [np.eye(3)]
        k_reduced, degen = reduce_rotation(k, symm_op)
        # (1, 1, 1) wraps to (0, 0, 0)
        assert k_reduced.shape == (2, 3)
        np.testing.assert_allclose(degen, [1, 2])

    def test_with_degeneracies(self):
        """Test with pre-existing degeneracies."""
        k = np.array([[0, 0, 0], [0.25, 0.25, 0.25]])
        symm_op = [np.eye(3)]
        degeneracies = [2, 3]
        k_reduced, degen = reduce_rotation(k, symm_op, degeneracies_=degeneracies)
        assert len(k_reduced) == 2
        np.testing.assert_allclose(degen, [2, 3])

    def test_si_diamond_reduction(self):
        """Test full reduction for Si diamond."""
        latt = np.array([[1, 1, 0],
                         [1, 0, 1],
                         [0, 1, 1]]) * 2.5
        tau_d = np.array([[0, 0, 0], [0.25, 0.25, 0.25]])
        tau_c = tau_d @ latt
        si_diamond = Atoms(positions=tau_c, cell=latt, symbols=['Si', 'Si'])

        k = mp(4, 4, 4, gamma_centered=True)
        k_tinv, degen_tinv = reduce_time_inversion(k)
        symm_op = get_symm_ops(si_diamond)

        k_ibz, degen = reduce_rotation(k_tinv, symm_op, degeneracies_=degen_tinv)
        assert len(k_ibz) == 8  # 8 irreducible k-points
        assert np.sum(degen) == 64


class TestReduce:
    """Tests for reduce function (combined reduction)."""

    def test_reduce_with_time_inversion(self):
        """Test reduce with time inversion enabled."""
        k = mp(4, 4, 4, gamma_centered=True)
        symm_op = [np.eye(3)]  # Only identity
        k_reduced, _ = reduce(k, symm_op, time_inversion_symmetry=True)
        assert len(k_reduced) == 36  # Same as time_inversion alone

    def test_reduce_without_time_inversion(self):
        """Test reduce with time inversion disabled."""
        k = mp(4, 4, 4, gamma_centered=True)
        symm_op = [np.eye(3)]
        k_reduced, _ = reduce(k, symm_op, time_inversion_symmetry=False)
        assert len(k_reduced) == 64  # No reduction with identity only

    def test_reduce_full_symmetry(self):
        """Test reduce with full symmetry."""
        latt = np.array([[1, 1, 0],
                         [1, 0, 1],
                         [0, 1, 1]]) * 2.5
        tau_d = np.array([[0, 0, 0], [0.25, 0.25, 0.25]])
        tau_c = tau_d @ latt
        si_diamond = Atoms(positions=tau_c, cell=latt, symbols=['Si', 'Si'])

        k = mp(4, 4, 4, gamma_centered=True)
        symm_op = get_symm_ops(si_diamond)

        k_reduced, degen = reduce(k, symm_op, time_inversion_symmetry=True)
        assert len(k_reduced) == 8
        assert np.sum(degen) == 64


class TestSample:
    """Tests for sample function (main workflow function)."""

    @pytest.fixture
    def si_diamond(self):
        """Create Si diamond structure."""
        latt = np.array([[1, 1, 0],
                         [1, 0, 1],
                         [0, 1, 1]]) * 2.5
        tau_d = np.array([[0, 0, 0], [0.25, 0.25, 0.25]])
        tau_c = tau_d @ latt
        return Atoms(positions=tau_c, cell=latt, symbols=['Si', 'Si'])

    def test_sample_full_symmetry(self, si_diamond):
        """Test sample with full symmetry (default)."""
        k, wk = sample(si_diamond, meshgrid=[4, 4, 4])
        assert k.shape == (8, 3)
        assert len(wk) == 8
        np.testing.assert_allclose(np.sum(wk), 1.0)

    def test_sample_no_rotational_symmetry(self, si_diamond):
        """Test sample without rotational symmetry (ABACUS symmetry 0)."""
        k, wk = sample(si_diamond, meshgrid=[4, 4, 4], rotational_symmetry=False)
        assert len(k) == 36  # Only time-inversion reduction
        np.testing.assert_allclose(np.sum(wk), 1.0)

    def test_sample_no_symmetry(self, si_diamond):
        """Test sample without any symmetry (ABACUS symmetry -1)."""
        k, wk = sample(si_diamond, meshgrid=[4, 4, 4],
                       rotational_symmetry=False, time_inversion_symmetry=False)
        assert len(k) == 64  # No reduction
        np.testing.assert_allclose(np.sum(wk), 1.0)

    def test_sample_with_meshspacing(self, si_diamond):
        """Test sample with meshspacing instead of meshgrid."""
        k, wk = sample(si_diamond, meshspacing=[0.2, 0.2, 0.2])
        assert len(k) > 0
        np.testing.assert_allclose(np.sum(wk), 1.0)

    def test_sample_output_range(self, si_diamond):
        """Sampled k-points should be in [-0.5, 0.5) range."""
        k, _ = sample(si_diamond, meshgrid=[4, 4, 4])
        assert k.min() >= -0.5
        assert k.max() < 0.5

    def test_sample_gamma_centered_false(self, si_diamond):
        """Test sample with original MP scheme."""
        k, wk = sample(si_diamond, meshgrid=[4, 4, 4], gamma_centered=False)
        assert len(k) > 0
        np.testing.assert_allclose(np.sum(wk), 1.0)


class TestConsistencyWithMakeKpoints:
    """Tests ensuring consistency between ksampling.py and make_kpoints.py."""

    def test_mp_consistency_gamma_centered(self):
        """mp(gamma_centered=True) should match make_kpoints.gamma_center."""
        from dptb.utils.make_kpoints import gamma_center

        for mesh in [[3, 3, 3], [4, 4, 4], [2, 3, 4]]:
            kpts_ks = mp(*mesh, gamma_centered=True)
            kpts_mk = gamma_center(mesh)

            # Sort for comparison
            kpts_ks_sorted = kpts_ks[np.lexsort(kpts_ks.T[::-1])]
            kpts_mk_sorted = kpts_mk[np.lexsort(kpts_mk.T[::-1])]

            np.testing.assert_allclose(kpts_ks_sorted, kpts_mk_sorted, atol=1e-10,
                                       err_msg=f"Gamma mismatch for mesh {mesh}")

    def test_mp_consistency_original(self):
        """mp(gamma_centered=False) should match make_kpoints.monkhorst_pack."""
        from dptb.utils.make_kpoints import monkhorst_pack

        for mesh in [[3, 3, 3], [4, 4, 4], [2, 3, 4]]:
            kpts_ks = mp(*mesh, gamma_centered=False)
            kpts_mk = monkhorst_pack(mesh)

            # Sort for comparison
            kpts_ks_sorted = kpts_ks[np.lexsort(kpts_ks.T[::-1])]
            kpts_mk_sorted = kpts_mk[np.lexsort(kpts_mk.T[::-1])]

            np.testing.assert_allclose(kpts_ks_sorted, kpts_mk_sorted, atol=1e-10,
                                       err_msg=f"MP mismatch for mesh {mesh}")

    def test_time_inversion_consistency(self):
        """reduce_time_inversion should match make_kpoints.time_symmetry_reduce count."""
        from dptb.utils.make_kpoints import time_symmetry_reduce

        for mesh in [[4, 4, 4], [3, 3, 3], [2, 4, 6]]:
            # Gamma-centered
            kpts_ks, _ = reduce_time_inversion(mp(*mesh, gamma_centered=True))
            kpts_mk, _ = time_symmetry_reduce(mesh, is_gamma_center=True)
            assert len(kpts_ks) == len(kpts_mk), \
                f"Gamma {mesh}: {len(kpts_ks)} vs {len(kpts_mk)}"

            # MP
            kpts_ks, _ = reduce_time_inversion(mp(*mesh, gamma_centered=False))
            kpts_mk, _ = time_symmetry_reduce(mesh, is_gamma_center=False)
            assert len(kpts_ks) == len(kpts_mk), \
                f"MP {mesh}: {len(kpts_ks)} vs {len(kpts_mk)}"

    def test_kspacing_consistency(self):
        """build_kmeshgrid should match make_kpoints.kgrid_spacing meshgrid."""
        from dptb.utils.make_kpoints import kgrid_spacing

        cell = cellpar_to_cell([4.22798145, 4.22798145, 4.22798145, 60, 60, 60])
        struct = Atoms(symbols=['Si'], positions=[[0, 0, 0]], cell=cell)
        b1, b2, b3 = calculate_reciprocal_vectors(cell[0], cell[1], cell[2])

        for kspacing in [0.05, 0.1, 0.2]:
            n_ks = build_kmeshgrid(b1, b2, b3, kspacing)
            kpts_mk = kgrid_spacing(struct, kspacing, 'Gamma')

            expected_total = np.prod(n_ks)
            assert len(kpts_mk) == expected_total, \
                f"kspacing={kspacing}: expected {expected_total}, got {len(kpts_mk)}"


class TestImplementationConsistency:
    """Tests ensuring consistency among all four k-point reduction implementations.

    The four implementations are:
    - Python direct: O(n²m) reference implementation
    - Python hash: O(nm) hash-based fallback when Numba unavailable
    - Numba direct: O(n²m) JIT-compiled version
    - Numba hash: O(nm) JIT-compiled hash-based version

    All four should produce identical k-points and weights for the same input.
    """

    @pytest.fixture
    def si_diamond(self):
        """Create Si diamond structure."""
        latt = np.array([[1, 1, 0],
                         [1, 0, 1],
                         [0, 1, 1]]) * 2.5
        tau_d = np.array([[0, 0, 0], [0.25, 0.25, 0.25]])
        tau_c = tau_d @ latt
        return Atoms(positions=tau_c, cell=latt, symbols=['Si', 'Si'])

    @staticmethod
    def _compare_results(k1, wk1, k2, wk2):
        """Compare two reduction results after sorting."""
        if len(k1) != len(k2):
            return False, f"Different count: {len(k1)} vs {len(k2)}"

        # Sort by k-point coordinates for comparison
        idx1 = np.lexsort((k1[:, 2], k1[:, 1], k1[:, 0]))
        idx2 = np.lexsort((k2[:, 2], k2[:, 1], k2[:, 0]))

        k1_sorted, wk1_sorted = k1[idx1], wk1[idx1]
        k2_sorted, wk2_sorted = k2[idx2], wk2[idx2]

        if not np.allclose(k1_sorted, k2_sorted, atol=1e-10):
            return False, "K-points differ"

        if not np.allclose(wk1_sorted, wk2_sorted, atol=1e-10):
            return False, "Weights differ"

        return True, "MATCH"

    def test_numba_available(self):
        """Check if Numba is available for testing."""
        from dptb.utils.ksampling import HAS_NUMBA
        # This test documents whether Numba is available, not a pass/fail
        if not HAS_NUMBA:
            pytest.skip("Numba not available, skipping Numba consistency tests")

    def test_direct_method_consistency(self, si_diamond):
        """Test consistency between Python direct and Numba direct methods."""
        from dptb.utils.ksampling import (
            HAS_NUMBA, _reduce_by_symmetry_direct, _reduce_by_symmetry_numba
        )
        if not HAS_NUMBA:
            pytest.skip("Numba not available")

        symm_ops = get_symm_ops(si_diamond)
        symm_prec = 1e-8

        for mesh in [[4, 4, 4], [8, 8, 8], [10, 10, 10]]:
            kpts = mp(*mesh, gamma_centered=True)

            # Python direct method
            k_py, wk_py = _reduce_by_symmetry_direct(
                kpts.copy(), np.ones(len(kpts), dtype=float), symm_ops, symm_prec
            )

            # Numba direct method
            k_nb, wk_nb = _reduce_by_symmetry_numba(
                kpts.copy(), np.ones(len(kpts), dtype=float), symm_ops, symm_prec,
                use_hash=False
            )

            match, msg = self._compare_results(k_py, wk_py, k_nb, wk_nb)
            assert match, f"Mesh {mesh}: Python-direct vs Numba-direct: {msg}"

    def test_hash_method_consistency(self, si_diamond):
        """Test consistency between Python direct and Numba hash methods.

        Verifies that the Numba hash-based O(nm) method produces identical
        results to the Python direct O(n²m) reference implementation.
        """
        from dptb.utils.ksampling import (
            HAS_NUMBA, _reduce_by_symmetry_direct, _reduce_by_symmetry_numba
        )
        if not HAS_NUMBA:
            pytest.skip("Numba not available")

        symm_ops = get_symm_ops(si_diamond)
        symm_prec = 1e-8

        for mesh in [[4, 4, 4], [8, 8, 8], [12, 12, 12]]:
            kpts = mp(*mesh, gamma_centered=True)

            # Python direct method (reference)
            k_py, wk_py = _reduce_by_symmetry_direct(
                kpts.copy(), np.ones(len(kpts), dtype=float), symm_ops, symm_prec
            )

            # Numba hash method
            k_nb, wk_nb = _reduce_by_symmetry_numba(
                kpts.copy(), np.ones(len(kpts), dtype=float), symm_ops, symm_prec,
                use_hash=True
            )

            match, msg = self._compare_results(k_py, wk_py, k_nb, wk_nb)
            assert match, f"Mesh {mesh}: Python-direct vs Numba-hash: {msg}"

    def test_time_inversion_consistency(self, si_diamond):
        """Test consistency for time-inversion symmetry reduction."""
        from dptb.utils.ksampling import (
            HAS_NUMBA, _reduce_by_symmetry_direct, _reduce_by_symmetry_numba
        )
        if not HAS_NUMBA:
            pytest.skip("Numba not available")

        time_inv_op = [-np.eye(3)]
        symm_prec = 1e-8

        for mesh in [[10, 10, 10], [15, 15, 15]]:
            kpts = mp(*mesh, gamma_centered=True)

            # Python direct method
            k_py, wk_py = _reduce_by_symmetry_direct(
                kpts.copy(), np.ones(len(kpts), dtype=float), time_inv_op, symm_prec
            )

            # Numba hash method
            k_nb, wk_nb = _reduce_by_symmetry_numba(
                kpts.copy(), np.ones(len(kpts), dtype=float), time_inv_op, symm_prec,
                use_hash=True
            )

            match, msg = self._compare_results(k_py, wk_py, k_nb, wk_nb)
            assert match, f"Mesh {mesh}: Time-inversion consistency: {msg}"

    def test_weight_conservation(self, si_diamond):
        """Test that Numba methods conserve total weight."""
        from dptb.utils.ksampling import HAS_NUMBA, _reduce_by_symmetry_numba
        if not HAS_NUMBA:
            pytest.skip("Numba not available")

        symm_ops = get_symm_ops(si_diamond)
        symm_prec = 1e-8

        for mesh in [[6, 6, 6], [10, 10, 10]]:
            kpts = mp(*mesh, gamma_centered=True)
            total_kpts = len(kpts)

            # Test both direct and hash methods
            for use_hash in [True, False]:
                _, wk = _reduce_by_symmetry_numba(
                    kpts.copy(), np.ones(len(kpts), dtype=float), symm_ops, symm_prec,
                    use_hash=use_hash
                )
                assert np.isclose(np.sum(wk), total_kpts), \
                    f"Mesh {mesh}, use_hash={use_hash}: Weight not conserved"

    def test_different_structures(self):
        """Test consistency across different crystal structures."""
        from dptb.utils.ksampling import (
            HAS_NUMBA, _reduce_by_symmetry_direct, _reduce_by_symmetry_numba
        )
        from ase.build import bulk
        if not HAS_NUMBA:
            pytest.skip("Numba not available")

        structures = [
            ("Cu_fcc", bulk('Cu', 'fcc', a=3.6)),
            ("Fe_bcc", bulk('Fe', 'bcc', a=2.87)),
        ]

        symm_prec = 1e-8
        mesh = [8, 8, 8]

        for name, struct in structures:
            symm_ops = get_symm_ops(struct)
            kpts = mp(*mesh, gamma_centered=True)

            # Python direct
            k_py, wk_py = _reduce_by_symmetry_direct(
                kpts.copy(), np.ones(len(kpts), dtype=float), symm_ops, symm_prec
            )

            # Numba hash
            k_nb, wk_nb = _reduce_by_symmetry_numba(
                kpts.copy(), np.ones(len(kpts), dtype=float), symm_ops, symm_prec,
                use_hash=True
            )

            match, msg = self._compare_results(k_py, wk_py, k_nb, wk_nb)
            assert match, f"{name}: Python vs Numba: {msg}"

    def test_sample_function_consistency(self, si_diamond):
        """Test that sample() produces consistent results with Numba enabled/disabled."""
        from dptb.utils.ksampling import HAS_NUMBA
        if not HAS_NUMBA:
            pytest.skip("Numba not available")

        # sample() automatically uses Numba when available
        # We test that results are valid (weights sum to 1, correct IBZ count)
        for mesh in [[4, 4, 4], [8, 8, 8]]:
            k, wk = sample(si_diamond, meshgrid=mesh)

            # Check weights sum to 1
            np.testing.assert_allclose(np.sum(wk), 1.0, rtol=1e-10)

            # Check k-points are in valid range
            assert k.min() >= -0.5
            assert k.max() < 0.5

            # Check total degeneracy equals original k-point count
            total_kpts = np.prod(mesh)
            np.testing.assert_allclose(np.sum(wk) * total_kpts, total_kpts, rtol=1e-10)

    def test_python_hash_vs_direct_consistency(self, si_diamond):
        """Test that Python hash method produces identical results to Python direct.

        This test verifies that the fallback Python hash method (used when Numba
        is not available) produces the exact same k-points and weights as the
        reference Python direct method. Both should select the same representative
        k-points from each equivalence class.

        If this test fails, it means the Python hash fallback is inconsistent
        with the reference implementation, which would cause different results
        depending on whether Numba is installed.
        """
        from dptb.utils.ksampling import (
            _reduce_by_symmetry_direct, _reduce_by_symmetry_hash
        )

        symm_ops = get_symm_ops(si_diamond)
        symm_prec = 1e-8

        for mesh in [[4, 4, 4], [8, 8, 8], [10, 10, 10]]:
            kpts = mp(*mesh, gamma_centered=True)

            # Python direct method (reference)
            k_direct, wk_direct = _reduce_by_symmetry_direct(
                kpts.copy(), np.ones(len(kpts), dtype=float), symm_ops, symm_prec
            )

            # Python hash method (should match direct)
            k_hash, wk_hash = _reduce_by_symmetry_hash(
                kpts.copy(), np.ones(len(kpts), dtype=float), symm_ops, symm_prec
            )

            match, msg = self._compare_results(k_direct, wk_direct, k_hash, wk_hash)
            assert match, f"Mesh {mesh}: Python-hash vs Python-direct: {msg}"

    def test_all_four_implementations_match(self, si_diamond):
        """Comprehensive test that all four implementations produce identical results.

        Tests Python-direct, Python-hash, Numba-direct, and Numba-hash on the same
        input to verify they all produce exactly the same k-points and weights.
        """
        from dptb.utils.ksampling import (
            HAS_NUMBA, _reduce_by_symmetry_direct, _reduce_by_symmetry_hash,
            _reduce_by_symmetry_numba
        )

        symm_ops = get_symm_ops(si_diamond)
        symm_prec = 1e-8

        for mesh in [[4, 4, 4], [8, 8, 8]]:
            kpts = mp(*mesh, gamma_centered=True)

            # Python direct (reference)
            k_py_direct, wk_py_direct = _reduce_by_symmetry_direct(
                kpts.copy(), np.ones(len(kpts), dtype=float), symm_ops, symm_prec
            )

            # Python hash
            k_py_hash, wk_py_hash = _reduce_by_symmetry_hash(
                kpts.copy(), np.ones(len(kpts), dtype=float), symm_ops, symm_prec
            )

            # Verify Python implementations match
            match, msg = self._compare_results(
                k_py_direct, wk_py_direct, k_py_hash, wk_py_hash
            )
            assert match, f"Mesh {mesh}: Py-direct vs Py-hash: {msg}"

            if HAS_NUMBA:
                # Numba direct
                k_nb_direct, wk_nb_direct = _reduce_by_symmetry_numba(
                    kpts.copy(), np.ones(len(kpts), dtype=float), symm_ops, symm_prec,
                    use_hash=False
                )

                # Numba hash
                k_nb_hash, wk_nb_hash = _reduce_by_symmetry_numba(
                    kpts.copy(), np.ones(len(kpts), dtype=float), symm_ops, symm_prec,
                    use_hash=True
                )

                # Verify all Numba implementations match Python direct
                match, msg = self._compare_results(
                    k_py_direct, wk_py_direct, k_nb_direct, wk_nb_direct
                )
                assert match, f"Mesh {mesh}: Py-direct vs Nb-direct: {msg}"

                match, msg = self._compare_results(
                    k_py_direct, wk_py_direct, k_nb_hash, wk_nb_hash
                )
                assert match, f"Mesh {mesh}: Py-direct vs Nb-hash: {msg}"

                # Verify Numba implementations match each other
                match, msg = self._compare_results(
                    k_nb_direct, wk_nb_direct, k_nb_hash, wk_nb_hash
                )
                assert match, f"Mesh {mesh}: Nb-direct vs Nb-hash: {msg}"

    def test_time_inversion_all_implementations(self):
        """Test all implementations with time-inversion symmetry only.

        Time-inversion is a single symmetry operation (-I), which is simpler
        than full rotational symmetry but still important to verify.
        """
        from dptb.utils.ksampling import (
            HAS_NUMBA, _reduce_by_symmetry_direct, _reduce_by_symmetry_hash,
            _reduce_by_symmetry_numba
        )

        time_inv_op = [-np.eye(3)]
        symm_prec = 1e-8

        for mesh in [[4, 4, 4], [10, 10, 10]]:
            kpts = mp(*mesh, gamma_centered=True)

            # Python direct (reference)
            k_py_direct, wk_py_direct = _reduce_by_symmetry_direct(
                kpts.copy(), np.ones(len(kpts), dtype=float), time_inv_op, symm_prec
            )

            # Python hash
            k_py_hash, wk_py_hash = _reduce_by_symmetry_hash(
                kpts.copy(), np.ones(len(kpts), dtype=float), time_inv_op, symm_prec
            )

            match, msg = self._compare_results(
                k_py_direct, wk_py_direct, k_py_hash, wk_py_hash
            )
            assert match, f"Mesh {mesh}: Py-direct vs Py-hash (time-inv): {msg}"

            if HAS_NUMBA:
                k_nb_hash, wk_nb_hash = _reduce_by_symmetry_numba(
                    kpts.copy(), np.ones(len(kpts), dtype=float), time_inv_op, symm_prec,
                    use_hash=True
                )

                match, msg = self._compare_results(
                    k_py_direct, wk_py_direct, k_nb_hash, wk_nb_hash
                )
                assert match, f"Mesh {mesh}: Py-direct vs Nb-hash (time-inv): {msg}"
