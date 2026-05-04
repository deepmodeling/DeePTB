import pytest
import numpy as np
from ase.atoms import Atoms
from ase.geometry import cellpar_to_cell
from dptb.utils.make_kpoints import (
    gamma_center, monkhorst_pack, kmesh_sampling, kmesh_sampling_negf,
    time_symmetry_reduce, kgrid_spacing, rot_revlatt_2D, kmesh_fs
)


class TestGammaCenter:
    """Tests for the gamma_center k-point mesh generation function."""

    def test_gamma_center_contains_gamma_point(self):
        """Gamma-centered mesh must always contain the gamma point (0,0,0)."""
        for mesh in [[3, 3, 3], [4, 4, 4], [5, 5, 5], [2, 3, 4]]:
            kpts = gamma_center(mesh)
            assert any(np.allclose(k, [0, 0, 0]) for k in kpts), \
                f"Gamma point not found in mesh {mesh}"

    def test_gamma_center_range(self):
        """K-points should be in the range [-0.5, 0.5)."""
        kpts = gamma_center([4, 4, 4])
        assert kpts.min() >= -0.5, "K-points should be >= -0.5"
        assert kpts.max() < 0.5, "K-points should be < 0.5"

    def test_gamma_center_odd_mesh_symmetric(self):
        """Odd meshes should produce symmetric k-points around gamma."""
        kpts = gamma_center([3, 3, 3])
        unique_k = np.unique(np.round(kpts[:, 0], 8))
        # For 3x3x3: should be -1/3, 0, 1/3
        expected = np.array([-1/3, 0, 1/3])
        np.testing.assert_allclose(unique_k, expected, atol=1e-10)

    def test_gamma_center_even_mesh(self):
        """Even meshes should produce correct k-points."""
        kpts = gamma_center([4, 4, 4])
        unique_k = np.unique(np.round(kpts[:, 0], 8))
        # For 4x4x4: should be -0.25, 0, 0.25 (0.5 wraps to -0.5)
        expected = np.array([-0.5, -0.25, 0, 0.25])
        np.testing.assert_allclose(unique_k, expected, atol=1e-10)

    def test_gamma_center_total_points(self):
        """Total number of k-points should be N1 * N2 * N3."""
        mesh = [3, 4, 5]
        kpts = gamma_center(mesh)
        assert len(kpts) == np.prod(mesh)

    def test_gamma_center_shape(self):
        """Output should have shape (N, 3)."""
        mesh = [2, 3, 4]
        kpts = gamma_center(mesh)
        assert kpts.shape == (np.prod(mesh), 3)

    def test_gamma_center_1d_mesh(self):
        """Test with 1D-like mesh (1x1xN)."""
        kpts = gamma_center([1, 1, 5])
        assert len(kpts) == 5
        assert all(np.allclose(k[:2], [0, 0]) for k in kpts)
        unique_kz = np.unique(np.round(kpts[:, 2], 8))
        expected = np.array([-0.4, -0.2, 0, 0.2, 0.4])
        np.testing.assert_allclose(unique_kz, expected, atol=1e-10)

    def test_gamma_center_single_point(self):
        """1x1x1 mesh should return only gamma point."""
        kpts = gamma_center([1, 1, 1])
        assert len(kpts) == 1
        np.testing.assert_allclose(kpts[0], [0, 0, 0])

    def test_gamma_center_invalid_mesh_length(self):
        """Should raise error for invalid mesh length."""
        with pytest.raises(ValueError):
            gamma_center([3, 3])
        with pytest.raises(ValueError):
            gamma_center([3, 3, 3, 3])

    def test_gamma_center_invalid_mesh_values(self):
        """Should raise error for non-positive mesh values."""
        with pytest.raises(ValueError):
            gamma_center([0, 3, 3])
        with pytest.raises(ValueError):
            gamma_center([3, -1, 3])

    def test_gamma_center_differs_from_monkhorst_pack_even(self):
        """Gamma-centered and MP meshes should differ for even grids."""
        mesh = [4, 4, 4]
        gamma_kpts = gamma_center(mesh)
        mp_kpts = monkhorst_pack(mesh)

        # Gamma-centered must contain gamma point
        gamma_has_gamma = any(np.allclose(k, [0, 0, 0]) for k in gamma_kpts)
        mp_has_gamma = any(np.allclose(k, [0, 0, 0]) for k in mp_kpts)

        assert gamma_has_gamma, "Gamma-centered mesh must contain gamma"
        assert not mp_has_gamma, "MP even mesh should not contain gamma"


class TestMonkhorstPack:
    """Tests for the monkhorst_pack k-point mesh generation function."""

    def test_monkhorst_pack_symmetric(self):
        """MP mesh should be symmetric around gamma."""
        kpts = monkhorst_pack([3, 3, 3])
        unique_k = np.unique(np.round(kpts[:, 0], 8))
        expected = np.array([-1/3, 0, 1/3])
        np.testing.assert_allclose(unique_k, expected, atol=1e-10)

    def test_monkhorst_pack_even_no_gamma(self):
        """Even MP mesh should not contain gamma point."""
        kpts = monkhorst_pack([4, 4, 4])
        has_gamma = any(np.allclose(k, [0, 0, 0]) for k in kpts)
        assert not has_gamma, "Even MP mesh should not contain gamma"

    def test_monkhorst_pack_odd_has_gamma(self):
        """Odd MP mesh should contain gamma point."""
        kpts = monkhorst_pack([3, 3, 3])
        has_gamma = any(np.allclose(k, [0, 0, 0]) for k in kpts)
        assert has_gamma, "Odd MP mesh should contain gamma"

    def test_monkhorst_pack_range(self):
        """MP k-points should be in [-0.5, 0.5) range."""
        kpts = monkhorst_pack([4, 4, 4])
        assert kpts.min() >= -0.5, "K-points should be >= -0.5"
        assert kpts.max() < 0.5, "K-points should be < 0.5"

    def test_monkhorst_pack_total_points(self):
        """Total number of k-points should be N1 * N2 * N3."""
        mesh = [3, 4, 5]
        kpts = monkhorst_pack(mesh)
        assert len(kpts) == np.prod(mesh)


class TestKmeshSampling:
    """Tests for kmesh_sampling function."""

    def test_kmesh_sampling_gamma_center(self):
        """kmesh_sampling with is_gamma_center=True should match gamma_center."""
        mesh = [4, 4, 4]
        kpts_sampling = kmesh_sampling(mesh, is_gamma_center=True)
        kpts_direct = gamma_center(mesh)
        np.testing.assert_allclose(
            np.sort(kpts_sampling, axis=0),
            np.sort(kpts_direct, axis=0)
        )

    def test_kmesh_sampling_monkhorst_pack(self):
        """kmesh_sampling with is_gamma_center=False should match monkhorst_pack."""
        mesh = [4, 4, 4]
        kpts_sampling = kmesh_sampling(mesh, is_gamma_center=False)
        kpts_direct = monkhorst_pack(mesh)
        np.testing.assert_allclose(
            np.sort(kpts_sampling, axis=0),
            np.sort(kpts_direct, axis=0)
        )


class TestTimeSymmetryReduce:
    """Tests for time_symmetry_reduce function."""

    def test_time_symmetry_reduce_gamma_center_count(self):
        """Time-reversal reduction should reduce k-point count for gamma-centered mesh."""
        mesh = [4, 4, 4]
        kpts, wk = time_symmetry_reduce(mesh, is_gamma_center=True)
        # 4x4x4 gamma-centered should reduce to 36 k-points
        assert len(kpts) == 36
        assert len(wk) == 36

    def test_time_symmetry_reduce_mp_count(self):
        """Time-reversal reduction should reduce k-point count for MP mesh."""
        mesh = [4, 4, 4]
        kpts, wk = time_symmetry_reduce(mesh, is_gamma_center=False)
        # 4x4x4 MP should reduce to 32 k-points
        assert len(kpts) == 32
        assert len(wk) == 32

    def test_time_symmetry_reduce_weight_sum(self):
        """Weights should sum to 1.0."""
        for is_gamma in [True, False]:
            _, wk = time_symmetry_reduce([4, 4, 4], is_gamma_center=is_gamma)
            np.testing.assert_allclose(wk.sum(), 1.0, atol=1e-10)

    def test_time_symmetry_reduce_output_range(self):
        """Reduced k-points should be in [-0.5, 0.5) range."""
        for is_gamma in [True, False]:
            kpts, _ = time_symmetry_reduce([4, 4, 4], is_gamma_center=is_gamma)
            assert kpts.min() >= -0.5, "K-points should be >= -0.5"
            assert kpts.max() < 0.5, "K-points should be < 0.5"

    def test_time_symmetry_reduce_single_point(self):
        """1x1x1 mesh should return single k-point with weight 1."""
        kpts, wk = time_symmetry_reduce([1, 1, 1], is_gamma_center=True)
        assert len(kpts) == 1
        np.testing.assert_allclose(kpts[0], [0, 0, 0])
        np.testing.assert_allclose(wk[0], 1.0)

    def test_time_symmetry_reduce_consistency_with_ksampling(self):
        """Reduced k-point count should match ksampling.reduce_time_inversion."""
        from dptb.utils.ksampling import reduce_time_inversion, mp

        for mesh in [[4, 4, 4], [3, 3, 3], [2, 4, 6]]:
            # Gamma-centered
            kpts_mk, _ = time_symmetry_reduce(mesh, is_gamma_center=True)
            kpts_ks, _ = reduce_time_inversion(mp(*mesh, gamma_centered=True))
            assert len(kpts_mk) == len(kpts_ks), \
                f"Gamma-centered {mesh}: {len(kpts_mk)} vs {len(kpts_ks)}"

            # MP
            kpts_mk, _ = time_symmetry_reduce(mesh, is_gamma_center=False)
            kpts_ks, _ = reduce_time_inversion(mp(*mesh, gamma_centered=False))
            assert len(kpts_mk) == len(kpts_ks), \
                f"MP {mesh}: {len(kpts_mk)} vs {len(kpts_ks)}"


class TestKmeshSamplingNegf:
    """Tests for kmesh_sampling_negf function."""

    def test_kmesh_sampling_negf_with_time_reversal(self):
        """With time reversal, should return reduced k-points."""
        kpts, wk = kmesh_sampling_negf([4, 4, 4], is_gamma_center=True, is_time_reversal=True)
        assert len(kpts) < 64  # Should be reduced from 4x4x4=64
        np.testing.assert_allclose(wk.sum(), 1.0, atol=1e-10)

    def test_kmesh_sampling_negf_without_time_reversal(self):
        """Without time reversal, should return full k-mesh."""
        kpts, wk = kmesh_sampling_negf([4, 4, 4], is_gamma_center=True, is_time_reversal=False)
        assert len(kpts) == 64
        np.testing.assert_allclose(wk.sum(), 1.0, atol=1e-10)


class TestKgridSpacing:
    """Tests for kgrid_spacing function."""

    @pytest.fixture
    def cubic_structure(self):
        """Create a cubic Si structure for testing."""
        cell = cellpar_to_cell([4.0, 4.0, 4.0, 90, 90, 90])
        return Atoms(symbols=['Si'], positions=[[0, 0, 0]], cell=cell)

    @pytest.fixture
    def fcc_structure(self):
        """Create an FCC Si structure for testing."""
        cell = cellpar_to_cell([4.22798145, 4.22798145, 4.22798145, 60, 60, 60])
        return Atoms(symbols=['Si'], positions=[[0, 0, 0]], cell=cell)

    def test_kgrid_spacing_cubic(self, cubic_structure):
        """Test k-grid generation for cubic cell."""
        kspacing = 0.1  # angstrom^-1
        kpts = kgrid_spacing(cubic_structure, kspacing, 'Gamma')
        # Should generate a reasonable number of k-points
        assert len(kpts) > 0
        # K-points should be in valid range
        assert kpts.min() >= -0.5
        assert kpts.max() < 0.5

    def test_kgrid_spacing_mp_vs_gamma(self, cubic_structure):
        """MP and Gamma sampling should give different results for even grids."""
        kspacing = 0.2
        kpts_mp = kgrid_spacing(cubic_structure, kspacing, 'MP')
        kpts_gamma = kgrid_spacing(cubic_structure, kspacing, 'Gamma')

        # Same number of points
        assert len(kpts_mp) == len(kpts_gamma)

        # But different positions (for most cases)
        # Gamma should contain [0,0,0], MP may not
        gamma_has_gamma = any(np.allclose(k, [0, 0, 0]) for k in kpts_gamma)
        assert gamma_has_gamma, "Gamma sampling should contain gamma point"

    def test_kgrid_spacing_invalid_sampling(self, cubic_structure):
        """Invalid sampling method should raise error."""
        with pytest.raises(ValueError):
            kgrid_spacing(cubic_structure, 0.1, 'invalid')

    def test_kgrid_spacing_consistency_with_ksampling(self, fcc_structure):
        """Meshgrid should match ksampling.build_kmeshgrid."""
        from dptb.utils.ksampling import build_kmeshgrid, calculate_reciprocal_vectors

        cell = np.array(fcc_structure.cell)
        b1, b2, b3 = calculate_reciprocal_vectors(cell[0], cell[1], cell[2])

        for kspacing in [0.05, 0.1, 0.2]:
            n_ks = build_kmeshgrid(b1, b2, b3, kspacing)
            kpts = kgrid_spacing(fcc_structure, kspacing, 'Gamma')

            # Check that the number of k-points matches expected
            expected_total = np.prod(n_ks)
            assert len(kpts) == expected_total, \
                f"kspacing={kspacing}: expected {expected_total}, got {len(kpts)}"


class TestRotRevlatt2D:
    """Tests for rot_revlatt_2D function."""

    def test_rot_revlatt_2d_accepts_ndarray(self):
        """Function should accept numpy ndarray input."""
        rev_latt = np.array([[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]])
        rev_latt_new, newcorr = rot_revlatt_2D(rev_latt, index=[0, 1])
        assert isinstance(rev_latt_new, np.ndarray)
        assert isinstance(newcorr, np.ndarray)

    def test_rot_revlatt_2d_shape(self):
        """Output should be 3x3 matrices."""
        rev_latt = np.array([[1.0, 0.5, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]])
        rev_latt_new, newcorr = rot_revlatt_2D(rev_latt, index=[0, 1])
        assert rev_latt_new.shape == (3, 3)
        assert newcorr.shape == (3, 3)

    def test_rot_revlatt_2d_invalid_shape(self):
        """Should raise error for non-3x3 input."""
        with pytest.raises(ValueError):
            rot_revlatt_2D(np.array([[1, 0], [0, 1]]), index=[0, 1])

    def test_rot_revlatt_2d_transformation(self):
        """Transformation should be invertible."""
        rev_latt = np.array([[1.0, 0.5, 0.0],
                             [0.0, 1.0, 0.3],
                             [0.0, 0.0, 1.0]])
        rev_latt_new, newcorr = rot_revlatt_2D(rev_latt, index=[0, 1])

        # Check transformation: rev_latt_new = rev_latt @ inv(newcorr)
        reconstructed = rev_latt @ np.linalg.inv(newcorr)
        np.testing.assert_allclose(rev_latt_new, reconstructed, atol=1e-10)


class TestKmeshFs:
    """Tests for kmesh_fs (Fermi surface) function."""

    def test_kmesh_fs_range(self):
        """K-points should be in [0, 1] range."""
        _, kgrids = kmesh_fs([4, 4, 4])
        assert kgrids.min() >= 0
        assert kgrids.max() <= 1

    def test_kmesh_fs_total_points(self):
        """Total k-points should be N1 * N2 * N3."""
        mesh = [3, 4, 5]
        _, kgrids = kmesh_fs(mesh)
        assert len(kgrids) == np.prod(mesh)

    def test_kmesh_fs_linspace_values(self):
        """Returned linspace values should be correct."""
        (lx, ly, lz), _ = kmesh_fs([5, 5, 5])
        expected = np.linspace(0, 1, 5)
        np.testing.assert_allclose(lx, expected)
        np.testing.assert_allclose(ly, expected)
        np.testing.assert_allclose(lz, expected)


class TestConsistencyWithKsampling:
    """Tests ensuring consistency between make_kpoints.py and ksampling.py."""

    def test_mp_consistency(self):
        """monkhorst_pack should match ksampling.mp(gamma_centered=False)."""
        from dptb.utils.ksampling import mp

        for mesh in [[3, 3, 3], [4, 4, 4], [2, 3, 4]]:
            kpts_mk = monkhorst_pack(mesh)
            kpts_ks = mp(*mesh, gamma_centered=False)

            # Sort for comparison
            kpts_mk_sorted = kpts_mk[np.lexsort(kpts_mk.T[::-1])]
            kpts_ks_sorted = kpts_ks[np.lexsort(kpts_ks.T[::-1])]

            np.testing.assert_allclose(kpts_mk_sorted, kpts_ks_sorted, atol=1e-10,
                                       err_msg=f"MP mismatch for mesh {mesh}")

    def test_gamma_center_consistency(self):
        """gamma_center should match ksampling.mp(gamma_centered=True)."""
        from dptb.utils.ksampling import mp

        for mesh in [[3, 3, 3], [4, 4, 4], [2, 3, 4]]:
            kpts_mk = gamma_center(mesh)
            kpts_ks = mp(*mesh, gamma_centered=True)

            # Sort for comparison
            kpts_mk_sorted = kpts_mk[np.lexsort(kpts_mk.T[::-1])]
            kpts_ks_sorted = kpts_ks[np.lexsort(kpts_ks.T[::-1])]

            np.testing.assert_allclose(kpts_mk_sorted, kpts_ks_sorted, atol=1e-10,
                                       err_msg=f"Gamma mismatch for mesh {mesh}")

    def test_output_range_consistency(self):
        """Both modules should output k-points in [-0.5, 0.5) range."""
        from dptb.utils.ksampling import mp, reduce_time_inversion

        mesh = [4, 4, 4]

        # make_kpoints
        kpts_mk_gamma = gamma_center(mesh)
        kpts_mk_mp = monkhorst_pack(mesh)
        kpts_mk_tr, _ = time_symmetry_reduce(mesh, is_gamma_center=True)

        # ksampling
        kpts_ks_gamma = mp(*mesh, gamma_centered=True)
        kpts_ks_mp = mp(*mesh, gamma_centered=False)
        kpts_ks_tr, _ = reduce_time_inversion(mp(*mesh, gamma_centered=True))

        # All should be in [-0.5, 0.5)
        for name, kpts in [("mk_gamma", kpts_mk_gamma), ("mk_mp", kpts_mk_mp),
                          ("mk_tr", kpts_mk_tr), ("ks_gamma", kpts_ks_gamma),
                          ("ks_mp", kpts_ks_mp), ("ks_tr", kpts_ks_tr)]:
            assert kpts.min() >= -0.5, f"{name}: min={kpts.min()} < -0.5"
            assert kpts.max() < 0.5, f"{name}: max={kpts.max()} >= 0.5"
