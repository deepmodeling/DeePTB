import numpy as np
import pytest
from ase.atoms import Atoms
from ase.geometry import cellpar_to_cell

from dptb.kpoints.geometry import (
    calculate_reciprocal_vectors,
    get_symm_ops,
    is_integer,
    rot_revlatt_2D,
)
from dptb.kpoints.mesh import (
    build_kmeshgrid,
    gamma_center,
    kgrid_spacing,
    kmesh_fs,
    kmesh_sampling,
    kmesh_sampling_negf,
    monkhorst_pack,
    monkhorst_pack_sampling,
    mp,
    time_symmetry_reduce,
)
from dptb.kpoints.path import abacus_kpath, ase_kpath, vasp_kpath
from dptb.kpoints.reduction import (
    _reduce_by_symmetry_direct,
    _reduce_by_symmetry_hash,
    reduce,
    reduce_rotation,
    reduce_time_inversion,
)
from dptb.kpoints.sampling import sample


def _sort_kpoints(kpts):
    return kpts[np.lexsort((kpts[:, 2], kpts[:, 1], kpts[:, 0]))]


@pytest.fixture
def cubic_structure():
    cell = cellpar_to_cell([4.0, 4.0, 4.0, 90, 90, 90])
    return Atoms(symbols=["Si"], positions=[[0, 0, 0]], cell=cell)


@pytest.fixture
def si_diamond():
    latt = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]]) * 2.5
    tau_d = np.array([[0, 0, 0], [0.25, 0.25, 0.25]])
    tau_c = tau_d @ latt
    return Atoms(positions=tau_c, cell=latt, symbols=["Si", "Si"])


class TestKpointsGeometry:
    def test_is_integer_tolerance(self):
        assert is_integer(1.0 + 1e-11)
        assert not is_integer(1.0 + 1e-9)

    def test_reciprocal_vectors_are_dual_to_cell(self):
        cell = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]]) * 2.5
        g1, g2, g3 = calculate_reciprocal_vectors(cell[0], cell[1], cell[2])
        gmat = np.stack([g1, g2, g3])

        np.testing.assert_allclose(cell @ gmat.T, 2 * np.pi * np.eye(3), atol=1e-10)

    def test_symmetry_operations_for_diamond(self, si_diamond):
        symm_ops = get_symm_ops(si_diamond)

        assert len(symm_ops) == 48
        assert any(np.allclose(op, np.eye(3)) for op in symm_ops)
        assert all(op.shape == (3, 3) for op in symm_ops)

    def test_rot_revlatt_2d_returns_expected_transform(self):
        rev_latt = np.array([
            [1.0, 0.5, 0.0],
            [0.0, 1.0, 0.3],
            [0.0, 0.0, 1.0],
        ])

        rev_latt_new, newcorr = rot_revlatt_2D(rev_latt, index=[0, 1])

        assert rev_latt_new.shape == (3, 3)
        assert newcorr.shape == (3, 3)
        np.testing.assert_allclose(rev_latt_new, rev_latt @ np.linalg.inv(newcorr), atol=1e-10)


class TestKpointsMesh:
    def test_gamma_center_contains_gamma_for_even_and_odd_meshes(self):
        for mesh in [[3, 3, 3], [4, 4, 4], [2, 3, 4]]:
            kpts = gamma_center(mesh)
            assert kpts.shape == (np.prod(mesh), 3)
            assert any(np.allclose(kpt, [0, 0, 0]) for kpt in kpts)
            assert kpts.min() >= -0.5
            assert kpts.max() < 0.5

    def test_shifted_monkhorst_pack_even_mesh_excludes_gamma(self):
        kpts = monkhorst_pack([4, 4, 4])

        assert kpts.shape == (64, 3)
        assert not any(np.allclose(kpt, [0, 0, 0]) for kpt in kpts)
        assert kpts.min() >= -0.5
        assert kpts.max() < 0.5

    def test_mp_direct_and_cartesian_outputs_are_consistent(self):
        cell = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]]) * 2.5
        b1, b2, b3 = calculate_reciprocal_vectors(cell[0], cell[1], cell[2])

        k_direct = mp(2, 2, 2, b1, b2, b3, direct=True)
        k_cart = mp(2, 2, 2, b1, b2, b3, direct=False)

        np.testing.assert_allclose(
            k_cart,
            np.tensordot(k_direct, np.array([b1, b2, b3]), axes=(1, 0)),
            atol=1e-10,
        )

    def test_monkhorst_pack_sampling_from_kspacing(self):
        cell = cellpar_to_cell([4.22798145, 4.22798145, 4.22798145, 60, 60, 60])

        kpts = monkhorst_pack_sampling(cell=cell, kspac=0.03 * 1.889725989)

        assert kpts.shape == (33**3, 3)

    def test_build_kmeshgrid_uses_minimum_one(self):
        nmesh = build_kmeshgrid(
            np.array([0.1, 0, 0]),
            np.array([0, 0.1, 0]),
            np.array([0, 0, 0.1]),
            1.0,
        )

        assert nmesh == [1, 1, 1]

    def test_kmesh_sampling_switches_between_gamma_and_shifted_mp(self):
        mesh = [4, 4, 4]

        np.testing.assert_allclose(_sort_kpoints(kmesh_sampling(mesh, True)), _sort_kpoints(gamma_center(mesh)))
        np.testing.assert_allclose(_sort_kpoints(kmesh_sampling(mesh, False)), _sort_kpoints(monkhorst_pack(mesh)))

    def test_mesh_time_symmetry_reduce_normalizes_weights(self):
        cases = [
            ([1, 1, 1], True, 1),
            ([4, 4, 4], True, 36),
            ([4, 4, 4], False, 32),
            ([2, 4, 6], True, 28),
            ([2, 4, 6], False, 24),
        ]

        for mesh, is_gamma_center, expected_count in cases:
            kpts, weights = time_symmetry_reduce(mesh, is_gamma_center=is_gamma_center)
            assert len(kpts) == expected_count
            np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-10)
            assert set(np.round(weights * np.prod(mesh), 8)).issubset({1.0, 2.0})

    def test_kmesh_sampling_negf_respects_time_reversal_flag(self):
        reduced_kpts, reduced_weights = kmesh_sampling_negf([4, 4, 4], is_time_reversal=True)
        full_kpts, full_weights = kmesh_sampling_negf([4, 4, 4], is_time_reversal=False)

        assert len(reduced_kpts) == 36
        assert len(full_kpts) == 64
        np.testing.assert_allclose(reduced_weights.sum(), 1.0)
        np.testing.assert_allclose(full_weights.sum(), 1.0)

    def test_kmesh_fs_is_endpoint_inclusive(self):
        (lx, ly, lz), kgrids = kmesh_fs([3, 4, 5])

        assert len(kgrids) == 3 * 4 * 5
        np.testing.assert_allclose(lx, np.linspace(0, 1, 3))
        np.testing.assert_allclose(ly, np.linspace(0, 1, 4))
        np.testing.assert_allclose(lz, np.linspace(0, 1, 5))

    def test_kgrid_spacing_uses_requested_sampling(self, cubic_structure):
        kpts_gamma = kgrid_spacing(cubic_structure, 0.2, "Gamma")
        kpts_mp = kgrid_spacing(cubic_structure, 0.2, "MP")

        assert len(kpts_gamma) == len(kpts_mp)
        assert any(np.allclose(kpt, [0, 0, 0]) for kpt in kpts_gamma)
        assert not np.allclose(_sort_kpoints(kpts_gamma), _sort_kpoints(kpts_mp))


class TestKpointsPath:
    def test_abacus_kpath_preserves_endpoint_and_distances(self, cubic_structure):
        kpath = np.array([
            [0.0, 0.0, 0.0, 4],
            [0.5, 0.0, 0.0, 3],
            [0.5, 0.5, 0.0, 1],
        ])

        kpts, xvals, high_sym = abacus_kpath(cubic_structure, kpath)

        assert kpts.shape == (8, 3)
        np.testing.assert_allclose(kpts[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(kpts[-1], [0.5, 0.5, 0.0])
        assert len(xvals) == len(kpts)
        assert len(high_sym) == len(kpath)

    def test_ase_kpath_returns_labels(self, cubic_structure):
        kpts, xvals, high_sym, labels = ase_kpath(cubic_structure, "GX", 5)

        assert kpts.shape == (5, 3)
        assert len(xvals) == 5
        assert len(high_sym) == 2
        assert labels == ["G", "X"]

    def test_vasp_kpath_returns_segment_labels(self, cubic_structure):
        high_sym_points = {
            "G": np.array([0.0, 0.0, 0.0]),
            "X": np.array([0.5, 0.0, 0.0]),
            "M": np.array([0.5, 0.5, 0.0]),
        }

        kpts, xvals, tick_positions, labels = vasp_kpath(
            cubic_structure,
            ["G-X", "X-M"],
            high_sym_points,
            4,
        )

        assert kpts.shape == (8, 3)
        assert len(xvals) == 8
        assert len(tick_positions) == 3
        assert labels == ["G", "X", "M"]


class TestKpointsReduction:
    def test_reduce_time_inversion_preserves_total_degeneracy(self):
        kpts = mp(4, 4, 4, gamma_centered=True)

        reduced, degeneracies = reduce_time_inversion(kpts)

        assert len(reduced) == 36
        assert degeneracies.sum() == len(kpts)
        assert set(degeneracies).issubset({1.0, 2.0})

    def test_reduce_time_inversion_handles_boundary_equivalence(self):
        kpts = np.array([
            [-0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [-0.25, 0.0, 0.0],
        ])

        reduced, degeneracies = reduce_time_inversion(kpts)

        assert len(reduced) == 2
        np.testing.assert_allclose(np.sort(degeneracies), [2, 2])

    def test_reduce_rotation_accepts_precomputed_degeneracies(self):
        kpts = np.array([[0, 0, 0], [0.25, 0.25, 0.25]])
        symm_ops = [np.eye(3)]

        reduced, degeneracies = reduce_rotation(kpts, symm_ops, degeneracies_=[2, 3])

        assert len(reduced) == 2
        np.testing.assert_allclose(degeneracies, [2, 3])

    def test_reduce_combines_time_inversion_and_rotation(self, si_diamond):
        kpts = mp(4, 4, 4, gamma_centered=True)
        symm_ops = get_symm_ops(si_diamond)

        reduced, degeneracies = reduce(kpts, symm_ops, time_inversion_symmetry=True)

        assert len(reduced) == 8
        assert np.sum(degeneracies) == len(kpts)

    def test_hash_and_direct_reduction_match_for_core_implementation(self, si_diamond):
        kpts = mp(4, 4, 4, gamma_centered=True)
        symm_ops = get_symm_ops(si_diamond)
        weights = np.ones(len(kpts), dtype=float)

        direct_kpts, direct_weights = _reduce_by_symmetry_direct(kpts.copy(), weights.copy(), symm_ops, 1e-8)
        hash_kpts, hash_weights = _reduce_by_symmetry_hash(kpts.copy(), weights.copy(), symm_ops, 1e-8)

        np.testing.assert_allclose(_sort_kpoints(direct_kpts), _sort_kpoints(hash_kpts), atol=1e-10)
        np.testing.assert_allclose(np.sort(direct_weights), np.sort(hash_weights), atol=1e-10)


class TestKpointsSampling:
    def test_sample_full_symmetry_normalizes_weights(self, si_diamond):
        kpts, weights = sample(si_diamond, meshgrid=[4, 4, 4])

        assert kpts.shape == (8, 3)
        np.testing.assert_allclose(weights.sum(), 1.0)

    def test_sample_without_symmetry_returns_full_mesh(self, si_diamond):
        kpts, weights = sample(
            si_diamond,
            meshgrid=[4, 4, 4],
            rotational_symmetry=False,
            time_inversion_symmetry=False,
        )

        assert kpts.shape == (64, 3)
        np.testing.assert_allclose(weights.sum(), 1.0)

    def test_sample_meshspacing_path(self, si_diamond):
        kpts, weights = sample(si_diamond, meshspacing=[0.2, 0.2, 0.2])

        assert len(kpts) > 0
        np.testing.assert_allclose(weights.sum(), 1.0)
