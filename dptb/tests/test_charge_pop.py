import pytest
import os
from pathlib import Path
import torch
import numpy as np
from dptb.data import AtomicData
from ase.io import read
from dptb.utils.constants import Harte2eV
from dptb.nn.dftbsk import DFTBSK
from dptb.postprocess.charge_pop import (
    Mulliken,
    bincount_sum,
    direct_diag_rhos,
    _bincount_sum_numpy,
    _direct_diag_rhos_numpy,
    NUMBA_AVAILABLE,
)


rootdir = os.path.join(Path(os.path.abspath(__file__)).parent, "data")


def test_get_mulcharge(rootdir = rootdir):
    

    sk_path = os.path.join(rootdir, 'dftb')
    struct = os.path.join(rootdir, 'hBN/hBN.vasp')
    basis = {'B':['2s','2p'],"N":["2s","2p"]}   
    AtomicData_options = {"r_max": {'B': 6.349479778742587, 'N': 5.366822193937187}}
    nel_atom = {'B': 3, 'N': 5}  # Valence electrons for B and N
    kmeshgrid = [20, 20, 1]  # k-mesh grid


    model = DFTBSK(basis=basis, skdata=sk_path,overlap=True)

    mulliken = Mulliken(model=model,
                        device='cpu',
                        eig_method='eigh')

    mulliken.get_mulcharge(data = struct,
                            kmeshgrid = kmeshgrid,
                            kgamma_center = True,
                            krotational_symmetry = False,
                            ktime_inversion_symmetry = True,
                            smearing_method = 'Fermi-Dirac',
                            nel_atom = nel_atom,
                            AtomicData_options = AtomicData_options)
    
    mul_charge_ref = np.array([5.3166588, 2.6833414])
    mul_delta_charge_ref = np.array([0.3166588, -0.3166588])
    mul_per_atom_norbs_ref = [4, 4]
    mul_per_atom_charge_ref = [5, 3]
    mul_per_atom_indices_ref = np.array([0, 4, 8])

    assert np.allclose(mulliken.mul_charge, mul_charge_ref, atol=1e-5)
    assert np.allclose(mulliken.delta_charge, mul_delta_charge_ref, atol=1e-5)
    assert mulliken.per_atom_norbs == mul_per_atom_norbs_ref
    assert mulliken.per_atom_charge == mul_per_atom_charge_ref
    assert np.array_equal(mulliken.per_atom_indices, mul_per_atom_indices_ref)


def test_cal_mul_charge_orthogonal_limit(rootdir=rootdir):
    sk_path = os.path.join(rootdir, 'dftb')
    basis = {'B': ['2s', '2p'], "N": ["2s", "2p"]}
    model = DFTBSK(basis=basis, skdata=sk_path, overlap=False)
    mulliken = Mulliken(model=model, device='cpu', eig_method='eigh', overlap=False)
    eigenvectors = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            [[0.6, 0.8], [1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.complex128,
    )
    occ = np.array([[2.0, 0.5], [1.5, 0.25]], dtype=np.float64)
    wk = np.array([0.4, 0.6], dtype=np.float64)
    per_atom_indices = np.array([0, 2, 3])

    mul_charge = mulliken.cal_mul_charge(
        per_atom_norbs=[2, 1],
        per_atom_indices=per_atom_indices,
        eigenvectors=eigenvectors,
        overlap_np=None,
        occ=occ,
        wk=wk,
    )

    diag_vals = np.real(np.sum(np.conj(eigenvectors) * eigenvectors * occ[:, None, :], axis=2))
    expected_trace = np.stack(
        [diag_vals[:, 0] + diag_vals[:, 1], diag_vals[:, 2]],
        axis=1,
    )
    assert np.allclose(mul_charge, wk @ expected_trace)


def test_direct_diag_rhos():
    """Test direct_diag_rhos computes correct diagonal values of Rho_S."""
    np.random.seed(42)
    nk, norb, nstate = 100, 8, 8

    # Generate random complex eigenvectors (orthonormal-like)
    V_real = np.random.randn(nk, norb, nstate)
    V_imag = np.random.randn(nk, norb, nstate)
    V = (V_real + 1j * V_imag).astype(np.complex128)

    # Generate random overlap @ V
    SV_real = np.random.randn(nk, norb, nstate)
    SV_imag = np.random.randn(nk, norb, nstate)
    SV = (SV_real + 1j * SV_imag).astype(np.complex128)

    # Generate random Fermi-Dirac occupations
    fermi_prop = np.random.rand(nk, nstate).astype(np.float64)

    # Test NumPy implementation
    diag_numpy = _direct_diag_rhos_numpy(V, SV, fermi_prop)
    assert diag_numpy.shape == (nk, norb)
    assert diag_numpy.dtype == np.float64

    # Test dispatcher (uses Numba if available)
    diag_dispatch = direct_diag_rhos(V, SV, fermi_prop)
    assert diag_dispatch.shape == (nk, norb)

    # Both implementations should give identical results
    assert np.allclose(diag_numpy, diag_dispatch, atol=1e-12)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
def test_direct_diag_rhos_numba_consistency():
    """Test Numba and NumPy implementations produce identical results."""
    from dptb.postprocess.charge_pop import _direct_diag_rhos_numba

    np.random.seed(123)
    nk, norb, nstate = 500, 16, 16

    V = (np.random.randn(nk, norb, nstate) + 1j * np.random.randn(nk, norb, nstate)).astype(np.complex128)
    SV = (np.random.randn(nk, norb, nstate) + 1j * np.random.randn(nk, norb, nstate)).astype(np.complex128)
    fermi_prop = np.random.rand(nk, nstate).astype(np.float64)

    # Ensure contiguous arrays for Numba
    V_c = np.ascontiguousarray(V, dtype=np.complex128)
    SV_c = np.ascontiguousarray(SV, dtype=np.complex128)
    fermi_c = np.ascontiguousarray(fermi_prop, dtype=np.float64)

    diag_numpy = _direct_diag_rhos_numpy(V, SV, fermi_prop)
    diag_numba = _direct_diag_rhos_numba(V_c, SV_c, fermi_c)

    assert np.allclose(diag_numpy, diag_numba, atol=1e-12)


def test_bincount_sum():
    """Test bincount_sum correctly sums diagonal values per atom."""
    np.random.seed(42)
    nk, norb, natoms = 100, 8, 2

    diag_vals = np.random.randn(nk, norb).astype(np.float64)
    orb2atom = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)  # 4 orbs per atom

    # Test NumPy implementation
    trace_numpy = _bincount_sum_numpy(diag_vals, orb2atom, natoms)
    assert trace_numpy.shape == (nk, natoms)

    # Test dispatcher
    trace_dispatch = bincount_sum(diag_vals, orb2atom, natoms)
    assert trace_dispatch.shape == (nk, natoms)

    # Both should give identical results
    assert np.allclose(trace_numpy, trace_dispatch, atol=1e-12)

    # Verify correctness: manual sum for first k-point
    expected_atom0 = np.sum(diag_vals[0, :4])
    expected_atom1 = np.sum(diag_vals[0, 4:])
    assert np.isclose(trace_numpy[0, 0], expected_atom0)
    assert np.isclose(trace_numpy[0, 1], expected_atom1)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
def test_bincount_sum_numba_consistency():
    """Test Numba and NumPy bincount_sum produce identical results."""
    from dptb.postprocess.charge_pop import _bincount_sum_numba

    np.random.seed(456)
    nk, norb, natoms = 1000, 32, 8

    diag_vals = np.random.randn(nk, norb).astype(np.float64)
    # Distribute orbitals unevenly across atoms
    orb2atom = np.repeat(np.arange(natoms), [3, 4, 5, 4, 3, 5, 4, 4]).astype(np.int64)

    diag_c = np.ascontiguousarray(diag_vals, dtype=np.float64)
    orb2atom_c = np.ascontiguousarray(orb2atom, dtype=np.int64)

    trace_numpy = _bincount_sum_numpy(diag_vals, orb2atom, natoms)
    trace_numba = _bincount_sum_numba(diag_c, orb2atom_c, natoms)

    assert np.allclose(trace_numpy, trace_numba, atol=1e-12)
