
import pytest
import os
import shutil
import torch
import numpy as np
from dptb.postprocess.unified.system import TBSystem
from dptb.postprocess.unified.properties.band import BandStructureData
from dptb.postprocess.unified.properties.dos import DosData
from dptb.postprocess.unified.utils import calculate_fermi_level
from dptb.data import AtomicDataDict

# Paths to example data
# Using relative paths from the project root (where tests are usually run from)
# or absolute paths if needed.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
MODEL_PATH = os.path.join(PROJECT_ROOT, "examples/ToW90_PythTB/models/nnsk.ep20.pth")
STRUCT_PATH = os.path.join(PROJECT_ROOT, "examples/ToW90_PythTB/silicon.vasp")

@pytest.fixture(scope="module")
def silicon_system():
    """
    Fixture to initialize the TBSystem once for all tests in this module.
    This avoids reloading the model multiple times, which is expensive.
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(STRUCT_PATH):
        pytest.skip("Example data files (silicon.vasp/nnsk.ep20.pth) not found. Skipping integration tests.")
    
    print(f"Initializing TBSystem with Model: {MODEL_PATH} and Structure: {STRUCT_PATH}")
    tbsys = TBSystem(data=STRUCT_PATH, calculator=MODEL_PATH)
    return tbsys

def test_system_initialization(silicon_system):
    """Test TBSystem initialization and basic properties."""
    tbsys = silicon_system
    assert tbsys is not None
    assert tbsys.atoms is not None
    assert tbsys.calculator is not None
    
    # Check atom_orbs (should be populated for real model)
    assert hasattr(tbsys, 'atom_orbs')
    assert len(tbsys.atom_orbs) > 0
    print(f"Atom Orbitals: {tbsys.atom_orbs}")

def test_band_calculation(silicon_system):
    """Test Band Structure calculation."""
    tbsys = silicon_system
    
    # 2. Band Structure Calculation
    kpath_config = {
        "method": "abacus",
        "kpath": [
            [0.0, 0.0, 0.0, 10],  # G
            [0.5, 0.0, 0.5, 1],   # X
            # Simplified path for testing speed
        ],
        "klabels": ["G", "X"],
    }
    
    tbsys.band.set_kpath(**kpath_config)
    bs = tbsys.band.compute()
    
    assert isinstance(bs, BandStructureData)
    assert tbsys.has_bands is True
    assert bs.eigenvalues is not None
    assert len(bs.eigenvalues.shape) >= 2 
    
    # Test Plotting (dry run)
    plot_file = "test_real_band.png"
    tbsys.band.plot(show=False, filename=plot_file)
    if os.path.exists(plot_file):
        os.remove(plot_file)

def test_dos_calculation(silicon_system):
    """Test DOS and PDOS calculation."""
    tbsys = silicon_system
    
    # 3. DOS Calculation
    kmesh = [4, 4, 4] 
    tbsys.dos.set_kpoints(kmesh=kmesh)
    tbsys.dos.set_dos_config(erange=[-10, 10], npts=100, pdos=True)
    
    dos_res = tbsys.dos.compute()
    
    assert isinstance(dos_res, DosData)
    assert tbsys.has_dos is True
    assert dos_res.total_dos is not None
    assert len(dos_res.total_dos) == 100
    
    assert dos_res.pdos is not None
    assert dos_res.pdos.shape[0] == 100
    assert dos_res.pdos.shape[1] == len(tbsys.atom_orbs)
    
    plot_file = "test_real_dos.png"
    tbsys.dos.plot(show=False, plot_pdos=True, filename=plot_file)
    if os.path.exists(plot_file):
        os.remove(plot_file)

def test_calculate_fermi_level_ignores_invalid_padding():
    eigenvalues = np.array([[0.0, 1.0], [0.0, 1.0]])
    padded_eigenvalues = np.array([[0.0, 1.0, 1e4], [0.0, 1.0, 1e4]])
    valid_mask = np.array([[True, True, False], [True, True, False]])

    reference = calculate_fermi_level(
        eigenvalues=eigenvalues,
        total_electrons=2,
        spindeg=2,
        weights=np.array([0.5, 0.5]),
    )
    masked = calculate_fermi_level(
        eigenvalues=padded_eigenvalues,
        total_electrons=2,
        spindeg=2,
        weights=np.array([0.5, 0.5]),
        eigenvalue_valid_mask=valid_mask,
    )

    assert abs(reference - masked) < 1e-10

def test_get_efermi_accepts_solver_kwargs(silicon_system):
    silicon_system.set_electrons({"Si": 4})
    efermi = silicon_system.get_efermi(kmesh=[2, 2, 2], solver="torch")
    assert np.isfinite(efermi)

def test_get_efermi_accepts_eig_solver_alias(silicon_system):
    silicon_system.set_electrons({"Si": 4})
    efermi = silicon_system.get_efermi(kmesh=[2, 2, 2], eig_solver="torch", nk=2)
    assert np.isfinite(efermi)

def test_get_bands_forwards_nk_and_eig_solver_alias(silicon_system, monkeypatch):
    captured = {}

    def fake_get_eigenvalues(data, **kwargs):
        captured.update(kwargs)
        kpoints = data[AtomicDataDict.KPOINT_KEY]
        num_k = kpoints[0].shape[0] if kpoints.is_nested else kpoints.shape[0]
        num_orb = len(silicon_system.atom_orbs)
        eigs = torch.zeros((num_k, num_orb), dtype=silicon_system.calculator.dtype)
        return data, eigs

    monkeypatch.setattr(silicon_system.calculator, "get_eigenvalues", fake_get_eigenvalues)
    bands = silicon_system.get_bands(
        kpath_config={
            "method": "abacus",
            "kpath": [[0.0, 0.0, 0.0, 4], [0.5, 0.0, 0.5, 1]],
            "klabels": ["G", "X"],
        },
        reuse=False,
        eig_solver="torch",
        nk=2,
    )

    assert isinstance(bands.band_data, BandStructureData)
    assert captured["solver"] == "torch"
    assert captured["nk"] == 2

def test_get_dos_forwards_nk_and_eig_solver_alias(silicon_system, monkeypatch):
    captured = {}

    def fake_get_eigenvalues(data, **kwargs):
        captured.update(kwargs)
        kpoints = data[AtomicDataDict.KPOINT_KEY]
        num_k = kpoints[0].shape[0] if kpoints.is_nested else kpoints.shape[0]
        num_orb = len(silicon_system.atom_orbs)
        eigs = torch.zeros((num_k, num_orb), dtype=silicon_system.calculator.dtype)
        return data, eigs

    monkeypatch.setattr(silicon_system.calculator, "get_eigenvalues", fake_get_eigenvalues)
    dos = silicon_system.get_dos(
        kmesh=[1, 1, 1],
        erange=[-1, 1],
        npts=10,
        pdos=False,
        reuse=False,
        eig_solver="numpy",
        nk=1,
    )

    assert isinstance(dos.dos_data, DosData)
    assert captured["solver"] == "numpy"
    assert captured["nk"] == 1

def test_get_dos_recomputes_when_solver_changes(silicon_system, monkeypatch):
    calls = []

    def fake_get_eigenvalues(data, **kwargs):
        calls.append(dict(kwargs))
        kpoints = data[AtomicDataDict.KPOINT_KEY]
        num_k = kpoints[0].shape[0] if kpoints.is_nested else kpoints.shape[0]
        num_orb = len(silicon_system.atom_orbs)
        eigs = torch.zeros((num_k, num_orb), dtype=silicon_system.calculator.dtype)
        return data, eigs

    monkeypatch.setattr(silicon_system.calculator, "get_eigenvalues", fake_get_eigenvalues)

    dos_kwargs = {
        "kmesh": [1, 1, 1],
        "erange": [-1, 1],
        "npts": 10,
        "pdos": False,
    }
    silicon_system.get_dos(**dos_kwargs, solver="torch", nk=1, reuse=False)
    silicon_system.get_dos(**dos_kwargs, solver="numpy", nk=1)
    silicon_system.get_dos(**dos_kwargs, solver="numpy", nk=2)

    assert len(calls) == 2
    assert calls[0]["solver"] == "torch"
    assert calls[0]["nk"] == 1
    assert calls[1]["solver"] == "numpy"
    assert calls[1]["nk"] == 1

def test_get_bands_reuses_when_only_nk_changes(silicon_system, monkeypatch):
    calls = []

    def fake_get_eigenvalues(data, **kwargs):
        calls.append(dict(kwargs))
        kpoints = data[AtomicDataDict.KPOINT_KEY]
        num_k = kpoints[0].shape[0] if kpoints.is_nested else kpoints.shape[0]
        num_orb = len(silicon_system.atom_orbs)
        eigs = torch.zeros((num_k, num_orb), dtype=silicon_system.calculator.dtype)
        return data, eigs

    monkeypatch.setattr(silicon_system.calculator, "get_eigenvalues", fake_get_eigenvalues)
    kpath_config = {
        "method": "abacus",
        "kpath": [[0.0, 0.0, 0.0, 4], [0.5, 0.0, 0.5, 1]],
        "klabels": ["G", "X"],
    }

    silicon_system.get_bands(kpath_config=kpath_config, solver="torch", nk=1, reuse=False)
    silicon_system.get_bands(kpath_config=kpath_config, solver="torch", nk=2)

    assert len(calls) == 1
    assert calls[0]["solver"] == "torch"
    assert calls[0]["nk"] == 1

def test_get_bands_recomputes_when_kpath_changes(silicon_system, monkeypatch):
    calls = []

    def fake_get_eigenvalues(data, **kwargs):
        calls.append(dict(kwargs))
        kpoints = data[AtomicDataDict.KPOINT_KEY]
        num_k = kpoints[0].shape[0] if kpoints.is_nested else kpoints.shape[0]
        num_orb = len(silicon_system.atom_orbs)
        eigs = torch.zeros((num_k, num_orb), dtype=silicon_system.calculator.dtype)
        return data, eigs

    monkeypatch.setattr(silicon_system.calculator, "get_eigenvalues", fake_get_eigenvalues)
    first_kpath = {
        "method": "abacus",
        "kpath": [[0.0, 0.0, 0.0, 4], [0.5, 0.0, 0.5, 1]],
        "klabels": ["G", "X"],
    }
    second_kpath = {
        "method": "abacus",
        "kpath": [[0.0, 0.0, 0.0, 3], [0.5, 0.5, 0.0, 1]],
        "klabels": ["G", "M"],
    }

    silicon_system.get_bands(kpath_config=first_kpath, solver="torch", nk=1, reuse=False)
    silicon_system.get_bands(kpath_config=second_kpath, solver="torch", nk=2)

    assert len(calls) == 2

def test_get_dos_reuses_when_only_nk_changes(silicon_system, monkeypatch):
    calls = []

    def fake_get_eigenvalues(data, **kwargs):
        calls.append(dict(kwargs))
        kpoints = data[AtomicDataDict.KPOINT_KEY]
        num_k = kpoints[0].shape[0] if kpoints.is_nested else kpoints.shape[0]
        num_orb = len(silicon_system.atom_orbs)
        eigs = torch.zeros((num_k, num_orb), dtype=silicon_system.calculator.dtype)
        return data, eigs

    monkeypatch.setattr(silicon_system.calculator, "get_eigenvalues", fake_get_eigenvalues)
    dos_kwargs = {
        "kmesh": [1, 1, 1],
        "erange": [-1, 1],
        "npts": 10,
        "pdos": False,
    }

    silicon_system.get_dos(**dos_kwargs, solver="torch", nk=1, reuse=False)
    silicon_system.get_dos(**dos_kwargs, solver="torch", nk=2)

    assert len(calls) == 1
    assert calls[0]["solver"] == "torch"
    assert calls[0]["nk"] == 1

def test_get_dos_recomputes_when_kmesh_changes(silicon_system, monkeypatch):
    calls = []

    def fake_get_eigenvalues(data, **kwargs):
        calls.append(dict(kwargs))
        kpoints = data[AtomicDataDict.KPOINT_KEY]
        num_k = kpoints[0].shape[0] if kpoints.is_nested else kpoints.shape[0]
        num_orb = len(silicon_system.atom_orbs)
        eigs = torch.zeros((num_k, num_orb), dtype=silicon_system.calculator.dtype)
        return data, eigs

    monkeypatch.setattr(silicon_system.calculator, "get_eigenvalues", fake_get_eigenvalues)
    dos_kwargs = {
        "erange": [-1, 1],
        "npts": 10,
        "pdos": False,
    }

    silicon_system.get_dos(kmesh=[1, 1, 1], **dos_kwargs, solver="torch", nk=1, reuse=False)
    silicon_system.get_dos(kmesh=[2, 1, 1], **dos_kwargs, solver="torch", nk=2)

    assert len(calls) == 2

def test_get_dos_does_not_reuse_when_config_changes_without_kmesh(silicon_system, monkeypatch):
    calls = []

    def fake_get_eigenvalues(data, **kwargs):
        calls.append(dict(kwargs))
        kpoints = data[AtomicDataDict.KPOINT_KEY]
        num_k = kpoints[0].shape[0] if kpoints.is_nested else kpoints.shape[0]
        num_orb = len(silicon_system.atom_orbs)
        eigs = torch.zeros((num_k, num_orb), dtype=silicon_system.calculator.dtype)
        return data, eigs

    monkeypatch.setattr(silicon_system.calculator, "get_eigenvalues", fake_get_eigenvalues)

    silicon_system.get_dos(
        kmesh=[1, 1, 1],
        erange=[-1, 1],
        npts=10,
        pdos=False,
        solver="torch",
        reuse=False,
    )
    with pytest.raises(AssertionError, match="kmesh must be provided"):
        silicon_system.get_dos(
            erange=[-2, 2],
            npts=10,
            pdos=False,
            solver="torch",
        )

    assert len(calls) == 1

def test_solver_and_eig_solver_conflict_raises(silicon_system):
    with pytest.raises(ValueError, match="solver and eig_solver"):
        silicon_system.get_bands(solver="torch", eig_solver="numpy")

def test_get_hamiltonian_gethk(silicon_system):
    """Test getting Hamiltonian H(k) at specific k-points."""
    tbsys = silicon_system
    
    # Define a k-point (Gamma)
    k_points = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]
    
    # Call calculator.get_hk
    hk, sk = tbsys.calculator.get_hk(tbsys.data, k_points=k_points)
    
    assert hk is not None
    assert isinstance(hk, torch.Tensor)
    
    # Check shape: Should be related to number of k-points and number of orbitals
    # Since it might be nested, we check appropriately
    if hk.is_nested:
         hk_tensor = hk.to_padded_tensor(0)
    else:
         hk_tensor = hk
         
    # Expected shape: [batch, nk, norb, norb] or [nk, norb, norb]
    # For silicon with ep20 model, norb should be consistent with atom_orbs
    n_orb = len(tbsys.atom_orbs)
    
    # Check last two dimensions match orbital count
    assert hk_tensor.shape[-1] == n_orb
    assert hk_tensor.shape[-2] == n_orb
    
    # Should have results for 2 k-points
    # If returned as batched [1, Nk, Norb, Norb] or similar
    assert hk_tensor.shape[-3] == 2 or hk_tensor.shape[0] == 2 

    if sk is not None:
        assert sk.shape == hk.shape

def test_get_hopping_gethr(silicon_system):
    """Test getting Hopping terms H(R)."""
    tbsys = silicon_system
    
    # Call calculator.get_hr
    hr_blocks, sr_blocks = tbsys.calculator.get_hr(tbsys.data)
    
    assert hr_blocks is not None
    assert isinstance(hr_blocks, dict) or isinstance(hr_blocks, list) or isinstance(hr_blocks, torch.Tensor)
    
    # get_hr usually returns blocks in a specific format (feature_to_block output)
    # It returns a dictionary of blocks where keys are edge indices or similar
    
    # The exact verification depends on return type of feature_to_block
    # Based on calculator.py: Hblocks = feature_to_block(...)
    
    # Just asserting it returns something valid related to the system size
    if isinstance(hr_blocks, dict):
        assert len(hr_blocks) > 0
    elif isinstance(hr_blocks, torch.Tensor):
         assert hr_blocks.numel() > 0
