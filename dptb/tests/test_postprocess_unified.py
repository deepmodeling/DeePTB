
import pytest
import os
import shutil
import torch
import numpy as np
from dptb.postprocess.unified.system import TBSystem
from dptb.postprocess.unified.properties.band import BandStructureData
from dptb.postprocess.unified.properties.dos import DosData
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

