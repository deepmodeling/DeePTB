
import pytest
import os
import numpy as np
import torch
from dptb.postprocess.unified.system import TBSystem
from dptb.postprocess.unified.utils import calculate_fermi_level
from dptb.data import AtomicDataDict

# Paths to example data (Same as test_postprocess_unified.py)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
MODEL_PATH = os.path.join(PROJECT_ROOT, "examples/ToW90_PythTB/models/nnsk.ep20.pth")
STRUCT_PATH = os.path.join(PROJECT_ROOT, "examples/ToW90_PythTB/silicon.vasp")

@pytest.fixture(scope="module")
def silicon_system():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(STRUCT_PATH):
        pytest.skip("Example data files not found.")
    
    print(f"Initializing TBSystem with Model: {MODEL_PATH} and Structure: {STRUCT_PATH}")
    tbsys = TBSystem(data=STRUCT_PATH, calculator=MODEL_PATH)
    return tbsys

def test_generic_fermi_utility():
    """Test the calculate_fermi_level utility with synthetic data."""
    # Scenario: 10 states, -5 eV to +4 eV energy.
    # Total electrons = 4. Spin degeneracy = 2. -> 2 lowest states fully occupied.
    # Eigs: [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
    # Ef should be between -4 and -3. mid point -3.5 eV.
    
    # Needs to be shape (1, 10) so it's 1 K-point with 10 bands.
    # Weights should be [1.0] for that 1 K-point.
    eigenvalues = np.array([[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]], dtype=float)
    total_electrons = 4.0
    spindeg = 2
    weights = np.array([1.0])
    
    ef = calculate_fermi_level(
        eigenvalues=eigenvalues, 
        total_electrons=total_electrons, 
        spindeg=spindeg, 
        weights=weights,
        temperature=0.01, # Low temp for sharp step
        smearing_method='FD'
    )
    
    print(f"Synthetic Ef: {ef}")
    assert -4.0 <= ef <= -3.0
    
    # Test with weights (half weight for first 5 states)
    weights = np.ones(10)
    weights[:5] = 0.5 # [-5..-1] have weight 0.5. Occupying them gives 2.5 electrons (spin=1) -> 5 electrons (spin=2)
    # Target 4 electrons.
    # occ of -5 (0.5), -4 (0.5), -3 (0.5), -2 (0.5). Sum = 2.0 (state weight) * 2 (spin) = 4.0 electrons.
    # So Ef should be just above -2.
    
    ef_weighted = calculate_fermi_level(
        eigenvalues=eigenvalues, 
        total_electrons=4.0, 
        spindeg=2,
        weights=weights,
        temperature=0.01,
        smearing_method='FD'
    )
    print(f"Weighted Synthetic Ef: {ef_weighted}")
    assert -2.0 <= ef_weighted <= -1.0

def test_system_get_total_electrons(silicon_system):
    """Test obtaining total electrons from system."""
    tbsys = silicon_system
    nel_atom = {'Si': 4}
    
    # Silicon primitive cell has 2 atoms. Total valence should be 8.
    tbsys.set_electrons(nel_atom)
    assert tbsys._total_electrons == 8.0
    
    # Test missing element error
    with pytest.raises(KeyError):
        tbsys.set_electrons({'H': 1})
