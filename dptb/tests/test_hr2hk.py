"""
Test suite for HR2HK module with different gauge conventions.
Tests both Wannier90 Gauge (gauge=False) and Physical Gauge (gauge=True).
"""
import pytest
import torch
import numpy as np
import os
from dptb.nn.hr2hk import HR2HK
from dptb.postprocess.unified import TBSystem
from dptb.data import AtomicDataDict
from ase.io import read


@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    """Get the root directory of the project."""
    return str(request.config.rootdir)


@pytest.fixture(scope='module')
def tb_system(root_directory):
    """
    Create a TBSystem using the unified postprocess framework.
    This is the recommended way to initialize models and structures.
    """
    model_path = root_directory + "/dptb/tests/data/silicon_1nn/nnsk.ep500.pth"
    structure_path = root_directory + "/dptb/tests/data/silicon_1nn/silicon.vasp"
    
    if not os.path.exists(model_path):
        pytest.skip(f"Model file not found at {model_path}.")
    
    if not os.path.exists(structure_path):
        pytest.skip(f"Structure file not found at {structure_path}.")
    
    # Use TBSystem - the unified way to handle TB models
    system = TBSystem(
        data=structure_path,
        calculator=model_path,
        device=torch.device("cpu")
    )
    
    return system


class TestHR2HK:
    """Test suite for HR2HK module using TBSystem from unified postprocess."""
    
    def test_initialization_gauge_false(self, tb_system):
        """Test HR2HK initialization with gauge=False (Wannier90 convention)."""
        basis = tb_system.model.idp.basis
        
        hr2hk = HR2HK(
            basis=basis,
            gauge=False,
            dtype=torch.float32,
            device=torch.device("cpu")
        )
        
        assert hr2hk.gauge == False
        assert hr2hk.overlap == False
        assert hr2hk.dtype == torch.float32
        
    def test_initialization_gauge_true(self, tb_system):
        """Test HR2HK initialization with gauge=True (Physical/Periodic convention)."""
        basis = tb_system.model.idp.basis
        
        hr2hk = HR2HK(
            basis=basis,
            gauge=True,
            dtype=torch.float32,
            device=torch.device("cpu")
        )
        
        assert hr2hk.gauge == True
        assert hr2hk.overlap == False
        
    def test_forward_gauge_false(self, tb_system):
        """Test forward pass with gauge=False using real model."""
        # Get data and model from TBSystem
        data_dict = tb_system.data.copy()
        basis = tb_system.model.idp.basis
        data_dict = tb_system.model(data_dict)
        # Add k-points
        kpoints = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float32)
        data_dict[AtomicDataDict.KPOINT_KEY] = kpoints
        # Run model to get edge/node features
        # Now test HR2HK
        hr2hk = HR2HK(basis=basis, gauge=False, dtype=torch.float32, device=torch.device("cpu"))
        output = hr2hk(data_dict)
        
        # Check output
        assert AtomicDataDict.HAMILTONIAN_KEY in output
        hk = output[AtomicDataDict.HAMILTONIAN_KEY]
        
        assert hk.shape[0] == kpoints.shape[0]
        assert hk.shape[1] == hk.shape[2]  # Square matrix
        
        # Check Hermiticity
        assert torch.allclose(hk, hk.transpose(1, 2).conj(), atol=1e-5)
        
    def test_forward_gauge_true(self, tb_system):
        """Test forward pass with gauge=True using real model."""
        # Get data and model from TBSystem
        data_dict = tb_system.data.copy()
        basis = tb_system.model.idp.basis
        
        # Add k-points
        kpoints = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]], dtype=torch.float32)
        data_dict[AtomicDataDict.KPOINT_KEY] = kpoints
        
        # Run model to get edge/node features
        data_dict = tb_system.model(data_dict)
        
        # Now test HR2HK
        hr2hk = HR2HK(basis=basis, gauge=True, dtype=torch.float32, device=torch.device("cpu"))
        output = hr2hk(data_dict)
        
        # Check output
        assert AtomicDataDict.HAMILTONIAN_KEY in output
        hk = output[AtomicDataDict.HAMILTONIAN_KEY]
        
        assert hk.shape[0] == kpoints.shape[0]
        assert hk.shape[1] == hk.shape[2]  # Square matrix
        
        # Check Hermiticity
        assert torch.allclose(hk, hk.transpose(1, 2).conj(), atol=1e-5)
        
        # Check that edge_vectors were computed
        assert AtomicDataDict.EDGE_VECTORS_KEY in output
        
    def test_gauge_eigenvalues_consistency(self, tb_system):
        """
        CRITICAL TEST: Verify that eigenvalues (band energies) are identical for both gauges.
        This is the most important test - eigenvalues are physical observables
        and must be gauge-invariant.
        """
        basis = tb_system.model.idp.basis
        
        # Test multiple k-points including Gamma and non-Gamma
        kpoints = torch.tensor([
            [0.0, 0.0, 0.0],   # Gamma - both gauges must be exactly same
            [0.25, 0.0, 0.0],  # Non-Gamma - test gauge transformation
            [0.5, 0.5, 0.0],   # M point
            [0.3, 0.3, 0.3],   # General point
        ], dtype=torch.float32)
        
        # Prepare two independent data copies
        data_dict_w90 = tb_system.data.copy()
        data_dict_w90[AtomicDataDict.KPOINT_KEY] = kpoints
        data_dict_w90 = tb_system.model(data_dict_w90)
        
        data_dict_phys = tb_system.data.copy()
        data_dict_phys[AtomicDataDict.KPOINT_KEY] = kpoints
        data_dict_phys = tb_system.model(data_dict_phys)
        
        # Create HR2HK with different gauges
        hr2hk_w90 = HR2HK(basis=basis, gauge=False, dtype=torch.float32)
        hr2hk_phys = HR2HK(basis=basis, gauge=True, dtype=torch.float32)
        
        # Transform to k-space
        output_w90 = hr2hk_w90(data_dict_w90)
        output_phys = hr2hk_phys(data_dict_phys)
        
        hk_w90 = output_w90[AtomicDataDict.HAMILTONIAN_KEY]
        hk_phys = output_phys[AtomicDataDict.HAMILTONIAN_KEY]
        
        print(f"\n{'='*70}")
        print(f"Testing gauge invariance of eigenvalues (band energies)")
        print(f"{'='*70}")
        
        # Compute eigenvalues for all k-points
        n_kpoints = hk_w90.shape[0]
        for ik in range(n_kpoints):
            # Compute eigenvalues using torch.linalg.eigvalsh for Hermitian matrices
            eigvals_w90 = torch.linalg.eigvalsh(hk_w90[ik])
            eigvals_phys = torch.linalg.eigvalsh(hk_phys[ik])
            
            # Sort eigenvalues (they may come in different order)
            eigvals_w90_sorted = torch.sort(eigvals_w90.real)[0]
            eigvals_phys_sorted = torch.sort(eigvals_phys.real)[0]
            
            max_diff = torch.max(torch.abs(eigvals_w90_sorted - eigvals_phys_sorted)).item()
            
            # CRITICAL: Eigenvalues must be identical regardless of gauge
            assert torch.allclose(eigvals_w90_sorted, eigvals_phys_sorted, atol=1e-4, rtol=1e-3), \
                f"Eigenvalues differ at k-point {ik}: {kpoints[ik].numpy()}!\n" \
                f"Max difference: {max_diff:.2e} eV\n" \
                f"This indicates a bug in gauge transformation!"
            
    def test_overlap_eigenvalues_gauge_consistency(self, tb_system):
        """Test that overlap matrix eigenvalues are gauge-invariant."""
        # Skip if model doesn't have overlap
        if not hasattr(tb_system.model, 'overlap') or tb_system.model.overlap is None:
            pytest.skip("Model does not have overlap matrix")
        
        basis = tb_system.model.idp.basis
        kpoints = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float32)
        
        # Prepare data for overlap
        data_dict_w90 = tb_system.data.copy()
        data_dict_w90[AtomicDataDict.KPOINT_KEY] = kpoints
        if hasattr(tb_system.model, 'overlap') and tb_system.model.overlap:
            data_dict_w90 = tb_system.model.overlap(data_dict_w90)
        
        data_dict_phys = tb_system.data.copy()
        data_dict_phys[AtomicDataDict.KPOINT_KEY] = kpoints
        if hasattr(tb_system.model, 'overlap') and tb_system.model.overlap:
            data_dict_phys = tb_system.model.overlap(data_dict_phys)
        
        s2k_w90 = HR2HK(basis=basis, overlap=True, gauge=False, dtype=torch.float32)
        s2k_phys = HR2HK(basis=basis, overlap=True, gauge=True, dtype=torch.float32)
        
        sk_w90 = s2k_w90(data_dict_w90)[AtomicDataDict.HAMILTONIAN_KEY]
        sk_phys = s2k_phys(data_dict_phys)[AtomicDataDict.HAMILTONIAN_KEY]
        
        # Check eigenvalues at all k-points
        for ik in range(sk_w90.shape[0]):
            eigvals_w90 = torch.sort(torch.linalg.eigvalsh(sk_w90[ik]))[0]
            eigvals_phys = torch.sort(torch.linalg.eigvalsh(sk_phys[ik]))[0]
            
            assert torch.allclose(eigvals_w90, eigvals_phys, atol=1e-4, rtol=1e-3), \
                f"Overlap eigenvalues differ at k-point {ik}!"
                
    def test_derivative_enforces_gauge(self, tb_system):
        """Test that derivative=True enforces gauge=True."""
        basis = tb_system.model.idp.basis
        
        hr2hk = HR2HK(
            basis=basis,
            derivative=True,
            gauge=False,  # Try to set gauge=False
            dtype=torch.float32
        )
        
        # derivative=True should force gauge=True
        assert hr2hk.gauge == True, "derivative=True should enforce gauge=True"
        assert hr2hk.derivative == True
        
    def test_complex_dtype(self, tb_system):
        """Test that complex dtype is properly handled."""
        data_dict = tb_system.data.copy()
        basis = tb_system.model.idp.basis
        
        kpoints = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        data_dict[AtomicDataDict.KPOINT_KEY] = kpoints
        data_dict = tb_system.model(data_dict)
        
        hr2hk = HR2HK(basis=basis, gauge=True, dtype=torch.float32, device=torch.device("cpu"))
        
        output = hr2hk(data_dict)
        hk = output[AtomicDataDict.HAMILTONIAN_KEY]
        
        # Output should be complex
        assert hk.dtype in [torch.complex64, torch.complex128], \
            "Hamiltonian in k-space should be complex"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
