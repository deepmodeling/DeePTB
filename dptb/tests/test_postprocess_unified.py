
import pytest
import torch
import numpy as np
from dptb.postprocess.unified import TBSystem, HamiltonianCalculator
from dptb.postprocess.unified.properties.band import BandStructureData
from dptb.data import AtomicData, AtomicDataDict
from ase import Atoms

class MockCalculator:
    """Mock calculator for testing purposes."""
    device = torch.device('cpu')
    dtype = torch.float32
    
    def get_hamiltonian(self, data):
        return data # Pass through
        
    def get_eigenvalues(self, data):
        # Fake eigenvalues: random numbers
        # Extract nk from nested tensor safely
        k_nested = data[AtomicDataDict.KPOINT_KEY]
        # NestedTensor created from list of tensors can be unbound
        # Depending on torch version/implementation, unbind might needed or shape access
        # In the failing run, shape access failed. Unbind is safer for prototype nested tensors.
        try:
             k_tensor = k_nested.unbind()[0]
        except:
             k_tensor = k_nested # Fallback if not nested/already tensor
             
        nk = k_tensor.shape[0]
        nb = 10
        eigs = torch.rand((1, nk, nb))
        
        # Inject into data
        data[AtomicDataDict.ENERGY_EIGENVALUE_KEY] = eigs
        return data, eigs[0]
    
    def get_orbital_info(self):
        return {}

def test_unified_system_import():
    atoms = Atoms('Si2', positions=[[0,0,0], [1.3, 1.3, 1.3]], cell=[5,5,5], pbc=True)
    calc = MockCalculator()
    sys = TBSystem(atoms, model=calc)
    assert sys is not None

def test_unified_band_structure():
    atoms = Atoms('Si2', positions=[[0,0,0], [1.3, 1.3, 1.3]], cell=[5,5,5], pbc=True)
    calc = MockCalculator()
    sys = TBSystem(atoms, model=calc)
    
    # 1. Set K-path (ASE uses compact string usually)
    # Using 'GM' instead of 'G-M' as ASE parser expects compact or explicit
    sys.band.set_kpath(method='ase', pathstr='GM', total_nkpoints=20)
    
    # 2. Compute
    bs = sys.band.compute()
    
    assert isinstance(bs, BandStructureData)
    assert bs.eigenvalues.shape[0] == 20
    assert bs.eigenvalues.shape[1] == 10
    
    # 3. Test Plotting (dry run)
    sys.band.plot(filename='test_band.png', show=False)
    
    # Cleanup
    import os
    if os.path.exists('test_band.png'):
        os.remove('test_band.png')

if __name__ == "__main__":
    test_unified_system_import()
    test_unified_band_structure()
    print("All tests passed!")
