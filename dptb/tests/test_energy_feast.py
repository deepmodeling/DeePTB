
import pytest
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from dptb.nn.energy import FEASTEig
from dptb.utils.feast_wrapper import _MKL_RT

class MockMat:
    def __init__(self, mat):
        self.mat = mat
    def to_scipy(self, format="csr"):
        return self.mat

class MockContainer:
    def __init__(self, mat):
        self.obj = MockMat(mat)
    def sample_k(self, k, symm=True):
        return self.obj

@pytest.mark.skipif(_MKL_RT is None, reason="MKL runtime not found")
class TestFEASTEig:
    def test_solve_standard(self):
        N = 50
        np.random.seed(42)
        A = np.random.rand(N, N)
        A = A + A.T
        Asp = sp.csr_matrix(A)
        
        h_container = MockContainer(Asp)
        kpoints = np.array([[0,0,0]]) # Dummy kpoint
        
        evals_ref = eigh(A, eigvals_only=True)
        emin, emax = evals_ref[0]-0.1, evals_ref[-1]+0.1
        
        solver = FEASTEig(emin=emin, emax=emax, m0=N)
        evals_list = solver.solve(h_container, None, kpoints)
        
        assert len(evals_list) == 1
        evals = evals_list[0]
        assert len(evals) == N
        np.testing.assert_allclose(np.sort(evals), np.sort(evals_ref), atol=1e-8)

    def test_solve_generalized(self):
        N = 30
        np.random.seed(43)
        A = np.random.rand(N, N)
        A = A + A.T
        M = np.random.rand(N, N)
        M = M @ M.T + np.eye(N)
        
        Asp = sp.csr_matrix(A)
        Msp = sp.csr_matrix(M)
        
        h_container = MockContainer(Asp)
        s_container = MockContainer(Msp)
        kpoints = np.array([[0,0,0]])
        
        evals_ref = eigh(A, M, eigvals_only=True)
        emin, emax = np.min(evals_ref)-0.1, np.max(evals_ref)+0.1
        
        solver = FEASTEig(emin=emin, emax=emax, m0=N)
        evals_list = solver.solve(h_container, s_container, kpoints)
        
        assert len(evals_list) == 1
        evals = evals_list[0]
        assert len(evals) == N
        np.testing.assert_allclose(np.sort(evals), np.sort(evals_ref), atol=1e-8)

    def test_solve_with_vectors(self):
        """Test returning eigenvectors."""
        N = 40
        np.random.seed(44)
        A = np.random.rand(N, N)
        A = A + A.T
        Asp = sp.csr_matrix(A)
        
        h_container = MockContainer(Asp)
        kpoints = np.array([[0,0,0]])
        
        # Reference evals
        evals_ref = eigh(A, eigvals_only=True)
        emin, emax = evals_ref[0]-0.1, evals_ref[-1]+0.1
        
        solver = FEASTEig(emin=emin, emax=emax, m0=N)
        
        # Test return_eigenvectors=True
        evals_list, evecs_list = solver.solve(h_container, None, kpoints, return_eigenvectors=True)
        
        assert len(evals_list) == 1
        assert len(evecs_list) == 1
        
        evals = evals_list[0]
        evecs = evecs_list[0]
        
        assert len(evals) == N
        assert evecs.shape == (N, N) # Found all
        
        # Check residual for first few
        for i in range(min(5, N)):
            val = evals[i]
            vec = evecs[:, i]
            resid = Asp @ vec - val * vec
            assert np.linalg.norm(resid) < 1e-8

if __name__ == "__main__":
    pytest.main([__file__])
