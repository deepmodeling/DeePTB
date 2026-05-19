
import pytest
import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from dptb.utils.feast_wrapper import FeastSolver, _MKL_RT

@pytest.mark.skipif(_MKL_RT is None, reason="MKL runtime not found")
class TestFeastWrapper:
    
    def test_standard_hermitian(self):
        """Test finding eigenvalues in interval for standard Hermitian problem."""
        N = 100
        np.random.seed(42)
        # Create random Hermitian matrix
        A = np.random.rand(N, N) + 1j * np.random.rand(N, N)
        A = A + A.T.conj()
        # Make sparse
        A[np.abs(A) < 0.8] = 0
        Asp = sp.csr_matrix(A)
        
        # Reference solution (dense)
        evals_ref, evecs_ref = eigh(A)
        
        # Define interval covering middle 10 eigenvalues
        mid = N // 2
        emin = evals_ref[mid-5] - 0.1
        emax = evals_ref[mid+5] + 0.1
        expected_indices = np.where((evals_ref >= emin) & (evals_ref <= emax))[0]
        n_expected = len(expected_indices)
        
        solver = FeastSolver()
        # Initial guess explicitly smaller to test resizing logic if needed, 
        # but here we use conservative m0=20 > 11 so it should work first try.
        evals, X = solver.solve(Asp, emin=emin, emax=emax, m0=max(n_expected + 5, 20))
        
        assert len(evals) == n_expected
        np.testing.assert_allclose(np.sort(evals), evals_ref[expected_indices], atol=1e-8)
        
        # Check eigenvector residual ||Ax - \lambda x||
        for i in range(len(evals)):
            val = evals[i]
            vec = X[:, i]
            resid = Asp @ vec - val * vec
            assert np.linalg.norm(resid) < 1e-8

    def test_generalized_hermitian(self):
        """Test generalized problem Ax = \lambda Mx where M is positive definite."""
        N = 50
        np.random.seed(123)
        
        # A: Hermitian
        A = np.random.rand(N, N) + 1j * np.random.rand(N, N)
        A = A + A.T.conj()
        A[np.abs(A) < 0.5] = 0
        Asp = sp.csr_matrix(A)
        
        # M: Hermitian Positive Definite
        M = np.random.rand(N, N) + 1j * np.random.rand(N, N)
        M = M @ M.T.conj() + np.eye(N) # Ensure pos def
        M[np.abs(M) < 0.1] = 0
        Msp = sp.csr_matrix(M)
        
        # Dense reference: eigh(a, b) solves generalized
        evals_ref = eigh(A, M, eigvals_only=True)
        
        # Interval
        emin, emax = np.min(evals_ref) - 0.1, np.max(evals_ref) + 0.1
        
        solver = FeastSolver()
        evals, X = solver.solve(Asp, M=Msp, emin=emin, emax=emax)
        
        assert len(evals) == N
        np.testing.assert_allclose(np.sort(evals), np.sort(evals_ref), atol=1e-8)
        
        # Check generalized residual ||Ax - \lambda Mx||
        for i in range(len(evals)):
            val = evals[i]
            vec = X[:, i]
            resid = Asp @ vec - val * (Msp @ vec)
            assert np.linalg.norm(resid) < 1e-7

    def test_subspace_resize(self):
        """Test if wrapper correctly handles small initial m0 (info=3)."""
        N = 50
        np.random.seed(999)
        A = np.random.rand(N, N)
        A = A + A.T
        Asp = sp.csr_matrix(A)
        
        evals_ref = eigh(A, eigvals_only=True)
        emin, emax = -100, 100 # All eigenvalues
        
        solver = FeastSolver()
        # Start with ridiculously small m0=2
        # Expected eigenvalues = 50.
        # FEAST should return info=3 and wrapper should retry.
        
        # Capture stdout to see print message? Or just verify result.
        evals, X = solver.solve(Asp, emin=emin, emax=emax, m0=2, max_refinement=10)
        
        assert len(evals) == N
        np.testing.assert_allclose(np.sort(evals), np.sort(evals_ref), atol=1e-8)


    def test_real_symmetric(self):
        """Test finding eigenvalues for real symmetric problem."""
        N = 100
        np.random.seed(42)
        A = np.random.rand(N, N)
        A = A + A.T
        Asp = sp.csr_matrix(A)
        
        evals_ref = eigh(A, eigvals_only=True)
        emin, emax = evals_ref[0]-0.1, evals_ref[-1]+0.1
        
        solver = FeastSolver()
        evals, X = solver.solve(Asp, emin=emin, emax=emax, m0=max(N, 20))
        
        assert len(evals) == N
        np.testing.assert_allclose(np.sort(evals), np.sort(evals_ref), atol=1e-8)
        
        # Check eigenvector residual
        for i in range(len(evals)):
            val = evals[i]
            vec = X[:, i]
            resid = Asp @ vec - val * vec
            assert np.linalg.norm(resid) < 1e-8

    def test_generalized_real_symmetric(self):
        """Test generalized real symmetric Ax = lambda Mx."""
        N = 50
        np.random.seed(123)
        A = np.random.rand(N, N)
        A = A + A.T
        Asp = sp.csr_matrix(A)
        
        M = np.random.rand(N, N)
        M = M @ M.T + np.eye(N)
        Msp = sp.csr_matrix(M)
        
        evals_ref = eigh(A, M, eigvals_only=True)
        emin, emax = np.min(evals_ref)-0.1, np.max(evals_ref)+0.1
        
        solver = FeastSolver()
        evals, X = solver.solve(Asp, M=Msp, emin=emin, emax=emax, m0=N, max_refinement=10)
        
        assert len(evals) == N
        np.testing.assert_allclose(np.sort(evals), np.sort(evals_ref), atol=1e-8)
        

        for i in range(len(evals)):
            val = evals[i]
            vec = X[:, i]
            resid = Asp @ vec - val * (Msp @ vec)
            assert np.linalg.norm(resid) < 1e-7

    def test_lower_triangular(self):
        """Test with uplo='L' and automatic extraction."""
        N = 50
        np.random.seed(99)
        A = np.random.rand(N, N)
        A = A + A.T
        Asp = sp.csr_matrix(A)
        
        evals_ref = eigh(A, eigvals_only=True)
        emin, emax = np.min(evals_ref)-0.1, np.max(evals_ref)+0.1
        
        solver = FeastSolver()
        # Find ALL eigenvalues
        evals, X = solver.solve(Asp, emin=emin, emax=emax, m0=N, uplo='L', extract_triangular=True)
        
        assert len(evals) == N
        np.testing.assert_allclose(np.sort(evals), np.sort(evals_ref), atol=1e-8)

    def test_manual_triangular(self):
        """Test with pre-processed triangular matrix and extract_triangular=False."""
        N = 50
        np.random.seed(101)
        A = np.random.rand(N, N)
        A = A + A.T
        # Manually extract upper
        A_triu = sp.triu(A, format='csr')
        
        evals_ref = eigh(A, eigvals_only=True)
        emin, emax = np.min(evals_ref)-0.1, np.max(evals_ref)+0.1
        
        solver = FeastSolver()
        # Pass triangular matrix, disable extraction, set uplo='U'
        evals, X = solver.solve(A_triu, emin=emin, emax=emax, m0=N, uplo='U', extract_triangular=False)
        
        assert len(evals) == N
        np.testing.assert_allclose(np.sort(evals), np.sort(evals_ref), atol=1e-8)

if __name__ == "__main__":
    pytest.main([__file__])
