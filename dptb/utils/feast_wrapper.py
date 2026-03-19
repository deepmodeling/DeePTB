import ctypes
import os
import sys
import glob
import site
import numpy as np
import scipy.sparse as sp
from ctypes.util import find_library
import warnings

# Use same MKL finding logic as pardiso_wrapper for consistency
def _find_mkl_rt():
    """Find and load mkl_rt shared library."""
    libmkl = None
    mkl_rt = os.environ.get('PYPARDISO_MKL_RT')
    if mkl_rt is None:
        mkl_rt = find_library('mkl_rt')
    if mkl_rt is None:
        mkl_rt = find_library('mkl_rt.1')
    
    if mkl_rt is None:
        globs = glob.glob(f'{sys.prefix}/[Ll]ib*/**/*mkl_rt*', recursive=True) or \
                glob.glob(f'{site.USER_BASE}/[Ll]ib*/**/*mkl_rt*', recursive=True)
        for path in sorted(globs, key=len):
            try:
                libmkl = ctypes.CDLL(path)
                break
            except (OSError, ImportError):
                pass
    else:
        try:
            libmkl = ctypes.CDLL(mkl_rt)
        except (OSError, ImportError):
            pass
    return libmkl

_MKL_RT = _find_mkl_rt()

if _MKL_RT:
    # Define function signatures for FEAST
    
    # void feastinit(int* fpm);
    _MKL_RT.feastinit.argtypes = [ctypes.POINTER(ctypes.c_int)]
    _MKL_RT.feastinit.restype = None

    # void zfeast_hcsrev(
    #   const char* uplo, const int* n, const void* a, const int* ia, const int* ja,
    #   const int* fpm, double* epsout, int* loop, const double* emin, const double* emax,
    #   int* m0, double* E, void* X, int* M, double* res, int* info
    # );
    _MKL_RT.zfeast_hcsrev.argtypes = [
        ctypes.POINTER(ctypes.c_char),  # uplo
        ctypes.POINTER(ctypes.c_int),   # n
        ctypes.c_void_p,                # a (complex128*)
        ctypes.POINTER(ctypes.c_int),   # ia
        ctypes.POINTER(ctypes.c_int),   # ja
        ctypes.POINTER(ctypes.c_int),   # fpm
        ctypes.POINTER(ctypes.c_double),# epsout
        ctypes.POINTER(ctypes.c_int),   # loop
        ctypes.POINTER(ctypes.c_double),# emin
        ctypes.POINTER(ctypes.c_double),# emax
        ctypes.POINTER(ctypes.c_int),   # m0
        ctypes.c_void_p,                # E (double*)
        ctypes.c_void_p,                # X (complex128*)
        ctypes.POINTER(ctypes.c_int),   # M (found eigs)
        ctypes.POINTER(ctypes.c_double),# res
        ctypes.POINTER(ctypes.c_int)    # info
    ]
    _MKL_RT.zfeast_hcsrev.restype = None

    # void zfeast_hcsrgv(...) see below

    # void dfeast_scsrev(
    #   const char* uplo, const int* n, const double* a, const int* ia, const int* ja,
    #   const int* fpm, double* epsout, int* loop, const double* emin, const double* emax,
    #   int* m0, double* E, double* X, int* M, double* res, int* info
    # );
    _MKL_RT.dfeast_scsrev.argtypes = [
        ctypes.POINTER(ctypes.c_char),  # uplo
        ctypes.POINTER(ctypes.c_int),   # n
        ctypes.c_void_p,                # a (double*)
        ctypes.POINTER(ctypes.c_int),   # ia
        ctypes.POINTER(ctypes.c_int),   # ja
        ctypes.POINTER(ctypes.c_int),   # fpm
        ctypes.POINTER(ctypes.c_double),# epsout
        ctypes.POINTER(ctypes.c_int),   # loop
        ctypes.POINTER(ctypes.c_double),# emin
        ctypes.POINTER(ctypes.c_double),# emax
        ctypes.POINTER(ctypes.c_int),   # m0
        ctypes.c_void_p,                # E (double*)
        ctypes.c_void_p,                # X (double*)
        ctypes.POINTER(ctypes.c_int),   # M (found eigs)
        ctypes.POINTER(ctypes.c_double),# res
        ctypes.POINTER(ctypes.c_int)    # info
    ]
    _MKL_RT.dfeast_scsrev.restype = None

    # void dfeast_scsrgv(
    #   const char* uplo, const int* n, const double* a, const int* ia, const int* ja,
    #   const double* b, const int* ib, const int* jb,
    #   const int* fpm, double* epsout, int* loop, const double* emin, const double* emax,
    #   int* m0, double* E, double* X, int* M, double* res, int* info
    # );
    _MKL_RT.dfeast_scsrgv.argtypes = [
        ctypes.POINTER(ctypes.c_char),  # uplo
        ctypes.POINTER(ctypes.c_int),   # n
        ctypes.c_void_p,                # a (double*)
        ctypes.POINTER(ctypes.c_int),   # ia
        ctypes.POINTER(ctypes.c_int),   # ja
        ctypes.c_void_p,                # b (double*)
        ctypes.POINTER(ctypes.c_int),   # ib
        ctypes.POINTER(ctypes.c_int),   # jb
        ctypes.POINTER(ctypes.c_int),   # fpm
        ctypes.POINTER(ctypes.c_double),# epsout
        ctypes.POINTER(ctypes.c_int),   # loop
        ctypes.POINTER(ctypes.c_double),# emin
        ctypes.POINTER(ctypes.c_double),# emax
        ctypes.POINTER(ctypes.c_int),   # m0
        ctypes.c_void_p,                # E (double*)
        ctypes.c_void_p,                # X (double*)
        ctypes.POINTER(ctypes.c_int),   # M (found eigs)
        ctypes.POINTER(ctypes.c_double),# res
        ctypes.POINTER(ctypes.c_int)    # info
    ]
    _MKL_RT.dfeast_scsrgv.restype = None

    # void zfeast_hcsrgv(
    #   const char* uplo, const int* n, const void* a, const int* ia, const int* ja,
    #   const void* b, const int* ib, const int* jb,
    #   const int* fpm, double* epsout, int* loop, const double* emin, const double* emax,
    #   int* m0, double* E, void* X, int* M, double* res, int* info
    # );
    _MKL_RT.zfeast_hcsrgv.argtypes = [
        ctypes.POINTER(ctypes.c_char),  # uplo
        ctypes.POINTER(ctypes.c_int),   # n
        ctypes.c_void_p,                # a (complex128*)
        ctypes.POINTER(ctypes.c_int),   # ia
        ctypes.POINTER(ctypes.c_int),   # ja
        ctypes.c_void_p,                # b (complex128*)
        ctypes.POINTER(ctypes.c_int),   # ib
        ctypes.POINTER(ctypes.c_int),   # jb
        ctypes.POINTER(ctypes.c_int),   # fpm
        ctypes.POINTER(ctypes.c_double),# epsout
        ctypes.POINTER(ctypes.c_int),   # loop
        ctypes.POINTER(ctypes.c_double),# emin
        ctypes.POINTER(ctypes.c_double),# emax
        ctypes.POINTER(ctypes.c_int),   # m0
        ctypes.c_void_p,                # E (double*)
        ctypes.c_void_p,                # X (complex128*)
        ctypes.POINTER(ctypes.c_int),   # M (found eigs)
        ctypes.POINTER(ctypes.c_double),# res
        ctypes.POINTER(ctypes.c_int)    # info
    ]
    _MKL_RT.zfeast_hcsrgv.restype = None

class FeastSolver:
    """
    Wrapper for MKL FEAST solver (zfeast_hcsrev) for complex Hermitian matrices.
    Finds all eigenvalues in a given interval [emin, emax].
    """
    
    def __init__(self):
        if _MKL_RT is None:
            raise ImportError("MKL runtime library (mkl_rt) not found. Cannot use FEAST.")
        
        # Initialize default FPM
        self.fpm = np.zeros(128, dtype=np.int32)
        _MKL_RT.feastinit(self.fpm.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        
        # Standard defaults suitable for contour integration
        # fpm[0] = 1 (Enable logging/print) - Turn off by default
        self.fpm[0] = 0 
        
    def _prepare_matrix(self, mat, dtype, uplo_char, extract_triangular, name="Matrix"):
        """Prepare matrix: ensure CSR, check dtype, extract triangular part."""
        # Check dtype
        if mat.dtype != dtype:
            warnings.warn(f"Converting {name} to {dtype} for FEAST")
            mat = mat.astype(dtype)
            
        # Ensure CSR
        if not sp.isspmatrix_csr(mat):
            mat = mat.tocsr()
            
        if extract_triangular:
            if uplo_char == b'U':
                return sp.triu(mat, format='csr')
            elif uplo_char == b'L':
                return sp.tril(mat, format='csr')
            else:
                 raise ValueError(f"Invalid uplo: {uplo_char}")
        else:
            return mat

    def solve(self, A, M=None, emin=-1.0, emax=1.0, m0=None, max_refinement=3, uplo='U', extract_triangular=True):
        """
        Solve eigenvalue problem Ax = \lambda x (or Ax = \lambda Mx) for \lambda in [emin, emax].
        
        Args:
            A: Scipy sparse CSR matrix (Complex Hermitian / Real Symmetric)
            M: Scipy sparse CSR matrix (Hermitian Positive Definite), optional.
            emin, emax: Eigenvalue interval.
            m0: Initial subspace size. If None, defaults to 10 or 1.5x expected if passed.
            max_refinement: Number of retries if subspace is too small (info=3).
            uplo: Upper ('U') or Lower ('L') triangle to use. Default 'U'.
            extract_triangular: If True (default), automatically extracts the specified triangular part 
                                using sp.triu/sp.tril. Set False if input is already triangular.
        
        Returns:
            evals: Array of found eigenvalues.
            X: Array of eigenvectors (column-wise).
        """
        if not sp.isspmatrix_csr(A):
            A = A.tocsr()
            
        N = A.shape[0]
        if A.shape[1] != N:
            raise ValueError("Matrix A must be square")
            
        # Check dtype
        # Prepare pointers
        if isinstance(uplo, str):
            uplo_char = uplo.upper().encode('ascii')
        elif isinstance(uplo, bytes):
            uplo_char = uplo.upper()
        else:
            uplo_char = b'U'
        
        uplo_c = ctypes.create_string_buffer(uplo_char) 

        # Detect Complexity
        is_complex = np.iscomplexobj(A) or (M is not None and np.iscomplexobj(M))
        
        if is_complex:
            # Complex Hermitian
            dtype = np.complex128
            fn_std = _MKL_RT.zfeast_hcsrev
            fn_gen = _MKL_RT.zfeast_hcsrgv
        else:
            # Real Symmetric
            dtype = np.float64
            fn_std = _MKL_RT.dfeast_scsrev
            fn_gen = _MKL_RT.dfeast_scsrgv
            
        # Prepare A
        A_triu = self._prepare_matrix(A, dtype, uplo_char, extract_triangular, name="A")
        
        # Prepare M (if generalized)
        if M is not None:
             M_triu = self._prepare_matrix(M, dtype, uplo_char, extract_triangular, name="M")

        if m0 is None:
            # If finding ALL eigenvalues (emin very small, emax very large), m0 should span the space.
            # But usually we find interval. Conservative estimate: 1.5x expected or somewhat large number.
            # For robustness, start with larger default if N small.
            m0 = min(N, max(N // 2, 20)) 
            
        # Generic preparation
        ia = A_triu.indptr.astype(np.int32) + 1
        ja = A_triu.indices.astype(np.int32) + 1
        a_data = A_triu.data

            
        loop = ctypes.c_int(0)
        epsout = ctypes.c_double(0.0)
        emin_c = ctypes.c_double(emin)
        emax_c = ctypes.c_double(emax)
        
        # Retry loop for m0 refinement
        for attempt in range(max_refinement + 1):
            
            # Prepare output arrays
            E = np.zeros(m0, dtype=np.float64)
            # MKL expects column-major (Fortran) storage for dense matrices.
            X = np.zeros((N, m0), dtype=dtype, order='F')
            res = np.zeros(m0, dtype=np.float64)
            info = ctypes.c_int(0)
            M_found = ctypes.c_int(0)
            m0_c = ctypes.c_int(m0)
    
            if M is None:
                # Standard problem
                fn_std(
                    uplo_c,
                    ctypes.byref(ctypes.c_int(N)),
                    a_data.ctypes.data_as(ctypes.c_void_p),
                    ia.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    ja.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    self.fpm.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    ctypes.byref(epsout),
                    ctypes.byref(loop),
                    ctypes.byref(emin_c),
                    ctypes.byref(emax_c),
                    ctypes.byref(m0_c),
                    E.ctypes.data_as(ctypes.c_void_p),
                    X.ctypes.data_as(ctypes.c_void_p),
                    ctypes.byref(M_found),
                    res.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    ctypes.byref(info)
                )
            else:
                 # Generalized problem
                 m_data = M_triu.data
                 ib = M_triu.indptr.astype(np.int32) + 1
                 jb = M_triu.indices.astype(np.int32) + 1
                 
                 fn_gen(
                    uplo_c,
                    ctypes.byref(ctypes.c_int(N)),
                    a_data.ctypes.data_as(ctypes.c_void_p),
                    ia.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    ja.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    m_data.ctypes.data_as(ctypes.c_void_p),
                    ib.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    jb.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    self.fpm.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    ctypes.byref(epsout),
                    ctypes.byref(loop),
                    ctypes.byref(emin_c),
                    ctypes.byref(emax_c),
                    ctypes.byref(m0_c),
                    E.ctypes.data_as(ctypes.c_void_p),
                    X.ctypes.data_as(ctypes.c_void_p),
                    ctypes.byref(M_found),
                    res.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    ctypes.byref(info)
                )
            
            if info.value == 0:
                # Success
                n_eig = M_found.value
                return E[:n_eig], X[:, :n_eig]
                
            elif info.value == 3:
                # Warning: Size of the subspace M0 is too small
                # MKL usually suggests a new size in m0_c
                new_m0 = m0_c.value
                if new_m0 <= m0: # If suggestion is not larger, force doubling
                     new_m0 = m0 * 2
                
                if new_m0 > N:
                    new_m0 = N
                if m0 == N:
                     raise RuntimeError(f"FEAST info=3: Subspace too small even at N={N}.")
                
                print(f"FEAST info=3 (Subspace too small/bad estimate). MKL suggested {m0_c.value}. Increasing m0 from {m0} to {new_m0} and retrying.")
                m0 = new_m0
                continue
                
            else:
                # Other errors
                raise RuntimeError(f"FEAST failed with info={info.value}. Check MKL documentation.")
                
        raise RuntimeError(f"FEAST failed to converge after {max_refinement} refinements of m0.")
