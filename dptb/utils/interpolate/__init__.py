"""
DFTB+-compatible interpolation methods for Slater-Koster integrals.

This module provides interpolation algorithms that match DFTB+ behavior:
- NevillePolyInterp: N-point polynomial interpolation using Neville's algorithm
- Poly5ToZero: 5th-order polynomial decay to zero with C² continuity
- DFTBPlusInterp1D: Complete DFTB+-compatible 1D interpolation
"""

from .poly_interp import NevillePolyInterp
from .poly5_decay import Poly5ToZero
from .dftbplus_interp import DFTBPlusInterp1D

__all__ = ['NevillePolyInterp', 'Poly5ToZero', 'DFTBPlusInterp1D']
