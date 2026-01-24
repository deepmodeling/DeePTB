"""
Structure class for managing atomic structure information in TBSystem.

This module provides a unified interface for structure data that:
1. Encapsulates ASE Atoms with TB-specific metadata
2. Provides coordinate transformations (Cartesian ↔ Fractional)
3. Manages orbital basis information per atom
4. Supports export to various formats
5. Maintains compatibility with ASE ecosystem
"""

import numpy as np
import ase
from ase import Atoms
from typing import Dict, List, Optional, Tuple
import logging

log = logging.getLogger(__name__)


class Structure:
    """
    Encapsulates atomic structure with tight-binding specific metadata.

    This class serves as a bridge between ASE Atoms and TB calculations,
    providing:
    - Cached coordinate transformations
    - Orbital basis management
    - Export utilities for different formats

    Attributes
    ----------
    atoms : ase.Atoms
        The underlying ASE Atoms object
    basis : Dict[str, List[str]]
        Orbital basis per element type, e.g., {'Si': ['3s', '3p']}
    """

    def __init__(self, atoms: Atoms, basis: Dict[str, List[str]]):
        """
        Initialize Structure from ASE Atoms and basis information.

        Parameters
        ----------
        atoms : ase.Atoms
            Atomic structure
        basis : Dict[str, List[str]]
            Orbital basis per element, e.g., {'C': ['2s', '2p']}
        """
        self._atoms = atoms
        self._basis = basis

        # Caches for expensive computations
        self._fractional_positions_cache = None
        self._inv_cell_cache = None
        self._site_norbits_cache = None
        self._atom_orbs_cache = None

    @property
    def atoms(self) -> Atoms:
        """Return the ASE Atoms object."""
        return self._atoms

    @property
    def basis(self) -> Dict[str, List[str]]:
        """Return the orbital basis dictionary."""
        return self._basis

    # ========== Basic Structure Properties ==========

    @property
    def positions(self) -> np.ndarray:
        """Return Cartesian atomic positions (Å)."""
        return self._atoms.get_positions()

    @property
    def fractional_positions(self) -> np.ndarray:
        """
        Return fractional coordinates.

        Cached for performance.
        """
        if self._fractional_positions_cache is None:
            self._fractional_positions_cache = self.positions @ self.inv_cell
        return self._fractional_positions_cache

    @property
    def cell(self) -> np.ndarray:
        """Return unit cell vectors as 3x3 array (Å)."""
        return self._atoms.get_cell().array

    @property
    def inv_cell(self) -> np.ndarray:
        """
        Return inverse of cell matrix.

        Cached for performance. Used for Cartesian → Fractional conversion.
        """
        if self._inv_cell_cache is None:
            self._inv_cell_cache = np.linalg.inv(self.cell)
        return self._inv_cell_cache

    @property
    def atomic_numbers(self) -> np.ndarray:
        """Return atomic numbers as numpy array."""
        return self._atoms.get_atomic_numbers()

    @property
    def symbols(self) -> List[str]:
        """Return chemical symbols as list."""
        return self._atoms.get_chemical_symbols()

    @property
    def natoms(self) -> int:
        """Return number of atoms."""
        return len(self._atoms)

    @property
    def pbc(self) -> np.ndarray:
        """Return periodic boundary conditions."""
        return self._atoms.get_pbc()

    # ========== Orbital Basis Properties ==========

    def _count_orbitals_per_type(self, orbitals: List[str]) -> int:
        """
        Count total number of orbitals for a given orbital list.

        Parameters
        ----------
        orbitals : List[str]
            List of orbital labels, e.g., ['3s', '3p', '3d']

        Returns
        -------
        int
            Total number of orbitals (s=1, p=3, d=5, f=7, g=9)
        """
        l_map = {'s': 1, 'p': 3, 'd': 5, 'f': 7, 'g': 9}
        count = 0
        for orb in orbitals:
            # Extract orbital type (s, p, d, f, g)
            orb_type = orb[-1] if orb[-1] in l_map else orb.split('_')[0][-1]
            count += l_map.get(orb_type, 0)
        return count

    @property
    def site_norbits(self) -> np.ndarray:
        """
        Return number of orbitals per atom site.

        Returns
        -------
        np.ndarray
            Array of shape (natoms,) with orbital count per atom

        Example
        -------
        For a system with 2 C atoms (basis: 2s, 2p):
        >>> structure.site_norbits
        array([4, 4])  # Each C has 1(s) + 3(p) = 4 orbitals
        """
        if self._site_norbits_cache is None:
            site_norbits = []
            for symbol in self.symbols:
                if symbol not in self._basis:
                    raise ValueError(f"Element {symbol} not found in basis: {list(self._basis.keys())}")
                norb = self._count_orbitals_per_type(self._basis[symbol])
                site_norbits.append(norb)
            self._site_norbits_cache = np.array(site_norbits, dtype=int)
        return self._site_norbits_cache

    @property
    def norbits(self) -> int:
        """Return total number of orbitals in the system."""
        return int(np.sum(self.site_norbits))

    @property
    def atom_orbs(self) -> List[str]:
        """
        Return orbital labels for each orbital in the system.

        Returns
        -------
        List[str]
            Orbital labels in format "atom_index-element-orbital"

        Example
        -------
        For a system with 2 C atoms (basis: 2s, 2p):
        >>> structure.atom_orbs
        ['0-C-2s', '0-C-2p_x', '0-C-2p_y', '0-C-2p_z',
         '1-C-2s', '1-C-2p_x', '1-C-2p_y', '1-C-2p_z']
        """
        if self._atom_orbs_cache is None:
            atom_orbs = []
            for i, symbol in enumerate(self.symbols):
                for orb in self._basis[symbol]:
                    atom_orbs.append(f"{i}-{symbol}-{orb}")
            self._atom_orbs_cache = atom_orbs
        return self._atom_orbs_cache

    # ========== Export Utilities ==========

    def get_basis_compressed(self) -> Dict[str, str]:
        """
        Return basis in compressed format for export.

        Returns
        -------
        Dict[str, str]
            Basis in format {'C': '1s2p', 'Si': '1s1p1d'}

        Example
        -------
        >>> structure.get_basis_compressed()
        {'C': '1s2p', 'Si': '1s1p1d'}
        """
        basis_compressed = {}
        for elem, orbitals in self._basis.items():
            counts = {'s': 0, 'p': 0, 'd': 0, 'f': 0, 'g': 0}
            for orb in orbitals:
                # Extract orbital type
                for orb_type in "spdfg":
                    if orb_type in orb:
                        counts[orb_type] += 1
                        break

            # Build compressed string
            compressed = ""
            for orb_type in "spdfg":
                if counts[orb_type] > 0:
                    compressed += f"{counts[orb_type]}{orb_type}"

            basis_compressed[elem] = compressed

        return basis_compressed

    def to_dict(self) -> Dict:
        """
        Export structure to dictionary format (JSON-serializable).

        Returns
        -------
        Dict
            Dictionary containing all structure information, with numpy arrays
            converted to lists for JSON compatibility.
        """
        return {
            'cell': self.cell.tolist(),
            'positions': self.positions.tolist(),
            'atomic_numbers': self.atomic_numbers.tolist(),
            'symbols': self.symbols,
            'natoms': self.natoms,
            'site_norbits': self.site_norbits.tolist(),
            'norbits': self.norbits,
            'basis': self._basis,
            'pbc': self.pbc.tolist()
        }

    def to_json(self, filepath: str):
        """
        Export structure to JSON file.

        Parameters
        ----------
        filepath : str
            Path to output JSON file

        Example
        -------
        >>> structure.to_json("structure.json")
        """
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        log.info(f"Structure saved to {filepath}")

    @classmethod
    def from_json(cls, filepath: str) -> 'Structure':
        """
        Load structure from JSON file.

        Parameters
        ----------
        filepath : str
            Path to JSON file

        Returns
        -------
        Structure
            Loaded Structure instance
        """
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Reconstruct ASE Atoms
        atoms = Atoms(
            symbols=data['symbols'],
            positions=data['positions'],
            cell=data['cell'],
            pbc=data['pbc']
        )

        return cls(atoms, data['basis'])

    def invalidate_cache(self):
        """Invalidate all cached properties (call after modifying atoms)."""
        self._fractional_positions_cache = None
        self._inv_cell_cache = None
        self._site_norbits_cache = None
        self._atom_orbs_cache = None

    @classmethod
    def from_ase(cls, atoms: Atoms, basis: Dict[str, List[str]]) -> 'Structure':
        """
        Create Structure from ASE Atoms object.

        Parameters
        ----------
        atoms : ase.Atoms
            Atomic structure
        basis : Dict[str, List[str]]
            Orbital basis per element

        Returns
        -------
        Structure
            New Structure instance
        """
        return cls(atoms, basis)

    def __repr__(self) -> str:
        return (f"Structure(natoms={self.natoms}, norbits={self.norbits}, "
                f"formula={self._atoms.get_chemical_formula()})")

