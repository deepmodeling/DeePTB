"""
Proposed architecture for Pardiso integration.

This document outlines a better design for integrating high-performance
eigensolvers (like Pardiso) into DeePTB's postprocessing workflow.

## Current Issues

1. **Tight coupling**: Julia script is standalone, requires manual invocation
2. **Data redundancy**: Full H(R), S(R) exported to files
3. **Fragmented workflow**: Python → files → Julia → files → Python
4. **Limited reusability**: Only works for band calculations

## Proposed Architecture

### Phase 1: Modular Julia Backend (Current PR)

```
dptb/postprocess/julia/
├── io/
│   ├── load_structure.jl    # Read structure.json
│   └── load_hamiltonian.jl  # Read H5 files
├── solvers/
│   ├── pardiso_solver.jl    # Pardiso eigenvalue solver
│   └── solver_utils.jl      # Common utilities
├── tasks/
│   ├── band_calculation.jl  # Band structure task
│   └── dos_calculation.jl   # DOS task
└── main.jl                  # Entry point (calls tasks)
```

**Benefits**:
- Modular: Easy to add new tasks (DOS, optical, etc.)
- Testable: Each module can be unit tested
- Maintainable: Clear separation of concerns

### Phase 2: Python Solver Abstraction

```python
# dptb/postprocess/unified/solvers/base.py
from abc import ABC, abstractmethod

class EigenSolver(ABC):
    \"\"\"Abstract base class for eigenvalue solvers.\"\"\"

    @abstractmethod
    def solve_at_k(self, H_k: np.ndarray, S_k: np.ndarray,
                   num_bands: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        \"\"\"
        Solve generalized eigenvalue problem H|ψ⟩ = E S|ψ⟩ at single k-point.

        Returns
        -------
        eigenvalues : np.ndarray (num_bands,)
        eigenvectors : np.ndarray (norb, num_bands)
        \"\"\"
        pass

    @abstractmethod
    def solve_batch(self, kpoints: np.ndarray, H_R: dict, S_R: dict,
                    num_bands: int, **kwargs) -> np.ndarray:
        \"\"\"
        Solve for multiple k-points efficiently.

        Returns
        -------
        eigenvalues : np.ndarray (nk, num_bands)
        \"\"\"
        pass


# dptb/postprocess/unified/solvers/numpy_solver.py
class NumpySolver(EigenSolver):
    \"\"\"Default numpy/scipy solver (current implementation).\"\"\"

    def solve_at_k(self, H_k, S_k, num_bands, **kwargs):
        if S_k is None:
            evals, evecs = np.linalg.eigh(H_k)
        else:
            evals, evecs = scipy.linalg.eigh(H_k, S_k)
        return evals[:num_bands], evecs[:, :num_bands]

    def solve_batch(self, kpoints, H_R, S_R, num_bands, **kwargs):
        eigenvalues = []
        for kpt in kpoints:
            H_k = self._construct_hk(kpt, H_R)
            S_k = self._construct_hk(kpt, S_R) if S_R else None
            evals, _ = self.solve_at_k(H_k, S_k, num_bands)
            eigenvalues.append(evals)
        return np.array(eigenvalues)


# dptb/postprocess/unified/solvers/pardiso_solver.py
class PardisoSolver(EigenSolver):
    \"\"\"Julia/Pardiso backend solver for large systems.\"\"\"

    def __init__(self, julia_script_path=None, use_pyjulia=False):
        self.use_pyjulia = use_pyjulia
        if use_pyjulia:
            self._init_pyjulia()
        else:
            self.julia_script = julia_script_path or self._default_script_path()

    def solve_batch(self, kpoints, H_R, S_R, num_bands, **kwargs):
        if self.use_pyjulia:
            # Direct Julia call (no file I/O)
            return self._solve_via_pyjulia(kpoints, H_R, S_R, num_bands)
        else:
            # File-based workflow (current approach)
            return self._solve_via_files(kpoints, H_R, S_R, num_bands)

    def _solve_via_files(self, kpoints, H_R, S_R, num_bands):
        # Export data
        temp_dir = self._export_data(H_R, S_R, kpoints)
        # Call Julia script
        result = subprocess.run(['julia', self.julia_script, temp_dir])
        # Load results
        eigenvalues = np.load(os.path.join(temp_dir, 'eigenvalues.npy'))
        return eigenvalues


# Usage in BandAccessor
class BandAccessor:
    def __init__(self, tbsystem, solver='numpy'):
        self.tbsystem = tbsystem
        self.solver = get_solver(solver)  # Factory function

    def compute(self, **kwargs):
        hr, sr = self.tbsystem.calculator.get_hr(self.tbsystem.data)
        eigenvalues = self.solver.solve_batch(
            self.kpoints, hr, sr, self.num_bands
        )
        self.eigenvalues = eigenvalues
        return self
```

**Benefits**:
- Unified interface: All properties (band, DOS, optical) use same solver
- Pluggable: Easy to add new solvers (cuSOLVER, ELPA, etc.)
- Backward compatible: Default numpy solver maintains current behavior

### Phase 3: Backend System (Future)

```python
# User-facing API
import dptb
dptb.set_backend('pardiso')  # Global setting

tbsys = TBSystem(data, calculator)
bands = tbsys.band.compute(kpath)  # Automatically uses Pardiso

# Or per-calculation
bands = tbsys.band.compute(kpath, solver='pardiso')
```

## Implementation Roadmap

### Immediate (Current PR)
1. ✅ Optimize data format (JSON instead of text files)
2. ✅ Use OrbitalMapper.atom_norb (avoid recomputation)
3. 🔲 Modularize Julia script
4. 🔲 Add Julia unit tests

### Short-term (Next PR)
1. Create `dptb/postprocess/unified/solvers/` module
2. Implement `EigenSolver` ABC
3. Refactor current code into `NumpySolver`
4. Implement `PardisoSolver` (file-based)
5. Integrate into `BandAccessor`

### Medium-term
1. Add PyJulia support (optional dependency)
2. Implement `PardisoSolver._solve_via_pyjulia()`
3. Add solver benchmarks
4. Documentation and examples

### Long-term
1. Add more solvers (cuSOLVER for GPU, ELPA for HPC)
2. Auto-selection based on system size
3. Distributed computing support

## Design Principles

1. **Separation of concerns**: I/O, solving, post-processing are separate
2. **Pluggable architecture**: Easy to add new solvers
3. **Backward compatibility**: Default behavior unchanged
4. **Performance**: Minimize data copying and file I/O
5. **Testability**: Each component can be unit tested
6. **Documentation**: Clear examples and API docs

## Example Usage

```python
# Current workflow (after Phase 1)
tbsys = TBSystem(data, calculator)
tbsys.export_for_julia('pardiso_input')
# Run Julia script manually
# Load results manually

# After Phase 2
tbsys = TBSystem(data, calculator)
bands = tbsys.band.compute(kpath, solver='pardiso')
bands.plot()

# After Phase 3
dptb.set_backend('pardiso')
tbsys = TBSystem(data, calculator)
bands = tbsys.band.compute(kpath)  # Automatically fast!
```
"""
