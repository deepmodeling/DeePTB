# DeePTB Context

This file records the domain language and maintenance assumptions used when reviewing DeePTB changes. It is written for maintainers, contributors, and AI-assisted review tools.

## Project Shape

DeePTB is a scientific machine learning package for tight-binding electronic structure simulation. Its core workflow is:

```text
AtomicData
  -> OrbitalMapper
  -> embedding
  -> prediction
  -> Hamiltonian / overlap transform
  -> eigenvalues, matrices, or exported electronic-structure artifacts
```

Correctness depends on tensor semantics, orbital indexing, physical invariants, and backward compatibility with existing configs, checkpoints, and datasets.

## Core Concepts

**AtomicData** represents structures and graph data used by DeePTB models.

**AtomicDataDict** is the shared dictionary interface between data, model, training, and postprocess code. Key names and tensor meanings are part of the public internal contract. Do not rename, repurpose, or silently reshape keys without tests and migration notes.

**OrbitalMapper** is the authority for translating basis definitions into orbital, bond, and block indices. Code that depends on orbital ordering should go through OrbitalMapper rather than rebuilding index logic locally.

**Reduced matrix elements** are direction-independent matrix elements predicted or parameterized before Hamiltonian expansion. Models should keep this representation until the Hamiltonian transform stage.

**Hamiltonian transform** expands reduced matrix elements into full orbital blocks. DeePTB has SK and E3 paths; changes here are high risk because small ordering, sign, dtype, or shape changes can alter physical outputs.

**Overlap** is optional but must remain consistent with the Hamiltonian path when enabled. Tests should cover overlap behavior when a change touches shared matrix construction logic.

## Model Types

Model type is determined from `model_options`.

**NNENV** uses embedding plus prediction layers to produce Hamiltonian terms. It supports SKTB and E3TB methods.

**NNSK** learns Slater-Koster parameters with analytic functional forms.

**DFTBSK** uses fixed Slater-Koster parameters from a DFTB database.

**MIX** combines a base SK model with neural corrections. Its multiplicative correction behavior is part of the model contract.

## Review Priorities

When reviewing a PR, focus first on:

- AtomicDataDict key meaning and tensor shape changes.
- OrbitalMapper ordering and basis mapping changes.
- Reduced matrix element conventions.
- SK/E3 Hamiltonian expansion behavior.
- Config schema, defaults, docs, and examples staying aligned.
- Checkpoint and dataset backward compatibility.
- Tests that cover changed scientific behavior, not just import success.

