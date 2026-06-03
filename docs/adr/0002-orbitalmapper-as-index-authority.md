# ADR 0002: Use OrbitalMapper As The Index Authority

## Status

Accepted

## Context

Basis definitions, orbital ordering, bond types, and block indices appear across data processing, model construction, Hamiltonian assembly, training, and postprocess code. Duplicating index logic makes it easy for two paths to disagree while still passing superficial shape checks.

## Decision

`OrbitalMapper` is the authority for basis-to-index and orbital-block mapping in DeePTB. Callers should use it rather than reconstructing orbital ordering or block index rules locally.

Any change to `OrbitalMapper` behavior should be reviewed as a compatibility-sensitive change.

## Consequences

- Index mapping behavior has locality in one module.
- Mixed-basis and multi-species behavior should be regression-tested through `OrbitalMapper`.
- Downstream modules should not infer orbital ordering from ad hoc string or list manipulation.
- AI-assisted code should be rejected if it duplicates index mapping logic outside the established mapper path.

