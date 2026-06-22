# ADR 0003: Use First-Brillouin-Zone Coordinates For K-Point Meshes

## Status

Accepted

## Context

K-point mesh generation, path construction, and symmetry reduction are used by band, DOS, SCC, transport, and export workflows. Historically, related helpers lived in `dptb.utils.make_kpoints` and `dptb.utils.ksampling`, with overlapping responsibilities and different assumptions about how fractional k-points should be represented.

Gamma-centered meshes previously exposed points in a `[0, 1)` fractional-coordinate convention in some utility paths. Symmetry reduction, especially time-reversal reduction, needs to compare `k` and `-k` while handling Brillouin-zone boundary points such as `0.5` and `-0.5`. Keeping multiple conventions in active implementation code makes boundary handling easy to get wrong.

## Decision

Core k-point implementation lives in `dptb.kpoints`, split by responsibility:

- `dptb.kpoints.mesh` for mesh generation;
- `dptb.kpoints.path` for high-symmetry paths;
- `dptb.kpoints.reduction` for symmetry and time-reversal reduction;
- `dptb.kpoints.sampling` for higher-level sampling.

Mesh k-points should be wrapped to the first-Brillouin-zone fractional-coordinate convention `[-0.5, 0.5)`. For example, a Gamma-centered `[4, 1, 1]` mesh is represented as:

```text
0.00, 0.25, -0.50, -0.25
```

rather than:

```text
0.00, 0.25, 0.50, 0.75
```

These are equivalent modulo reciprocal lattice vectors (`0.75 == -0.25`, `0.50 == -0.50`), but `[-0.5, 0.5)` is the canonical representation inside DeePTB k-point utilities.

`dptb.utils.make_kpoints` and `dptb.utils.ksampling` remain compatibility wrappers. New code should import from `dptb.kpoints` directly.

## Consequences

- Symmetry reduction has one canonical coordinate convention for comparing equivalent k-points.
- Time-reversal reduction can treat Brillouin-zone boundary points consistently.
- Physical calculations that consume k-points through periodic Hamiltonian evaluation should be unchanged by this representation choice.
- User code that assumes Gamma-centered mesh coordinates are non-negative or lie in `[0, 1)` may observe different raw k-point arrays and should wrap or compare k-points modulo reciprocal lattice vectors.
- Regression tests should cover both `dptb.kpoints` core functions and the compatibility wrappers under `dptb.utils`.
- AI-assisted refactors should not reintroduce duplicated mesh or reduction logic in downstream modules.
