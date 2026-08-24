# Carbon BCC Self-Consistent Total-Energy K-Point Convergence Test Plan

## Status

In progress. The current scope is the convergence of the self-consistent total
energy of one standalone Carbon BCC calculation with respect to k-point
density. Inner SCC charge convergence is a required diagnostic but is not the
target convergence criterion. This is a validation plan, not an implementation
proposal.

## Motivation

DeePTB's SCC-DFTB functionality should give a stable, reproducible total energy
for a fixed Carbon BCC input as the k-point density is increased. An individual
mesh reaching its SCC charge tolerance only proves that mesh was solved
self-consistently; it does not prove Brillouin-zone integration or total-energy
convergence. This work is not a direct test against the existing multi-
structure DFTB+ pytest benchmark.

The downstream UniSK workflow exposed a practical concern. Its legacy Carbon
Bayesian-optimization script uses per-structure k-point spacings such as
`[0.05, 0.05, 0.05]` for BCC and FCC. For the Carbon BCC cell used by the
DeePTB DFTB+ benchmark, that spacing generates a `75 x 75 x 75` mesh
(`421875` raw points), whereas the existing DeePTB benchmark uses
`20 x 20 x 20` (`8000` raw points). The former was copied from an old
production-oriented workflow with the comment that it was chosen to "ensure
convergence"; no k-point convergence study was preserved with that script.

This difference must not be interpreted as proof that DeePTB SCC is slow or
incorrect. It is first a numerical-configuration question. A mesh that is
overly fine can make a correct SCC calculation impractical inside a Bayesian
optimization loop.

## Existing Assets

The repository contains a broader correctness regression:

- `dptb/tests/test_dftb_scc.py::test_dftbscc_matches_dftbp_benchmarks`
  compares DeePTB SCC electronic energies against DFTB+ reference data for
  Carbon BCC, graphene, and dimer EOS scans.
- The scan contains seven lattice/bond-length scales for each structure.
- DFTB+ reference tables and the corresponding structures/SK files live in
  `dptb/tests/data/dftb/structs_eos` and `dptb/tests/data/dftb`.
- The current acceptance threshold is a maximum electronic-energy difference
  below `1e-2 eV` at every scale.
- Its direct command is:

  ```bash
  python -m pytest -q dptb/tests/test_dftb_scc.py -k matches_dftbp_benchmarks
  ```

That regression is useful background and supplies the canonical Carbon
structure and SK data, but it is not the execution target for the present
study. The current standalone entry point is:

- `examples/dftb_scc/carbon_bcc_scc_convergence/run.py`;
- `examples/dftb_scc/carbon_bcc_scc_convergence/POSCAR`, containing only the
  scale-1.000 primitive Carbon BCC cell; and
- the canonical `C-C.skf` under `dptb/tests/data/dftb`, shared without invoking
  any pytest helper or multi-structure EOS scan.

The standalone example records self-consistent total, electronic, and repulsive
energies; energy changes between successive meshes and against all denser
meshes in the scan; k-point counts; SCC iteration count and charge residual
history; electron count; Mulliken charge; Fermi level; and wall time. Existing
unit tests for SCC state, mixers, API equivalence, and k-point symmetry
reduction remain useful but do not define this study's convergence criterion.

## Questions To Answer

The following effects are coupled in an SCC calculation but need separate
tests and diagnostics:

1. **K-point integration:** What mesh keeps the self-consistent total energy per
   atom within the agreed tolerance of every denser mesh tested for the fixed
   Carbon BCC cell?
2. **Occupations and smearing:** Are Fermi-Dirac and Gaussian paths correctly
   normalized, electron-number conserving, and stable in the relevant
   zero-/finite-temperature limits?
3. **Fermi-level solve:** Does the root solve use the same k-point weights and
   occupation convention as the band-energy and charge calculations?
4. **Symmetry reduction:** Do full, time-reversal-reduced, and
   rotational-symmetry-reduced meshes give equivalent observables with correct
   accumulated weights?
5. **Inner SCC loop:** Does every k mesh reach the charge-residual tolerance,
   and can SCC numerical noise contaminate the outer total-energy trend?
6. **Performance:** Which component dominates runtime after the numerical
   configuration is fixed: Hamiltonian construction, eigensolve, occupation,
   Mulliken population, Gamma/SCC shift, or mixing?

## Important Configuration Distinction

The current UniSK Carbon workflow and the DeePTB DFTB+ benchmark are not a
direct apples-to-apples timing or numerical comparison. Among other settings,
the legacy UniSK path uses Gaussian smearing (`"G"`) and a `0.05` spacing,
while the DFTB+ benchmark uses Fermi-Dirac smearing (`"FD"`) and fixed meshes
such as Carbon BCC `20 x 20 x 20`. Future comparisons must explicitly record
all physical and numerical settings, including SK files, repulsive parameters,
cutoffs, temperature, mesh, symmetry flags, mixer, tolerance, and maximum
iteration count.

## Phased Work

### Phase 0: Lock the standalone Carbon BCC baseline

- Use only the scale-1.000 Carbon BCC structure in the standalone example.
- Do not call `test_dftbscc_matches_dftbp_benchmarks` or scan graphene, dimer,
  or the seven-point Carbon EOS in this phase.
- Record wall time, SCC iteration count, full residual history, final residual,
  electron count, Fermi level, total/electronic/repulsive energies, SCC shift
  energy, and Mulliken charge for every numerical case.
- Keep the command runnable independently of notebooks and without generating
  or duplicating SK tables.

### Phase 1: K-point convergence matrix

For the fixed Carbon BCC cell, scan deliberately chosen fixed meshes and, where
useful, equivalent spacing-derived meshes. At minimum record:

- total-energy change per atom from the preceding mesh;
- maximum absolute total-energy change per atom relative to every denser mesh
  in the same run;
- an explicit k-point convergence flag evaluated against a documented energy
  threshold; the last and densest point cannot validate itself;
- charge and Fermi-level change relative to the finest accepted mesh;
- SCC iteration count and convergence failures;
- raw and symmetry-reduced k-point counts;
- wall time split by SCC iteration and total calculation.

Use the results to define separate **smoke**, **development**, and
**production** mesh presets. Do not retain `0.05` as a default solely because
it appeared in a legacy script.

The initial `20^3, 24^3, 28^3, 32^3, 36^3, 40^3` even-grid scan shows that
`20^3` differs from a denser result by up to `2.252 meV/atom`. It therefore does
not meet a `1 meV/atom` criterion. `24^3` and denser tested even grids remain
within `1 meV/atom`, but neighbouring odd grids and spacing-derived meshes must
still be checked before fixing a production preset.

### Phase 2: Occupation and Fermi-level regression tests

- Add small, deterministic tests for occupation normalization and exact
  electron-number conservation with non-uniform k-point weights.
- Test Fermi-Dirac and Gaussian implementations independently.
- Verify expected low-temperature and high-temperature limiting behaviour.
- First record Fermi level, band energy, SCC charge, and electron-count error
  for the standalone Carbon BCC input. Broader material-class coverage is a
  separate follow-up.

### Phase 3: Symmetry-reduction equivalence

- For the Carbon BCC cell, compare full meshes with time-reversal and
  rotational reductions.
- Assert equivalence of electron count, Fermi level, charge, total/electronic
  energy, SCC shift energy, and converged SCC shift within explicit tolerances.
- Assert that the sum of reduced k-point weights equals the full mesh weight.
- Report the reduction factor as a performance metric.

Symmetry reduction is a performance optimization, not a change to the target
physical result.

### Phase 4: SCC robustness and performance work

Only after Phases 0-3 identify the limiting configuration should solver work
begin. Candidate changes include static-data caching across SCC iterations,
batched k-point eigensolves, warm starts, and mixer/preconditioner tuning.
Every change must run the numerical regression matrix and a separate timing
benchmark.

The primitive Carbon BCC cell has one atom, so charge conservation and symmetry
largely pin its Mulliken charge. It can reveal occupation-noise effects in a
very tight SCC tolerance, but it is not a difficult charge-mixing problem. A
true mixer stress test will require a separate cell with non-equivalent charge
degrees of freedom or an explicitly perturbed initial charge; that must not be
silently substituted into this Carbon BCC baseline.

## Tetrahedron Integration

Tetrahedron integration may be useful for zero-temperature Brillouin-zone
integration, DOS, and difficult metallic cases, but it is not a drop-in
replacement for smearing. It requires a separate design covering band
interpolation, degeneracies, weights, self-consistency, and eventually force
consistency. It should be evaluated only after the existing Fermi-Dirac and
Gaussian paths are fully benchmarked and tested.

## Acceptance Criteria For Future Changes

A proposed SCC numerical or performance change is acceptable only when:

1. the standalone Carbon BCC total-energy matrix remains within its documented
   meV/atom tolerance against all denser meshes tested, while inner SCC charge
   residuals also meet their separate tolerance;
2. symmetry-reduced and full-mesh results agree within documented tolerances;
3. electron-number conservation is verified for every occupation path;
4. convergence failures and iteration counts do not regress without a recorded
   numerical reason; and
5. any claimed speedup reports the mesh, symmetry flags, precision, hardware,
   warm-up policy, and the same numerical acceptance criteria.

## Non-Goals

- Covering graphene, the Carbon dimer, or the seven-point Carbon EOS in the
  current standalone SCC convergence study.
- Using `test_dftbscc_matches_dftbp_benchmarks` as the current execution entry
  point.
- Changing default UniSK production parameters in this task.
- Declaring a legacy `kmeshspacing` value physically necessary without a
  convergence study.
- Replacing the SCC algorithm with a tetrahedron method before the current
  occupation and k-point paths have a complete regression suite.
