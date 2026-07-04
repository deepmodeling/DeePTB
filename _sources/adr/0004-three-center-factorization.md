# ADR 0004: Two-Center-Factorized Three-Center Hamiltonian Terms

## Status

Accepted

## Context

The fixed (non-SCF) LCAO Hamiltonian contains two-center and three-center contributions. The
kinetic term is two-center; the neutral-atom/local potential and nonlocal-pseudopotential parts
generate three-center contributions when the potential/projector center `C` differs from the
orbital centers `A` and `B`. DeePTB's existing two-body SK path parameterizes only two-center
reduced matrix elements and expands them with a CG basis + `wigner_D` rotation
(`dptb/nn/hamiltonian.py`). Message-passing embeddings (`Slem`/`Lem`/`Trinity`) capture many-body
effects, but they mix `(Σ_C P_AC)(Σ_{C'} P_{BC'})` through node features and do not represent a
genuine, shared-center three-center object.

We want an explicit three-center term that (a) stays inside DeePTB's "parameterize reduced values,
never integrate" philosophy (ADR 0001) and (b) reuses the existing CG/rotation machinery and
`OrbitalMapper` indexing (ADR 0002).

## Decision

Represent the three-center block by a **two-center factorization**:

    H^(3)_{AB} = Σ_{C ≠ A,B}  P_{AC} · D_C · P_{CB},   P_{CB} = P_{BC}^†

where `P_{AC}^{μα} = <φ_{μA}|β_{Cα}>` is a two-center orbital↔projector "integral value" and `D_C`
is a small center-local, species-`C` coupling matrix. Consistent with ADR 0001, **the network emits
the reduced (direction-independent) values of `P_{AC}` and the matrix `D_C` directly** — there are
no orbitals on `C` and nothing is integrated. `P_{AC}` is placed into the local A–C frame by the
same SK selection matrices and rotated to the global frame with `wigner_D`; `D_C` is block-diagonal
in `l` (identity in `m`), hence rotation invariant.

The term is **additive** in block form, exactly like Trinity's existing two-body SK contribution
(`deeptb.py`: `EDGE_FEATURES += EDGE_ATTRS`): `H^0_{AB} = H^(2)_{AB} + H^(3)_{AB}`.

Two cutoffs are used, matching the physics and DeePTB's existing neighbour lists: the target blocks
`H^(3)_{AB}` are produced **only for the basis-cutoff bonds** (`EDGE_INDEX`, the user's `r_max`), so
they occupy exactly the same block set as `<A|B>`; the projector reach `A-C`/`C-B` uses the separate,
user-tunable **environment cutoff** (`ENV_INDEX` at `er_max`, built by
`AtomicData.from_points(er_max=...)`). `er_max` is a knob alongside the projector basis size and
`l_max`.

Implementation lives in `dptb/nn/threecenter.py` (`ThreeCenterFactorized`, `CenterCouplingMatrix`);
it consumes the graph already in the `AtomicDataDict` (edge list, edge vectors) and uses
`OrbitalMapper` for all orbital and **projector** indexing (the projector set is simply another
`OrbitalMapper` basis), reusing `wigner_D`, `BesselBasis`, `ScalarMLPFunction`, `polynomial_cutoff`
and `scatter` rather than re-implementing them.

Both the **hopping** correction `H^(3)_{AB}` (basis-bond edge blocks) and the **onsite** correction
`H^(3)_{AA} = sum_{C != A} P_{AC} D_C P_{CA}` (node blocks) are assembled; the onsite term is the
`A==B` "triangle", i.e. each env edge `C->A` on its own, scattered to atom `A`.

It is a **Trinity-only** feature, wired **entirely inside `dptb/nn/embedding/trinity.py`** (plus its
config schema in `dptb/utils/argcheck.py`); `dptb/nn/deeptb.py` and the rest of the pipeline are
untouched. An optional `three_center` block **inside the trinity embedding config** builds the module
in `Trinity.__init__`, and `Trinity.forward` assembles the correction and adds it to the reduced
`EDGE_FEATURES`/`NODE_FEATURES`. Because Trinity emits *reduced-equivariant* features (the block
transform happens later in `NNENV`), the correction is added via `to_reduced` — the CG decomposition
that is the exact inverse of `E3Hamiltonian(decompose=False)` — so the existing `NNENV` transform then
turns (many-body + three-center) into blocks together, with the two-body `EDGE_ATTRS` added on top as
before. The config lives under the trinity variant, so `three_center` on any other embedding is an
argcheck error.

A single trinity `mode` (Trinity-only; replaces the old `only2b`/`exclusive`) selects the channels on
top of the always-present two-body base: `"2b"` = 2b only; `"3b"`/`"2b+3b"` = add the three-body term;
`"full"` = add three-body + env (message passing). Following the original convention the 2b base is
always produced by NNENV, so `deeptb.py` is untouched; the 3b/2b+3b/full modes require a `three_center`
block, and the projector reach needs the environment neighbour list, so `er_max` must be set. The
`three_center` module (hence its parameters) is built whenever the block is configured, *independent of*
`mode`, so the `state_dict` is identical across modes and one can warm-start progressively.

Trainability is controlled by a separate, orthogonal `freeze` list (any subset of `["2b","3b","env"]`)
that sets `requires_grad=False` on the two-body term, the three-center term, and/or the message-passing
pathway respectively. This makes progressive training explicit — e.g. train `2b`, then `freeze:["2b"]`
with `mode:"2b+3b"` to train only the three-center term, then `freeze:["2b","3b"]` with `mode:"full"` to
train only env. If `freeze` is unset the historical default is kept: `"full"` freezes `["2b","3b"]`
(train only env) and every other mode freezes nothing; pass `freeze:[]` to train everything.

## Consequences

- Heterogeneous species are handled in `OrbitalMapper.full_basis` (the union of all species' shells)
  with `mask_to_basis` zeroing absent channels, so tensors stay uniform (batched GEMM unchanged) and
  blocks come out in full_basis order like `E3Hamiltonian`. Single shared layouts are the special
  case where the masks are all-ones.
- Hermiticity is structural (`P_{CB}=P_{BC}^†`, `D_C=D_C^†`), so upper/lower blocks stay consistent.
- Rotation covariance follows from `wigner_D` on `P` plus rotation-invariant `D_C`.
- Auxiliary rank (`N_l` projectors per `l`) is the accuracy/cost knob; the factorization rank of a
  triangle block is bounded by the projector count.
- Assembly is fully vectorized for GPU: `P_{XC}` is built for all edges at once (CG selection +
  `wigner_D`), the triplet index (all `(A,C,B)` sharing center `C`) is built with
  `argsort`/`cumsum`/`repeat_interleave` with **no Python loop over centers**, target edges are
  matched with `searchsorted` (no `N_atoms^2` map), and `sum_C P_AC D_C P_CB` runs as one batched
  GEMM over all triangles. Cost tracks the triplet count `sum_C deg(C)^2`, inherent to three-center.
- Changes to the assembler are compatibility-sensitive once wired into a trainable model and must be
  covered by `dptb/tests/test_threecenter.py`.
