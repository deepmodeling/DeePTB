# ADR 0005: Trinity Enhancements Toward State-of-the-Art Hamiltonian Prediction

## Status

Phase 1 accepted/implemented; Phases 2–3 proposed.

## Context: survey of the SOTA model designs

We want Trinity to outperform the leading DFT-Hamiltonian predictors. The relevant designs and what
each one demonstrated:

| Model | Key design elements | Measured/reported effect |
|---|---|---|
| **DeepH** (Nat. Comput. Sci. '22) | local-coordinate MPNN, invariant net + block rotation | superseded by E3 variants |
| **DeepH-E3** (Nat. Commun. '23) | full O(3) equivariance with parity, Wigner–Eckart block output, **per-channel target standardization**, gate activations | sub-meV on materials; standardization is load-bearing |
| **QHNet** (ICML '23) | pair (off-diagonal) features built once after the node backbone; CG expansion module; NormGate | large speedup at similar accuracy |
| **SPHNet** ('24) | sparse gating that prunes tensor-product paths | ~2× faster than QHNet, similar/better MAE |
| **DeepH-2** (arXiv:2401.17015) | equivariant local-coordinate **transformer** (attention) hybrid | claims SOTA over DeepH-E3 in accuracy+efficiency |
| **TraceGrad** (arXiv:2405.05722) | auxiliary supervision of the SO(3)-invariant **T = tr(H_b H_b†)** per block + gradient-boosted equivariant features, auto-balanced loss `loss_H + λ·nograd(loss_H/loss_T)·loss_T` | up to 34–48% MAE reduction on DeepH-E3, ~40% on QHNet |
| **QHNetV2** (arXiv:2506.09398) | **everything in SO(2) local frames**: off-diagonal features live in their pair frame across layers; SO(2) TP (m₁±m₂), SO(2) gate, SO(2) LayerNorm; no global CG TPs | 33.7% better than SPHNet, 58%→ over QHNet, 4.34× faster; ablation: SO(2) TP + SO(2) FFN both matter |
| E3 theory (eSCN '23, EquiformerV2) | SO(2) reduction of TPs (O(L³) vs O(L⁶)), S2 activations, equivariant attention | foundation of all the above efficiency gains |

Trinity's standing before this ADR: it already has (a) SO(2)-reduced tensor products (`SO2_Linear`,
the eSCN trick QHNetV2 is built on), (b) an Allegro-style scalar latent track with cutoff-gated
updates, and (c) two physics priors none of the molecular models have — the **always-on two-body SK
base** and the **three-center factorized term** (ADR 0004), which bought a measured 2.9× loss
reduction on the Al-AlO₂-Al junction. What it lacked (measured on that junction, see below) was the
entire normalization stack and several SOTA tricks.

### Measured pathologies driving the design (Al66O36 junction data)

- Onsite scalar means up to **−142.7 eV** were *learned by gradient descent* from init=1 (the
  trinity e3tb path had identity `lambda x: x` prediction heads — `E3statistics` was skipped).
- Within one (bond type, irrep) channel, hopping norms decay **10³–10⁶×** between r=1.7 and 7.4 Å;
  across channels 36–53×. A constant per-channel scale cannot absorb the r-decay.
- Uniform element-wise L1+RMSE ⇒ per-channel gradient share ∝ channel scale (NTK view: time to
  relative error ε on a channel of scale σ ∝ Σσ/σ), i.e. an effective condition number of 10⁶⁺ —
  the "fast start, weeks-long tail" convergence common to all models on H data.
- `onsite_shift: true` on single-gauge data leaves the μ·S direction unpenalized: one production run
  drifted μ = −25.6 eV while logging a 75× smaller loss than its true unshifted error.

## Decision — Phase 1 (implemented)

Three data-driven, architecture-light changes; all are set once from `E3statistics` at train start,
cost O(1) per element at runtime, and default to the exact historical behaviour when unset.

1. **Real scale/shift prediction heads for every e3tb embedding** (`dptb/nn/deeptb.py`): the trinity
   identity-head hack is removed; `E3PerSpeciesScaleShift`/`E3PerEdgeSpeciesScaleShift` receive
   per-(species/bond-type, irrep) **scales** (target norms / scalar stds) and per-species scalar
   **shifts** (target means) from `E3statistics`. This is DeepH-E3's standardization, applied to the
   residual the trinity env/3b channels must learn on top of the SK base. Measured: **7.4× lower
   loss at iteration 1, 5× lower at equal wall time; 250 iterations ≈ the previous 15 000.**

2. **Radial decay envelope** (`E3PerEdgeSpeciesScaleShift.set_decay`, fitted in
   `_E3edgespecies_stat`): the effective edge scale becomes `scale_ch · exp(−κ_ch (r − r₀_ch))`,
   with κ_ch ≥ 0 per (bond type, irrep) from a least-squares fit of log‖H_ch‖ vs r, and `scale_ch`
   anchored to the **measured near-bond norm** (mean over edges within 0.5 Å of the closest — NOT
   the extrapolated log-linear intercept, which overshoots ~10× because the fit is dominated by
   mid/far edges; A/B-tested). The network then learns O(1) quantities at **every** bond length,
   not magnitudes spanning the 10³–10⁶ decay window; extrapolation beyond the training r inherits
   the physical exponential decay. Fitted κ on the Al66O36 junction: 1.0–3.1 Å⁻¹ (196/308 channels)
   — textbook LCAO decay constants. Benchmark (trinity 2b+3b, 400 iters, identical seed/optimizer):
   total loss 0.0709 vs 0.0739 without the envelope, with **2.2× lower far-edge (6.0–7.4 Å) MAE**
   (7.5e-3 vs 1.63e-2 eV) and 24% lower at 4.5–6.0 Å; the margin widens with training. The near bin
   (1.5–3.0 Å) is currently ~25% worse — a two-segment/binned envelope is a candidate refinement.
   This is our materials-specific contribution — molecular benchmarks (QH9/MD17) have too narrow an
   r-range to expose the need, which is why none of the surveyed models have it. κ=0 (default, and
   for channels without a reliable fit) is exactly the old behaviour; buffers are always registered
   so checkpoints stay layout-stable.

3. **TraceGrad-style invariant auxiliary loss** (`HamilLossAbs(trace_weight=...)`): supervise
   T = tr(H_b H_b†) per irrep block. In DeePTB's orthonormal CG representation this is *exactly* the
   sum of squared reduced coefficients over an irrep slice — one square + segment-sum, no extra CG
   work. Auto-balanced as `loss_H + λ·nograd(loss_H/loss_T)·loss_T` per the TraceGrad paper (their
   reported gains: 34–48% on DeepH-E3, ~40% on QHNet baselines; we adopt the supervision half — the
   gradient-boosted-feature half is Phase 2). Config: `loss_options.train.trace_weight` (default 0).
   Our own 250-iteration A/B showed **no gain at short horizon** (the aux term diverts optimization
   budget early), so it stays off by default; the literature gains are reported for full-length
   trainings — try `trace_weight: 1.0` on production runs only.

Also fixed alongside (see git history): dataset-cache invalidation no longer keyed to
`dptb.__version__`; `three_center` params always built (progressive 2b→2b+3b→full training with a
mode/freeze split); `from_reference` tolerates older checkpoints (strict=False fallback).

## Phase 2 — first tranche (implemented)

1. **SO(2) gate** (`SO2_Linear(so2_gate=True)`, trinity config `so2_gate`): every |m|>0 output
   channel of the message-passing tensor products is gated by a sigmoid of the *in-frame m=0*
   components — QHNetV2's in-frame nonlinearity (their ablation: a main accuracy driver). The plain
   `SO2_Linear` is purely linear per m; the gate adds cross-channel nonlinearity *between* the
   rotate-in and rotate-out steps, so SO(3) equivariance is preserved (verified to 4e-6 under random
   rotations; note the convention: `R[:,[1,2,0]]` lives in the e3nn `1o` space). Gates are
   initialized OPEN (zero weight, bias 2 → sigmoid ≈ 0.88) so the module starts near-identity — a
   zero-init bias measurably degraded the starting loss (0.46 vs 0.34). Off by default — it adds
   parameters, so old full-mode checkpoints keep loading strict.
   **Measured (honest):** a 150-iteration full-mode A/B on the Al66O36 junction showed no gain from
   gate+hermitian over the Phase-1 stack (0.0526 vs 0.0483, within single-seed noise); QHNetV2-scale
   gains are reported on large datasets with full-length training. Treat `so2_gate` as a
   to-be-validated knob for production runs, not a proven local win.
2. **Exact hermiticity of the env edge channel** (trinity config `hermitian`, default True):
   LCAO H is hermitian; in the reduced CG representation the constraint lands **only on the
   same-shell-pair (io==jo) channels** of reversed edges: `r_ji = (−1)^J · r_ij` — verified
   numerically against `E3Hamiltonian` (the io≠jo channels of the two directions carry independent
   physics, since only io≤jo pairs are stored per directed edge). The SK two-body and three-center
   terms satisfy the relation by construction; the env output does not, so Trinity now averages
   `r ← (r + (−1)^J r[rev])/2` (reversed partner found by an O(E log E) shift-aware match).
   Zero parameters, exact constraint, halves the function space in the constrained subspace.
3. **Wigner-D cache**: all `SO2_Linear` calls of all layers rotate with the same per-edge Wigner
   matrices; Trinity now computes them once per forward and passes them down (2×n_layers redundant
   `batch_wigner_D` computations saved; bitwise-identical outputs, verified). **Measured:** no
   wall-time change on CPU (1.00×) — the Wigner build is a negligible share of the forward there;
   kept because it is free, exact, and the saving may matter on GPU where kernel-launch/assembly
   overhead weighs differently.

4. **Cutoff-boundary continuity** (`boundary_envelope` in `dptb/nn/cutoff.py`): the model used to be
   discontinuous in r at the cutoff-activation boundary, and float32 evaluation of the p=6
   polynomial cutoff at r/r_max ≈ 1 has ~3e-6 absolute noise (catastrophic cancellation) vs true
   coefficients 1e-6–1e-11, so active-set membership flipped between dtypes: measured 74/14682
   edges (all within 25 mÅ of r_max) with up to **0.7 eV** dtype-dependent output noise. Two leak
   paths were fixed with a C²-smooth envelope (exactly 1 below 0.95·r_max, stable factored form
   `(1−t)³(6t²+3t+1)`): (i) the env pathway (InitLayer features, node message weights, `out_edge`
   output — MLP biases otherwise emit O(1) features for ε-active edges); (ii) **the dominant one**:
   `E3PerEdgeSpeciesScaleShift` skips exactly-zero rows, so an inactive edge got NO shift while an
   ε-active one got the full per-channel shift (up to 0.7 eV) — the shift is now multiplied by
   `boundary_envelope²` when `r_max` is known. Result: **0/14682 outliers; max float32-vs-float64
   deviation 1.4e-5 eV** (was 0.698). Regression tests: envelope properties, env-channel continuity
   at the boundary, and the head-level zero-row/ε-row shift-jump test.

## Phase 2 — remaining (proposed, ordered by expected value/effort)

1. **Persistent SO(2) local-frame edge track** (QHNetV2's core): keep the off-diagonal features in
   their pair SO(2) frame across all layers (rotate node contributions in per layer; rotate out once
   before the output head) — removes the remaining per-op rotate-in/rotate-out pairs and enables
   in-frame SO(2) LayerNorm and the SO(2) tensor product (m_out = m₁±m₂; QHNetV2 ablation: −15.9%
   without the TP, −58% without TP+FFN).
2. **TraceGrad gradient features**: feed ∂z/∂f (z = invariant trace head) back as an equivariant
   feature boost — the second half of TraceGrad's gains.
3. **Equivariant attention** for `UpdateNode`/`UpdateEdge` (DeepH-2 / EquiformerV2 style): the
   latent-conditioned `env_embed_mlps` weights are already attention-shaped; add softmax
   normalization over neighbours and a temperature.
4. **Parity-complete hidden irreps** as the documented default (e.g. add `0o/1e/2o/…` multiplicities)
   — DeepH-E3 showed odd/even completeness matters for H targets; today's default config carries
   only one parity per l in hidden layers.

## Numerical audit of the SO(2) MPNN (measured at init, full mode, Al66O36)

Signal-propagation and gradient-flow instrumentation of the production configuration revealed:

1. **Spectral collapse with depth**: per-l feature RMS across the 3 layers — edge l=2: 0.101→0.054,
   l=3: 0.140→0.076, node l=1: 0.80→0.37, l=3: 0.42→0.16, while l=0 GROWS (edge 0.127→0.158, node
   0.73→1.34). The Gate (sigmoid ≈ 0.5 average on l>0) and the scalar-vs-pooled-l>0 split in
   `SeperableLayerNorm` drain exactly the high-l channels that carry the bond-anisotropy content of
   H. Depth currently *hurts* the hardest channels.
2. **20× per-block effective-learning-rate imbalance**: `AtomicResNet` heads (`edge_prediction_h2`,
   `edge_prediction_s`) have |w|_rms ≈ 0.05 (their out-layers are init with std=1e-3) while the env
   stack has |w|_rms ≈ 1.0 — with a single Adam lr the 2b/overlap heads move ~10%/step while the
   env stack moves ~0.5%/step: the heads sit at a high lr-noise floor while the env crawls.
3. **Norm robustness**: `SeperableLayerNorm` divides by the row norm with eps added to the NORM
   (5e-3), so near-zero rows are amplified up to 1/eps = 200× (the boundary-instability amplifier).
4. **Fixed neighbour normalization**: `env_sum_normalizations = rsqrt(avg_num_neighbors)` is a
   config constant (was set 273 vs measured 144) and cannot adapt across diverse systems.
5. **Parity**: `SO2_Linear` ignores irrep parity (SO(3)-equivariant, not O(3)) and the default
   hidden irreps carry one parity per l; mirror symmetry of H is not enforced (same class as
   QHNet/QHNetV2, weaker prior than DeepH-E3).
6. Residual path applies `linear_res` every layer (no identity gradient highway); radial basis is
   10 Bessel over 7.4 Å (~0.7 Å resolution); trainer has no gradient clipping and no weight EMA.

### Production-numerics roadmap (ordered by measured impact)

P1. **Per-l normalization groups + gate gain correction**: normalize each l (not scalars vs pooled
    rest) and restore post-gate RMS (learnable per-l gain, init 1/E[sigmoid]) — directly attacks the
    measured spectral collapse. Variance-form eps (sqrt(ms+eps²)) fixes the near-zero amplification.
P2. **Per-group optimizer scaling**: parameter groups with lr scaled by group |w|_rms (μP-style), or
    re-init the AtomicResNet out-layers to O(1/sqrt(fan)) and absorb the 1e-3 into a fixed output
    gain — equalizes the measured 20× imbalance.
P3. **Trainer hardening for production**: gradient-norm clipping (batch=1 spikes), weight EMA for
    eval/checkpoints, warmup + cosine decay to ~1e-5, full-dataset batch or accumulation, TF32 off,
    float64 fine-tune stage below ~1e-3.
P4. **Adaptive neighbour normalization**: per-node deg^{-1/2} (or E3statistics-measured average at
    from-scratch init) instead of the config constant — required for transferability across diverse
    densities.
P5. **Identity residuals** when irreps match; wider radial basis (16–32) for the 2b/overlap heads.
P6. **Parity-complete hidden irreps** (config) and, longer term, parity-restricted SO(2) maps for a
    true O(3) prior.

## Phase 3 (research directions)

- **Long-range electrostatics channel** for junctions/interfaces: onsite levels track the local
  electrostatic potential; all surveyed models are strictly local. A cheap scalar Ewald/damped-
  Coulomb message on the latent track feeding the node head would be a genuine differentiator
  (4G-HDNNP-style), directly aimed at band alignment across heterostructures.
- **Sparse TP path gating** (SPHNet) inside `SO2_Linear` for further speed at high l.
- **float64 fine-tune stage** below ~1e-3 loss (float32 gradient noise becomes binding when eV-scale
  channels need 1e-4 relative accuracy).

## Consequences

- All Phase-1 features are opt-out/no-op by default at the module level and are populated by
  `E3statistics(model=...)` at training start; inference-time cost is negligible.
- Old checkpoints load (missing buffers = identity behaviour + warning); new checkpoints carry the
  statistics buffers and restore bit-exactly.
- slem/lem inherit improvements 1–2 automatically (shared heads/statistics path).
- Benchmarks: `tools/`-level A/B on Al66O36 (this repo, see ADR + session logs). Claiming
  superiority over QHNetV2/DeepH-2 requires running the public benchmarks (QH9, DeepH's materials
  sets) after Phase 2 — Phase 1 targets the materials-specific pathologies those benchmarks do not
  probe.
