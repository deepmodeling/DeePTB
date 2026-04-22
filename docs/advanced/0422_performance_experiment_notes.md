# 2026-04-22 Performance Experiment Notes

This note records optimization routes that were tested during the 0422 DeePTB
memory/performance work but were not kept in the production code path.

## Removed Routes

- `so2_tp_chunk_size` / Wigner-D chunking:
  - Production-like bs32 smoke did not reduce peak memory.
  - Baseline peak allocated was about `28.7 GB`; chunked Wigner/SO2 was about
    `28.8 GB` and increased peak reserved memory from about `39.4 GB` to
    `41.7 GB`.
  - SO2 module peak delta was also slightly worse.
  - Conclusion: extra chunk orchestration adds complexity without useful memory
    relief for the current `lem_moe_v3_h0` workload.

- `MOLELinear.forward_chunk_size`:
  - Chunking per-system `F.linear` did not address the measured SO2 peak source.
  - It kept the same mixed-weight materialization and added another branch in the
    hot path.
  - Conclusion: not worth keeping until a profile shows per-system GEMM itself is
    the dominant memory or launch problem.

- `SO2_m_Linear.flatten_output`:
  - This was only useful for the removed SO2 chunk path.
  - It did not independently reduce the large SO2 TP peak.
  - Conclusion: removed with the chunk route.

- SO2 activation checkpointing prototype:
  - Earlier smoke attempts were slower or failed before showing a stable memory
    win.
  - Conclusion: not exposed as a config option in this PR.

## Deferred Routes

- OpenEquivariance/cuEquivariance direct replacement:
  - Local and online review showed these libraries mainly target standard CG
    tensor products, not DeePTB's current eSCN-like SO2 path plus soft
    element-level MoE weight fusion.
  - They remain useful for isolated microbenchmarks or future kernel prototypes,
    but are not a direct replacement for the current SO2/MoE hot path.

## Kept Work

- Regular CUDA memory monitoring and TensorBoard logging.
- CPU/GPU synchronization cleanup in LEM/MoE hot paths.
- Precompute of fixed-geometry LEM active edges and cutoff coefficients.
- Wigner static tensor caching and SO2 mask registration, which reduce repeated
  setup overhead without introducing the failed chunking branch.
