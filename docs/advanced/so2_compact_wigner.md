# SO2 Compact Wigner Apply

This note records the validation data for the SO2 Wigner apply memory
optimization introduced by `so2_wigner_apply_mode`.

## Behavior

`SO2_Linear` previously materialized the full block-diagonal Wigner matrix as
`[num_edges, D, D]`, where `D = (lmax + 1)^2`. For `lmax=6`, this is
`[num_edges, 49, 49]`.

The default mode is now:

```json
"so2_wigner_apply_mode": "compact_blocks"
```

`compact_blocks` stores only the per-degree Wigner blocks:

```text
[E,1,1], [E,3,3], [E,5,5], ..., [E,13,13]
```

The previous dense path remains available with:

```json
"so2_wigner_apply_mode": "full_dense"
```

## Production-like Smoke Data

Environment summary:

- GPU: 2x NVIDIA L40S
- Model: `lem_moe_v3_h0`
- DDP: enabled on 2 GPUs
- Dataset scale: production-like internal smoke set
- CUDA memory monitor: enabled

### Batch Size 32

| mode | result | train wall time | peak allocated | peak reserved |
| --- | --- | ---: | ---: | ---: |
| `full_dense` | pass | 54.675 s | 28716.0 MB | 41348.0 MB |
| `compact_blocks` | pass | 55.642 s | 28430.6 MB | 39396.0 MB |
| default config, field omitted | pass | 55.829 s | 28430.6 MB | 39396.0 MB |

Delta for explicit `compact_blocks` at batch size 32:

- `peak allocated`: -285.4 MB
- `peak reserved`: -1952.0 MB
- training wall time: +0.967 s for this one-epoch smoke

### Batch Size 48

| mode | result | train wall time | peak allocated | peak reserved |
| --- | --- | ---: | ---: | ---: |
| `compact_blocks` | pass | 56.482 s | 41094.9 MB | 44628.0 MB |
| `full_dense` | OOM during backward | n/a | 41525.0 MB before OOM | 44032.0 MB before OOM |

The `full_dense` run failed during backward while trying to allocate another
5.26 GiB. At failure time, PyTorch reported 24.26 GiB allocated and 14.85 GiB
reserved but unallocated.

## Interpretation

`compact_blocks` does not dramatically reduce global `peak allocated` at batch
size 32 because the whole training step has other large live activations.
However, it reduces allocator pressure enough that batch size 48 completes in
the production-like smoke where `full_dense` fails.
