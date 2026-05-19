# Ill-conditioned overlap example

This example shows how to enable DeePTB's opt-in fallback for ill-conditioned
overlap matrices during inference and post-processing.

For non-orthogonal models, the band solver handles the generalized eigenvalue
problem:

```text
H(k) c = E S(k) c
```

The default dense solver uses Cholesky factorization of `S(k)`. This is the
right fail-fast behavior for training and for well-conditioned overlap matrices.
When a large or redundant basis makes `S(k)` nearly singular, inference can
instead use `ill_threshold` to project out near-linearly-dependent overlap
modes.

## Files

- `band_with_ill_overlap.json`: a `dptb run` band configuration with
  `ill_threshold` and `ill_pad_value`.
- `tbsystem_ill_overlap.ipynb`: the same workflow through the unified
  `TBSystem` API.

The example reuses the E3 silicon model and data under `examples/e3`.

## Run with the CLI

From this directory:

```bash
uv run dptb run band_with_ill_overlap.json \
  -i ../e3/ref_model/nnenv.ep1474.pth \
  -stu ../e3/data/Si64.vasp \
  -o ./output
```

The relevant options are:

```json
{
    "ill_threshold": 5e-4,
    "ill_pad_value": 10000.0
}
```

`ill_threshold` is disabled by default. Set it only for inference or
post-processing when Cholesky fails because the overlap matrix is ill
conditioned.

## Run with `TBSystem`

Open `tbsystem_ill_overlap.ipynb`, or use the same pattern in Python:

```python
from pathlib import Path
from dptb.postprocess.unified.system import TBSystem

root = Path("..").resolve()

system = TBSystem(
    data=str(root / "e3/data/Si64.vasp"),
    calculator=str(root / "e3/ref_model/nnenv.ep1474.pth"),
    override_overlap=str(root / "e3/data/Si64.0/overlaps.h5"),
    device="cpu",
)

bands = system.get_bands(
    kpath_config={
        "method": "abacus",
        "kpath": [
            [0.0, 0.0, 0.0, 20],
            [0.5, 0.0, 0.0, 1],
        ],
        "klabels": ["G", "X"],
    },
    ill_threshold=5e-4,
)
```

If any overlap modes are projected out, DeePTB logs a warning and stores a
valid-eigenvalue mask in the returned atomic data under
`AtomicDataDict.EIGENVALUE_VALID_MASK_KEY`.

## Notes

- Keep `ill_threshold=None` during training unless you intentionally want the
  loss to match a projected-subspace solver.
- If many modes are projected out, inspect the basis, overlap source, and model
  instead of simply increasing the threshold.
- PDOS/eigenvector workflows are not covered by this fallback yet, because they
  require a corresponding projected eigenvector treatment.
