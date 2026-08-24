# Carbon BCC self-consistent total-energy k-point convergence

This is a standalone numerical diagnostic for one fixed Carbon BCC structure.
It does **not** invoke
`dptb/tests/test_dftb_scc.py::test_dftbscc_matches_dftbp_benchmarks`, scan an
EOS, or include graphene and the Carbon dimer.

There are two distinct convergence loops:

1. **Inner SCC convergence**: at one fixed k mesh, the charge residual reaches
   `--tol`. This only says that particular discrete Brillouin-zone problem was
   solved self-consistently.
2. **Outer k-point convergence**: the converged total energy per atom stops
   changing as the k mesh is made denser. This is the primary criterion in this
   example.

An SCC-converged result at `20 x 20 x 20` is not automatically a k-point-
converged result.

The script reports, for every case:

- requested and resolved k-point mesh, raw and symmetry-reduced k-point counts,
  and the reduced-weight sum;
- self-consistent total energy, electronic energy, and repulsive energy, using
  `sigma_rep = {"C": 0.561}`;
- total-energy changes from the preceding mesh and maximum absolute change
  against every denser mesh in the same scan;
- `kpoint_energy_converged_within_scan`, evaluated against the configurable
  `--energy-tol-mev-per-atom` threshold (default `1 meV/atom`);
- inner SCC status, iteration count, final charge residual, and the full
  residual history in JSON output;
- electron count/error, Mulliken and delta charges, Fermi level, electronic
  energy, H0 band energy, and SCC shift energy;
- wall time for the SCC calculation.

The densest point in a scan has no denser validation point and is therefore
reported with `kpoint_energy_converged_within_scan = null`, rather than being
declared converged automatically. For this fixed structure, the repulsive
energy is independent of k sampling, so total- and electronic-energy changes
are identical; the script still calculates and records the actual total energy.

The committed BCC POSCAR is the scale-1.000 primitive cell used by the existing
Carbon data. The script shares the repository's canonical `C-C.skf` file under
`dptb/tests/data/dftb` by default, avoiding a second copy of the same 104 KiB
table. Override it with `--sk-path` if another Carbon parameter set is being
tested.

## Important limitation

This primitive BCC cell contains one Carbon atom. Charge conservation and
symmetry therefore pin its Mulliken population close to 4 electrons, so SCC
charge mixing normally converges in the first residual check. The example is a
valid total-energy k-point convergence test even though its inner SCC loop is
trivial. It is not, by itself, a difficult mixer-robustness benchmark. A later
mixer stress test needs a structure with non-equivalent charge degrees of
freedom or an explicitly perturbed initial charge.

## Run

From the repository root:

```bash
uv run python examples/dftb_scc/carbon_bcc_scc_convergence/run.py \
  --scan kmesh \
  --meshes 20 24 28 32 36 40 \
  --energy-tol-mev-per-atom 1.0 \
  --output /tmp/carbon_bcc_kmesh.json
```

If `uv` is unavailable but the repository virtual environment already exists:

```bash
.venv/bin/python examples/dftb_scc/carbon_bcc_scc_convergence/run.py \
  --scan kmesh --meshes 12 16 20 24
```

Preview spacing-derived meshes without allocating or diagonalizing them:

```bash
uv run python examples/dftb_scc/carbon_bcc_scc_convergence/run.py \
  --scan spacing --spacings 0.20 0.10 0.05 --dry-run
```

The `0.05` case should be previewed first because it resolves to a very large
mesh for this cell. It is intentionally not included in the default spacing
list and must be requested explicitly.

Compare mixers at one fixed mesh:

```bash
uv run python examples/dftb_scc/carbon_bcc_scc_convergence/run.py \
  --scan mixer --fixed-mesh 8 \
  --mixers simple pulay broyden diis \
  --mix-rates 0.1 0.3 \
  --output /tmp/carbon_bcc_mixers.json
```

Compare Fermi-Dirac and Gaussian occupations:

```bash
uv run python examples/dftb_scc/carbon_bcc_scc_convergence/run.py \
  --scan smearing --fixed-mesh 8 \
  --smearing-methods FD G --temperatures 0.1 300
```

Use `--help` for convergence tolerance, maximum iterations, symmetry controls,
structure override, and other numerical settings. The default charge-residual
tolerance is `1e-10 e`; tighter tolerances such as `1e-12` or the historical
`1e-14` should be requested explicitly because occupation/charge numerical
noise can otherwise dominate the inner SCC iteration count. This charge
tolerance is independent of `--energy-tol-mev-per-atom`, which controls the
outer k-point convergence assessment.

## Current observation

For the tested even-grid sequence `20, 24, 28, 32, 36, 40`, `20^3` differs from
denser meshes by as much as `2.252 meV/atom`, so it is not converged at the
`1 meV/atom` threshold. Starting from `24^3`, all tested denser even grids stay
within `1 meV/atom`. This is evidence only within that scan; neighbouring odd
meshes and the spacing-derived `75^3` case remain separate checks before a
production mesh is declared.
