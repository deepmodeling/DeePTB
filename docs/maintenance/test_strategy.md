# Test Strategy

DeePTB tests are grouped by review purpose. The markers help maintainers pick a useful test subset for a PR without changing the existing full test workflow.

## Markers

**smoke** tests are quick checks for imports, config generation, model build, and core mapping entrypoints. They should be stable and low-dependency.

**regression** tests protect known behavior, numerical semantics, tensor shape/order, and compatibility. High-risk changes should usually add or update a regression test.

**slow** tests cover training, complex postprocess, external or optional dependency paths, and workflows that are noticeably longer-running.

## Local Commands

```bash
python -m pytest ./dptb/tests -m smoke
python -m pytest ./dptb/tests -m regression
python -m pytest ./dptb/tests -m "not slow"
python -m pytest ./dptb/tests
```

## Review Guidance

- Run `smoke` for quick confidence before reviewing a broad PR.
- Run `regression` when a PR touches high-risk files from `docs/maintenance/risk_map.md`.
- Run `not slow` when you want a wider local check without the heaviest workflows.
- Run the full test suite before merging changes that affect model behavior, training, Hamiltonian construction, datasets, or public workflows.

The first version of this strategy does not make markers a required CI gate. The existing full CI workflow remains unchanged.
