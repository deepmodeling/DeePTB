# Risk Map

This map helps maintainers decide how much review and testing a PR needs. It is intentionally practical: files are high risk when a small change can alter scientific results, compatibility, or many downstream workflows.

## High Risk

Changes here should normally include regression tests and maintainer review.

| Area | Files | Review focus |
| --- | --- | --- |
| AtomicDataDict contract | `dptb/data/_keys.py` | key names, key meaning, tensor shape, downstream callers |
| Orbital indexing | `dptb/data/transforms.py` | basis ordering, orbital block indices, mixed-basis behavior |
| Hamiltonian expansion | `dptb/nn/hamiltonian.py` | SK/E3 transforms, signs, ordering, dtype/device, overlap |
| Model assembly | `dptb/nn/build.py` | model type detection, checkpoint loading, config interpretation |
| Config schema | `dptb/utils/argcheck.py` | defaults, backward compatibility, docs/examples alignment |
| Training behavior | `dptb/nnops/trainer.py` | task detection, validation behavior, reference datasets |
| Loss behavior | `dptb/nnops/loss.py` | numerical targets, masks, reductions, task-specific semantics |

## Medium Risk

Changes here may need focused tests depending on scope.

| Area | Files | Review focus |
| --- | --- | --- |
| Embedding and prediction | `dptb/nn/embedding/`, `dptb/nn/prediction/` | tensor shape, irreps, batch behavior, model compatibility |
| Dataset backends | `dptb/data/` except high-risk files | format compatibility, missing fields, cutoff behavior |
| CLI entrypoints | `dptb/entrypoints/` | args, defaults, config loading, user-facing behavior |
| Postprocess and export | `dptb/postprocess/` | exported formats, unit conventions, compatibility |
| Plugins | `dptb/plugins/` | checkpointing, logging, training lifecycle |

## Lower Risk

These changes still need review, but usually do not require broad regression testing.

| Area | Files | Review focus |
| --- | --- | --- |
| Documentation | `docs/`, `README.md`, examples README files | stale commands, stale paths, config consistency |
| Issue templates | `.github/ISSUE_TEMPLATE/` | clarity and triage usefulness |
| Packaging metadata | `pyproject.toml`, release docs | dependency compatibility, install behavior |

## Escalation Rules

Treat a PR as high risk if it:

- changes a high-risk file;
- changes public CLI/config behavior;
- changes tensor shapes, dtype, device movement, batching, or key names;
- changes checkpoint or dataset compatibility;
- changes numerical behavior in Hamiltonian, overlap, eigenvalue, loss, or export paths;
- touches both data representation and model behavior in the same PR.

