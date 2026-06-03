# AI Coding Guide

This guide defines how AI-assisted coding should be used in DeePTB. AI can help draft code, tests, docs, and reviews, but maintainers remain responsible for scientific correctness.

## Required Discipline

- Treat tensor shape, dtype, device, batching, and key semantics as correctness issues.
- Prefer existing DeePTB interfaces and domain concepts over new helper layers.
- Keep changes scoped to the PR goal. Avoid drive-by cleanup in high-risk modules.
- Update tests, examples, and docs when config, CLI, data format, or public behavior changes.
- Mark AI-assisted PRs in the PR template.

## Red Lines

AI-generated changes must not silently:

- Rename or repurpose `AtomicDataDict` keys.
- Change orbital ordering or basis mapping outside `OrbitalMapper`.
- Change reduced matrix element conventions.
- Change SK or E3 Hamiltonian block expansion semantics.
- Change config defaults or schema without docs and tests.
- Change checkpoint or dataset compatibility without an explicit migration path.
- Move domain logic into generic utilities where reviewers cannot see the physics meaning.

## High-Risk Changes Need Regression Tests

Add or update regression tests when touching:

- `dptb/data/_keys.py`
- `dptb/data/transforms.py`
- `dptb/nn/hamiltonian.py`
- `dptb/nn/build.py`
- `dptb/utils/argcheck.py`
- `dptb/nnops/loss.py`
- `dptb/nnops/trainer.py`

Useful regression tests usually check concrete behavior: tensor keys, shapes, ordering, config parsing, checkpoint loading, numerical outputs, or CLI behavior.

## Review Prompts For AI Tools

Use the prompts in `docs/maintenance/ai_review_prompts.md` for repeatable review. The recommended minimum set is:

- Maintainer Review: correctness, compatibility, risk, and merge recommendation.
- Test Gap Review: changed behavior that is missing regression, smoke, or example coverage.
