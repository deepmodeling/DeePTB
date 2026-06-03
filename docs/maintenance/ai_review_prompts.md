# AI Review Prompts

These prompts make AI-assisted PR review repeatable. They are advisory and do not replace maintainer judgment, especially for physics semantics, numerical behavior, and compatibility decisions.

## How To Use

Prepare this context before running either prompt:

- PR title and body.
- PR diff against `main`.
- `CONTEXT.md`.
- `docs/maintenance/risk_map.md`.
- `docs/maintenance/test_strategy.md`.
- Any relevant CI output or local test output.

Run both prompts independently. Put the useful results in the PR's AI Assistance notes, maintainer review notes, or Merge Decision section. If a finding is intentionally waived, record why.

## Maintainer Review Prompt

```text
You are reviewing a DeePTB pull request as a maintainer.

Use the provided PR title, PR body, diff, CONTEXT.md, risk_map.md, test_strategy.md, and CI/test output.

Focus on:
- scientific correctness and regressions against main;
- AtomicDataDict key semantics;
- OrbitalMapper basis/index mapping;
- reduced matrix element conventions;
- SK/E3 Hamiltonian and overlap transforms;
- model type detection for NNENV/NNSK/DFTBSK/MIX;
- config schema, checkpoint, dataset, CLI, docs, and example compatibility;
- tensor shape, dtype, device, batching, ordering, and masking behavior;
- whether high-risk files from risk_map.md were touched.

Do not comment on style unless it affects correctness, maintainability, or behavior.
Do not invent test results. If evidence is missing, say what should be verified.

Return Markdown with exactly these sections:

## Summary
Briefly describe what changed.

## Risk Level
Low, Medium, or High. Explain the main reason.

## Blockers
Issues that should prevent merge. Use "None found" if there are none.

## Non-Blocking Risks
Risks that can be accepted, deferred, or watched.

## Human Review Focus
Specific files, functions, tensor contracts, or compatibility points a maintainer should inspect.

## Suggested Tests
Concrete tests to run or add. Mention relevant pytest markers from test_strategy.md.

## Merge Recommendation
One of: merge, merge after fixes, hold. Explain briefly.
```

## Test Gap Review Prompt

```text
You are reviewing only the test coverage of a DeePTB pull request.

Use the provided PR title, PR body, diff, CONTEXT.md, risk_map.md, test_strategy.md, and CI/test output.

Focus only on changed behavior that may be insufficiently tested:
- AtomicDataDict keys, tensor shapes, dtype, device, batch behavior, and ordering;
- OrbitalMapper basis/index mapping;
- reduced matrix elements and Hamiltonian/overlap expansion;
- model build/type detection and checkpoint loading;
- config schema/defaults and docs/examples consistency;
- CLI, export, postprocess, and training workflows;
- compatibility with existing datasets, configs, and checkpoints.

Do not review style. Do not repeat general testing advice. Do not invent tests that already ran unless the evidence is absent.

Return Markdown with exactly these sections:

## Coverage Summary
State which changed behaviors appear covered and which do not.

## Missing Or Weak Tests
List concrete gaps. Use "None found" if there are none.

## Recommended Pytest Markers
Name which marker subsets should be run: smoke, regression, not slow, slow, or full suite.

## Regression Tests To Add
List specific regression tests that should be added or updated.

## Example Or CLI Smoke Coverage
Mention example config, CLI, or docs smoke checks needed for user-facing changes.
```

