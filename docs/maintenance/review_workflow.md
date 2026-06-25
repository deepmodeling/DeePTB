# Review Workflow

DeePTB has more code paths than the current maintainer capacity can review manually from scratch every time. The review workflow should make AI useful, CI mechanically strict, and human review focused on design and scientific risk.

## PR Review Pipeline

Use three layers:

1. **AI-assisted review** identifies likely risks, missing tests, and compatibility concerns.
2. **CI** catches mechanical failures, regressions, and smoke-test breakage.
3. **Human maintainer review** decides whether the remaining risk is acceptable.

AI review is advisory. It should not replace maintainer judgment, especially for physics semantics and numerical behavior.

For the step-by-step PR procedure, use `docs/maintenance/maintainer_sop.md`.

## Minimal AI Review Roles

Before running prompts, generate a file-based review plan when reviewing a local branch:

```bash
git diff --name-only main...HEAD | python scripts/ci/pr_review_plan.py --stdin
```

The same script can run in GitHub Actions as an advisory job summary. Treat it as a checklist generator, not as an automated approval.

Start with two roles before adding more automation. Use the prompt text in `docs/maintenance/ai_review_prompts.md` so manual review and future automation follow the same rules.

**Maintainer Review**

- correctness bugs
- regressions against main
- public API, config, checkpoint, and dataset compatibility
- high-risk modules touched
- merge recommendation

**Test Gap Review**

- changed behavior without tests
- examples or CLI paths that should be smoke-tested
- regression tests needed for scientific behavior

These two prompts are the recommended minimum AI review set. Future CodeRabbit, OpenClaw, Hermes, or GitHub Action automation should reuse them rather than creating a separate review standard.

## Maintainer Checklist

- [ ] PR scope is clear and not mixed with unrelated cleanup.
- [ ] High-risk files were reviewed using `docs/maintenance/risk_map.md`.
- [ ] Main branch behavior is preserved unless explicitly changed.
- [ ] Public API, CLI, docs, examples, and config schema are consistent.
- [ ] Tests cover the changed behavior.
- [ ] AI review findings were handled or explicitly waived.
- [ ] CI passed.
- [ ] Remaining risks are recorded in the PR merge decision.

## Repository Hygiene Check

The repository hygiene check is a low-noise CI guard for hard errors only. It verifies that example JSON files parse, required maintenance documents exist, and the maintenance docs remain linked from the docs index.

Run it locally with:

```bash
python scripts/ci/check_repository_hygiene.py
```

This check does not replace full unit tests, Sphinx docs builds, notebook validation, or human review.

## Test Layering

Use pytest markers to match test effort to PR risk:

```bash
.venv/bin/python -m pytest ./dptb/tests -m smoke
.venv/bin/python -m pytest ./dptb/tests -m regression
.venv/bin/python -m pytest ./dptb/tests -m "not slow"
.venv/bin/python -m pytest ./dptb/tests
```

See `docs/maintenance/test_strategy.md` for marker definitions and review guidance.

## Merge Decision Template

Use the PR template's Merge Decision section before merging:

```md
Recommendation: merge after CI passes.

Reason:
- Core workflow is covered by regression tests.
- AI review concerns were resolved.
- No unresolved compatibility issues.

Remaining risks:
- Real MKL/Pardiso path requires Linux + MKL environment and is not covered in this PR.
```
