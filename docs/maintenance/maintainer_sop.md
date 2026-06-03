# Maintainer PR Review SOP

This SOP turns DeePTB's maintenance documents into a concrete PR review flow. It is meant for maintainers reviewing alone or with AI assistance.

## 1. Intake

Start by reading the PR title, body, changed files, and CI status.

- Confirm the scope is clear and not mixed with unrelated cleanup.
- Check the DeePTB Impact Area section in the PR template.
- Check whether AI assistance was used and whether the author documented manual review.
- Check compatibility answers for public API, CLI, config schema, checkpoints, and datasets.
- Ask for clarification before reviewing deeply if the PR intent is unclear.

## 2. Risk Triage

Use `docs/maintenance/risk_map.md` to classify the PR.
For a first-pass advisory plan, run:

```bash
git diff --name-only main...HEAD | python scripts/ci/pr_review_plan.py --stdin
```

Use the script output as a prompt for human review, not as an automatic merge decision.
On GitHub, the advisory PR Review Plan may also appear as a bot comment. That comment is generated from changed file names only; it should start the review, not end it.

**Low risk**

- Documentation-only changes.
- Issue template or maintenance text changes.
- Packaging metadata changes without dependency or install behavior changes.

**Medium risk**

- Embedding, prediction, dataset backend, CLI, postprocess, export, or plugin changes.
- Docs/examples changes that affect user commands or config snippets.

**High risk**

- Any high-risk file from `risk_map.md`.
- Changes to tensor shapes, dtype, device movement, batching, ordering, masks, or key names.
- Changes to config schema/defaults, checkpoint compatibility, dataset compatibility, Hamiltonian/overlap, eigenvalues, loss, training behavior, or export semantics.
- PRs that touch both data representation and model behavior.

If the PR risk level in the template disagrees with this triage, use the higher risk level unless the reason is clearly documented.

## 3. AI Review

For medium and high risk PRs, run both prompts from `docs/maintenance/ai_review_prompts.md`:

- Maintainer Review Prompt.
- Test Gap Review Prompt.

Use the PR title, PR body, diff against `main`, `CONTEXT.md`, `risk_map.md`, `test_strategy.md`, and relevant CI or local test output as context.

AI findings are advisory. Handle each actionable finding by fixing it, asking the author to fix it, or recording why it is waived.

## 4. Test Selection

Choose the smallest test set that matches the PR risk, then expand when evidence is missing.

```bash
python scripts/ci/check_repository_hygiene.py
uv run pytest ./dptb/tests -m smoke
uv run pytest ./dptb/tests -m regression
uv run pytest ./dptb/tests -m "not slow"
uv run pytest ./dptb/tests
```

- Run repository hygiene for examples, docs navigation, maintenance docs, or workflow changes.
- Run `smoke` for broad PRs before deep review.
- Run `regression` for high-risk files or compatibility-sensitive behavior.
- Run `not slow` when you need wider confidence without the heaviest workflows.
- Run the full suite before merging changes to model behavior, training, Hamiltonian construction, datasets, checkpoint behavior, or public workflows.

Do not treat missing local optional dependencies as silent success. Record the skipped environment and whether CI covers it.

## 5. Hold Conditions

Hold the PR instead of merging when any of these apply:

- PR scope is unclear or mixes unrelated changes.
- High-risk behavior changed without a regression test or a documented waiver.
- Config, checkpoint, dataset, CLI, or public API compatibility is unclear.
- CI failed and the failure is not explained.
- AI review identified a plausible correctness issue that was not resolved or waived.
- Tensor/key/order semantics changed but the PR does not explain the intended behavior.
- Remaining risk is too large to describe in the Merge Decision.

## 6. Merge Decision

Before merging, fill the PR template's Merge Decision section.

```md
Recommendation: merge / merge after fixes / hold

Reason:
- ...

Tests:
- ...

Remaining risks:
- ...

Known limitations:
- ...

Follow-up issue needed: yes / no
```

The merge decision should make future debugging easier. If a risk is accepted, name it directly.
