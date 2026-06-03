# PR Review Plan

`scripts/ci/pr_review_plan.py` turns changed files into an advisory review plan. It is the active counterpart to the maintainer SOP: instead of expecting the maintainer to remember every rule, it prints the likely risk level, review focus, AI prompt suggestions, tests, and hold conditions.

The script is stdlib-only and does not call GitHub, AI services, or DeePTB runtime dependencies.

## Local Usage

Pass files directly:

```bash
python scripts/ci/pr_review_plan.py --changed-files dptb/nn/hamiltonian.py
```

Or pipe changed files from git:

```bash
git diff --name-only main...HEAD | python scripts/ci/pr_review_plan.py --stdin
```

The output is Markdown and can be pasted into a PR's AI Assistance or Merge Decision notes.

For a shorter PR-page comment format, use:

```bash
git diff --name-only main...HEAD | python scripts/ci/pr_review_plan.py --stdin --format github-comment
```

The default CLI format is intentionally detailed. The GitHub comment format is optimized for maintainer scanning: the visible summary is bilingual, while detailed file lists, local commands, and hold conditions are kept in collapsible sections.

## GitHub Automation

DeePTB has two advisory workflows for this plan:

- `.github/workflows/pr_review_plan.yml` runs on `pull_request` and writes the plan to the Actions job summary.
- `.github/workflows/pr_review_plan_comment.yml` runs on `pull_request_target` and writes or updates one PR comment.

The comment workflow is designed for fork PRs. It checks out the trusted base commit from the main repository, reads changed file names through the GitHub API, then runs the base-branch copy of `scripts/ci/pr_review_plan.py`. It must not checkout or execute code from the fork branch.

The comment workflow may also add an Evidence section based on GitHub check status. That evidence is owned by the workflow, not by `pr_review_plan.py`, because the script only knows changed file names.

Required permissions for the comment workflow are intentionally narrow:

```yaml
permissions:
  contents: read
  pull-requests: write
  issues: write
```

`issues: write` is needed because PR timeline comments use the issues comments API. `pull-requests: write` is needed for repositories where GitHub requires both issue-comment and pull-request write scopes for PR timeline comments. The workflow updates the existing bot comment using a hidden marker instead of creating a new comment on every PR update.

## How To Read The Output

- **Risk level** follows `docs/maintenance/risk_map.md`.
- **Touched risk areas** show which mapped areas were changed.
- **Required maintainer focus** lists what a human should inspect.
- **Suggested AI review** points to the minimum recommended prompts.
- **Suggested checks** lists likely local commands for this PR.
- **Hold conditions** names situations that should stop merge until resolved or explicitly waived.
- **GitHub comment summary** shows bilingual risk, reason, and review focus, with details folded by default.

The plan is advisory. When the PR body, maintainer judgment, or scientific impact suggests higher risk than the file map, use the higher risk level.

## Examples

Hamiltonian changes should be treated as high risk:

```bash
printf "dptb/nn/hamiltonian.py\n" | python scripts/ci/pr_review_plan.py --stdin
```

Docs-only changes should normally stay low risk:

```bash
printf "docs/index.rst\n" | python scripts/ci/pr_review_plan.py --stdin
```

GitHub comment preview:

```bash
printf "dptb/utils/argcheck.py\n" | python scripts/ci/pr_review_plan.py --stdin --format github-comment
```
