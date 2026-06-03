#!/usr/bin/env python3
"""Low-noise repository hygiene checks for DeePTB."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_FILES = (
    ".github/workflows/pr_review_plan.yml",
    ".github/workflows/pr_review_plan_comment.yml",
    ".github/workflows/repository_hygiene.yml",
    "CONTEXT.md",
    "docs/AI_CODING_GUIDE.md",
    "docs/maintenance/ai_review_prompts.md",
    "docs/maintenance/docs_warning_triage.md",
    "docs/maintenance/index.md",
    "docs/maintenance/maintainer_sop.md",
    "docs/maintenance/pr_review_plan.md",
    "docs/maintenance/risk_map.md",
    "docs/maintenance/review_workflow.md",
    "docs/maintenance/test_strategy.md",
    "docs/adr/index.md",
)

REQUIRED_DOCS_INDEX_ENTRIES = (
    "maintenance/index",
)

REQUIRED_MAINTENANCE_INDEX_ENTRIES = (
    "ai_review_prompts",
    "docs_warning_triage",
    "maintainer_sop",
    "pr_review_plan",
    "test_strategy",
)


def check_examples_json() -> list[str]:
    errors: list[str] = []
    json_files = sorted((REPO_ROOT / "examples").rglob("*.json"))

    for path in json_files:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - report any parser/read failure.
            rel_path = path.relative_to(REPO_ROOT)
            errors.append(f"{rel_path}: invalid JSON ({type(exc).__name__}: {exc})")

    print(f"Checked {len(json_files)} example JSON files.")
    return errors


def check_required_files() -> list[str]:
    errors: list[str] = []

    for rel_path in REQUIRED_FILES:
        path = REPO_ROOT / rel_path
        if not path.is_file():
            errors.append(f"{rel_path}: required maintenance file is missing")

    print(f"Checked {len(REQUIRED_FILES)} required maintenance files.")
    return errors


def check_docs_index() -> list[str]:
    errors: list[str] = []
    index_path = REPO_ROOT / "docs/index.rst"

    if not index_path.is_file():
        return ["docs/index.rst: required docs index is missing"]

    content = index_path.read_text(encoding="utf-8")
    for entry in REQUIRED_DOCS_INDEX_ENTRIES:
        if entry not in content:
            errors.append(f"docs/index.rst: missing toctree entry '{entry}'")

    print(f"Checked docs/index.rst for {len(REQUIRED_DOCS_INDEX_ENTRIES)} required entries.")
    return errors


def check_maintenance_index() -> list[str]:
    errors: list[str] = []
    index_path = REPO_ROOT / "docs/maintenance/index.md"

    if not index_path.is_file():
        return ["docs/maintenance/index.md: required maintenance index is missing"]

    content = index_path.read_text(encoding="utf-8")
    for entry in REQUIRED_MAINTENANCE_INDEX_ENTRIES:
        if entry not in content:
            errors.append(f"docs/maintenance/index.md: missing toctree entry '{entry}'")

    print(
        "Checked docs/maintenance/index.md for "
        f"{len(REQUIRED_MAINTENANCE_INDEX_ENTRIES)} required entries."
    )
    return errors


def main() -> int:
    errors: list[str] = []
    errors.extend(check_examples_json())
    errors.extend(check_required_files())
    errors.extend(check_docs_index())
    errors.extend(check_maintenance_index())

    if errors:
        print("\nRepository hygiene check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Repository hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
