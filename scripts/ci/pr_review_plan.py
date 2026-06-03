#!/usr/bin/env python3
"""Generate an advisory DeePTB PR review plan from changed files."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from enum import IntEnum


class Risk(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


RISK_LABELS = {
    Risk.LOW: "Low",
    Risk.MEDIUM: "Medium",
    Risk.HIGH: "High",
}


@dataclass(frozen=True)
class AreaRule:
    name: str
    risk: Risk
    focus: str
    exact: tuple[str, ...] = ()
    prefixes: tuple[str, ...] = ()


AREA_RULES = (
    AreaRule(
        name="AtomicDataDict contract",
        risk=Risk.HIGH,
        exact=("dptb/data/_keys.py",),
        focus="key names, key meaning, tensor shape, and downstream callers",
    ),
    AreaRule(
        name="Orbital indexing",
        risk=Risk.HIGH,
        exact=("dptb/data/transforms.py",),
        focus="basis ordering, orbital block indices, and mixed-basis behavior",
    ),
    AreaRule(
        name="Hamiltonian expansion",
        risk=Risk.HIGH,
        exact=("dptb/nn/hamiltonian.py",),
        focus="SK/E3 transforms, signs, ordering, dtype/device, and overlap",
    ),
    AreaRule(
        name="Model assembly",
        risk=Risk.HIGH,
        exact=("dptb/nn/build.py",),
        focus="model type detection, checkpoint loading, and config interpretation",
    ),
    AreaRule(
        name="Config schema",
        risk=Risk.HIGH,
        exact=("dptb/utils/argcheck.py",),
        focus="defaults, backward compatibility, docs/examples alignment",
    ),
    AreaRule(
        name="Training behavior",
        risk=Risk.HIGH,
        exact=("dptb/nnops/trainer.py",),
        focus="task detection, validation behavior, and reference datasets",
    ),
    AreaRule(
        name="Loss behavior",
        risk=Risk.HIGH,
        exact=("dptb/nnops/loss.py",),
        focus="numerical targets, masks, reductions, and task-specific semantics",
    ),
    AreaRule(
        name="Embedding and prediction",
        risk=Risk.MEDIUM,
        prefixes=("dptb/nn/embedding/", "dptb/nn/prediction/"),
        focus="tensor shape, irreps, batch behavior, and model compatibility",
    ),
    AreaRule(
        name="Dataset backends",
        risk=Risk.MEDIUM,
        prefixes=("dptb/data/",),
        focus="format compatibility, missing fields, and cutoff behavior",
    ),
    AreaRule(
        name="CLI entrypoints",
        risk=Risk.MEDIUM,
        prefixes=("dptb/entrypoints/",),
        focus="args, defaults, config loading, and user-facing behavior",
    ),
    AreaRule(
        name="Postprocess and export",
        risk=Risk.MEDIUM,
        prefixes=("dptb/postprocess/",),
        focus="exported formats, unit conventions, and compatibility",
    ),
    AreaRule(
        name="Plugins",
        risk=Risk.MEDIUM,
        prefixes=("dptb/plugins/",),
        focus="checkpointing, logging, and training lifecycle",
    ),
    AreaRule(
        name="GitHub Actions and CI",
        risk=Risk.MEDIUM,
        prefixes=(".github/workflows/",),
        focus="trigger scope, required checks, dependency installation, and runtime noise",
    ),
    AreaRule(
        name="Tests",
        risk=Risk.MEDIUM,
        prefixes=("dptb/tests/",),
        focus="marker choice, regression coverage, skipped environments, and assertions",
    ),
    AreaRule(
        name="Examples and configs",
        risk=Risk.LOW,
        prefixes=("examples/",),
        focus="valid JSON, command consistency, and config/schema compatibility",
    ),
    AreaRule(
        name="Documentation",
        risk=Risk.LOW,
        prefixes=("docs/",),
        exact=("README.md",),
        focus="stale commands, stale paths, config consistency, and Sphinx warnings",
    ),
    AreaRule(
        name="Issue templates",
        risk=Risk.LOW,
        prefixes=(".github/ISSUE_TEMPLATE/",),
        focus="clarity and triage usefulness",
    ),
    AreaRule(
        name="Packaging metadata",
        risk=Risk.LOW,
        exact=("pyproject.toml", ".github/release-check.md"),
        focus="dependency compatibility and install behavior",
    ),
    AreaRule(
        name="Maintenance governance",
        risk=Risk.LOW,
        exact=("CONTEXT.md",),
        prefixes=("docs/maintenance/", "docs/adr/", "scripts/ci/"),
        focus="review guidance consistency and low-noise maintainer workflow",
    ),
)


def normalize_path(path: str) -> str:
    path = path.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


def collect_paths(args: argparse.Namespace) -> list[str]:
    paths: list[str] = []
    if args.stdin:
        paths.extend(sys.stdin.read().splitlines())
    paths.extend(args.changed_files or [])

    seen: set[str] = set()
    normalized: list[str] = []
    for path in paths:
        clean = normalize_path(path)
        if clean and clean not in seen:
            seen.add(clean)
            normalized.append(clean)
    return normalized


def matches_rule(path: str, rule: AreaRule) -> bool:
    return path in rule.exact or any(path.startswith(prefix) for prefix in rule.prefixes)


def classify(paths: list[str]) -> tuple[Risk, list[tuple[AreaRule, list[str]]], list[str]]:
    rule_files: dict[AreaRule, list[str]] = {rule: [] for rule in AREA_RULES}

    for path in paths:
        path_rules = [rule for rule in AREA_RULES if matches_rule(path, rule)]
        if not path_rules:
            continue
        max_risk = max(rule.risk for rule in path_rules)
        for rule in path_rules:
            if rule.risk == max_risk:
                rule_files[rule].append(path)

    matched: list[tuple[AreaRule, list[str]]] = []
    for rule in AREA_RULES:
        files = rule_files[rule]
        if files:
            matched.append((rule, files))

    matched_paths = {path for _, files in matched for path in files}
    unmatched = [path for path in paths if path not in matched_paths]

    risk = max((rule.risk for rule, _ in matched), default=Risk.LOW)

    touches_data = any(path.startswith("dptb/data/") for path in paths)
    touches_model = any(path.startswith(("dptb/nn/", "dptb/nnops/")) for path in paths)
    if touches_data and touches_model:
        risk = Risk.HIGH

    return risk, matched, unmatched


def has_any(paths: list[str], prefixes: tuple[str, ...] = (), exact: tuple[str, ...] = ()) -> bool:
    return any(path in exact or any(path.startswith(prefix) for prefix in prefixes) for path in paths)


def suggested_checks(risk: Risk, paths: list[str]) -> list[str]:
    checks: list[str] = []
    touches_docs_or_examples = has_any(
        paths,
        prefixes=("docs/", "examples/", ".github/workflows/", "scripts/ci/"),
        exact=("CONTEXT.md", ".github/pull_request_template.md"),
    )
    touches_docs = has_any(paths, prefixes=("docs/",), exact=("README.md",))
    touches_behavior = has_any(paths, prefixes=("dptb/",), exact=("pyproject.toml",))

    if touches_docs_or_examples:
        checks.append("python scripts/ci/check_repository_hygiene.py")
    if touches_docs:
        checks.append("uv run python -m sphinx -b html docs docs/_build/pr-review-plan-html")

    if risk == Risk.HIGH:
        checks.append("uv run pytest ./dptb/tests -m regression")
        checks.append("uv run pytest ./dptb/tests")
    elif risk == Risk.MEDIUM:
        checks.append("uv run pytest ./dptb/tests -m smoke")
        if touches_behavior:
            checks.append("uv run pytest ./dptb/tests -m regression")
        else:
            checks.append('uv run pytest ./dptb/tests -m "not slow"')
    else:
        if touches_behavior:
            checks.append("uv run pytest ./dptb/tests -m smoke")

    if not checks:
        checks.append("uv run pytest ./dptb/tests -m smoke")

    deduped: list[str] = []
    for check in checks:
        if check not in deduped:
            deduped.append(check)
    return deduped


def ai_prompt_suggestions(risk: Risk) -> list[str]:
    if risk == Risk.LOW:
        return [
            "Optional: run the Test Gap Review Prompt if examples, docs commands, or test markers changed.",
        ]
    return [
        "Run the Maintainer Review Prompt from docs/maintenance/ai_review_prompts.md.",
        "Run the Test Gap Review Prompt from docs/maintenance/ai_review_prompts.md.",
    ]


def maintainer_focus(risk: Risk, matched: list[tuple[AreaRule, list[str]]], paths: list[str]) -> list[str]:
    focus = [f"{rule.name}: {rule.focus}" for rule, _ in matched]
    if any(path.startswith("dptb/data/") for path in paths) and any(
        path.startswith(("dptb/nn/", "dptb/nnops/")) for path in paths
    ):
        focus.append("Cross-boundary risk: data representation and model/training behavior changed together.")
    if risk == Risk.HIGH:
        focus.append("Confirm compatibility for public API, CLI/config schema, checkpoints, and datasets.")
    if not focus:
        focus.append("Confirm the PR scope is clear and no unrelated cleanup is mixed in.")
    return focus


def hold_conditions(risk: Risk) -> list[str]:
    holds = [
        "CI failed and the failure is not explained.",
        "The PR scope is unclear or mixes unrelated changes.",
    ]
    if risk >= Risk.MEDIUM:
        holds.append("AI review found a plausible correctness issue that was not fixed or explicitly waived.")
    if risk == Risk.HIGH:
        holds.extend(
            [
                "High-risk behavior changed without a regression test or documented waiver.",
                "Config, checkpoint, dataset, tensor key/order, or numerical compatibility is unclear.",
            ]
        )
    return holds


def render_plan(paths: list[str]) -> str:
    risk, matched, unmatched = classify(paths)
    lines: list[str] = [
        "# DeePTB PR Review Plan",
        "",
        f"Risk level: **{RISK_LABELS[risk]}**",
        f"Changed files: **{len(paths)}**",
        "",
        "## Touched Risk Areas",
    ]

    if matched:
        for rule, files in matched:
            preview = ", ".join(files[:5])
            suffix = "" if len(files) <= 5 else f", ... ({len(files)} files)"
            lines.append(f"- **{rule.name}** ({RISK_LABELS[rule.risk]}): {preview}{suffix}")
    else:
        lines.append("- No mapped DeePTB risk area matched. Review scope manually.")

    if unmatched:
        preview = ", ".join(unmatched[:8])
        suffix = "" if len(unmatched) <= 8 else f", ... ({len(unmatched)} files)"
        lines.append(f"- **Unmapped files**: {preview}{suffix}")

    sections = (
        ("Required Maintainer Focus", maintainer_focus(risk, matched, paths)),
        ("Suggested AI Review", ai_prompt_suggestions(risk)),
        ("Suggested Checks", suggested_checks(risk, paths)),
        ("Hold Conditions To Check", hold_conditions(risk)),
    )
    for title, items in sections:
        lines.extend(["", f"## {title}"])
        lines.extend(f"- {item}" for item in items)

    lines.extend(
        [
            "",
            "## Notes",
            "- This plan is advisory and does not replace maintainer judgment.",
            "- Use the higher risk level when the PR body and changed files disagree.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an advisory DeePTB PR review plan from changed files."
    )
    parser.add_argument(
        "--changed-files",
        nargs="*",
        default=[],
        help="Changed files to classify. May be combined with --stdin.",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read newline-separated changed files from stdin.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = collect_paths(args)
    if not paths:
        print(
            "No changed files provided. Use --changed-files <path...> or pipe paths with --stdin.",
            file=sys.stderr,
        )
        return 2

    print(render_plan(paths), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
