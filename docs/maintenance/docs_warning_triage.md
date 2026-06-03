# Docs Warning Triage

This page tracks known Sphinx warning classes so documentation review can focus on meaningful regressions instead of rediscovering existing noise.

## Local Check

```bash
uv run python -m sphinx -b html docs /private/tmp/deeptb-docs-check
```

## Warning Classes

| Warning class | Current policy |
| --- | --- |
| Missing toctree entries | Fix low-risk documentation pages when they are already part of the public docs. Keep design notes unlisted unless they are ready for public navigation. |
| Image path warnings | Fix direct path mistakes. Prefer paths that resolve from the Markdown file location. |
| Markdown heading warnings | Fix obvious cases where a Markdown document starts at `##` instead of `#`. |
| Code block highlighting warnings | Track only for now. Many examples contain JSON-like snippets with comments or ellipses; fixing them requires broader historical docs cleanup. |
| Notebook execution or output warnings | Track only for now. Do not change notebook contents or execution policy in maintenance-only PRs. |
| Third-party Sphinx deprecation warnings | Track only for now unless they break the build. |

## Fixed In This Maintenance Pass

- Added low-risk public pages to toctrees for electronic properties, community docs, and quick examples.
- Fixed the smoothing-function image path in `docs/advanced/sktb/dptb_env.md`.
- Converted `docs/pardiso_architecture.md` to start with a real H1 Markdown heading.

## Last Observed Full Build

Command:

```bash
uv run python -m sphinx -b html docs /private/tmp/deeptb-docs-check
```

Latest maintenance validation succeeded with 41 warnings. The warnings are still in the tracked classes above:

- missing `_static` directory from `html_static_path`;
- notebook execution failures for quick-start notebooks, with tracebacks written under the build `reports/` directory;
- unlisted design/status notes: `docs/pardiso_architecture.md` and `docs/phase1_summary.md`;
- definition-list formatting in `docs/input_params/run_options.rst`;
- historical code block highlighting warnings from JSON-like snippets with comments, ellipses, tree glyphs, traceback text, or non-ASCII math symbols;
- missing MyST cross references in `docs/phase1_summary.md` and `docs/quick_start/hands_on/tutorial4_E3_si.ipynb`;
- `deepmodeling_sphinx` deprecation warnings for Sphinx 9 APIs.

This maintenance pass does not attempt to execute or repair notebooks, rewrite historical code snippets, or promote design/status notes into public navigation.

## Known Warnings Left For Later

- Code block highlighting warnings in historical docs that use JSON-like examples with comments or ellipses.
- Notebook execution or notebook output warnings.
- Unlisted design/status notes such as `docs/phase1_summary.md`.
- Sphinx/deepmodeling_sphinx deprecation or static-copy warnings that do not fail the build.
