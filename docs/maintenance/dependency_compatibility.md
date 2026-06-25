# Dependency Compatibility Matrix

DeePTB depends on PyTorch, PyG, and the compiled `torch-scatter` extension.
Compatibility testing should therefore verify the exact Python ABI, Torch
version, and PyG binary wheel combination instead of only checking package
version specifiers.

## Local Matrix Tool

Use the matrix runner from the repository root:

```bash
python tools/compat/test_matrix.py --job py312-torch251-cpu
python tools/compat/test_matrix.py --job py312-torch210-cpu
```

The tool creates one isolated uv environment per job under `.compat-envs/`,
installs DeePTB in editable mode, runs an import and `torch_scatter.scatter_add`
smoke check, then runs the default smoke tests:

```bash
python -m pytest dptb/tests -m smoke -q
```

Results are written as JSON files under `tools/compat/results/`.

## Installation Policy

DeePTB now separates the installation surface into two layers:

- `install.sh`: tested installation channel for new machines. It detects or
  accepts the backend, pins the corresponding Torch / PyG / `torch-scatter`
  combination, refuses source builds for critical compiled dependencies, and
  creates a standalone `.venv` under the DeePTB repository.
- `pyproject.toml`: standard Python package contract and developer
  compatibility range. It supports source checkout installs such as
  `pip install .`, `pip install -e .`, `uv sync`, and downstream projects that
  manage their own compatible Torch environment, but it does not encode
  CUDA-driver-specific wheel choices. Published package installs such as
  `pip install dptb` are a separate release-validation path and were not part
  of this compatibility pass.

The tested installer is the recommended standalone DeePTB path. Library or
downstream installs should use standard package installation and ensure the
current environment has a matching PyTorch / `torch-scatter` binary wheel. The
project metadata range is intentionally broader so that intermediate Torch
releases, such as Torch 2.7, can be used when a matching `torch-scatter` binary
wheel is available.

CI should exercise the same installer path as users. Do not run `uv sync` after
`install.sh`, because that re-resolves the environment from `uv.lock` /
`pyproject.toml` and can replace the Torch / PyG / `torch-scatter` combination
selected by the installer. Activate the standalone `.venv` and run normal
commands from that environment. For tests that need optional dependencies, use:

```bash
bash install.sh cpu --extra pythtb
source .venv/bin/activate
python -m pytest ./dptb/tests/
```

The CI runner must also be new enough for the selected binary wheels. The
current CPU installer uses Torch 2.12.1, and the matching Linux
`torch-scatter` wheel requires a newer glibc than the old Ubuntu 20.04-based
test container provides. The unit-test workflow therefore runs on the
`ubuntu-latest` host instead of the legacy `ghcr.io/deepmodeling/deeptb:latest`
container.

Runtime dependencies should avoid exact pins unless the package is tied to a
binary wheel matrix or a known API break. `torch-scatter==2.1.2` is intentionally
fixed because the available PyG wheels define the supported Torch/CUDA
combinations. `pytest` and `pytest-order` are installed by default because new
installations are expected to run the test suite before use.

After relaxing runtime pins, run at least a resolver check and one install/import
check before accepting the change:

```bash
uv lock --dry-run
python tools/compat/test_matrix.py --job py312-torch2121-cpu --no-tests
python tools/compat/test_matrix.py --job py313-torch2121-cpu-modern-cdeps --no-tests
```

## Binary Wheel Rule

`torch-scatter` must be installed from a PyG binary wheel for supported matrix
entries. Do not treat a source build as a supported installation path.

Each job passes a matching PyG wheel page, for example:

```bash
https://data.pyg.org/whl/torch-2.10.0+cpu.html
https://data.pyg.org/whl/torch-2.10.0+cu128.html
```

The runner uses `--only-binary torch-scatter` and fails if the installed
`torch-scatter` distribution metadata is not a platform binary wheel. On Linux
CUDA jobs, the version may include a local suffix such as `2.1.2+pt210cu128`.
On macOS CPU jobs, the wheel can still be binary even when the package version
is just `2.1.2`; inspect the wheel tag instead of relying only on the version
string.

## Suggested Exploration Order

For local macOS CPU checks:

1. `py310-torch251-cpu`: current baseline.
2. `py312-torch251-cpu`: current declared Python upper range with current Torch.
3. `py312-torch210-cpu`: primary Torch upgrade probe.
4. `py313-torch210-cpu`: primary Python and Torch upgrade probe.
5. `py314-torch211-cpu`: experimental only.

For RTX 50 / Blackwell machines, reuse the same runner with CUDA jobs that use
`cu128` or newer PyG wheel pages. The decision order should be:

```text
GPU model -> compute capability -> PyTorch CUDA wheel -> torch-scatter PyG wheel
```

## Tested Support Summary

- macOS CPU: Python 3.12 / 3.13 with Torch 2.10.0 and 2.12.1 passed install,
  import, smoke, regression, and `not slow` pytest checks.
- RTX 5090 with driver 570.211.01 / CUDA 12.8: Python 3.12 / 3.13 with
  `torch 2.10.0+cu128`, `torch-geometric 2.8.0`, and
  `torch-scatter 2.1.2+pt210cu128` passed install, import, smoke, regression,
  and `not slow` pytest checks.
- RTX 5090 with the current driver cannot use `torch 2.12.x+cu130`; the
  wheels install, but CUDA is unavailable because the driver is too old for the
  CUDA 13 runtime.
- `cu132` is not exposed by uv/PyTorch as a Torch backend at the time of this
  test, so the installer does not advertise it as a supported path.
- `torch 2.12.x+cu128` is not a supported 5090 path because the PyG index does
  not provide a matching `torch-scatter` binary wheel.
- Python 3.14 remains experimental. Torch warns that `torch.jit.script` is not
  supported there, so Python 3.14 needs a separate TorchScript migration task.
