#!/usr/bin/env bash
# DeePTB tested one-command installer.
#
# This script is the conservative installation path for new machines. It pins a
# tested Torch / PyG / torch-scatter combination for the selected backend. The
# broader source-compatibility ranges live in pyproject.toml for developers who
# intentionally manage their own environments.
#
# Usage:
#   ./install.sh              # auto: GPU if available, otherwise CPU
#   ./install.sh auto         # same as default
#   ./install.sh cpu          # force CPU
#   ./install.sh gpu          # auto-detect CUDA backend from nvidia-smi
#   ./install.sh cu128        # force CUDA 12.8 wheel path
#   ./install.sh cu130        # force CUDA 13.0 wheel path
#   ./install.sh cpu --extra pythtb
#
# Optional:
#   PYTHON_BIN=/path/to/python ./install.sh auto

set -euo pipefail

REQUESTED_BACKEND="auto"
if [[ "$#" -gt 0 && "$1" != --* ]]; then
    REQUESTED_BACKEND="$1"
    shift
fi

extras=()
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --extra)
            if [[ "$#" -lt 2 || "$2" == --* ]]; then
                echo "ERROR: --extra requires an extra name."
                exit 1
            fi
            extras+=("$2")
            shift 2
            ;;
        -h|--help)
            sed -n '1,20p' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: unknown option '$1'."
            echo "Allowed options: --extra <name>"
            exit 1
            ;;
    esac
done

if [[ -z "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="$(command -v python || command -v python3 || true)"
fi

if [[ -z "${PYTHON_BIN}" ]]; then
    echo "ERROR: python was not found. Set PYTHON_BIN=/path/to/python."
    exit 1
fi

python_version_full="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
python_major_minor="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

case "${python_major_minor}" in
    3.10|3.11|3.12|3.13)
        ;;
    *)
        echo "ERROR: DeePTB tested installer supports Python 3.10-3.13."
        echo "Detected Python: ${python_version_full} (${PYTHON_BIN})"
        echo "Python 3.14 needs a separate TorchScript migration task."
        exit 1
        ;;
esac

install_uv_if_needed() {
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv not found. Installing uv with ${PYTHON_BIN} -m pip ..."
        "${PYTHON_BIN}" -m pip install uv
    fi
}

detect_cuda_version() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo ""
        return
    fi
    nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' | head -1
}

cuda_to_backend() {
    local cuda_version="$1"
    if [[ -z "${cuda_version}" ]]; then
        echo "cpu"
        return
    fi
    "${PYTHON_BIN}" - "${cuda_version}" <<'PY'
import sys

parts = sys.argv[1].split(".")
major = int(parts[0])
minor = int(parts[1]) if len(parts) > 1 else 0
cuda = (major, minor)

if cuda >= (13, 0):
    print("cu130")
elif cuda >= (12, 8):
    print("cu128")
elif cuda >= (12, 4):
    print("cu124")
elif cuda >= (12, 1):
    print("cu121")
elif cuda >= (11, 8):
    print("cu118")
else:
    print("unsupported")
PY
}

select_backend() {
    local requested="$1"
    case "${requested}" in
        auto)
            local cuda_version
            cuda_version="$(detect_cuda_version)"
            cuda_to_backend "${cuda_version}"
            ;;
        gpu)
            local cuda_version
            cuda_version="$(detect_cuda_version)"
            if [[ -z "${cuda_version}" ]]; then
                echo "ERROR: requested GPU install, but nvidia-smi was not found or CUDA could not be detected." >&2
                exit 1
            fi
            cuda_to_backend "${cuda_version}"
            ;;
        cpu|cu118|cu121|cu124|cu128|cu130)
            echo "${requested}"
            ;;
        cu132)
            echo "ERROR: cu132 is not supported by uv/PyTorch as a torch backend yet." >&2
            echo "Use cu130 on CUDA 13.x drivers until a tested PyTorch cu132 backend exists." >&2
            exit 1
            ;;
        *)
            echo "ERROR: invalid backend '${requested}'." >&2
            echo "Allowed: auto, cpu, gpu, cu118, cu121, cu124, cu128, cu130" >&2
            exit 1
            ;;
    esac
}

backend="$(select_backend "${REQUESTED_BACKEND}")"
if [[ "${backend}" == "unsupported" ]]; then
    echo "ERROR: detected CUDA is older than 11.8, which is not supported by this installer."
    exit 1
fi

torch_version=""
pyg_index_torch_version=""
torch_scatter_pin="2.1.2"
torch_geometric_pin=">=2.8.0"

case "${backend}" in
    cpu)
        torch_version="2.12.1"
        pyg_index_torch_version="2.12.1"
        torch_scatter_pin="2.1.2"
        ;;
    cu128)
        # Verified on RTX 5090 with driver 570 / CUDA 12.8.
        # torch-scatter has cu128 binary wheels for torch 2.10, but not torch 2.12.
        torch_version="2.10.0"
        pyg_index_torch_version="2.10.0"
        torch_scatter_pin="2.1.2+pt210cu128"
        ;;
    cu130)
        # Requires a driver new enough for CUDA 13.0 runtime.
        torch_version="2.12.1"
        pyg_index_torch_version="2.12.1"
        torch_scatter_pin="2.1.2+pt212cu130"
        ;;
    cu118|cu121|cu124)
        # Legacy CUDA paths retained for older clusters.
        torch_version="2.5.1"
        pyg_index_torch_version="2.5.0"
        torch_geometric_pin=">=2.7.0"
        torch_scatter_pin="2.1.2+pt25${backend}"
        ;;
esac

find_links_url="https://data.pyg.org/whl/torch-${pyg_index_torch_version}+${backend}.html"

override_file="$(mktemp "${TMPDIR:-/tmp}/deeptb-install-overrides.XXXXXX")"
trap 'rm -f "${override_file}"' EXIT

{
    echo "torch==${torch_version}"
    echo "torch-scatter==${torch_scatter_pin}"
    echo "torch-geometric${torch_geometric_pin}"
    if [[ "${python_major_minor}" == "3.13" ]]; then
        echo "lmdb>=2.2.1"
        echo "h5py>=3.16.0"
        echo "numpy>=2.5.0,<3"
        echo "scipy>=1.18.0"
    fi
} > "${override_file}"

echo "======================================"
echo "DeePTB installer"
echo "======================================"
echo "Python:        ${python_version_full} (${PYTHON_BIN})"
echo "Requested:     ${REQUESTED_BACKEND}"
echo "Selected:      ${backend}"
echo "Torch:         ${torch_version}"
echo "torch-scatter: ${torch_scatter_pin}"
echo "PyG index:     ${find_links_url}"
if [[ "${python_major_minor}" == "3.13" ]]; then
    echo "C extensions:  modern pins for Python 3.13"
fi
if [[ "${#extras[@]}" -gt 0 ]]; then
    echo "Extras:        ${extras[*]}"
fi
echo "======================================"

install_uv_if_needed

uv venv --python "${PYTHON_BIN}" .venv

project_spec="."
if [[ "${#extras[@]}" -gt 0 ]]; then
    IFS=,
    project_spec=".[${extras[*]}]"
    unset IFS
fi

uv pip install \
    --python .venv/bin/python \
    --overrides "${override_file}" \
    --find-links "${find_links_url}" \
    --torch-backend "${backend}" \
    --only-binary torch-scatter \
    --only-binary lmdb \
    --only-binary h5py \
    --only-binary numpy \
    --only-binary scipy \
    -e "${project_spec}"

DEEPTB_SELECTED_BACKEND="${backend}" .venv/bin/python - <<'PY'
import os
import sys
from importlib import metadata

import torch
import torch_scatter
import torch_geometric

dist = metadata.distribution("torch-scatter")
wheel = dist.read_text("WHEEL") or ""
if "Root-Is-Purelib: false" not in wheel or "Tag:" not in wheel:
    raise SystemExit("ERROR: torch-scatter is not installed as a platform binary wheel.")

print("Installed versions:")
print(f"  Python:          {sys.version.split()[0]}")
print(f"  torch:           {torch.__version__}")
print(f"  torch_geometric: {torch_geometric.__version__}")
print(f"  torch_scatter:   {torch_scatter.__version__}")
print(f"  CUDA available:  {torch.cuda.is_available()}")
print(f"  CUDA runtime:    {torch.version.cuda}")

backend = os.environ["DEEPTB_SELECTED_BACKEND"]
if backend != "cpu" and not torch.cuda.is_available():
    raise SystemExit(
        "ERROR: GPU backend was selected, but torch.cuda.is_available() is false. "
        "Use a backend matching the installed NVIDIA driver, or update the driver."
    )
PY

echo ""
echo "Installation complete."
echo "This standalone environment lives in: .venv"
echo "Activate it:"
echo "  source .venv/bin/activate"
echo "Then run:"
echo "  dptb --help"
echo "  python -m pytest ./dptb/tests/"
