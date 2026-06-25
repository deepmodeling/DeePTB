"""Install torch-scatter for the currently installed PyTorch.

This helper is for library/downstream installs where the environment already
owns its PyTorch stack. For standalone DeePTB installs, prefer ``install.sh``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from importlib import metadata


SCATTER_VERSION = "2.1.2"


def normalize_torch_version(version: str) -> tuple[str, str]:
    match = re.match(r"^(\d+\.\d+\.\d+)(?:\+([A-Za-z0-9_]+))?", version)
    if not match:
        raise SystemExit(f"ERROR: cannot parse torch version: {version}")

    base_version = match.group(1)
    backend = match.group(2) or "cpu"
    if backend.startswith("cu"):
        return base_version, backend
    return base_version, "cpu"


def scatter_requirement(torch_base: str, backend: str) -> str:
    if backend == "cpu":
        return f"torch-scatter=={SCATTER_VERSION}"

    major, minor, _patch = torch_base.split(".")
    return f"torch-scatter=={SCATTER_VERSION}+pt{major}{minor}{backend}"


def installed_scatter_version() -> str | None:
    try:
        return metadata.version("torch-scatter")
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Install a torch-scatter binary wheel matching the current torch "
            "version and CPU/CUDA backend."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pip command without installing anything.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reinstall torch-scatter even if it is already importable.",
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "ERROR: torch is not installed in this environment. Install the "
            "PyTorch build required by your project first, then rerun this "
            "helper."
        ) from exc

    torch_base, backend = normalize_torch_version(torch.__version__)
    find_links = f"https://data.pyg.org/whl/torch-{torch_base}+{backend}.html"
    requirement = scatter_requirement(torch_base, backend)

    current_scatter = installed_scatter_version()
    print("Detected environment:")
    print(f"  Python:          {sys.version.split()[0]}")
    print(f"  torch:           {torch.__version__}")
    print(f"  CUDA runtime:    {torch.version.cuda}")
    print(f"  CUDA available:  {torch.cuda.is_available()}")
    print(f"  PyG wheel index: {find_links}")
    print(f"  torch-scatter:   {requirement}")

    if current_scatter and not args.force:
        required_version = requirement.split("==", 1)[1].lower()
        if current_scatter.lower() == required_version:
            print(f"torch-scatter is already installed: {current_scatter}")
            return
        print(f"torch-scatter is already installed: {current_scatter}")
        print(f"The detected torch backend expects: {required_version}")
        print("Use --force to reinstall it for the detected torch backend.")
        raise SystemExit(1)

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--only-binary",
        "torch-scatter",
        requirement,
        "-f",
        find_links,
    ]

    print("Command:")
    print("  " + " ".join(command))
    if args.dry_run:
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
