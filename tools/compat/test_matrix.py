#!/usr/bin/env python3
"""Run DeePTB dependency compatibility jobs in isolated uv environments."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = ROOT / "tools" / "compat" / "matrix.json"
DEFAULT_ENVS = ROOT / ".compat-envs"
DEFAULT_RESULTS = ROOT / "tools" / "compat" / "results"


def run(cmd: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> dict:
    started = time.time()
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.time() - started
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "seconds": round(elapsed, 3),
        "output": proc.stdout,
    }


def fail_on(step: dict) -> None:
    if step["returncode"] != 0:
        raise RuntimeError(f"command failed with exit code {step['returncode']}")


def python_bin(env_dir: Path) -> Path:
    if platform.system() == "Windows":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def load_matrix(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def selected_jobs(matrix: dict, names: list[str]) -> list[dict]:
    jobs = matrix["jobs"]
    if not names:
        return jobs
    by_name = {job["name"]: job for job in jobs}
    missing = sorted(set(names) - set(by_name))
    if missing:
        raise SystemExit(f"unknown job(s): {', '.join(missing)}")
    return [by_name[name] for name in names]


def scatter_version(torch_version: str, torch_backend: str) -> str:
    """Construct torch-scatter local version suffix from torch version and backend."""
    if torch_backend == "cpu":
        return "2.1.2"
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", torch_version)
    if not m:
        raise ValueError(f"Cannot parse torch version: {torch_version}")
    pt = f"pt{m.group(1)}{m.group(2)}"
    return f"2.1.2+{pt}{torch_backend}"


def make_override_file(job: dict, work_dir: Path) -> Path:
    override = work_dir / "overrides.txt"
    backend = job.get("torch_backend", "cpu")
    scat_ver = scatter_version(job["torch"], backend)
    lines = [
        f"torch=={job['torch']}",
        f"torch-scatter=={scat_ver}",
    ]
    if job.get("torch_geometric"):
        lines.append(f"torch-geometric{job['torch_geometric']}")
    lines.extend(job.get("extra_overrides", []))
    override.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return override


def version_probe_code() -> str:
    return r"""
import importlib
import json
import platform
import sys
from importlib import metadata

packages = [
    "dptb",
    "torch",
    "torch-geometric",
    "torch-scatter",
    "numpy",
    "scipy",
    "e3nn",
    "h5py",
    "pytest",
]

result = {
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "packages": {},
}

for package in packages:
    try:
        dist = metadata.distribution(package)
        result["packages"][package] = {
            "version": dist.version,
            "location": str(dist.locate_file("")),
            "installer": dist.read_text("INSTALLER"),
            "wheel": dist.read_text("WHEEL"),
        }
    except metadata.PackageNotFoundError:
        result["packages"][package] = {"missing": True}

for module in ["torch", "torch_geometric", "torch_scatter"]:
    try:
        imported = importlib.import_module(module)
        result[module] = {"version": getattr(imported, "__version__", None)}
    except Exception as exc:
        result[module] = {"error": f"{type(exc).__name__}: {exc}"}

try:
    import torch
    result["torch_backend"] = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "mps_built": torch.backends.mps.is_built() if hasattr(torch.backends, "mps") else None,
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else None,
        "compile_available": hasattr(torch, "compile"),
    }
except Exception as exc:
    result["torch_backend_error"] = f"{type(exc).__name__}: {exc}"

print(json.dumps(result, indent=2, sort_keys=True))
"""


def smoke_probe_code() -> str:
    return r"""
import torch
import torch_scatter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(16, device=device)
index = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3, 0, 1, 2, 3], device=device)
out = torch_scatter.scatter_add(x, index)
assert out.shape == (4,)

import dptb
import dptb.data.AtomicData
import dptb.nn.build

print("scatter_add", out.shape, out.dtype, out.device)
print("dptb", getattr(dptb, "__version__", "unknown"))
"""


def is_binary_wheel(package_info: dict) -> bool:
    wheel = package_info.get("wheel") or ""
    return "Root-Is-Purelib: false" in wheel and "Tag:" in wheel


def parse_probe_json(output: str) -> dict:
    """Parse a JSON object from probe output that may include warnings."""
    start = output.find("{")
    end = output.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("probe output did not contain a JSON object")
    return json.loads(output[start : end + 1])


def run_job(job: dict, matrix: dict, args: argparse.Namespace) -> dict:
    env_dir = args.env_root / job["name"]
    result_dir = args.results_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    work_dir = result_dir / f".{job['name']}"
    work_dir.mkdir(parents=True, exist_ok=True)

    py = python_bin(env_dir)
    result = {
        "job": job,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "host_python": sys.version,
        "steps": [],
        "status": "running",
    }
    child_env = os.environ.copy()
    if job.get("mkl_feast"):
        child_env["DPTB_TEST_MKL_FEAST"] = "1"
    else:
        child_env.pop("DPTB_TEST_MKL_FEAST", None)

    try:
        if not args.skip_install:
            step = run(["uv", "venv", "--clear", "--python", job["python"], str(env_dir)], env=child_env)
            result["steps"].append({"name": "venv", **step})
            fail_on(step)

            override = make_override_file(job, work_dir)
            install_cmd = [
                "uv",
                "pip",
                "install",
                "--python",
                str(py),
                "--overrides",
                str(override),
                "--find-links",
                job["pyg_find_links"],
                "--torch-backend",
                job.get("torch_backend", "cpu"),
                "--only-binary",
                "torch-scatter",
                "-e",
                ".",
            ]
            step = run(install_cmd, env=child_env)
            result["steps"].append({"name": "install", **step})
            fail_on(step)

        step = run([str(py), "-c", version_probe_code()], env=child_env)
        result["steps"].append({"name": "version-probe", **step})
        fail_on(step)
        result["version_probe"] = parse_probe_json(step["output"])
        backend = job.get("torch_backend", "cpu")
        torch_backend = result["version_probe"].get("torch_backend", {})
        if backend != "cpu" and not torch_backend.get("cuda_available"):
            raise RuntimeError(
                f"{job['name']} installed {backend} wheels, but torch.cuda.is_available() is false"
            )

        scatter_installed_version = (
            result["version_probe"]
            .get("packages", {})
            .get("torch-scatter", {})
            .get("version", "")
        )
        scatter_info = (
            result["version_probe"]
            .get("packages", {})
            .get("torch-scatter", {})
        )
        if not is_binary_wheel(scatter_info):
            raise RuntimeError(
                "torch-scatter was not installed from a platform binary wheel"
            )
        result["torch_scatter_binary_wheel"] = {
            "version": scatter_installed_version,
            "wheel": scatter_info.get("wheel"),
        }

        step = run([str(py), "-c", smoke_probe_code()], env=child_env)
        result["steps"].append({"name": "import-and-scatter-smoke", **step})
        fail_on(step)

        if args.no_tests:
            test_command = []
        elif args.test_command is None:
            test_command = matrix["default_test_command"]
        elif args.test_command == [""]:
            test_command = []
        else:
            test_command = args.test_command
        if test_command:
            cmd = [str(py) if part == "python" else part for part in test_command]
            step = run(cmd, env=child_env)
            result["steps"].append({"name": "tests", **step})
            fail_on(step)

        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        out = result_dir / f"{job['name']}.json"
        out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {out}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--job", action="append", default=[], help="Run one named job.")
    parser.add_argument("--env-root", type=Path, default=DEFAULT_ENVS)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--no-tests", action="store_true", help="Skip pytest/test command execution.")
    parser.add_argument(
        "--test-command",
        nargs=argparse.REMAINDER,
        help="Override the matrix test command. Use --test-command '' to skip.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.test_command == [""]:
        args.test_command = []
    args.env_root = args.env_root.resolve()
    args.results_dir = args.results_dir.resolve()
    matrix = load_matrix(args.matrix)
    jobs = selected_jobs(matrix, args.job)
    args.env_root.mkdir(parents=True, exist_ok=True)

    results = [run_job(job, matrix, args) for job in jobs]
    failed = [result["job"]["name"] for result in results if result["status"] != "passed"]
    print()
    print("Compatibility matrix summary")
    for result in results:
        print(f"- {result['job']['name']}: {result['status']}")
    if failed:
        print(f"failed jobs: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
