#!/usr/bin/env python3
"""
Robust installer for PyG binary dependencies:
  torch-scatter, torch-sparse, torch-cluster, torch-spline-conv

Usage: run with the Python interpreter you want these packages installed into:
  python install_pyg_binaries.py
"""
import subprocess
import sys
import importlib
import traceback

def is_importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False

def format_torch_base_version(torch_ver: str) -> str:
    # remove +cuXXX or other build tags: "2.4.1+cu118" -> "2.4.1"
    return torch_ver.split('+')[0]

def format_cuda_tag(cuda_ver: str) -> str:
    # torch.version.cuda usually like "11.8" or None
    if not cuda_ver:
        return "cpu"
    parts = cuda_ver.split('.')
    if len(parts) >= 2:
        return "cu" + parts[0] + parts[1]
    else:
        return "cu" + parts[0]

def try_pip_install(packages, url):
    cmd = [sys.executable, "-m", "pip", "install", *packages, "-f", url]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"pip install failed for url: {url}")
        print("Return code:", e.returncode)
        return False
    except Exception as e:
        print("Unexpected error while running pip:", e)
        return False

def main():
    try:
        import torch
    except ImportError:
        print("The torch module is not found. Please install a compatible PyTorch first.")
        print('Example: pip install "torch>=2.0.0,<=2.5.1"  (or visit https://pytorch.org for the recommended command for your CUDA).')
        return

    torch_ver_raw = torch.__version__
    torch_base = format_torch_base_version(torch_ver_raw)
    cuda_raw = torch.version.cuda  # e.g. "11.8" or None
    cuda_tag = format_cuda_tag(cuda_raw)

    print(f"Detected torch: {torch_ver_raw} -> using base '{torch_base}'")
    print(f"Detected CUDA for torch: {cuda_raw} -> using wheel tag '{cuda_tag}'")

    # packages: pip package name -> Python import name
    packages_map = {
        "torch-scatter": "torch_scatter",
        "torch-sparse": "torch_sparse",
        "torch-cluster": "torch_cluster",
        "torch-spline-conv": "torch_spline_conv",
    }

    missing_pkg_names = [pkg for pkg, imp in packages_map.items() if not is_importable(imp)]
    if not missing_pkg_names:
        print("All target PyG binary dependencies already import successfully:")
        for pkg, imp in packages_map.items():
            print(f"  - {pkg} (import as {imp}) => OK")
        print("If you want, you can still install/update torch-geometric: pip install torch-geometric")
        return

    print("Missing packages (to be installed):", missing_pkg_names)

    # Build candidate wheel-list URLs to try (order matters: most specific first)
    candidates = []
    # 1) exact base version with cuda tag
    candidates.append(f"https://data.pyg.org/whl/torch-{torch_base}+{cuda_tag}.html")
    # 2) try major.minor.0 with cuda tag (some wheels published per minor release)
    try:
        major, minor, _ = (torch_base.split('.') + ["0", "0"])[:3]
        mm0 = f"{major}.{minor}.0"
        if mm0 != torch_base:
            candidates.append(f"https://data.pyg.org/whl/torch-{mm0}+{cuda_tag}.html")
    except Exception:
        pass
    # 3) fallback without a cuda suffix (older naming / sometimes available)
    candidates.append(f"https://data.pyg.org/whl/torch-{torch_base}.html")
    # 4) final fallback: cpu-only page (if CUDA page fails, try cpu)
    if cuda_tag != "cpu":
        candidates.append(f"https://data.pyg.org/whl/torch-{torch_base}+cpu.html")

    print("Will try the following wheel index pages (in order):")
    for c in candidates:
        print("  ", c)

    success = False
    for url in candidates:
        print("\nAttempting pip install from:", url)
        ok = try_pip_install(missing_pkg_names, url)
        # re-check imports on success
        if ok:
            still_missing = [pkg for pkg, imp in packages_map.items() if not is_importable(imp)]
            if not still_missing:
                print("All packages installed and importable.")
                success = True
                break
            else:
                print("After install attempt, these are still missing:", still_missing)
                # try next candidate if any
        else:
            print("Install attempt failed for", url)
    if not success:
        print("\nAutomatic installation attempts did not succeed for all packages.")
        print("Suggestions:")
        print("  1) Visit the PyG installation guide to find the exact wheel URL that matches your PyTorch version and CUDA: https://pytorch-geometric.readthedocs.io")
        print("  2) If you are using a nightly/dev PyTorch build, prebuilt binaries may not be available; consider installing a stable PyTorch or building these packages from source.")
        print("  3) You can try installing packages individually to see detailed build logs, e.g.:")
        print(f"       {sys.executable} -m pip install torch-sparse -f https://data.pyg.org/whl/torch-{torch_base}+{cuda_tag}.html")
        print("  4) As a last resort, build from source (watch for compilers and CUDA toolkit).")
        print("\nIf you want, paste the full error output here and I can help interpret it.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("Unexpected exception:")
        traceback.print_exc()
        sys.exit(1)
