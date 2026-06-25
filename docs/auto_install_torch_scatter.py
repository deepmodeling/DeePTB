"""Deprecated torch-scatter helper.

Use the repository-level install.sh instead. It selects the tested PyTorch,
PyG, and torch-scatter binary-wheel combination for the requested backend.
"""

from __future__ import annotations


def main() -> None:
    print("This helper is deprecated.")
    print("Use one of the tested installer commands from the repository root:")
    print("  bash install.sh cpu")
    print("  bash install.sh gpu")
    print("  bash install.sh cu128")


if __name__ == "__main__":
    main()
