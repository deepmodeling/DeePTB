"""Deprecated torch-scatter helper.

Use the repository-level install.sh instead. It selects the tested PyTorch,
PyG, and torch-scatter binary-wheel combination for the requested backend.
"""

from __future__ import annotations


def main() -> None:
    print("This helper is deprecated.")
    print(
        "It no longer installs torch-scatter directly because torch-scatter must "
        "match the selected PyTorch, CUDA backend, and PyG wheel index."
    )
    print(
        "此脚本不再单独安装 torch-scatter；torch-scatter 必须和 PyTorch 版本、"
        "CUDA 后端、PyG wheel index 成组匹配。"
    )
    print("Use one of the tested installer commands from the repository root:")
    print("请在仓库根目录使用下面任一测试过的安装命令：")
    print("  bash install.sh cpu")
    print("  bash install.sh gpu")
    print("  bash install.sh cu128")


if __name__ == "__main__":
    main()
