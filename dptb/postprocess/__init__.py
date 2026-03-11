from typing import TYPE_CHECKING

# 定义模块映射关系
_import_structure = {
    "bandstructure": ["Band"],
    "totbplas": ["TBPLaS"],
    "write_block": ["write_block"],
    "write_abacus_csr_file": ["write_blocks_to_abacus_csr"],
}

# 仅在类型检查期间导入（IDE补全用），运行时不执行
if TYPE_CHECKING:
    from .bandstructure import Band
    from .totbplas import TBPLaS
    from .write_block import write_block
    from .write_abacus_csr_file import write_blocks_to_abacus_csr

# 核心逻辑：定义 __all__ 和 __getattr__
__all__ = [
    "Band",
    "TBPLaS",
    "write_block",
    "write_blocks_to_abacus_csr"
]


def __getattr__(name):
    import importlib

    # 查找 name 属于哪个子模块
    for module, items in _import_structure.items():
        if name in items:
            # 动态导入子模块
            mod = importlib.import_module(f".{module}", __package__)
            return getattr(mod, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")