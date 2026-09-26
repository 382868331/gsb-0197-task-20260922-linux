"""稀疏体数据的游程连通组件（run-length connected components）。

公开接口见模块顶层文档与 README。
"""

from .cavities import Cavity, CavityResult, find_cavities
from .core import (
    BoundingBox,
    Component,
    InvalidVolumeError,
    LabelResult,
    Run,
    label,
)

__all__ = [
    "BoundingBox",
    "Cavity",
    "CavityResult",
    "Component",
    "InvalidVolumeError",
    "LabelResult",
    "Run",
    "find_cavities",
    "label",
]
