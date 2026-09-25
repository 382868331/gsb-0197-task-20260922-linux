"""稀疏体数据的游程连通组件（run-length connected components）。

公开接口见模块顶层文档与 README。
"""

from .core import (
    BoundingBox,
    Component,
    ClosedComponent,
    FinishResult,
    InvalidVolumeError,
    LabelResult,
    Run,
    SliceResult,
    StreamLabeler,
    label,
)

__all__ = [
    "BoundingBox",
    "Component",
    "ClosedComponent",
    "FinishResult",
    "InvalidVolumeError",
    "LabelResult",
    "Run",
    "SliceResult",
    "StreamLabeler",
    "label",
]
