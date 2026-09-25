"""稀疏体数据的游程连通组件（run-length connected components）。

公开接口见模块顶层文档与 README。
"""

from .core import (
    BoundingBox,
    ClosedComponent,
    Component,
    InvalidVolumeError,
    LabelResult,
    Run,
    SliceResult,
    StreamLabeler,
    StreamResult,
    label,
)

__all__ = [
    "BoundingBox",
    "ClosedComponent",
    "Component",
    "InvalidVolumeError",
    "LabelResult",
    "Run",
    "SliceResult",
    "StreamLabeler",
    "StreamResult",
    "label",
]
