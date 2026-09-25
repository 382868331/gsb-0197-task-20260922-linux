"""稀疏体数据的游程连通组件（run-length connected components）。

公开接口见模块顶层文档与 README。
"""

from .core import (
    BoundingBox,
    Component,
    ContactPair,
    InvalidVolumeError,
    LabelResult,
    MaterialComponent,
    MaterialLabelResult,
    MaterialRun,
    Run,
    label,
    label_materials,
)

__all__ = [
    "BoundingBox",
    "Component",
    "ContactPair",
    "InvalidVolumeError",
    "LabelResult",
    "MaterialComponent",
    "MaterialLabelResult",
    "MaterialRun",
    "Run",
    "label",
    "label_materials",
]
