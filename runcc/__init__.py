"""稀疏二值体的游程连通组件标记。

公开接口见 :func:`runcc.core.label_volume`。仅使用 Python 标准库。
"""

from .core import (
    CONNECTIVITIES,
    MAX_DIM,
    MAX_RUNS,
    BoundingBox,
    Component,
    LabelingResult,
    Run,
    label_volume,
)

label = label_volume

__all__ = [
    "MAX_DIM",
    "MAX_RUNS",
    "CONNECTIVITIES",
    "Run",
    "BoundingBox",
    "Component",
    "LabelingResult",
    "label_volume",
    "label",
]
