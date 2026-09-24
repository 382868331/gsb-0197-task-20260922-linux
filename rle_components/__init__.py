"""稀疏体数据的游程连通组件（run-length connected components）。

公开接口见模块顶层文档与 README：

* :func:`label`：无状态一次性标记（首轮能力，保持不变）；
* :class:`ComponentModel`：有状态模型，支持批量增删、删除桥接后
  重算连通性与 split/merge 事件（见 ``rle_components.dynamic``）。
"""

from .core import (
    BoundingBox,
    Component,
    InvalidVolumeError,
    LabelResult,
    Run,
    label,
)
from .dynamic import (
    ComponentModel,
    ComponentSnapshot,
    MergeEvent,
    SplitEvent,
    UpdateResult,
)

__all__ = [
    "BoundingBox",
    "Component",
    "ComponentModel",
    "ComponentSnapshot",
    "InvalidVolumeError",
    "LabelResult",
    "MergeEvent",
    "Run",
    "SplitEvent",
    "UpdateResult",
    "label",
]
