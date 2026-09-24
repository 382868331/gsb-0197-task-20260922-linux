"""有状态体素体：批量增删 + 组件分裂/合并事件。

在 :mod:`rle_components.core` 的无状态纯函数 ``label`` 之上提供一个可原地
更新的体素体 :class:`VoxelVolume`：

* 内部仍以按行有序半开 x 游程存储（与 ``label`` 输入同构），不展开体素；
* ``add_voxels`` / ``remove_voxels`` / ``apply`` 接受一批 ``(z, y, x)``
  体素坐标，校验全部通过后一次性应用，随后**全量重算**连通性
  （组件 ID 仍按组件最小体素坐标 (z, y, x) 字典序分配）；
* 每次成功更新返回 :class:`UpdateResult`，内含新的 ``LabelResult`` 与
  本次更新引起的组件事件（``split`` / ``merge`` / ``created`` /
  ``removed``）。事件通过新旧游程的区间求交在体素级建立
  “旧组件 → 新组件”的流向得到，与 ID 重编号无关。

原子性约定：任何非法输入（坐标类型/越界、同一批中同时增删同一体素、
结果游程数超过 20000 等）都在应用前抛出 ``InvalidVolumeError``，
体素体保持调用前状态，不存在半更新。
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import inf
from typing import NamedTuple

from .core import (
    _VALID_CONNECTIVITY,
    MAX_RUNS,
    InvalidVolumeError,
    LabelResult,
    Run,
    _check_dim,
    _normalize_runs,
    label,
)

__all__ = [
    "ComponentEvent",
    "UpdateResult",
    "VoxelVolume",
]


class ComponentEvent(NamedTuple):
    """一次批量更新引起的组件级事件。

    * ``kind="split"``：旧组件 ``old_ids[0]`` 的体素分散到多个新组件
      ``new_ids``；
    * ``kind="merge"``：多个旧组件 ``old_ids`` 的体素汇入新组件
      ``new_ids[0]``；
    * ``kind="created"``：新组件没有任何旧组件体素（全新出现）；
    * ``kind="removed"``：旧组件的体素全部被删除。

    事件按体素集合重叠定义，与组件 ID 重编号无关：每次更新后 ID 都按
    新全量结果的最小体素坐标重新分配，未增删体素的组件若仅 ID 变化
    不产生事件。
    """

    kind: str
    old_ids: tuple[int, ...]
    new_ids: tuple[int, ...]


@dataclass(frozen=True)
class UpdateResult:
    """一次成功批量更新的返回值。

    Attributes:
        result: 更新后的全量 :class:`LabelResult`。
        events: 本次更新引起的组件事件，顺序固定为
            split（按旧 ID 升序）、merge（按新 ID 升序）、
            created（按新 ID 升序）、removed（按旧 ID 升序）。
        added: 本批实际新增的体素数（已存在的体素为无操作，不计入）。
        removed: 本批实际删除的体素数（不存在的体素为无操作，不计入）。
    """

    result: LabelResult
    events: tuple[ComponentEvent, ...]
    added: int
    removed: int


# ---------------------------------------------------------------------------
# 行内游程的单体素增删（行始终保持 start 升序、互不重叠）
# ---------------------------------------------------------------------------


def _add_to_row(row: list[tuple[int, int]], x: int) -> bool:
    """向有序游程行加入体素 x；返回是否真正改变了行。"""
    i = bisect_left(row, (x + 1, 0)) - 1  # 最后一条 start <= x 的游程
    if i >= 0 and row[i][1] > x:
        return False  # 已覆盖：无操作
    merge_left = i >= 0 and row[i][1] == x
    j = i + 1
    merge_right = j < len(row) and row[j][0] == x + 1
    if merge_left and merge_right:
        row[i] = (row[i][0], row[j][1])
        del row[j]
    elif merge_left:
        row[i] = (row[i][0], x + 1)
    elif merge_right:
        row[j] = (x, row[j][1])
    else:
        row.insert(j, (x, x + 1))
    return True


def _remove_from_row(row: list[tuple[int, int]], x: int) -> bool:
    """从有序游程行删除体素 x；返回是否真正改变了行。"""
    i = bisect_left(row, (x + 1, 0)) - 1
    if i < 0 or row[i][1] <= x:
        return False  # 未覆盖：无操作
    s, e = row[i]
    if s == x and e == x + 1:
        del row[i]
    elif s == x:
        row[i] = (x + 1, e)
    elif e == x + 1:
        row[i] = (s, x)
    else:
        row[i] = (s, x)
        row.insert(i + 1, (x + 1, e))
    return True


# ---------------------------------------------------------------------------
# 新旧状态之间的体素级组件流向
# ---------------------------------------------------------------------------


def _voxel_flow(
    old_runs: list[list[list[tuple[int, int]]]],
    old_result: LabelResult,
    new_runs: list[list[list[tuple[int, int]]]],
    new_result: LabelResult,
) -> tuple[dict[int, dict[int, int]], dict[int, dict[int, int]]]:
    """区间求交得到组件间体素流向。

    返回 ``(forward, backward)``：``forward[旧ID][新ID] = 体素数``，
    ``backward`` 为其反向。新旧两行游程各自有序，双指针线性求交。
    """

    forward: dict[int, dict[int, int]] = {}
    backward: dict[int, dict[int, int]] = {}
    old_labels = old_result.run_labels
    new_labels = new_result.run_labels
    for z in range(len(old_runs)):
        for y in range(len(old_runs[z])):
            orow = old_runs[z][y]
            nrow = new_runs[z][y]
            if not orow or not nrow:
                continue
            olab = old_labels[z][y]
            nlab = new_labels[z][y]
            i = j = 0
            while i < len(orow) and j < len(nrow):
                s = orow[i][0] if orow[i][0] > nrow[j][0] else nrow[j][0]
                e = orow[i][1] if orow[i][1] < nrow[j][1] else nrow[j][1]
                if s < e:
                    oc, nc = olab[i], nlab[j]
                    f = forward.setdefault(oc, {})
                    f[nc] = f.get(nc, 0) + (e - s)
                    b = backward.setdefault(nc, {})
                    b[oc] = b.get(oc, 0) + (e - s)
                if orow[i][1] < nrow[j][1]:
                    i += 1
                elif orow[i][1] > nrow[j][1]:
                    j += 1
                else:
                    i += 1
                    j += 1
    return forward, backward


def _build_events(
    old_result: LabelResult,
    new_result: LabelResult,
    forward: dict[int, dict[int, int]],
    backward: dict[int, dict[int, int]],
) -> tuple[ComponentEvent, ...]:
    splits: list[ComponentEvent] = []
    removed: list[ComponentEvent] = []
    for oc in range(1, len(old_result.components) + 1):
        targets = forward.get(oc)
        if not targets:
            removed.append(ComponentEvent("removed", (oc,), ()))
        elif len(targets) > 1:
            splits.append(
                ComponentEvent("split", (oc,), tuple(sorted(targets)))
            )
    merges: list[ComponentEvent] = []
    created: list[ComponentEvent] = []
    for nc in range(1, len(new_result.components) + 1):
        sources = backward.get(nc)
        if not sources:
            created.append(ComponentEvent("created", (), (nc,)))
        elif len(sources) > 1:
            merges.append(
                ComponentEvent("merge", tuple(sorted(sources)), (nc,))
            )
    return tuple(splits + merges + created + removed)


# ---------------------------------------------------------------------------
# 有状态体素体
# ---------------------------------------------------------------------------


class VoxelVolume:
    """可批量增删体素、自动重算连通组件的有状态三维二值体。

    尺寸与连通性在构造时固定；初始为空体。每次成功的批量更新都会
    全量重算组件（ID 按组件最小体素坐标升序）并返回事件。
    """

    __slots__ = ("_depth", "_height", "_width", "_connectivity",
                 "_runs", "_result")

    def __init__(
        self, depth: int, height: int, width: int, connectivity: int = 6
    ) -> None:
        self._depth = _check_dim(depth, "depth")
        self._height = _check_dim(height, "height")
        self._width = _check_dim(width, "width")
        if (
            isinstance(connectivity, bool)
            or not isinstance(connectivity, int)
            or connectivity not in _VALID_CONNECTIVITY
        ):
            raise InvalidVolumeError(
                "connectivity 必须为整数 6/18/26 之一（拒绝 bool/float），"
                f"得到 {type(connectivity).__name__}: {connectivity!r}"
            )
        self._connectivity = connectivity
        self._runs: list[list[list[tuple[int, int]]]] = [
            [[] for _ in range(height)] for _ in range(depth)
        ]
        self._result = label(
            self._runs, depth, height, width, connectivity
        )

    @classmethod
    def from_runs(
        cls,
        runs: object,
        depth: int,
        height: int,
        width: int,
        connectivity: int = 6,
    ) -> VoxelVolume:
        """以与 ``label`` 相同的 RLE 输入构造（先整体校验，非法即抛错）。"""
        vol = cls(depth, height, width, connectivity)
        normalized = _normalize_runs(runs, depth, height, width)
        vol._runs = normalized
        vol._result = label(normalized, depth, height, width, connectivity)
        return vol

    # -- 只读属性 ----------------------------------------------------------

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    @property
    def connectivity(self) -> int:
        return self._connectivity

    @property
    def voxel_count(self) -> int:
        return sum(
            e - s for sl in self._runs for row in sl for s, e in row
        )

    @property
    def num_runs(self) -> int:
        return sum(len(row) for sl in self._runs for row in sl)

    def result(self) -> LabelResult:
        """当前状态的全量标记结果（与最近一次更新返回的相同）。"""
        return self._result

    def runs_snapshot(self) -> list[list[list[Run]]]:
        """当前体数据的深拷贝，形状与 ``label`` 的 ``runs`` 输入一致。"""
        return [
            [[Run(s, e) for s, e in row] for row in sl]
            for sl in self._runs
        ]

    # -- 批量更新 ----------------------------------------------------------

    def add_voxels(self, voxels: object) -> UpdateResult:
        """批量加入体素；已存在的体素为无操作。"""
        return self.apply(add_voxels=voxels)

    def remove_voxels(self, voxels: object) -> UpdateResult:
        """批量删除体素；不存在的体素为无操作。"""
        return self.apply(remove_voxels=voxels)

    def apply(
        self, add_voxels: object = (), remove_voxels: object = ()
    ) -> UpdateResult:
        """在一个批次内先删后增，随后全量重算并给出组件事件。

        整批坐标先全部校验；同一体素不得同时出现在增、删两列。
        任何非法输入都抛 :class:`InvalidVolumeError` 且状态不变。
        两列都为空时不触发重算，返回当前结果与空事件序列。
        """

        adds = self._check_voxel_batch(add_voxels, "add_voxels")
        rems = self._check_voxel_batch(remove_voxels, "remove_voxels")
        both = adds & rems
        if both:
            sample = sorted(both)[0]
            raise InvalidVolumeError(
                f"同一体素不能同批增删：{sample} 同时出现在 "
                f"add_voxels 与 remove_voxels（共 {len(both)} 个冲突）"
            )
        if not adds and not rems:
            return UpdateResult(self._result, (), 0, 0)

        new_runs = [[list(row) for row in sl] for sl in self._runs]
        removed = 0
        for z, y, x in rems:
            if _remove_from_row(new_runs[z][y], x):
                removed += 1
        added = 0
        for z, y, x in adds:
            if _add_to_row(new_runs[z][y], x):
                added += 1

        total_runs = sum(len(row) for sl in new_runs for row in sl)
        if total_runs > MAX_RUNS:
            raise InvalidVolumeError(
                f"更新后游程总数 {total_runs} 超过上限 {MAX_RUNS}；"
                f"本次批量更新被拒绝，体素体保持原状"
            )

        new_result = label(
            new_runs, self._depth, self._height, self._width,
            self._connectivity,
        )
        forward, backward = _voxel_flow(
            self._runs, self._result, new_runs, new_result
        )
        events = _build_events(self._result, new_result, forward, backward)

        self._runs = new_runs
        self._result = new_result
        return UpdateResult(new_result, events, added, removed)

    # -- 输入校验 ------------------------------------------------------------

    def _check_voxel_batch(self, voxels: object, what: str) -> set[tuple[int, int, int]]:
        if isinstance(voxels, (str, bytes)) or not isinstance(
            voxels, Iterable
        ):
            raise InvalidVolumeError(
                f"{what} 必须为 (z, y, x) 坐标序列，"
                f"得到 {type(voxels).__name__}"
            )
        limits = (("z", self._depth), ("y", self._height), ("x", self._width))
        out: set[tuple[int, int, int]] = set()
        for i, v in enumerate(voxels):
            where = f"{what}[{i}]"
            if (
                not isinstance(v, Sequence)
                or isinstance(v, (str, bytes))
                or len(v) != 3
            ):
                raise InvalidVolumeError(
                    f"{where}: 体素必须为 (z, y, x) 三元组，得到 {v!r}"
                )
            coords = []
            for raw, (axis, lim) in zip(v, limits):
                aw = f"{where}.{axis}"
                if isinstance(raw, float):
                    if raw != raw or raw in (inf, -inf):
                        raise InvalidVolumeError(
                            f"{aw} 不能为 NaN 或 Infinity，得到 {raw!r}"
                        )
                    raise InvalidVolumeError(
                        f"{aw} 坐标必须为整数，得到 float: {raw!r}"
                    )
                if isinstance(raw, bool) or not isinstance(raw, int):
                    raise InvalidVolumeError(
                        f"{aw} 坐标必须为整数，"
                        f"得到 {type(raw).__name__}: {raw!r}"
                    )
                if not 0 <= raw < lim:
                    raise InvalidVolumeError(
                        f"{aw} 坐标越界：要求 0 <= {axis} < {lim}，得到 {raw}"
                    )
                coords.append(raw)
            out.add((coords[0], coords[1], coords[2]))
        return out
