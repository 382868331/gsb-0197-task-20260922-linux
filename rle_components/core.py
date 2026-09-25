"""核心算法：稀疏 RLE 体数据上的连通组件标记。

输入约定
--------
三维二值体按行（固定 z、y，沿 x 轴）压缩为有序、互不重叠的半开游程
``[start, end)``。体数据表示为 ``runs[z][y] = [(start, end), ...]``，
每个切片是长度为 ``height`` 的序列；每行内游程按 ``start`` 升序、互不
重叠（端点恰好相接合法，重叠非法）。算法不展开体素，任何时刻只保留
当前切片与前一切片的游程。

连通性
------
* 6：仅面接触（|dx|+|dy|+|dz| = 1）；
* 18：面 + 边（恰两个坐标差为 1），不含角；
* 26：面 + 边 + 角（三个坐标差均为 1）。

扫描时每条游程在并查集中建一个节点，与同行前一条游程（x 向）、同切片
y+1 行、前一切片的 y 与 y-1 行做区间触达判定并合并；每个游程对恰好
在一个方向上被检查一次。

输出
----
* 每条游程的组件 ID；
* 每组件体素数、半开包围盒、暴露面面积（``6*体素数 - 2*内部面数``，
  内部面在三种方向扫描时各计一次）；
* 组件 ID 1 起始，按组件最小体素坐标 (z, y, x) 字典序升序。

全部输出为整数，模块不产生任何浮点结果。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import inf
from typing import NamedTuple

__all__ = [
    "Run",
    "BoundingBox",
    "Component",
    "LabelResult",
    "ClosedComponent",
    "SliceResult",
    "FinishResult",
    "StreamLabeler",
    "InvalidVolumeError",
    "label",
]

# 接口边界（题面给定）
MAX_DIM = 512
MAX_RUNS = 20000

_VALID_CONNECTIVITY = (6, 18, 26)


class Run(NamedTuple):
    """单条半开 x 游程：覆盖 x in [start, end)。"""

    start: int
    end: int


class BoundingBox(NamedTuple):
    """组件轴对齐包围盒，各坐标区间均为半开 [min, max)。"""

    min_z: int
    min_y: int
    min_x: int
    max_z: int
    max_y: int
    max_x: int


@dataclass(frozen=True)
class Component:
    """一个连通组件的聚合结果。

    Attributes:
        id: 1 起始组件 ID，按 (min_z, min_y, min_x) 升序。
        voxel_count: 组件体素总数。
        bbox: 轴对齐包围盒（半开区间）。
        exposed_area: 暴露面面积（体素面单位）= 6*voxel_count - 2*内部面数。
        min_z/min_y/min_x: 组件字典序最小体素坐标（= bbox 的三个 min）。
    """

    id: int
    voxel_count: int
    bbox: BoundingBox
    exposed_area: int
    min_z: int
    min_y: int
    min_x: int


@dataclass(frozen=True)
class LabelResult:
    """``label`` 的返回值。

    Attributes:
        run_labels: 与输入同形的嵌套列表 ``[z][y] -> [component_id, ...]``，
            每行 ID 顺序对应该行输入游程顺序。
        components: 组件列表，ID 为 1..N，按 (min_z, min_y, min_x) 升序。
        connectivity: 实际使用的连通性（6/18/26）。
        num_runs: 输入游程总数。
    """

    run_labels: list[list[list[int]]]
    components: list[Component]
    connectivity: int
    num_runs: int

    def component_at(self, z: int, y: int, run_index: int) -> Component:
        """返回 (z, y) 行内第 run_index 条游程所属的组件。"""

        cid = self.run_labels[z][y][run_index]
        return self.components[cid - 1]


@dataclass(frozen=True)
class ClosedComponent:
    """流式处理中已封闭、不可能再连接未来切片的组件。

    ``temporary_id`` 仅为流式阶段的稳定临时身份，不是最终组件 ID；
    最终 ID 在 :meth:`StreamLabeler.finish` 中按最小体素坐标统一分配。
    """

    temporary_id: int
    voxel_count: int
    bbox: BoundingBox
    exposed_area: int
    min_z: int
    min_y: int
    min_x: int


@dataclass(frozen=True)
class SliceResult:
    """一次 :meth:`StreamLabeler.feed_slice` 的结果。"""

    z: int
    run_labels: list[list[int]]
    closed: list[ClosedComponent]


@dataclass(frozen=True)
class FinishResult:
    """:meth:`StreamLabeler.finish` 的结算结果。"""

    components: list[Component]
    id_map: dict[int, int]
    closed: list[ClosedComponent]


class InvalidVolumeError(ValueError):
    """输入体数据或参数不满足契约。

    消息中带 ``z=.. y=.. run=..`` 等定位前缀；非法输入绝不被静默纠正。
    """


# ---------------------------------------------------------------------------
# 流式主流程
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _CompState:
    """仅为仍可能接触未来切片的未封闭组件保留的边界聚合状态。"""

    tid: int
    voxels: int
    internal_faces: int
    min_z: int
    min_y: int
    min_x: int
    max_z: int
    max_y: int
    max_x: int
    key_z: int
    key_y: int
    key_x: int
    parent: int = 0
    on_boundary: bool = False

    def __post_init__(self) -> None:
        self.parent = self.tid

    def closed(self) -> ClosedComponent:
        bbox = BoundingBox(
            self.min_z,
            self.min_y,
            self.min_x,
            self.max_z,
            self.max_y,
            self.max_x,
        )
        return ClosedComponent(
            temporary_id=self.tid,
            voxel_count=self.voxels,
            bbox=bbox,
            exposed_area=6 * self.voxels - 2 * self.internal_faces,
            min_z=self.key_z,
            min_y=self.key_y,
            min_x=self.key_x,
        )


class StreamLabeler:
    """逐 z 切片喂入的游程连通组件标记器。

    切片必须通过 :meth:`feed_slice` 按连续、递增的 z 顺序显式喂入，
    整层没有体素时也要传入一个全为 ``[]`` 的空切片。每次喂入后，
    不再接触最新切片的组件会随返回值封闭；其活跃合并状态随即释放。
    临时 ID 在整个流中稳定且不重用，最终 ID 只在 :meth:`finish` 时
    按组件最小体素坐标统一分配。
    """

    def __init__(
        self,
        depth: int,
        height: int,
        width: int,
        connectivity: int = 6,
    ) -> None:
        self.depth = _check_dim(depth, "depth")
        self.height = _check_dim(height, "height")
        self.width = _check_dim(width, "width")
        if (
            isinstance(connectivity, bool)
            or not isinstance(connectivity, int)
            or connectivity not in _VALID_CONNECTIVITY
        ):
            raise InvalidVolumeError(
                "connectivity 必须为整数 6/18/26 之一（拒绝 bool/float），"
                f"得到 {type(connectivity).__name__}: {connectivity!r}"
            )
        self.connectivity = connectivity
        self._edge = connectivity >= 18
        self._corner = connectivity == 26
        self._next_tid = 1
        self._next_z = 0
        self._total_runs = 0
        self._prev_rows: list[list[tuple[int, int]]] | None = None
        self._prev_tids: list[list[int]] | None = []
        self._states: dict[int, _CompState] = {}
        self._closed: list[ClosedComponent] = []
        self._aliases: dict[int, int] = {}
        self._finished = False

    @property
    def active_temporary_ids(self) -> tuple[int, ...]:
        """当前仍接触最新切片、可与未来切片合并的临时身份。"""

        if self._finished:
            return ()
        return tuple(
            tid for tid, st in self._states.items() if st.on_boundary
        )

    @property
    def num_active_components(self) -> int:
        """边界上的未封闭组件数；整层空白后为 0。"""

        return len(self.active_temporary_ids)

    @property
    def num_boundary_runs(self) -> int:
        """最新切片中的游程数；只保留边界游程，不保留历史展开体。"""

        if self._finished or self._prev_tids is None:
            return 0
        return sum(len(row) for row in self._prev_tids)

    def feed_slice(self, z: object, rows: object) -> SliceResult:
        """喂入一个切片，返回该切片游程标签和本切片后封闭的组件。"""

        if self._finished:
            raise InvalidVolumeError("finish 已调用，不能继续喂入切片")
        if isinstance(z, bool) or not isinstance(z, int):
            raise InvalidVolumeError(
                f"z 必须为整数（拒绝 bool），得到 {type(z).__name__}: {z!r}"
            )
        if z != self._next_z:
            expected = self._next_z
            raise InvalidVolumeError(
                f"切片 z 必须连续递增：期望 z={expected}，得到 z={z}；"
                "空切片也必须显式喂入"
            )

        normalized, slice_runs = _normalize_slice(
            rows, z, self.height, self.width
        )
        if self._total_runs + slice_runs > MAX_RUNS:
            raise InvalidVolumeError(
                f"游程总数超过上限 {MAX_RUNS}（z={z}）"
            )

        for state in self._states.values():
            state.on_boundary = False

        cur_tids: list[list[int]] = [[] for _ in range(self.height)]
        run_labels: list[list[int]] = [[] for _ in range(self.height)]

        for y, row in enumerate(normalized):
            ids = cur_tids[y]
            labels = run_labels[y]
            for s, e in row:
                tid = self._next_tid
                self._next_tid += 1
                self._states[tid] = _CompState(
                    tid=tid,
                    voxels=e - s,
                    internal_faces=max(0, e - s - 1),
                    min_z=z,
                    min_y=y,
                    min_x=s,
                    max_z=z + 1,
                    max_y=y + 1,
                    max_x=e,
                    key_z=z,
                    key_y=y,
                    key_x=s,
                    on_boundary=True,
                )
                ids.append(tid)
                labels.append(tid)

            for ri in range(1, len(ids)):
                if normalized[y][ri - 1][1] == row[ri][0]:
                    self._merge(ids[ri - 1], ids[ri], 1)

            if y > 0:
                for i, j, overlap in _pairs(row, normalized[y - 1]):
                    if overlap > 0:
                        self._merge(ids[i], cur_tids[y - 1][j], overlap)
                    elif self._edge:
                        self._merge(ids[i], cur_tids[y - 1][j], 0)

            if self._prev_rows is not None and self._prev_tids:
                for dy in (-1, 0, 1):
                    py = y + dy
                    if not 0 <= py < self.height:
                        continue
                    for i, j, overlap in _pairs(
                        row, self._prev_rows[py]
                    ):
                        face_count = overlap if dy == 0 and overlap > 0 else 0
                        connected = (
                            (dy == 0 and (overlap > 0 or self._edge))
                            or (dy != 0 and overlap > 0 and self._edge)
                            or (dy != 0 and overlap == 0 and self._corner)
                        )
                        if connected:
                            self._merge(
                                ids[i],
                                self._prev_tids[py][j],
                                face_count,
                            )

        for y in range(self.height):
            resolved: list[int] = []
            for tid in cur_tids[y]:
                root = self._find(tid)
                resolved.append(root)
            cur_tids[y] = resolved
            if run_labels[y]:
                run_labels[y] = resolved

        closed: list[ClosedComponent] = []
        stale = [tid for tid, st in self._states.items() if not st.on_boundary]
        for tid in stale:
            root = self._find(tid)
            if root == tid:
                closed.append(self._states.pop(tid).closed())
            else:
                self._states.pop(tid, None)
                self._aliases[tid] = root
        closed.sort(key=lambda c: (c.min_z, c.min_y, c.min_x))
        self._closed.extend(closed)

        self._prev_rows = normalized if slice_runs else None
        self._prev_tids = cur_tids if slice_runs else None
        self._next_z += 1
        self._total_runs += slice_runs

        return SliceResult(z=z, run_labels=run_labels, closed=closed)

    def finish(self) -> FinishResult:
        """结算剩余未封闭组件，并返回最终 ID 映射。"""

        if self._finished:
            raise InvalidVolumeError("finish 只能调用一次")

        remaining_roots = {
            self._find(tid)
            for tid, state in self._states.items()
            if state.on_boundary
        }
        remaining = [self._states[tid].closed() for tid in remaining_roots]
        all_closed = self._closed + remaining
        all_closed.sort(key=lambda c: (c.min_z, c.min_y, c.min_x))

        components: list[Component] = []
        id_map: dict[int, int] = {}
        for final_id, closed in enumerate(all_closed, start=1):
            id_map[closed.temporary_id] = final_id
            components.append(
                Component(
                    id=final_id,
                    voxel_count=closed.voxel_count,
                    bbox=closed.bbox,
                    exposed_area=closed.exposed_area,
                    min_z=closed.min_z,
                    min_y=closed.min_y,
                    min_x=closed.min_x,
                )
            )

        for alias, target in self._aliases.items():
            while target not in id_map:
                target = self._aliases[target]
            id_map[alias] = id_map[target]

        self._states.clear()
        self._prev_rows = None
        self._prev_tids = []
        self._finished = True

        return FinishResult(
            components=components,
            id_map=id_map,
            closed=all_closed,
        )

    def _find(self, tid: int) -> int:
        if tid not in self._states:
            target = self._aliases[tid]
            root = self._find(target)
            self._aliases[tid] = root
            return root
        path = []
        root = tid
        while root in self._states:
            parent = self._states[root].parent
            if parent == root:
                break
            path.append(root)
            root = parent
        if root not in self._states:
            root = self._find(root)
        for node in path:
            self._states[node].parent = root
        return root

    def _merge(self, a: int, b: int, face_count: int) -> None:
        ra = self._find(a)
        rb = self._find(b)
        if ra == rb:
            sa = self._states[ra]
            sa.internal_faces += face_count
            return

        sa = self._states[ra]
        sb = self._states[rb]
        if (sa.key_z, sa.key_y, sa.key_x) > (
            sb.key_z,
            sb.key_y,
            sb.key_x,
        ):
            ra, rb = rb, ra
            sa, sb = sb, sa

        sa.voxels += sb.voxels
        sa.internal_faces += sb.internal_faces + face_count
        sa.min_z = min(sa.min_z, sb.min_z)
        sa.min_y = min(sa.min_y, sb.min_y)
        sa.min_x = min(sa.min_x, sb.min_x)
        sa.max_z = max(sa.max_z, sb.max_z)
        sa.max_y = max(sa.max_y, sb.max_y)
        sa.max_x = max(sa.max_x, sb.max_x)
        sa.on_boundary = True
        sb.parent = ra
        sb.on_boundary = False
        self._aliases[rb] = ra


def label(
    runs: object,
    depth: int,
    height: int,
    width: int,
    connectivity: int = 6,
) -> LabelResult:
    """对 RLE 二值体做连通组件标记（旧的一次性批处理入口）。"""

    depth = _check_dim(depth, "depth")
    height = _check_dim(height, "height")
    width = _check_dim(width, "width")
    if (
        isinstance(connectivity, bool)
        or not isinstance(connectivity, int)
        or connectivity not in _VALID_CONNECTIVITY
    ):
        raise InvalidVolumeError(
            f"connectivity 必须为整数 6/18/26 之一（拒绝 bool/float），"
            f"得到 {type(connectivity).__name__}: {connectivity!r}"
        )

    vol = _normalize_runs(runs, depth, height, width)
    stream = StreamLabeler(depth, height, width, connectivity)
    run_labels: list[list[list[int]]] = []
    temporary_labels: list[list[list[int]]] = []
    for z, slice_rows in enumerate(vol):
        result = stream.feed_slice(z, slice_rows)
        temporary_labels.append(result.run_labels)
    finished = stream.finish()
    for sl in temporary_labels:
        run_labels.append(
            [[finished.id_map[tid] for tid in row] for row in sl]
        )

    return LabelResult(
        run_labels=run_labels,
        components=finished.components,
        connectivity=connectivity,
        num_runs=sum(len(row) for sl in vol for row in sl),
    )


# ---------------------------------------------------------------------------
# 输入校验（全部成功才返回；失败时无任何已产生的对外状态）
# ---------------------------------------------------------------------------


def _check_dim(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidVolumeError(
            f"{what} 必须为整数（拒绝 bool），得到 {type(value).__name__}: {value!r}"
        )
    if not 1 <= value <= MAX_DIM:
        raise InvalidVolumeError(f"{what} 必须在 [1, {MAX_DIM}] 内，得到 {value}")
    return value


def _check_coord(value: object, where: str) -> int:
    if isinstance(value, float):
        if value != value or value in (inf, -inf):
            raise InvalidVolumeError(
                f"{where} 不能为 NaN 或 Infinity，得到 {value!r}"
            )
        raise InvalidVolumeError(
            f"{where} 坐标必须为整数，得到 float: {value!r}"
        )
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidVolumeError(
            f"{where} 坐标必须为整数，得到 {type(value).__name__}: {value!r}"
        )
    return value


def _normalize_runs(
    runs: object, depth: int, height: int, width: int
) -> list[list[list[tuple[int, int]]]]:
    if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)):
        raise InvalidVolumeError(
            f"runs 必须为按切片组织的序列，得到 {type(runs).__name__}"
        )
    if len(runs) != depth:
        raise InvalidVolumeError(
            f"runs 切片数 {len(runs)} 与 depth={depth} 不一致"
        )

    normalized: list[list[list[tuple[int, int]]]] = []
    total = 0

    for z, slice_rows in enumerate(runs):
        out_slice, slice_total = _normalize_slice(
            slice_rows, z, height, width
        )
        if total + slice_total > MAX_RUNS:
            raise InvalidVolumeError(
                f"游程总数超过上限 {MAX_RUNS}（z={z}）"
            )
        normalized.append(out_slice)
        total += slice_total

    return normalized


def _normalize_slice(
    slice_rows: object, z: int, height: int, width: int
) -> tuple[list[list[tuple[int, int]]], int]:
    """校验并规范化一个切片；成功前不产生对外状态。"""

    total = 0
    out_slice: list[list[tuple[int, int]]] = []
    if not isinstance(slice_rows, Sequence) or isinstance(
        slice_rows, (str, bytes)
    ):
        raise InvalidVolumeError(
            f"z={z}: 切片必须为按行组织的序列，"
            f"得到 {type(slice_rows).__name__}"
        )
    if len(slice_rows) != height:
        raise InvalidVolumeError(
            f"z={z}: 行数 {len(slice_rows)} 与 height={height} 不一致"
        )

    for y, row in enumerate(slice_rows):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise InvalidVolumeError(
                f"z={z} y={y}: 行必须为游程序列，得到 {type(row).__name__}"
            )

        out_row: list[tuple[int, int]] = []
        prev_end = 0
        for ri, run in enumerate(row):
            where = f"z={z} y={y} run={ri}"
            if isinstance(run, dict):
                if set(run) != {"start", "end"}:
                    raise InvalidVolumeError(
                        f"{where}: 字典游程只能含键 start/end，得到 {sorted(run)}"
                    )
                raw0, raw1 = run["start"], run["end"]
            elif isinstance(run, Sequence) and not isinstance(
                run, (str, bytes)
            ) and len(run) == 2:
                raw0, raw1 = run[0], run[1]
            else:
                raise InvalidVolumeError(
                    f"{where}: 游程必须为 (start, end) 二元组，得到 {run!r}"
                )

            x0 = _check_coord(raw0, where + ".start")
            x1 = _check_coord(raw1, where + ".end")
            if not 0 <= x0 < x1 <= width:
                raise InvalidVolumeError(
                    f"{where}: 要求 0 <= start < end <= width({width})，"
                    f"得到 start={x0}, end={x1}（空游程/越界均非法）"
                )
            if ri > 0 and x0 < prev_end:
                raise InvalidVolumeError(
                    f"{where}: 游程必须按 start 升序且互不重叠，"
                    f"start={x0} 早于上游 end={prev_end}"
                )
            out_row.append((x0, x1))
            prev_end = x1
            total += 1
            if total > MAX_RUNS:
                raise InvalidVolumeError(
                    f"切片 z={z} 内游程数超过全局上限 {MAX_RUNS}"
                    f"（y={y} run={ri}）"
                )
        out_slice.append(out_row)
    return out_slice, total


# ---------------------------------------------------------------------------
# 并查集
# ---------------------------------------------------------------------------


def _pairs(a_runs, b_runs):
    """触达游程对扫描。

    产出 ``(i, j, overlap)``：b 游程与 a 游程的体素 x 距离 <= 1，即
    半开区间满足 ``max(starts) <= min(ends)``。

    * overlap > 0：投影严格重叠的体素列数（dy/dz=1 时为面或边接触）；
    * overlap == 0：端点恰好相接（最近体素 dx=1：同切片为边接触，
      跨切片 dy=0 为边接触，跨切片 dy=1 为角接触）。

    借助单调游标，整行扫描为 O(len(a) + len(b) + 触达对数)。
    """

    if not a_runs or not b_runs:
        return
    n = len(b_runs)
    j = 0
    for i, (s0, e0) in enumerate(a_runs):
        while j < n and b_runs[j][1] < s0:
            j += 1
        k = j
        while k < n and b_runs[k][0] <= e0:
            s1, e1 = b_runs[k]
            ov = e0 if e0 < e1 else e1
            ms = s0 if s0 > s1 else s1
            yield i, k, ov - ms
            k += 1


# ---------------------------------------------------------------------------
