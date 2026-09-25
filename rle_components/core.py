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

流式接口
--------
:class:`StreamLabeler` 按 z 递增逐切片喂入（``feed_slice``），空切片也
必须显式传入。某组件一旦不再接触最新切片，它不可能再与未来切片相连，
立即以**稳定临时 ID** 输出终结记录并释放其活跃合并状态；活跃内存只与
边界切片及未封闭组件相关，不保留全部已展开历史。``finish`` 结算最后
切片，把所有临时 ID 按组件最小 (z, y, x) 映射为与 :func:`label` 一致
的最终 ID。
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
    "StreamResult",
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


class InvalidVolumeError(ValueError):
    """输入体数据或参数不满足契约。

    消息中带 ``z=.. y=.. run=..`` 等定位前缀；非法输入绝不被静默纠正。
    """


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


def _normalize_slice(
    slice_rows: object, z: int, height: int, width: int, total: int
) -> tuple[list[list[tuple[int, int]]], int]:
    """校验并规范化单个切片，返回 (切片, 累计游程数)。"""

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

    out_slice: list[list[tuple[int, int]]] = []
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
                    f"游程总数超过上限 {MAX_RUNS}（z={z} y={y} run={ri}）"
                )
        out_slice.append(out_row)
    return out_slice, total


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
        out_slice, total = _normalize_slice(slice_rows, z, height, width, total)
        normalized.append(out_slice)
    return normalized


# ---------------------------------------------------------------------------
# 并查集
# ---------------------------------------------------------------------------


class _DSU:
    __slots__ = ("parent", "size")

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [0] * n  # 节点对应游程的体素数，合并后为组件体素数

    def find(self, x: int) -> int:
        parent = self.parent
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != x:
            parent[x], x = root, parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


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
# 主流程
# ---------------------------------------------------------------------------


def label(
    runs: object,
    depth: int,
    height: int,
    width: int,
    connectivity: int = 6,
) -> LabelResult:
    """对 RLE 二值体做连通组件标记。

    Args:
        runs: ``[z][y] -> [(start, end), ...]``；游程可为 :class:`Run`、
            二元组/列表，或 ``{"start": int, "end": int}`` 字典。
        depth: z 方向尺寸（切片数），1..512。
        height: y 方向尺寸（每切片行数），1..512。
        width: x 方向尺寸，1..512；空体请用空行表示。
        connectivity: 6（仅面）/ 18（面+边）/ 26（面+边+角）。

    Returns:
        :class:`LabelResult`，含每游程组件 ID 与每组件统计。

    Raises:
        InvalidVolumeError: 参数或数据不满足契约（bool 冒称整数、
            NaN/Infinity、越界、空游程、重叠/乱序、结构尺寸不符、
            游程数超过 20000 等），消息带 z/y/run 定位。
    """

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
    edge_mode = connectivity >= 18   # 18 或 26 合并边接触
    corner_mode = connectivity == 26  # 仅 26 合并角接触

    total_runs = sum(len(row) for sl in vol for row in sl)
    dsu = _DSU(total_runs)
    # 面邻居体素对数，最终在并查集根上汇总；每对仅在一个方向计一次
    face_pairs: list[tuple[int, int, int]] = []

    # 节点编号即游程创建顺序（z, y, 行内序 = 字典序）
    node_z: list[int] = []
    node_y: list[int] = []
    node_x0: list[int] = []
    node_x1: list[int] = []
    # 输出用 [z][y] -> [node_id,...]
    nodes_by: list[list[list[int]]] = []

    # 只保留两个切片的节点表
    prev_nodes: list[list[int]] | None = None
    next_id = 0

    for z in range(depth):
        cur_nodes: list[list[int]] = []
        nodes_by.append(cur_nodes)
        for y in range(height):
            row = vol[z][y]
            ids: list[int] = []
            for s, e in row:
                nid = next_id
                next_id += 1
                dsu.size[nid] = e - s
                # 游程内部 x 方向体素面邻居（length-1 个），面积统计用
                face_pairs.append((nid, nid, e - s - 1))
                node_z.append(z)
                node_y.append(y)
                node_x0.append(s)
                node_x1.append(e)
                ids.append(nid)
            cur_nodes.append(ids)

            # --- x 方向：同行游程，仅端点相接（面）才合并 ---
            for ri in range(1, len(ids)):
                if row[ri - 1][1] == row[ri][0]:
                    dsu.union(ids[ri - 1], ids[ri])
                    face_pairs.append((ids[ri - 1], ids[ri], 1))

            # --- y 方向：同切片 y-1 行（其节点已创建；行对 (y-1,y) ---
            # 在处理 y 时恰好检查一次）
            if y - 1 >= 0:
                other = vol[z][y - 1]
                other_ids = cur_nodes[y - 1]
                for i, j, ov in _pairs(row, other):
                    if ov > 0:
                        # x 重叠 → 面接触：所有连通性都合并，逐体素计面
                        dsu.union(ids[i], other_ids[j])
                        face_pairs.append((ids[i], other_ids[j], ov))
                    elif edge_mode:
                        # 端点相接（dy=1, dx=1）→ 边接触
                        dsu.union(ids[i], other_ids[j])

            # --- z 方向：前一切片的 y-1 / y / y+1 三行 ---
            # 注意与同切片 y 方向不同：(cur y, prev y+1) 这一游程对
            # 无法靠处理其他行覆盖（两行分属不同切片的不同行），三行
            # 都必须在本行扫描；每个 (cur_y, prev_y) 组合只在此出现一次。
            if prev_nodes is not None:
                for dy in (-1, 0, 1):
                    py = y + dy
                    if not 0 <= py < height:
                        continue
                    prow = vol[z - 1][py]
                    pids = prev_nodes[py]
                    for i, j, ov in _pairs(row, prow):
                        if dy == 0:
                            if ov > 0:
                                # dz=1, dy=0, x 重叠 → 面接触
                                dsu.union(ids[i], pids[j])
                                face_pairs.append((ids[i], pids[j], ov))
                            elif edge_mode:
                                # dz=1, dy=0, dx=1 → 边接触
                                dsu.union(ids[i], pids[j])
                        else:
                            if ov > 0:
                                # dz=1, |dy|=1, x 重叠 → 边接触
                                if edge_mode:
                                    dsu.union(ids[i], pids[j])
                            else:
                                # dz=1, |dy|=1, dx=1 → 角接触
                                if corner_mode:
                                    dsu.union(ids[i], pids[j])

        prev_nodes = cur_nodes

    # ------------------------------------------------------------------
    # 聚合：节点按创建顺序遍历，根第一次出现时的 (z,y,x0) 即组件最小坐标
    # ------------------------------------------------------------------

    n = total_runs
    # 排序键：组件字典序最小体素（节点创建序 = (z,y,x) 序，根首次出现处）
    key_z = [0] * n
    key_y = [0] * n
    key_x = [0] * n
    # 几何包围盒（与排序键可能不同，26 邻接下尤甚）
    min_z = [0] * n
    min_y = [0] * n
    min_x = [0] * n
    max_z = [0] * n
    max_y = [0] * n
    max_x = [0] * n
    seen: set[int] = set()

    for nid in range(n):
        r = dsu.find(nid)
        z, y, x0, x1 = node_z[nid], node_y[nid], node_x0[nid], node_x1[nid]
        if r not in seen:
            seen.add(r)
            key_z[r] = z
            key_y[r] = y
            key_x[r] = x0
            min_z[r] = z
            min_y[r] = y
            min_x[r] = x0
            max_z[r] = z + 1
            max_y[r] = y + 1
            max_x[r] = x1
        else:
            if z < min_z[r]:
                min_z[r] = z
            if y < min_y[r]:
                min_y[r] = y
            if x0 < min_x[r]:
                min_x[r] = x0
            if z + 1 > max_z[r]:
                max_z[r] = z + 1
            if y + 1 > max_y[r]:
                max_y[r] = y + 1
            if x1 > max_x[r]:
                max_x[r] = x1

    internal_faces = [0] * n
    for a, b, count in face_pairs:
        internal_faces[dsu.find(a)] += count

    roots = sorted(seen, key=lambda r: (key_z[r], key_y[r], key_x[r]))
    root_to_cid = {r: cid for cid, r in enumerate(roots, start=1)}

    components: list[Component] = []
    for cid, r in enumerate(roots, start=1):
        voxels = dsu.size[r]
        area = 6 * voxels - 2 * internal_faces[r]
        bbox = BoundingBox(
            min_z[r], min_y[r], min_x[r], max_z[r], max_y[r], max_x[r]
        )
        components.append(
            Component(
                id=cid,
                voxel_count=voxels,
                bbox=bbox,
                exposed_area=area,
                min_z=key_z[r],
                min_y=key_y[r],
                min_x=key_x[r],
            )
        )

    run_labels = [
        [[root_to_cid[dsu.find(nid)] for nid in row_ids] for row_ids in sl]
        for sl in nodes_by
    ]

    return LabelResult(
        run_labels=run_labels,
        components=components,
        connectivity=connectivity,
        num_runs=total_runs,
    )


# ---------------------------------------------------------------------------
# 流式接口：逐切片喂入，组件封闭即交付终结记录并释放活跃状态
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClosedComponent:
    """流式阶段某组件的终结记录（临时身份）。

    Attributes:
        temp_id: 稳定临时 ID（1 起始，按组件创建先后分配，终身不变）；
            不是最终 ID，最终 ID 由 :meth:`StreamLabeler.finish` 统一映射。
        voxel_count / bbox / exposed_area / min_z / min_y / min_x:
            与 :class:`Component` 同义；组件封闭时这些值即最终确定。
    """

    temp_id: int
    voxel_count: int
    bbox: BoundingBox
    exposed_area: int
    min_z: int
    min_y: int
    min_x: int


@dataclass(frozen=True)
class SliceResult:
    """一次 :meth:`StreamLabeler.feed_slice` 的返回。

    Attributes:
        z: 本次喂入切片的 z 下标（从 0 开始，每次调用递增 1）。
        run_labels: 本切片每行游程所属组件的临时 ID，
            形状 `[y] -> [temp_id, ...]`，顺序对应该行输入游程。
        closed: 本切片处理后被判定封闭的组件终结记录（可能为空）。
    """

    z: int
    run_labels: list[list[int]]
    closed: list[ClosedComponent]


@dataclass(frozen=True)
class StreamResult:
    """`StreamLabeler.finish` 的返回。

    Attributes:
        components: 最终组件列表，ID 1 起始、按 (min_z, min_y, min_x)
            升序，与批式 :func:`label` 的编号口径完全一致。
        id_of_temp: 临时 ID -> 最终 ID 的映射（覆盖全部组件）。
        connectivity: 实际使用的连通性。
        num_runs: 累计喂入的游程总数。
    """

    components: list[Component]
    id_of_temp: dict[int, int]
    connectivity: int
    num_runs: int


class _Agg:
    """活跃组件的聚合状态（仅未封闭组件持有）。"""

    __slots__ = (
        "temp_id", "voxels", "faces",
        "key_z", "key_y", "key_x",
        "min_z", "min_y", "min_x", "max_z", "max_y", "max_x",
    )

    def __init__(self, temp_id: int, z: int, y: int, x0: int, x1: int) -> None:
        self.temp_id = temp_id
        self.voxels = x1 - x0
        self.faces = x1 - x0 - 1  # 游程内部 x 方向面邻居
        self.key_z = z
        self.key_y = y
        self.key_x = x0
        self.min_z = z
        self.min_y = y
        self.min_x = x0
        self.max_z = z + 1
        self.max_y = y + 1
        self.max_x = x1

    def absorb(self, other: "_Agg", shared_faces: int) -> None:
        self.voxels += other.voxels
        self.faces += other.faces + shared_faces
        if (other.key_z, other.key_y, other.key_x) < (
            self.key_z, self.key_y, self.key_x
        ):
            self.key_z, self.key_y, self.key_x = (
                other.key_z, other.key_y, other.key_x,
            )
        self.min_z = min(self.min_z, other.min_z)
        self.min_y = min(self.min_y, other.min_y)
        self.min_x = min(self.min_x, other.min_x)
        self.max_z = max(self.max_z, other.max_z)
        self.max_y = max(self.max_y, other.max_y)
        self.max_x = max(self.max_x, other.max_x)

    def close(self) -> ClosedComponent:
        return ClosedComponent(
            temp_id=self.temp_id,
            voxel_count=self.voxels,
            bbox=BoundingBox(
                self.min_z, self.min_y, self.min_x,
                self.max_z, self.max_y, self.max_x,
            ),
            exposed_area=6 * self.voxels - 2 * self.faces,
            min_z=self.key_z,
            min_y=self.key_y,
            min_x=self.key_x,
        )


class StreamLabeler:
    """逐切片流式连通组件标记。

    用法::

        st = StreamLabeler(height, width, connectivity=6)
        for slice_rows in slices:          # z 必须连续递增；空切片也要
            out = st.feed_slice(slice_rows)  # 显式传 [[]]*height
            ...out.closed...               # 本切片封闭的组件终结记录
        final = st.finish()                # 结算尾部，给出最终 ID 映射

    组件一旦不接触最新切片即封闭：其终结记录在当次 `feed_slice` 返回，
    活跃合并状态随即释放。任意时刻的活跃状态只包含最近切片的边界游程
    节点与尚未封闭的组件聚合，不保留全部历史游程。
    """

    def __init__(self, height: int, width: int, connectivity: int = 6) -> None:
        self._height = _check_dim(height, "height")
        self._width = _check_dim(width, "width")
        if (
            isinstance(connectivity, bool)
            or not isinstance(connectivity, int)
            or connectivity not in _VALID_CONNECTIVITY
        ):
            raise InvalidVolumeError(
                f"connectivity 必须为整数 6/18/26 之一（拒绝 bool/float），"
                f"得到 {type(connectivity).__name__}: {connectivity!r}"
            )
        self._connectivity = connectivity
        self._edge_mode = connectivity >= 18
        self._corner_mode = connectivity == 26

        self._z = 0
        self._finished = False
        self._num_runs = 0
        self._next_temp = 1
        self._next_node = 0
        # 活跃状态：node 为最近两个切片内游程的编号
        self._parent: dict[int, int] = {}
        self._aggs: dict[int, _Agg] = {}       # 并查集根 -> 聚合
        self._prev_rows: list[list[tuple[int, int]]] | None = None
        self._prev_nodes: list[list[int]] | None = None
        self._closed: list[_Agg] = []
        # 被合并吸收、但可能已在历史切片 run_labels 中发布的临时 ID
        self._alias: dict[int, int] = {}  # 消亡 temp_id -> 幸存 temp_id

    # --  introspection（监控/测试用） --

    def active_node_count(self) -> int:
        """当前活跃并查集节点数（仅边界切片游程）。"""
        return len(self._parent)

    def active_component_count(self) -> int:
        """当前未封闭组件数。"""
        return len(self._aggs)

    # --  并查集（dict 版，节点随切片滚动释放） --

    def _find(self, n: int) -> int:
        parent = self._parent
        root = n
        while parent[root] != root:
            root = parent[root]
        while parent[n] != root:
            parent[n], n = root, parent[n]
        return root

    def _union(self, a: int, b: int, shared_faces: int = 0) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            self._aggs[ra].faces += shared_faces
            return
        aa, ab = self._aggs[ra], self._aggs[rb]
        if aa.voxels < ab.voxels:
            ra, rb, aa, ab = rb, ra, ab, aa
        aa.absorb(ab, shared_faces)
        # 临时身份保持稳定：组件终身沿用最早创建时的 temp_id；
        # 消亡的 temp_id 记录别名，finish 时一并映射到最终 ID
        if ab.temp_id < aa.temp_id:
            aa.temp_id, ab.temp_id = ab.temp_id, aa.temp_id
        self._alias[ab.temp_id] = aa.temp_id
        self._parent[rb] = ra
        del self._aggs[rb]

    # --  主流程 --

    def feed_slice(self, slice_rows: object) -> SliceResult:
        """喂入下一张切片（z 隐式连续递增），返回本切片结果。

        空切片必须显式传入 `[[]] * height`；跳过 z 层会破坏
        “不接触最新切片即封闭”的判定，属于契约违背。

        Raises:
            InvalidVolumeError: finish 后再次喂入、切片结构/坐标非法、
                切片数超过 512 或累计游程超过 20000。
        """
        if self._finished:
            raise InvalidVolumeError("finish 已结算，不能再 feed_slice")
        if self._z >= MAX_DIM:
            raise InvalidVolumeError(
                f"切片数超过上限 depth<={MAX_DIM}（z={self._z}）"
            )
        z = self._z
        height, width = self._height, self._width
        vol, self._num_runs = _normalize_slice(
            slice_rows, z, height, width, self._num_runs
        )

        cur_nodes: list[list[int]] = []
        flat: list[int] = []
        for y in range(height):
            row = vol[y]
            ids: list[int] = []
            for s, e in row:
                nid = self._next_node
                self._next_node += 1
                self._parent[nid] = nid
                self._aggs[nid] = _Agg(self._next_temp, z, y, s, e)
                self._next_temp += 1
                ids.append(nid)
                flat.append(nid)
            cur_nodes.append(ids)

            # x 方向：同行端点相接（面）
            for ri in range(1, len(ids)):
                if row[ri - 1][1] == row[ri][0]:
                    self._union(ids[ri - 1], ids[ri], 1)

            # y 方向：同切片 y-1 行（面/边）
            if y - 1 >= 0:
                other = vol[y - 1]
                other_ids = cur_nodes[y - 1]
                for i, j, ov in _pairs(row, other):
                    if ov > 0:
                        self._union(ids[i], other_ids[j], ov)
                    elif self._edge_mode:
                        self._union(ids[i], other_ids[j])

            # z 方向：前一切片 y-1 / y / y+1 三行（面/边/角）
            if self._prev_rows is not None:
                for dy in (-1, 0, 1):
                    py = y + dy
                    if not 0 <= py < height:
                        continue
                    prow = self._prev_rows[py]
                    pids = self._prev_nodes[py]
                    for i, j, ov in _pairs(row, prow):
                        if dy == 0:
                            if ov > 0:
                                self._union(ids[i], pids[j], ov)
                            elif self._edge_mode:
                                self._union(ids[i], pids[j])
                        elif ov > 0:
                            if self._edge_mode:
                                self._union(ids[i], pids[j])
                        elif self._corner_mode:
                            self._union(ids[i], pids[j])

        # 本切片每游程的临时组件 ID（封闭判定之前取，保证同切片一致）
        run_labels = [
            [self._aggs[self._find(nid)].temp_id for nid in ids]
            for ids in cur_nodes
        ]

        # 封闭判定：不接触最新切片的组件不可能再与未来切片相连
        present = {self._find(nid) for nid in flat}
        closed: list[ClosedComponent] = []
        for root in list(self._aggs):
            if root not in present:
                agg = self._aggs.pop(root)
                closed.append(agg.close())
                self._closed.append(agg)

        # 释放活跃状态：把仍活跃组件的根重挂到本切片节点上，
        # 随后丢弃前一切片的全部节点，只保留边界与未封闭聚合
        cur_set = set(flat)
        groups: dict[int, list[int]] = {}
        for nid in flat:
            groups.setdefault(self._find(nid), []).append(nid)
        for root, members in groups.items():
            new_root = members[0]
            self._parent[new_root] = new_root
            for m in members[1:]:
                self._parent[m] = new_root
            if root != new_root:
                self._parent[root] = new_root
                self._aggs[new_root] = self._aggs.pop(root)
        if self._prev_nodes is not None:
            for row_ids in self._prev_nodes:
                for nid in row_ids:
                    self._parent.pop(nid, None)
        # 上一步可能把旧根重定向到本切片节点后又删掉旧根条目，无影响

        self._prev_rows = vol
        self._prev_nodes = cur_nodes
        self._z += 1
        return SliceResult(z=z, run_labels=run_labels, closed=closed)

    def finish(self) -> StreamResult:
        """结算最后切片：封闭全部剩余组件，返回最终 ID 映射。

        最终 ID 1 起始、按组件最小 (z, y, x) 升序，与批式 :func:`label`
        的编号一致；临时 ID 只按创建次序分配，不按提前结束次序冒充原 ID。
        只能调用一次。
        """
        if self._finished:
            raise InvalidVolumeError("finish 只能调用一次")
        self._finished = True

        for root in list(self._aggs):
            self._closed.append(self._aggs.pop(root))
        self._parent.clear()
        self._prev_rows = None
        self._prev_nodes = None

        ordered = sorted(
            self._closed, key=lambda a: (a.key_z, a.key_y, a.key_x)
        )
        components: list[Component] = []
        id_of_temp: dict[int, int] = {}
        for cid, agg in enumerate(ordered, start=1):
            id_of_temp[agg.temp_id] = cid
            rec = agg.close()
            components.append(
                Component(
                    id=cid,
                    voxel_count=rec.voxel_count,
                    bbox=rec.bbox,
                    exposed_area=rec.exposed_area,
                    min_z=rec.min_z,
                    min_y=rec.min_y,
                    min_x=rec.min_x,
                )
            )
        # 消亡临时 ID 沿别名链解析到幸存身份，再映射到同一最终 ID
        for temp in self._alias:
            root = temp
            while root in self._alias:
                root = self._alias[root]
            id_of_temp[temp] = id_of_temp[root]
        return StreamResult(
            components=components,
            id_of_temp=id_of_temp,
            connectivity=self._connectivity,
            num_runs=self._num_runs,
        )
