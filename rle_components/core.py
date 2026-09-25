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

多材料扩展（``label_materials``）
--------------------------------
入口接受每行 ``(start, end, material)`` 游程（material 为非负整数），
空白区域仍由没有游程表示。扫描与区间触达框架与二值版一致，但仅当两条
触达游程材料相同时才合并；连通模式 6/18/26 语义不变。除每游程组件 ID、
体素数、包围盒外，输出每组件材料、对空白（含体积外侧）的暴露面积，以及
组件对之间的接触面积表。接触面积只计共享的方形体素面——边/角接触面积
为 0，不因选择 18/26 邻接而多计；每对组件按较小 ID 在前只输出一次。

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
    "MaterialRun",
    "MaterialComponent",
    "ContactPair",
    "MaterialLabelResult",
    "InvalidVolumeError",
    "label",
    "label_materials",
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
    游程元素只按下标取 [0]/[1]，二元组与三元组（带材料）均适用。
    """

    if not a_runs or not b_runs:
        return
    n = len(b_runs)
    j = 0
    for i, a_run in enumerate(a_runs):
        s0, e0 = a_run[0], a_run[1]
        while j < n and b_runs[j][1] < s0:
            j += 1
        k = j
        while k < n and b_runs[k][0] <= e0:
            s1, e1 = b_runs[k][0], b_runs[k][1]
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
# 多材料扩展：分材料连通 + 组件间接触面积
# ---------------------------------------------------------------------------


class MaterialRun(NamedTuple):
    """单条带材料的半开 x 游程：覆盖 x in [start, end)，材料为 material。"""

    start: int
    end: int
    material: int


@dataclass(frozen=True)
class ContactPair:
    """两个组件之间的接触面积（共享方形体素面数）。

    Attributes:
        component_a / component_b: 组件 ID，保证 component_a < component_b。
        area: 两组件共享的体素面数量。边/角接触不共享面，不计入；
            共享面的两组件必为不同材料（同材料面相邻必然已合并）。
    """

    component_a: int
    component_b: int
    area: int


@dataclass(frozen=True)
class MaterialComponent:
    """多材料标记下的一个连通组件。

    Attributes:
        id: 1 起始组件 ID，按 (min_z, min_y, min_x) 升序。
        material: 组件材料（非负整数；组件内所有体素材料一致）。
        voxel_count: 组件体素总数。
        bbox: 轴对齐包围盒（半开区间）。
        exposed_area: 对空白的暴露面面积（体积外侧按空白计），
            不含与其他组件的接触面。
        min_z/min_y/min_x: 组件字典序最小体素坐标（= ID 排序键）。
    """

    id: int
    material: int
    voxel_count: int
    bbox: BoundingBox
    exposed_area: int
    min_z: int
    min_y: int
    min_x: int


@dataclass(frozen=True)
class MaterialLabelResult:
    """``label_materials`` 的返回值。

    Attributes:
        run_labels: 与输入同形的嵌套列表 ``[z][y] -> [component_id, ...]``。
        components: 组件列表，ID 为 1..N，按 (min_z, min_y, min_x) 升序。
        contacts: 组件间接触面积表，每对仅出现一次（较小 ID 在前），
            按 (component_a, component_b) 升序；只含面积 > 0 的对。
        connectivity: 实际使用的连通性（6/18/26）。
        num_runs: 输入游程总数。
    """

    run_labels: list[list[list[int]]]
    components: list[MaterialComponent]
    contacts: list[ContactPair]
    connectivity: int
    num_runs: int

    def component_at(self, z: int, y: int, run_index: int) -> MaterialComponent:
        """返回 (z, y) 行内第 run_index 条游程所属的组件。"""

        cid = self.run_labels[z][y][run_index]
        return self.components[cid - 1]

    def contact_area(self, component_a: int, component_b: int) -> int:
        """查询两个组件的接触面积；无共享面时返回 0。"""

        a, b = component_a, component_b
        if a > b:
            a, b = b, a
        for c in self.contacts:
            if (c.component_a, c.component_b) == (a, b):
                return c.area
        return 0


def _check_material(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidVolumeError(
            f"{where} 材料必须为非负整数（拒绝 bool/float），"
            f"得到 {type(value).__name__}: {value!r}"
        )
    if value < 0:
        raise InvalidVolumeError(
            f"{where} 材料必须为非负整数，得到 {value}"
        )
    return value


def _normalize_material_runs(
    runs: object, depth: int, height: int, width: int
) -> list[list[list[tuple[int, int, int]]]]:
    if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)):
        raise InvalidVolumeError(
            f"runs 必须为按切片组织的序列，得到 {type(runs).__name__}"
        )
    if len(runs) != depth:
        raise InvalidVolumeError(
            f"runs 切片数 {len(runs)} 与 depth={depth} 不一致"
        )

    normalized: list[list[list[tuple[int, int, int]]]] = []
    total = 0

    for z, slice_rows in enumerate(runs):
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

        out_slice: list[list[tuple[int, int, int]]] = []
        for y, row in enumerate(slice_rows):
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                raise InvalidVolumeError(
                    f"z={z} y={y}: 行必须为游程序列，得到 {type(row).__name__}"
                )

            out_row: list[tuple[int, int, int]] = []
            prev_end = 0
            for ri, run in enumerate(row):
                where = f"z={z} y={y} run={ri}"
                if isinstance(run, dict):
                    if set(run) != {"start", "end", "material"}:
                        raise InvalidVolumeError(
                            f"{where}: 字典游程只能含键 start/end/material，"
                            f"得到 {sorted(run)}"
                        )
                    raw0, raw1, rawm = (
                        run["start"],
                        run["end"],
                        run["material"],
                    )
                elif isinstance(run, Sequence) and not isinstance(
                    run, (str, bytes)
                ) and len(run) == 3:
                    raw0, raw1, rawm = run[0], run[1], run[2]
                else:
                    raise InvalidVolumeError(
                        f"{where}: 游程必须为 (start, end, material) 三元组，"
                        f"得到 {run!r}"
                    )

                x0 = _check_coord(raw0, where + ".start")
                x1 = _check_coord(raw1, where + ".end")
                m = _check_material(rawm, where + ".material")
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
                out_row.append((x0, x1, m))
                prev_end = x1
                total += 1
                if total > MAX_RUNS:
                    raise InvalidVolumeError(
                        f"游程总数超过上限 {MAX_RUNS}（z={z} y={y} run={ri}）"
                    )
            out_slice.append(out_row)
        normalized.append(out_slice)

    return normalized


def label_materials(
    runs: object,
    depth: int,
    height: int,
    width: int,
    connectivity: int = 6,
) -> MaterialLabelResult:
    """对 RLE 多材料体做分材料连通组件标记与接触面积统计。

    同种材料的体素按 connectivity 连通规则合并为组件；不同材料即使
    面贴面也分属不同组件，其共享面计入组件对接触面积。

    Args:
        runs: ``[z][y] -> [(start, end, material), ...]``；游程可为
            :class:`MaterialRun`、三元组/列表，或
            ``{"start": int, "end": int, "material": int}`` 字典。
            material 为非负整数；空白区域由没有游程表示（空行/空切片）。
            每行游程按 start 升序、互不重叠（端点相接合法）。
        depth / height / width: 三方向尺寸，整数，均在 [1, 512]。
        connectivity: 6（仅面）/ 18（面+边）/ 26（面+边+角）。
            连通性只影响同材料体素是否合并；接触面积永远只计共享面。

    Returns:
        :class:`MaterialLabelResult`，含每游程组件 ID、每组件材料/
        体素数/包围盒/对空白暴露面积，以及组件对接触面积表。

    Raises:
        InvalidVolumeError: 参数或数据不满足契约（材料为负/bool/float、
            游程缺 material、坐标非法、重叠/乱序、超上限等），
            消息带 z/y/run 定位。
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

    vol = _normalize_material_runs(runs, depth, height, width)
    edge_mode = connectivity >= 18   # 18 或 26 合并边接触
    corner_mode = connectivity == 26  # 仅 26 合并角接触

    total_runs = sum(len(row) for sl in vol for row in sl)
    dsu = _DSU(total_runs)
    # 所有共享体素面的 (游程节点a, 节点b, 面数)，不分材料；扫描后按
    # 并查集根拆分：同根 → 组件内部面，异根 → 组件间接触面。
    face_pairs: list[tuple[int, int, int]] = []

    node_z: list[int] = []
    node_y: list[int] = []
    node_x0: list[int] = []
    node_x1: list[int] = []
    node_mat: list[int] = []
    nodes_by: list[list[list[int]]] = []

    prev_nodes: list[list[int]] | None = None
    next_id = 0

    for z in range(depth):
        cur_nodes: list[list[int]] = []
        nodes_by.append(cur_nodes)
        for y in range(height):
            row = vol[z][y]
            ids: list[int] = []
            for s, e, m in row:
                nid = next_id
                next_id += 1
                dsu.size[nid] = e - s
                # 游程内部 x 方向体素面邻居（length-1 个）
                face_pairs.append((nid, nid, e - s - 1))
                node_z.append(z)
                node_y.append(y)
                node_x0.append(s)
                node_x1.append(e)
                node_mat.append(m)
                ids.append(nid)
            cur_nodes.append(ids)

            # --- x 方向：同行端点相接的游程共享 1 个面；同材料才合并 ---
            for ri in range(1, len(ids)):
                if row[ri - 1][1] == row[ri][0]:
                    face_pairs.append((ids[ri - 1], ids[ri], 1))
                    if row[ri - 1][2] == row[ri][2]:
                        dsu.union(ids[ri - 1], ids[ri])

            # --- y 方向：同切片 y-1 行 ---
            if y - 1 >= 0:
                other = vol[z][y - 1]
                other_ids = cur_nodes[y - 1]
                for i, j, ov in _pairs(row, other):
                    if ov > 0:
                        # x 重叠 → 面接触：计面；同材料才合并
                        face_pairs.append((ids[i], other_ids[j], ov))
                        if row[i][2] == other[j][2]:
                            dsu.union(ids[i], other_ids[j])
                    elif edge_mode and row[i][2] == other[j][2]:
                        # 端点相接（dy=1, dx=1）→ 边接触，不计面积
                        dsu.union(ids[i], other_ids[j])

            # --- z 方向：前一切片的 y-1 / y / y+1 三行 ---
            if prev_nodes is not None:
                for dy in (-1, 0, 1):
                    py = y + dy
                    if not 0 <= py < height:
                        continue
                    prow = vol[z - 1][py]
                    pids = prev_nodes[py]
                    for i, j, ov in _pairs(row, prow):
                        same = row[i][2] == prow[j][2]
                        if dy == 0:
                            if ov > 0:
                                # dz=1, dy=0, x 重叠 → 面接触
                                face_pairs.append((ids[i], pids[j], ov))
                                if same:
                                    dsu.union(ids[i], pids[j])
                            elif edge_mode and same:
                                # dz=1, dy=0, dx=1 → 边接触
                                dsu.union(ids[i], pids[j])
                        else:
                            if ov > 0:
                                # dz=1, |dy|=1, x 重叠 → 边接触
                                if edge_mode and same:
                                    dsu.union(ids[i], pids[j])
                            else:
                                # dz=1, |dy|=1, dx=1 → 角接触
                                if corner_mode and same:
                                    dsu.union(ids[i], pids[j])

        prev_nodes = cur_nodes

    # ------------------------------------------------------------------
    # 聚合：节点按创建顺序遍历，根第一次出现时的 (z,y,x0) 即组件最小坐标
    # ------------------------------------------------------------------

    n = total_runs
    key_z = [0] * n
    key_y = [0] * n
    key_x = [0] * n
    min_z = [0] * n
    min_y = [0] * n
    min_x = [0] * n
    max_z = [0] * n
    max_y = [0] * n
    max_x = [0] * n
    root_mat = [0] * n
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
            root_mat[r] = node_mat[nid]
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

    # 拆分共享面：同根为组件内部面，异根为组件间接触面
    internal_faces = [0] * n
    contact: dict[tuple[int, int], int] = {}
    for a, b, count in face_pairs:
        ra, rb = dsu.find(a), dsu.find(b)
        if ra == rb:
            internal_faces[ra] += count
        else:
            key = (ra, rb) if ra < rb else (rb, ra)
            contact[key] = contact.get(key, 0) + count

    contact_total = [0] * n
    for (ra, rb), area in contact.items():
        contact_total[ra] += area
        contact_total[rb] += area

    roots = sorted(seen, key=lambda r: (key_z[r], key_y[r], key_x[r]))
    root_to_cid = {r: cid for cid, r in enumerate(roots, start=1)}

    components: list[MaterialComponent] = []
    for cid, r in enumerate(roots, start=1):
        voxels = dsu.size[r]
        # 暴露给空白的面 = 总面 - 内部面*2 - 与他组件的接触面
        exposed = 6 * voxels - 2 * internal_faces[r] - contact_total[r]
        bbox = BoundingBox(
            min_z[r], min_y[r], min_x[r], max_z[r], max_y[r], max_x[r]
        )
        components.append(
            MaterialComponent(
                id=cid,
                material=root_mat[r],
                voxel_count=voxels,
                bbox=bbox,
                exposed_area=exposed,
                min_z=key_z[r],
                min_y=key_y[r],
                min_x=key_x[r],
            )
        )

    contacts: list[ContactPair] = []
    for (ra, rb), area in contact.items():
        ca, cb = root_to_cid[ra], root_to_cid[rb]
        if ca > cb:
            ca, cb = cb, ca
        contacts.append(ContactPair(ca, cb, area))
    contacts.sort(key=lambda c: (c.component_a, c.component_b))

    run_labels = [
        [[root_to_cid[dsu.find(nid)] for nid in row_ids] for row_ids in sl]
        for sl in nodes_by
    ]

    return MaterialLabelResult(
        run_labels=run_labels,
        components=components,
        contacts=contacts,
        connectivity=connectivity,
        num_runs=total_runs,
    )
