"""游程连通组件标记核心算法。

输入为按行给出的半开 x 游程，算法不展开整个体：逐切片扫描，仅在当前切片
相邻行与前一切片相关行之间做区间求交，通过并查集合并。所有统计量（体素数、
包围盒、暴露面数）均为精确整数运算，不使用浮点。
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math

MAX_DIM = 512
MAX_RUNS = 20000
CONNECTIVITIES = (6, 18, 26)


@dataclass(frozen=True)
class Run:
    """一行内的半开 x 游程 [x0, x1)，位于体素行 (z, y)。"""

    z: int
    y: int
    x0: int
    x1: int

    def __iter__(self):
        return iter((self.z, self.y, self.x0, self.x1))


@dataclass(frozen=True)
class BoundingBox:
    """半开包围盒：x∈[x0,x1), y∈[y0,y1), z∈[z0,z1)。"""

    x0: int
    y0: int
    z0: int
    x1: int
    y1: int
    z1: int

    def __iter__(self):
        return iter((self.x0, self.y0, self.z0, self.x1, self.y1, self.z1))


@dataclass(frozen=True)
class Component:
    """一个连通组件。

    id: 1 起始、按组件最小体素 (z,y,x) 排序后的编号。
    origin: 组件的最小体素坐标 (z,y,x)。
    voxel_count: 体素总数。
    bbox: 半开轴对齐包围盒。
    exposed_area: 暴露面数量（单位体素面，精确整数；
        恒等于 6*voxel_count - 2*组件内 6 邻接体素面对数）。
    """

    id: int
    origin: tuple[int, int, int]
    voxel_count: int
    bbox: BoundingBox
    exposed_area: int


@dataclass(frozen=True)
class LabelingResult:
    """label_volume 的结果。

    labels: 与输入 runs 等长、按下标对齐的组件 id；长度为 0 的空游程对应 None。
    components: 按 id 升序的组件元组（components[cid-1].id == cid）。
    """

    dims: tuple[int, int, int]
    connectivity: int
    labels: tuple[int | None, ...]
    components: tuple[Component, ...]

    @property
    def num_components(self) -> int:
        return len(self.components)


def _check_int(value, what: str) -> int:
    """整数参数校验：拒绝 bool、NaN/Infinity 与非整数浮点。"""
    if isinstance(value, bool):
        raise TypeError(f"{what} 必须是 int，不能是 bool")
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"{what} 不能是 NaN 或 Infinity")
        raise TypeError(f"{what} 必须是 int，收到 float: {value!r}")
    if not isinstance(value, int):
        raise TypeError(f"{what} 必须是 int，收到 {type(value).__name__}")
    return value


def _validate_dims(dims) -> tuple[int, int, int]:
    if isinstance(dims, (str, bytes)) or not isinstance(dims, (tuple, list)):
        raise TypeError("dims 必须是长度为 3 的 (nx, ny, nz) 序列")
    if len(dims) != 3:
        raise ValueError(f"dims 必须恰好包含 3 个维度，收到 {len(dims)} 个")
    out = []
    for axis, value in zip(("nx", "ny", "nz"), dims):
        v = _check_int(value, f"dims.{axis}")
        if not 1 <= v <= MAX_DIM:
            raise ValueError(f"dims.{axis}={v} 超出接口边界 [1, {MAX_DIM}]")
        out.append(v)
    return tuple(out)


def _validate_connectivity(connectivity) -> int:
    c = _check_int(connectivity, "connectivity")
    if c not in CONNECTIVITIES:
        raise ValueError(f"connectivity 只支持 {CONNECTIVITIES}，收到 {c}")
    return c


def _parse_runs(runs, nx: int, ny: int, nz: int):
    """解析并完整校验游程序列。

    返回 (records, labels, by_row)：
    records 为正长度游程 (z, y, x0, x1, input_index) 列表（保持输入顺序）；
    labels 为等长占位列表，空游程位置固定为 None；
    by_row 按 (z,y) 分组且组内按 x 升序。
    全部校验在任何标记计算之前完成：失败时不产生任何结果对象。
    """
    if isinstance(runs, (str, bytes)) or not hasattr(runs, "__len__"):
        raise TypeError("runs 必须是由 Run 或 4 元组构成的有限序列")
    total = len(runs)
    if total > MAX_RUNS:
        raise ValueError(f"游程数量 {total} 超过接口上限 {MAX_RUNS}")

    labels: list[int | None] = [None] * total
    records = []
    for i in range(total):
        item = runs[i]
        if isinstance(item, Run):
            z, y, x0, x1 = item.z, item.y, item.x0, item.x1
        elif isinstance(item, (tuple, list)):
            if len(item) != 4:
                raise ValueError(
                    f"runs[{i}] 必须含 4 个字段 (z, y, x0, x1)，"
                    f"实际长度 {len(item)}"
                )
            z, y, x0, x1 = item
        else:
            raise TypeError(
                f"runs[{i}] 必须是 Run 或 4 元组，收到 {type(item).__name__}"
            )

        where = f"runs[{i}]"
        z = _check_int(z, f"{where}.z")
        y = _check_int(y, f"{where}.y")
        x0 = _check_int(x0, f"{where}.x0")
        x1 = _check_int(x1, f"{where}.x1")
        if not 0 <= z < nz:
            raise ValueError(f"{where}.z={z} 超出 [0, {nz - 1}]")
        if not 0 <= y < ny:
            raise ValueError(f"{where}.y={y} 超出 [0, {ny - 1}]")
        if not 0 <= x0 <= nx:
            raise ValueError(f"{where}.x0={x0} 超出 [0, {nx}]")
        if not 0 <= x1 <= nx:
            raise ValueError(f"{where}.x1={x1} 超出 [0, {nx}]")
        if x0 > x1:
            raise ValueError(f"{where}: 半开区间要求 x0 <= x1，收到 [{x0}, {x1})")
        if x0 == x1:
            # 长度 0 的空游程：契约允许，不参与标记，labels 对应 None。
            continue
        records.append((z, y, x0, x1, i))

    by_row: dict[tuple[int, int], list] = {}
    for rec in records:
        by_row.setdefault((rec[0], rec[1]), []).append(rec)
    for (z, y), row in by_row.items():
        row.sort(key=lambda r: (r[2], r[3]))
        for j in range(1, len(row)):
            px0, px1 = row[j - 1][2], row[j - 1][3]
            cx0, cx1 = row[j][2], row[j][3]
            if cx0 < px1:
                raise ValueError(
                    f"行 (z={z}, y={y}) 游程重叠: "
                    f"[{px0},{px1}) 与 [{cx0},{cx1})"
                )
    return records, labels, by_row


class _DSU:
    __slots__ = ("parent",)

    def __init__(self, n: int):
        self.parent = list(range(n))

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
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def label_volume(dims, runs, connectivity: int = 6) -> LabelingResult:
    """对稀疏二值体做游程连通组件标记。

    参数:
        dims: (nx, ny, nz)，每维 1..512。
        runs: Run 或 (z, y, x0, x1) 的序列；[x0,x1) 为半开区间。
            允许空游程（x0==x1，标签为 None）；输入顺序任意，同一
            (z,y) 行内游程不得重叠（端点相接合法，算面邻接）。
            游程总数（含空游程）<= 20000。
        connectivity: 6（仅面）/18（面+边，不含角）/26（面+边+角）。

    返回:
        LabelingResult。组件 id 从 1 开始，按组件最小体素 (z,y,x) 升序。

    异常:
        TypeError: 类型错误（含把 bool 当 int、把 float 当整数坐标等）。
        ValueError: 取值越界、NaN/Infinity、区间非法、游程重叠、超上限等。
        所有错误在任何标记计算之前抛出，不存在部分结果。
    """
    nx, ny, nz = _validate_dims(dims)
    conn = _validate_connectivity(connectivity)
    _records, labels, by_row = _parse_runs(runs, nx, ny, nz)

    n_runs = sum(len(row) for row in by_row.values())
    dsu = _DSU(n_runs)
    # 每个游程的初始暴露面：6L 减去同行内部邻接重复计数 2(L-1) = 4L+2。
    exposed: list[int] = []
    rz: list[int] = []
    ry: list[int] = []
    rx0: list[int] = []
    rx1: list[int] = []
    rlen: list[int] = []
    rid: dict[int, int] = {}
    # 仅保留当前切片与前一切片的行：(ids, x0s, x1s)，x 单调升序。
    row_struct: dict[tuple[int, int], tuple[list, list, list]] = {}
    next_id = 0

    def merge(a: int, b: int, faces: int) -> None:
        # 每处体素级面邻接从双方各扣 1（初始 6N 记账中该面被计了两次）。
        # 边/角接触 faces=0：只合并，不影响面积。
        dsu.union(a, b)
        if faces:
            exposed[a] -= faces
            exposed[b] -= faces

    def scan(row_key, a0, a1, closed, face_capable):
        """在相关行 row_key 中找邻接游程：合并并按面邻接数扣面积。

        closed: 是否承认 |dx|=1 的端点相接（边/角连通）。
        face_capable: 该方向行差是否可能共享面
            （同切片 |dy|=1 或跨切片 dy=0 为 True；跨切片 |dy|=1 恒为 False）。
        """
        struct = row_struct.get(row_key)
        if struct is None:
            return
        ids, xs0, xs1 = struct
        # 首个可能候选：closed 需 b1 >= a0；否则需 b1 > a0。
        start = bisect_right(xs1, a0 - 1) if closed else bisect_right(xs1, a0)
        for j in range(start, len(ids)):
            b0, b1 = xs0[j], xs1[j]
            if closed:
                if b0 > a1:  # |dx|>=2，之后的游程只会更远
                    break
            elif b0 >= a1:
                break
            dsu.union(next_id_local[0], ids[j])
            if face_capable:
                overlap = min(a1, b1) - max(a0, b0)
                if overlap > 0:
                    exposed[next_id_local[0]] -= overlap
                    exposed[ids[j]] -= overlap

    # 闭包内引用“当前游程 id”，用单元素列表避免 nonlocal 声明噪音。
    next_id_local = [0]

    prev_z = -1
    for z, y in sorted(by_row.keys()):
        if z != prev_z:
            # 进入新切片：z-2 及更早的行永不再需要（6/18/26 都只看 dz<=1）。
            if prev_z >= 0:
                stale = [k for k in row_struct if k[0] <= z - 2]
                for k in stale:
                    del row_struct[k]
            prev_z = z

        cur_ids: list[int] = []
        cur_x0: list[int] = []
        cur_x1: list[int] = []
        for _rz, _ry, a0, a1, input_index in by_row[(z, y)]:
            my = next_id
            next_id += 1
            next_id_local[0] = my
            rid[input_index] = my
            rz.append(z)
            ry.append(y)
            rx0.append(a0)
            rx1.append(a1)
            rlen.append(a1 - a0)
            # 初始暴露面：6L 个面减去同行内部 (L-1) 对邻接各重复计的 2 面
            # = 4L+2（±x 两端面 + y/z 四个沿长面）。
            exposed.append(4 * (a1 - a0) + 2)
            cur_ids.append(my)
            cur_x0.append(a0)
            cur_x1.append(a1)

            # 1) 同行前一游程：端点相接即面邻接（三种邻接均连通）。
            if len(cur_ids) >= 2 and rx1[cur_ids[-2]] == a0:
                merge(my, cur_ids[-2], 1)

            # 2) 当前切片相邻行 (dz=0, |dy|=1)：
            #    面邻接需 x 开相交；18/26 额外承认端点相接（边）。
            if y - 1 >= 0:
                scan((z, y - 1), a0, a1, conn >= 18, True)
            if y + 1 < ny:
                scan((z, y + 1), a0, a1, conn >= 18, True)

            # 3) 前一切片 (|dz|=1)。
            if z - 1 >= 0:
                # dy=0：6 需开相交（面），18/26 承认端点相接（边）。
                scan((z - 1, y), a0, a1, conn >= 18, True)
                if conn >= 18:
                    # |dy|=1：开相交为边（18 承认），端点相接为角（仅 26）。
                    # 该方向不可能共享面。
                    if y - 1 >= 0:
                        scan((z - 1, y - 1), a0, a1, conn >= 26, False)
                    if y + 1 < ny:
                        scan((z - 1, y + 1), a0, a1, conn >= 26, False)

        row_struct[(z, y)] = (cur_ids, cur_x0, cur_x1)

    n = next_id
    if n == 0:
        return LabelingResult(
            dims=(nx, ny, nz),
            connectivity=conn,
            labels=tuple(labels),
            components=(),
        )

    roots = [dsu.find(i) for i in range(n)]
    comp_vox: dict[int, int] = {}
    comp_area: dict[int, int] = {}
    xmin: dict[int, int] = {}
    xmax: dict[int, int] = {}
    ymin: dict[int, int] = {}
    ymax: dict[int, int] = {}
    zmin: dict[int, int] = {}
    zmax: dict[int, int] = {}
    origin: dict[int, tuple[int, int, int]] = {}

    for i in range(n):
        r = roots[i]
        comp_vox[r] = comp_vox.get(r, 0) + rlen[i]
        comp_area[r] = comp_area.get(r, 0) + exposed[i]
        if r not in xmin or rx0[i] < xmin[r]:
            xmin[r] = rx0[i]
        if r not in xmax or rx1[i] > xmax[r]:
            xmax[r] = rx1[i]
        if r not in ymin or ry[i] < ymin[r]:
            ymin[r] = ry[i]
        if r not in ymax or ry[i] > ymax[r]:
            ymax[r] = ry[i]
        if r not in zmin or rz[i] < zmin[r]:
            zmin[r] = rz[i]
        if r not in zmax or rz[i] > zmax[r]:
            zmax[r] = rz[i]
        o = (rz[i], ry[i], rx0[i])
        if r not in origin or o < origin[r]:
            origin[r] = o

    ordered_roots = sorted(comp_vox, key=lambda r: origin[r])
    remap: dict[int, int] = {}
    components: list[Component] = []
    for new_id, r in enumerate(ordered_roots, start=1):
        remap[r] = new_id
        bbox = BoundingBox(
            xmin[r], ymin[r], zmin[r],
            xmax[r], ymax[r] + 1, zmax[r] + 1,
        )
        components.append(
            Component(
                id=new_id,
                origin=origin[r],
                voxel_count=comp_vox[r],
                bbox=bbox,
                exposed_area=comp_area[r],
            )
        )

    for input_index, local in rid.items():
        labels[input_index] = remap[roots[local]]

    return LabelingResult(
        dims=(nx, ny, nz),
        connectivity=conn,
        labels=tuple(labels),
        components=tuple(components),
    )
