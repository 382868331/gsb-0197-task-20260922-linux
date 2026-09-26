"""封闭空腔查询：RLE 二值体中被前景完全包围的背景区域。

定义（固定邻接对，不可配置）
---------------------------
* 前景：6 邻接（仅面接触），组件 ID 沿用 label(connectivity=6) 的原
  规则（按最小 (z, y, x) 字典序 1 起始）。
* 背景：26 邻接（面 + 边 + 角）。因此只开一个角点的孔道会让内外背景
  连通，内部不再是空腔。
* 范围：调用方给出的有限包围盒 box（半开区间）。只统计盒内体素；
  与盒外连通的背景（即接触盒的任一边界面）不是空腔。边界上的洞口
  因此天然保留：开口凹槽的背景总能触达盒边界面，不会被误判为闭腔。

实现要点
--------
背景不逐体素展开：每行的背景表示为前景游程在盒内 x 区间的补集游程，
行间用与前景相同的单调游标重叠扫描做并查集合并。背景游程总数
<= 前景游程数 + 盒内行数，巨大空白盒只会产生每行一条背景游程。

接触前景组件：空腔体素与盒内前景体素面接触（6 邻接）时，该前景组件
计入 touches。盒外前景不参与接触统计（盒外体素本就不在范围内）。
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass

from .core import (
    BoundingBox,
    InvalidVolumeError,
    LabelResult,
    _check_dim,
    _DSU,
    _normalize_runs,
    _pairs,
    label,
)

__all__ = [
    "Cavity",
    "CavityResult",
    "find_cavities",
]

_BOX_KEYS = ("min_z", "min_y", "min_x", "max_z", "max_y", "max_x")


@dataclass(frozen=True)
class Cavity:
    """一个封闭空腔的聚合结果。

    Attributes:
        id: 1 起始空腔 ID，按空腔最小体素 (z, y, x) 字典序升序。
        voxel_count: 空腔体素总数（体积，仅计盒内体素）。
        bbox: 轴对齐包围盒（半开区间，各坐标均在 box 内）。
        touches: 与该空腔面接触的盒内前景组件 ID（升序元组，
            ID 规则与 label(..., connectivity=6) 一致）。
    """

    id: int
    voxel_count: int
    bbox: BoundingBox
    touches: tuple[int, ...]


@dataclass(frozen=True)
class CavityResult:
    """find_cavities 的返回值。

    Attributes:
        cavities: 封闭空腔列表，ID 为 1..N，按最小 (z, y, x) 升序。
        box: 实际使用的查询范围（半开包围盒）。
        foreground: 整体前景在 6 邻接下的标记结果（组件 ID 即
            Cavity.touches 引用的 ID）。
    """

    cavities: list[Cavity]
    box: BoundingBox
    foreground: LabelResult


# ---------------------------------------------------------------------------
# box 校验
# ---------------------------------------------------------------------------


def _check_box_coord(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidVolumeError(
            f"{where} 必须为整数（拒绝 bool/float），"
            f"得到 {type(value).__name__}: {value!r}"
        )
    return value


def _normalize_box(box: object, depth: int, height: int, width: int) -> BoundingBox:
    if isinstance(box, BoundingBox):
        raw: list[object] = list(box)
    elif isinstance(box, dict):
        if set(box) != set(_BOX_KEYS):
            raise InvalidVolumeError(
                f"box 字典只能含键 {list(_BOX_KEYS)}，得到 {sorted(box)}"
            )
        raw = [box[k] for k in _BOX_KEYS]
    elif isinstance(box, Sequence) and not isinstance(box, (str, bytes)):
        raw = list(box)
        if len(raw) != 6:
            raise InvalidVolumeError(
                f"box 必须为 6 元序列 (min_z, min_y, min_x, max_z, max_y, max_x)，"
                f"得到长度 {len(raw)}"
            )
    else:
        raise InvalidVolumeError(
            f"box 必须为 BoundingBox、6 元序列或含 {list(_BOX_KEYS)} 的字典，"
            f"得到 {type(box).__name__}: {box!r}"
        )

    vals = [_check_box_coord(v, f"box.{k}") for k, v in zip(_BOX_KEYS, raw)]
    limits = (depth, height, width)
    names = ("depth", "height", "width")
    for axis in range(3):
        lo, hi = vals[axis], vals[axis + 3]
        if not 0 <= lo < hi <= limits[axis]:
            raise InvalidVolumeError(
                f"box.{'zyx'[axis]} 区间必须满足 0 <= min < max <= "
                f"{names[axis]}({limits[axis]})，得到 min={lo}, max={hi}"
            )
    return BoundingBox(*vals)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def find_cavities(
    runs: object,
    depth: int,
    height: int,
    width: int,
    box: object,
) -> CavityResult:
    """在有限包围盒内查找被前景完全包围的封闭空腔。

    Args:
        runs: 与 label 相同的 RLE 输入。
        depth / height / width: 体尺寸，1..512。
        box: 查询范围，BoundingBox、(min_z, min_y, min_x, max_z,
            max_y, max_x) 六元序列或同名字典，半开区间，必须落在
            体数据范围内且各轴非空。

    Returns:
        CavityResult。空腔为盒内 26 邻接背景中不接触盒边界面的
        连通区域；touches 为面接触的盒内前景组件 ID。

    Raises:
        InvalidVolumeError: 体数据或 box 不满足契约（定位消息同
            label；box 错误带 box.<轴> 前缀）。
    """

    depth = _check_dim(depth, "depth")
    height = _check_dim(height, "height")
    width = _check_dim(width, "width")
    box = _normalize_box(box, depth, height, width)
    vol = _normalize_runs(runs, depth, height, width)

    # 前景组件：固定 6 邻接，ID 规则与基线 label 完全一致
    fg = label(vol, depth, height, width, connectivity=6)

    z0, y0, x0 = box.min_z, box.min_y, box.min_x
    z1, y1, x1 = box.max_z, box.max_y, box.max_x

    # ------------------------------------------------------------------
    # 盒内背景游程：每行 = 前景游程在 [x0, x1) 上的补集
    # ------------------------------------------------------------------
    bg: list[list[list[tuple[int, int]]]] = []  # 相对盒坐标的 [z][y]
    for z in range(z0, z1):
        bg_slice: list[list[tuple[int, int]]] = []
        for y in range(y0, y1):
            row: list[tuple[int, int]] = []
            cursor = x0
            for s, e in vol[z][y]:
                cs = s if s > x0 else x0
                ce = e if e < x1 else x1
                if cs >= x1 or ce <= x0:
                    continue
                if cursor < cs:
                    row.append((cursor, cs))
                cursor = ce
            if cursor < x1:
                row.append((cursor, x1))
            bg_slice.append(row)
        bg.append(bg_slice)

    # ------------------------------------------------------------------
    # 背景游程并查集（26 邻接）：同切片 y-1 行 + 前一切片 y-1/y/y+1 行，
    # 凡是 x 距离 <= 1 的游程对全部合并（_pairs 的触达语义）。
    # ------------------------------------------------------------------
    total = sum(len(row) for sl in bg for row in sl)
    dsu = _DSU(total)
    node_z: list[int] = []
    node_y: list[int] = []
    node_x0: list[int] = []
    node_x1: list[int] = []
    node_outside: list[bool] = []  # 接触盒边界面 -> 与盒外连通

    prev_nodes: list[list[int]] | None = None
    next_id = 0
    for zi, z in enumerate(range(z0, z1)):
        cur_nodes: list[list[int]] = []
        for yi, y in enumerate(range(y0, y1)):
            row = bg[zi][yi]
            ids: list[int] = []
            for s, e in row:
                nid = next_id
                next_id += 1
                dsu.size[nid] = e - s
                node_z.append(z)
                node_y.append(y)
                node_x0.append(s)
                node_x1.append(e)
                node_outside.append(
                    z == z0 or z == z1 - 1
                    or y == y0 or y == y1 - 1
                    or s == x0 or e == x1
                )
                ids.append(nid)
            cur_nodes.append(ids)

            # 同行背景游程之间必有前景体素相隔，不可能 26 邻接，无需合并
            if yi > 0:
                other = bg[zi][yi - 1]
                other_ids = cur_nodes[yi - 1]
                for i, j, _ov in _pairs(row, other):
                    dsu.union(ids[i], other_ids[j])

            if prev_nodes is not None:
                for dy in (-1, 0, 1):
                    py = yi + dy
                    if not 0 <= py < y1 - y0:
                        continue
                    prow = bg[zi - 1][py]
                    pids = prev_nodes[py]
                    for i, j, _ov in _pairs(row, prow):
                        dsu.union(ids[i], pids[j])

        prev_nodes = cur_nodes

    # ------------------------------------------------------------------
    # 接触前景检测：对每条背景游程查 6 邻接的盒内前景游程。
    # 每行前景用 starts/ends 两个有序数组 + 二分定位，不展开体素。
    # ------------------------------------------------------------------
    fg_starts: dict[tuple[int, int], list[int]] = {}
    fg_ends: dict[tuple[int, int], list[int]] = {}
    fg_cids: dict[tuple[int, int], list[int]] = {}
    for z in range(z0, z1):
        for y in range(y0, y1):
            frow = vol[z][y]
            fg_starts[(z, y)] = [s for s, _ in frow]
            fg_ends[(z, y)] = [e for _, e in frow]
            fg_cids[(z, y)] = fg.run_labels[z][y]

    touches: list[set[int]] = [set() for _ in range(total)]

    def add_overlap_contacts(nid: int, s: int, e: int, z: int, y: int) -> None:
        # 把 (z, y) 行内与 [s, e) 严格重叠的前景组件记入 touches
        starts = fg_starts[(z, y)]
        ends = fg_ends[(z, y)]
        cids = fg_cids[(z, y)]
        lo = bisect_right(ends, s)       # 第一个 end > s
        hi = bisect_left(starts, e)      # 第一个 start >= e
        for k in range(lo, hi):
            touches[nid].add(cids[k])

    for nid in range(total):
        z, y, s, e = node_z[nid], node_y[nid], node_x0[nid], node_x1[nid]
        # 同行左右邻居（x = s-1 与 x = e 处的前景体素）
        if s > x0:
            ends = fg_ends[(z, y)]
            k = bisect_left(ends, s)
            if k < len(ends) and ends[k] == s:
                touches[nid].add(fg_cids[(z, y)][k])
        if e < x1:
            starts = fg_starts[(z, y)]
            k = bisect_left(starts, e)
            if k < len(starts) and starts[k] == e:
                touches[nid].add(fg_cids[(z, y)][k])
        # 同切片 y±1 行、前/后切片同 y 行的 x 重叠前景
        if y > y0:
            add_overlap_contacts(nid, s, e, z, y - 1)
        if y < y1 - 1:
            add_overlap_contacts(nid, s, e, z, y + 1)
        if z > z0:
            add_overlap_contacts(nid, s, e, z - 1, y)
        if z < z1 - 1:
            add_overlap_contacts(nid, s, e, z + 1, y)

    # ------------------------------------------------------------------
    # 聚合：节点创建序 = (z, y, x) 字典序，根首次出现处即空腔最小体素
    # ------------------------------------------------------------------
    key_z = [0] * total
    key_y = [0] * total
    key_x = [0] * total
    min_z = [0] * total
    min_y = [0] * total
    min_x = [0] * total
    max_z = [0] * total
    max_y = [0] * total
    max_x = [0] * total
    root_touches: dict[int, set[int]] = {}

    for nid in range(total):
        r = dsu.find(nid)
        z, y, s, e = node_z[nid], node_y[nid], node_x0[nid], node_x1[nid]
        if r not in root_touches:
            root_touches[r] = set()
            key_z[r] = z
            key_y[r] = y
            key_x[r] = s
            min_z[r] = z
            min_y[r] = y
            min_x[r] = s
            max_z[r] = z + 1
            max_y[r] = y + 1
            max_x[r] = e
        else:
            if z < min_z[r]:
                min_z[r] = z
            if y < min_y[r]:
                min_y[r] = y
            if s < min_x[r]:
                min_x[r] = s
            if z + 1 > max_z[r]:
                max_z[r] = z + 1
            if y + 1 > max_y[r]:
                max_y[r] = y + 1
            if e > max_x[r]:
                max_x[r] = e
        root_touches[r] |= touches[nid]

    outside_roots = {
        dsu.find(nid) for nid in range(total) if node_outside[nid]
    }

    roots = sorted(
        (r for r in root_touches if r not in outside_roots),
        key=lambda r: (key_z[r], key_y[r], key_x[r]),
    )

    cavities: list[Cavity] = []
    for cid, r in enumerate(roots, start=1):
        cavities.append(
            Cavity(
                id=cid,
                voxel_count=dsu.size[r],
                bbox=BoundingBox(
                    min_z[r], min_y[r], min_x[r],
                    max_z[r], max_y[r], max_x[r],
                ),
                touches=tuple(sorted(root_touches[r])),
            )
        )

    return CavityResult(cavities=cavities, box=box, foreground=fg)
