"""流式逐切片的游程连通组件标记。

与一次性接口 rle_components.label 等价的增量形态：调用方按 z
递增顺序逐切片调用 append_slice，模块在每张切片处理后**尽早**返回
已经确定不会再与后续切片相连的组件；仍有机会连接下一切片的组件继续
保留。输入以 finish() 明确结束后，结算并返回全部剩余组件。

一致性保证
----------
* 每个组件的体素数、暴露面面积、包围盒、最小体素坐标与 label
  的离线结果**完全一致**（同一套校验、触达判定与面积计数代码路径）；
* 组件 ID 按**结算顺序** 1 起始连续分配，一般不等于离线 ID；把流式
  输出的全部组件按 (min_z, min_y, min_x) 排序后，即逐一对应
  离线 label 的组件顺序与 ID；
* 每个结算组件附带 runs：其包含的 (z, y, run_index) 游程
  定位，与离线 run_labels 逐游程对应。

内存与释放
----------
任何时刻只保留当前切片与前一切片的游程数据；已结算组件的并查集聚合
状态（包围盒、面积、游程定位等）在返回时即删除，不再持有历史切片。

错误语义
--------
单切片校验与 label 完全相同（同一 _normalize_slice），非法
切片抛 InvalidVolumeError 且**不改动任何内部状态**，修正后
可继续追加。切片数上限 512、游程总数上限 20000 与一次性接口一致。
"""

from __future__ import annotations

from dataclasses import dataclass

from .core import (
    MAX_DIM,
    BoundingBox,
    InvalidVolumeError,
    _check_connectivity,
    _check_dim,
    _DSU,
    _normalize_slice,
    _pairs,
)

__all__ = ["StreamComponent", "StreamingLabeler"]


@dataclass(frozen=True)
class StreamComponent:
    """流式接口结算出的一个连通组件。

    统计字段含义与 rle_components.Component 完全一致；
    id 按结算顺序分配（见模块文档的排序说明）。

    Attributes:
        runs: 该组件包含的全部游程定位 (z, y, run_index)，
            按字典序升序。
    """

    id: int
    voxel_count: int
    bbox: BoundingBox
    exposed_area: int
    min_z: int
    min_y: int
    min_x: int
    runs: tuple[tuple[int, int, int], ...]


class StreamingLabeler:
    """按 z 递增逐切片追加的连通组件标记器。

    Args:
        height: y 方向尺寸（每切片行数），1..512。
        width: x 方向尺寸，1..512。
        connectivity: 6（仅面）/ 18（面+边）/ 26（面+边+角）。

    用法::

        lab = StreamingLabeler(height, width, connectivity=26)
        for rows in slices:                # rows: [y] -> [(start, end), ...]
            settled = lab.append_slice(rows)   # 尽早返回已确定的组件
        settled = lab.finish()                 # 结算剩余组件
    """

    def __init__(self, height: int, width: int, connectivity: int = 6) -> None:
        self._height = _check_dim(height, "height")
        self._width = _check_dim(width, "width")
        self._connectivity = _check_connectivity(connectivity)
        self._edge = connectivity >= 18
        self._corner = connectivity == 26

        self._dsu = _DSU(0)
        # 并查集根上的聚合状态；组件结算时即从各表删除
        self._min_key: dict[int, tuple[int, int, int]] = {}
        self._bbox: dict[int, list[int]] = {}
        self._faces: dict[int, int] = {}
        self._runs: dict[int, list[tuple[int, int, int]]] = {}
        self._last_z: dict[int, int] = {}

        self._prev_rows: list[list[tuple[int, int]]] | None = None
        self._prev_ids: list[list[int]] | None = None
        self._z = 0
        self._num_runs = 0
        self._next_cid = 1
        self._finished = False

    # ------------------------------------------------------------------
    # 只读状态
    # ------------------------------------------------------------------

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
    def num_slices(self) -> int:
        """已追加的切片数。"""

        return self._z

    @property
    def num_runs(self) -> int:
        """已追加的游程总数。"""

        return self._num_runs

    @property
    def pending_count(self) -> int:
        """仍有机会连接后续切片、尚未结算的组件数。"""

        return len(self._last_z)

    @property
    def finished(self) -> bool:
        return self._finished

    # ------------------------------------------------------------------
    # 内部：并查集合并 + 根聚合
    # ------------------------------------------------------------------

    def _union(self, a: int, b: int) -> int:
        dsu = self._dsu
        ra, rb = dsu.find(a), dsu.find(b)
        if ra == rb:
            return ra
        root = dsu.union(a, b)
        absorbed = rb if root == ra else ra

        if self._min_key[absorbed] < self._min_key[root]:
            self._min_key[root] = self._min_key[absorbed]
        ba = self._bbox[root]
        bb = self._bbox[absorbed]
        for k in (0, 1, 2):
            if bb[k] < ba[k]:
                ba[k] = bb[k]
        for k in (3, 4, 5):
            if bb[k] > ba[k]:
                ba[k] = bb[k]
        self._faces[root] += self._faces.pop(absorbed)

        root_runs = self._runs[root]
        absorbed_runs = self._runs.pop(absorbed)
        if len(root_runs) < len(absorbed_runs):
            root_runs, absorbed_runs = absorbed_runs, root_runs
        root_runs.extend(absorbed_runs)
        self._runs[root] = root_runs

        if self._last_z[absorbed] > self._last_z[root]:
            self._last_z[root] = self._last_z[absorbed]

        del self._min_key[absorbed]
        del self._bbox[absorbed]
        del self._last_z[absorbed]
        return root

    def _add_faces(self, nid: int, count: int) -> None:
        """把 count 个内部面邻居计入 nid 当前所在根（须先完成合并）。"""

        root = self._dsu.find(nid)
        self._faces[root] += count

    def _finalize_root(self, root: int) -> StreamComponent:
        """结算一个根：生成输出组件并释放其全部聚合状态。"""

        voxels = self._dsu.size[root]
        faces = self._faces.pop(root)
        min_z, min_y, min_x = self._min_key.pop(root)
        b = self._bbox.pop(root)
        runs = tuple(sorted(self._runs.pop(root)))
        del self._last_z[root]
        comp = StreamComponent(
            id=self._next_cid,
            voxel_count=voxels,
            bbox=BoundingBox(*b),
            exposed_area=6 * voxels - 2 * faces,
            min_z=min_z,
            min_y=min_y,
            min_x=min_x,
            runs=runs,
        )
        self._next_cid += 1
        return comp

    # ------------------------------------------------------------------
    # 公开操作
    # ------------------------------------------------------------------

    def append_slice(self, rows: object) -> list[StreamComponent]:
        """追加下一张切片，返回本次处理后确定结算的组件（可为空）。

        Args:
            rows: 单切片 [y] -> [(start, end), ...]，游程形式与
                rle_components.label 相同；空行用 []。

        Raises:
            InvalidVolumeError: 已 finish()、切片数达 512 上限、
                或切片数据非法（与 label 相同的定位错误）。非法
                切片不改动内部状态，修正后可继续追加。
        """

        if self._finished:
            raise InvalidVolumeError(
                "输入已由 finish() 明确结束，不能再追加切片"
            )
        z = self._z
        if z >= MAX_DIM:
            raise InvalidVolumeError(
                f"切片数超过上限 {MAX_DIM}（depth 必须在 [1, {MAX_DIM}] 内）"
            )
        # 先完整校验，再改动任何内部状态：失败时不留部分变更
        cur_rows, self._num_runs = _normalize_slice(
            rows, self._height, self._width, z, self._num_runs
        )

        height = self._height
        cur_ids: list[list[int]] = []
        for y in range(height):
            row = cur_rows[y]
            ids: list[int] = []
            for ri, (s, e) in enumerate(row):
                nid = self._dsu.add(e - s)
                self._min_key[nid] = (z, y, s)
                self._bbox[nid] = [z, y, s, z + 1, y + 1, e]
                self._faces[nid] = e - s - 1  # 游程内部 x 向面邻居
                self._runs[nid] = [(z, y, ri)]
                self._last_z[nid] = z
                ids.append(nid)
            cur_ids.append(ids)

            # --- x 方向：同行游程，仅端点相接（面）才合并 ---
            for ri in range(1, len(ids)):
                if row[ri - 1][1] == row[ri][0]:
                    self._union(ids[ri - 1], ids[ri])
                    self._add_faces(ids[ri], 1)

            # --- y 方向：同切片 y-1 行 ---
            if y - 1 >= 0:
                other = cur_rows[y - 1]
                other_ids = cur_ids[y - 1]
                for i, j, ov in _pairs(row, other):
                    if ov > 0:
                        self._union(ids[i], other_ids[j])
                        self._add_faces(ids[i], ov)
                    elif self._edge:
                        self._union(ids[i], other_ids[j])

            # --- z 方向：前一切片的 y-1 / y / y+1 三行 ---
            if self._prev_rows is not None:
                for dy in (-1, 0, 1):
                    py = y + dy
                    if not 0 <= py < height:
                        continue
                    prow = self._prev_rows[py]
                    pids = self._prev_ids[py]
                    for i, j, ov in _pairs(row, prow):
                        if dy == 0:
                            if ov > 0:
                                self._union(ids[i], pids[j])
                                self._add_faces(ids[i], ov)
                            elif self._edge:
                                self._union(ids[i], pids[j])
                        elif ov > 0:
                            if self._edge:
                                self._union(ids[i], pids[j])
                        elif self._corner:
                            self._union(ids[i], pids[j])

        # 前一切片的数据此后不再被任何合并引用，立即释放
        self._prev_rows = cur_rows
        self._prev_ids = cur_ids
        self._z = z + 1

        # 尽早结算：处理后仍不含当前切片游程的根，不可能再与后续
        # 切片（只能触及当前切片游程）相连
        roots = [
            root for root, last in list(self._last_z.items()) if last < z
        ]
        # 同一批内按最小体素序结算，保证 ID 确定性且批次内与离线序一致
        roots.sort(key=lambda r: self._min_key[r])
        return [self._finalize_root(root) for root in roots]

    def finish(self) -> list[StreamComponent]:
        """明确结束输入，结算并返回全部剩余组件（按最小体素排序）。

        调用后不可再 append_slice，也不可重复 finish。
        """

        if self._finished:
            raise InvalidVolumeError("finish() 只能调用一次，输入已结束")
        self._finished = True
        roots = sorted(self._last_z, key=lambda r: self._min_key[r])
        settled = [self._finalize_root(root) for root in roots]
        # 释放不再需要的切片数据
        self._prev_rows = None
        self._prev_ids = None
        return settled
