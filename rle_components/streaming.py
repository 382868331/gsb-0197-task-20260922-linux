"""流式（逐切片追加）的游程连通组件标记。

与离线 label() 的对应关系：
  * 合并逻辑（同行 x 相接、同切片 y-1 行、前一切片 y-1/y/y+1 三行）与
    面积计数（6*体素 - 2*内部面）完全复用 core 中同一套触达判定，因此
    逐切片追加得到的组件体积、暴露面积、包围盒与一次性离线处理逐一相等。
  * 组件只有在“不可能再与后续切片相连”时才结算：连接只发生在相邻
    切片之间，故处理完切片 z 后，所有体素都位于 z 之前的组件已经封闭。
    仍触及最新切片的组件继续保留，绝不提前结束。
  * 组件 ID 与离线一致地按最小 (z,y,x) 字典序分配：已封闭组件只有在其
    排序键小于所有仍活跃组件的排序键时才发出，保证全局 ID 顺序与
    label() 的输出相同。
  * 内存：任何时刻只保留前一切片的游程与活跃组件的聚合状态；已结算
    组件与更早切片的历史立即释放。
"""

from __future__ import annotations

from .core import (
    MAX_DIM,
    BoundingBox,
    Component,
    InvalidVolumeError,
    _check_connectivity,
    _check_dim,
    _normalize_slice,
    _pairs,
)

__all__ = ["StreamingLabeler"]


class StreamingLabeler:
    """逐切片追加的连通组件标记器（增量版 label()）。

    用法::

        streamer = StreamingLabeler(height, width, connectivity=6)
        for rows in slices:                 # rows: [y] -> [(start, end), ...]
            done = streamer.append_slice(rows)   # 已封闭的组件（可能为空）
        rest = streamer.finish()            # 输入结束，结算剩余组件

    append_slice 返回本次可以确定“不会再与后面任何切片相连”的
    Component 列表（按组件 ID 升序）；finish 返回剩余全部组件。
    把各次返回值按顺序拼接，即得到与离线 label(全部切片, ...) 完全相同
    的组件序列（ID、体素数、包围盒、暴露面积逐项相等）。

    Raises:
        InvalidVolumeError: 尺寸/连通性/切片数据不满足契约（与 label()
            相同的错误语义与定位前缀），或追加切片数超过 512、累计游程数
            超过 20000。
        RuntimeError: 在 finish() 之后继续追加或重复结算。
    """

    def __init__(self, height: int, width: int, connectivity: int = 6) -> None:
        self._height = _check_dim(height, "height")
        self._width = _check_dim(width, "width")
        self._connectivity = _check_connectivity(connectivity)
        self._edge_mode = connectivity >= 18
        self._corner_mode = connectivity == 26

        self._finished = False
        self._z = 0               # 下一个切片的 z 下标
        self._total_runs = 0
        self._next_cid = 1

        # 并查集 + 根上聚合状态（按节点下标平行数组）
        self._parent: list[int] = []
        self._voxels: list[int] = []
        self._faces: list[int] = []     # 组件内部共享面数
        self._min_z: list[int] = []
        self._min_y: list[int] = []
        self._min_x: list[int] = []
        self._max_z: list[int] = []
        self._max_y: list[int] = []
        self._max_x: list[int] = []
        self._key: list[tuple[int, int, int]] = []   # 排序键：最小 (z,y,x)
        self._last_z: list[int] = []    # 组件出现体素的最大 z

        # 仍可能与后续切片相连的根，以及已封闭但等待按序发出的根
        self._active: set[int] = set()
        self._pending: list[int] = []

        # 仅保留前一切片：规范后的游程与对应节点号
        self._prev_rows: list[list[tuple[int, int]]] | None = None
        self._prev_ids: list[list[int]] | None = None

    # -- 只读状态 ------------------------------------------------------

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
        return self._total_runs

    @property
    def finalized_count(self) -> int:
        """已结算（发出）的组件数。"""
        return self._next_cid - 1

    # -- 并查集 --------------------------------------------------------

    def _find(self, x: int) -> int:
        parent = self._parent
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != x:
            parent[x], x = root, parent[x]
        return root

    def _union(self, a: int, b: int) -> int:
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return ra
        if self._voxels[ra] < self._voxels[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        self._voxels[ra] += self._voxels[rb]
        self._faces[ra] += self._faces[rb]
        if self._min_z[rb] < self._min_z[ra]:
            self._min_z[ra] = self._min_z[rb]
        if self._min_y[rb] < self._min_y[ra]:
            self._min_y[ra] = self._min_y[rb]
        if self._min_x[rb] < self._min_x[ra]:
            self._min_x[ra] = self._min_x[rb]
        if self._max_z[rb] > self._max_z[ra]:
            self._max_z[ra] = self._max_z[rb]
        if self._max_y[rb] > self._max_y[ra]:
            self._max_y[ra] = self._max_y[rb]
        if self._max_x[rb] > self._max_x[ra]:
            self._max_x[ra] = self._max_x[rb]
        if self._key[rb] < self._key[ra]:
            self._key[ra] = self._key[rb]
        if self._last_z[rb] > self._last_z[ra]:
            self._last_z[ra] = self._last_z[rb]
        self._active.discard(rb)
        self._active.add(ra)
        return ra

    # -- 主流程 --------------------------------------------------------

    def append_slice(self, rows: object) -> list[Component]:
        """追加一个切片，返回本次确定封闭的组件（按 ID 升序）。

        Args:
            rows: 单个切片的 [y] -> [(start, end), ...]，行数必须等于
                构造时的 height；游程格式与 label() 相同。
        """

        if self._finished:
            raise RuntimeError("finish() 已调用，不能再追加切片")
        z = self._z
        if z >= MAX_DIM:
            raise InvalidVolumeError(
                f"切片数超过上限 depth<={MAX_DIM}（试图追加第 {z + 1} 片）"
            )
        vol_slice, self._total_runs = _normalize_slice(
            rows, z, self._height, self._width, self._total_runs
        )

        cur_ids: list[list[int]] = []
        for y in range(self._height):
            row = vol_slice[y]
            ids: list[int] = []
            for s, e in row:
                nid = len(self._parent)
                self._parent.append(nid)
                self._voxels.append(e - s)
                self._faces.append(e - s - 1)  # 游程内部 x 向共享面
                self._min_z.append(z)
                self._min_y.append(y)
                self._min_x.append(s)
                self._max_z.append(z + 1)
                self._max_y.append(y + 1)
                self._max_x.append(e)
                self._key.append((z, y, s))
                self._last_z.append(z)
                self._active.add(nid)
                ids.append(nid)
            cur_ids.append(ids)

            # --- x 方向：同行游程端点相接（面接触） ---
            for ri in range(1, len(ids)):
                if row[ri - 1][1] == row[ri][0]:
                    root = self._union(ids[ri - 1], ids[ri])
                    self._faces[root] += 1

            # --- y 方向：同切片 y-1 行 ---
            if y - 1 >= 0:
                other = vol_slice[y - 1]
                other_ids = cur_ids[y - 1]
                for i, j, ov in _pairs(row, other):
                    if ov > 0:
                        root = self._union(ids[i], other_ids[j])
                        self._faces[root] += ov
                    elif self._edge_mode:
                        self._union(ids[i], other_ids[j])

            # --- z 方向：前一切片的 y-1 / y / y+1 三行 ---
            if self._prev_rows is not None:
                for dy in (-1, 0, 1):
                    py = y + dy
                    if not 0 <= py < self._height:
                        continue
                    prow = self._prev_rows[py]
                    pids = self._prev_ids[py]
                    for i, j, ov in _pairs(row, prow):
                        if dy == 0:
                            if ov > 0:
                                root = self._union(ids[i], pids[j])
                                self._faces[root] += ov
                            elif self._edge_mode:
                                self._union(ids[i], pids[j])
                        elif ov > 0:
                            if self._edge_mode:
                                self._union(ids[i], pids[j])
                        elif self._corner_mode:
                            self._union(ids[i], pids[j])

        self._prev_rows = vol_slice
        self._prev_ids = cur_ids
        self._z = z + 1
        return self._finalize_closed()

    def finish(self) -> list[Component]:
        """声明输入结束，结算并返回剩余全部组件（按 ID 升序）。"""

        if self._finished:
            raise RuntimeError("finish() 只能调用一次")
        self._finished = True
        roots = sorted(self._pending + list(self._active),
                       key=lambda r: self._key[r])
        self._pending = []
        self._active = set()
        self._prev_rows = None   # 释放最后的切片历史
        self._prev_ids = None
        return self._emit(roots)

    # -- 结算 ----------------------------------------------------------

    def _finalize_closed(self) -> list[Component]:
        """结算“不可能再与后续切片相连”的组件。

        处理完切片 z 后，只有触及切片 z 的组件还可能生长（连接只发生在
        相邻切片之间）；其余根已封闭。为保证组件 ID 与离线一致（按最小
        (z,y,x) 字典序），已封闭根只有在排序键小于全部活跃根的排序键时
        才发出，否则留待后续切片推进后再发。
        """

        # self._z 已指向下一切片；刚处理完的切片号为 self._z - 1。
        # 连接只发生在相邻切片之间，故最后出现于 self._z - 2 或更早的
        # 组件已不可能再生长。
        closed = [r for r in self._active if self._last_z[r] < self._z - 1]
        for r in closed:
            self._active.discard(r)
        self._pending.extend(closed)

        if self._active:
            threshold = min(self._key[r] for r in self._active)
            ready = [r for r in self._pending if self._key[r] < threshold]
        else:
            ready = self._pending
        if not ready:
            return []
        ready_set = set(ready)
        self._pending = [r for r in self._pending if r not in ready_set]
        ready.sort(key=lambda r: self._key[r])
        return self._emit(ready)

    def _emit(self, roots: list[int]) -> list[Component]:
        out: list[Component] = []
        for r in roots:
            cid = self._next_cid
            self._next_cid += 1
            voxels = self._voxels[r]
            kz, ky, kx = self._key[r]
            out.append(
                Component(
                    id=cid,
                    voxel_count=voxels,
                    bbox=BoundingBox(
                        self._min_z[r],
                        self._min_y[r],
                        self._min_x[r],
                        self._max_z[r],
                        self._max_y[r],
                        self._max_x[r],
                    ),
                    exposed_area=6 * voxels - 2 * self._faces[r],
                    min_z=kz,
                    min_y=ky,
                    min_x=kx,
                )
            )
        return out
