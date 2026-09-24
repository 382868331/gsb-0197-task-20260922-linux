"""有状态动态层：批量增删体素后的连通组件重算与分裂/合并事件。

在无状态纯函数 :func:`rle_components.core.label` 之上提供一个可修改的
体素模型 :class:`ComponentModel`：

* :meth:`ComponentModel.add` / :meth:`ComponentModel.remove` /
  :meth:`ComponentModel.update` 接受一批 ``(z, y, x)`` 体素坐标，
  也允许把坐标写成半开 x 游程 ``(z, y, start, end)``；
* 每次提交后**全量重算**连通组件（复用 ``label``），组件 ID 始终按组件
  最小体素坐标 (z, y, x) 字典序 1 起始重新分配；
* 返回 :class:`UpdateResult`，报告本次提交造成的 **split / merge**
  组件事件；
* 任何非法输入都在状态被修改之前拒绝（先全量校验、后写状态；写后重算
  失败也回滚到提交前快照），绝不留下半更新状态。

事件按"旧组件集合 -> 新组件集合"的体素归属变化定义（允许全量重算）：

* **split**：某个提交前组件的存活体素在提交后落到多个新组件
  （删除桥接的典型情形）；
* **merge**：某个提交后组件的存活体素来自多个提交前组件
  （新增桥接的典型情形）。

事件只在真正发生分裂/合并时产生；单纯增删但组件拓扑未变（例如在组件
内部加一个体素、或从棒端点删一个体素）不产生事件。一个旧组件的体素被
整批删光（旧组件 -> 0 个新组件）不属于分裂，不产生 split 事件。

注意：组件 ID 在每次重算后按最小坐标重新分配，因此组件身份不保证跨更新
稳定；事件中同时给出按最小坐标标识的旧/新组件摘要
（:class:`ComponentSnapshot`），供调用方跟随。本模块不提供持久组件谱系
（不在本题范围内）。
"""

from __future__ import annotations

from dataclasses import dataclass

from .core import (
    Component,
    InvalidVolumeError,
    MAX_DIM,
    MAX_RUNS,
    label,
)

__all__ = [
    "ComponentModel",
    "ComponentSnapshot",
    "MergeEvent",
    "SplitEvent",
    "UpdateResult",
]


# ---------------------------------------------------------------------------
# 校验工具（与 core 保持同一套口径：bool/float 不冒称整数）
# ---------------------------------------------------------------------------


def _is_real_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_coord(value: object, where: str) -> int:
    if not _is_real_int(value):
        raise InvalidVolumeError(
            f"{where} 坐标必须为整数（拒绝 bool/float），"
            f"得到 {type(value).__name__}: {value!r}"
        )
    return value  # type: ignore[return-value]


def _check_dim(value: object, what: str) -> int:
    if not _is_real_int(value):
        raise InvalidVolumeError(
            f"{what} 必须为整数（拒绝 bool），得到 "
            f"{type(value).__name__}: {value!r}"
        )
    if not 1 <= value <= MAX_DIM:  # type: ignore[operator]
        raise InvalidVolumeError(
            f"{what} 必须在 [1, {MAX_DIM}] 内，得到 {value}"
        )
    return value  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 事件数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentSnapshot:
    """跨更新识别组件用的不可变摘要（按最小体素坐标唯一）。"""

    min_z: int
    min_y: int
    min_x: int
    voxel_count: int

    @classmethod
    def of(cls, component: Component) -> ComponentSnapshot:
        return cls(
            component.min_z,
            component.min_y,
            component.min_x,
            component.voxel_count,
        )


@dataclass(frozen=True)
class SplitEvent:
    """一次提交造成的组件分裂事件。

    提交前组件 ``affected_components`` 的存活体素，在提交后分散到
    ``resulting_components`` 多个组件；``added_voxels`` 为落入这些
    新组件的本次新增体素数（纯删除型分裂为 0，增删混合提交可非 0）。
    各元组按组件最小坐标升序。
    """

    affected_components: tuple[ComponentSnapshot, ...]
    resulting_components: tuple[ComponentSnapshot, ...]
    added_voxels: int


@dataclass(frozen=True)
class MergeEvent:
    """一次提交造成的组件合并事件。

    提交前的多个组件（``merged_components``）经本次提交后连同
    ``added_voxels`` 个新增体素合并为一个新组件
    ``resulting_component``；``removed_voxels`` 为这些旧组件区域内
    本次被删除的体素数（纯新增型合并为 0，增删混合提交可非 0）。
    """

    merged_components: tuple[ComponentSnapshot, ...]
    resulting_component: ComponentSnapshot
    added_voxels: int
    removed_voxels: int


@dataclass(frozen=True)
class UpdateResult:
    """:meth:`ComponentModel.update` 的返回值。

    Attributes:
        added: 实际生效（原本不存在）的新增体素数。
        removed: 实际生效（原本存在）的删除体素数。
        components: 重算后的完整 :class:`Component` 元组（与
            :attr:`ComponentModel.components` 同一对象）。
        splits: 本次提交产生的分裂事件（按受影响旧组件最小坐标升序）。
        merges: 本次提交产生的合并事件（按合并后新组件最小坐标升序）。
        connectivity: 实际使用的连通性。
        num_voxels: 提交后模型中的体素总数。
    """

    added: int
    removed: int
    components: tuple[Component, ...]
    splits: tuple[SplitEvent, ...]
    merges: tuple[MergeEvent, ...]
    connectivity: int
    num_voxels: int

    @property
    def num_components(self) -> int:
        return len(self.components)


# ---------------------------------------------------------------------------
# 批次解析与编码
# ---------------------------------------------------------------------------


def _parse_batch(batch: object, where: str) -> set[tuple[int, int, int]]:
    """把一批混合坐标/游程展开为体素集合。

    批次按集合语义处理：重复指定同一体素幂等去重，不报错。
    坐标的尺寸越界检查由调用方按模型 depth/height/width 统一完成
    （游程的 end 与 width 相关，在此先做形状与整数校验）。
    """

    if (
        isinstance(batch, (str, bytes, dict))
        or not hasattr(batch, "__iter__")
    ):
        raise InvalidVolumeError(
            f"{where} 必须为坐标/游程组成的可迭代批（list/tuple/set/"
            f"生成器等），得到 {type(batch).__name__}"
        )

    voxels: set[tuple[int, int, int]] = set()

    for index, item in enumerate(batch):
        tag = f"{where}[{index}]"
        if not isinstance(item, (list, tuple)) or isinstance(item, str):
            raise InvalidVolumeError(
                f"{tag} 必须为 (z,y,x) 三元组或 (z,y,start,end) "
                f"半开游程，得到 {item!r}"
            )
        n = len(item)
        if n == 3:
            z = _check_coord(item[0], f"{tag}.z")
            y = _check_coord(item[1], f"{tag}.y")
            x = _check_coord(item[2], f"{tag}.x")
            if z < 0 or y < 0 or x < 0:
                raise InvalidVolumeError(
                    f"{tag}: 坐标不能为负，得到 (z={z},y={y},x={x})"
                )
            voxels.add((z, y, x))
        elif n == 4:
            z = _check_coord(item[0], f"{tag}.z")
            y = _check_coord(item[1], f"{tag}.y")
            s = _check_coord(item[2], f"{tag}.start")
            e = _check_coord(item[3], f"{tag}.end")
            if z < 0 or y < 0:
                raise InvalidVolumeError(
                    f"{tag}: z/y 不能为负，得到 z={z} y={y}"
                )
            if not 0 <= s < e:
                raise InvalidVolumeError(
                    f"{tag}: 要求 0 <= start < end，"
                    f"得到 start={s}, end={e}（空游程/反向均非法）"
                )
            for x in range(s, e):
                voxels.add((z, y, x))
        else:
            raise InvalidVolumeError(
                f"{tag} 必须为长度 3 (z,y,x) 或 4 (z,y,start,end) "
                f"的序列，得到长度 {n}: {item!r}"
            )

    return voxels


def _validate_within(
    voxels: set[tuple[int, int, int]],
    depth: int,
    height: int,
    width: int,
    where: str,
) -> None:
    for z, y, x in voxels:
        if z >= depth:
            raise InvalidVolumeError(f"{where}: z={z} 超出 depth={depth}")
        if y >= height:
            raise InvalidVolumeError(f"{where}: y={y} 超出 height={height}")
        if x >= width:
            raise InvalidVolumeError(f"{where}: x={x} 超出 width={width}")


def _encode_runs(
    voxels: set[tuple[int, int, int]], depth: int, height: int
) -> list[list[list[tuple[int, int]]]]:
    """把体素集合编码为 label() 需要的 [z][y] -> [(start, end), ...]。"""

    by_row: dict[tuple[int, int], list[int]] = {}
    for z, y, x in voxels:
        by_row.setdefault((z, y), []).append(x)

    vol: list[list[list[tuple[int, int]]]] = []
    for z in range(depth):
        rows: list[list[tuple[int, int]]] = []
        for y in range(height):
            xs = by_row.get((z, y))
            if not xs:
                rows.append([])
                continue
            xs.sort()
            row: list[tuple[int, int]] = []
            start = prev = xs[0]
            for x in xs[1:]:
                if x != prev + 1:
                    row.append((start, prev + 1))
                    start = x
                prev = x
            row.append((start, prev + 1))
            rows.append(row)
        vol.append(rows)
    return vol


def _snapshot_key(s: ComponentSnapshot) -> tuple[int, int, int]:
    return (s.min_z, s.min_y, s.min_x)


# ---------------------------------------------------------------------------
# 有状态模型
# ---------------------------------------------------------------------------


class ComponentModel:
    """可批量增删体素、删除桥接后自动重算连通性的有状态模型。

    Args:
        depth/height/width: z/y/x 方向尺寸，整数，各在 ``[1, 512]``。
        connectivity: 6 / 18 / 26，语义与 :func:`label` 完全一致，
            模型生命周期内固定。
        voxels: 可选初始体素，元素为 ``(z, y, x)`` 三元组或
            ``(z, y, start, end)`` 半开 x 游程（与 update 同格式）。

    初始数据即做一次完整校验，非法即抛 :class:`InvalidVolumeError`。
    后续每次提交同样要求当前配置编码出的游程总数不超过 ``MAX_RUNS``
    （20000，沿用无状态接口边界）；超限时提交被拒绝且状态不变。
    """

    __slots__ = (
        "_depth",
        "_height",
        "_width",
        "_connectivity",
        "_voxels",
        "_components",
        "_voxel_cid",
    )

    def __init__(
        self,
        depth: object,
        height: object,
        width: object,
        connectivity: object = 6,
        voxels: object = (),
    ) -> None:
        self._depth = _check_dim(depth, "depth")
        self._height = _check_dim(height, "height")
        self._width = _check_dim(width, "width")
        if not _is_real_int(connectivity) or connectivity not in (6, 18, 26):
            raise InvalidVolumeError(
                "connectivity 必须为整数 6/18/26 之一（拒绝 bool/float），"
                f"得到 {type(connectivity).__name__}: {connectivity!r}"
            )
        self._connectivity = connectivity  # type: ignore[assignment]
        self._voxels: set[tuple[int, int, int]] = set()
        self._components: tuple[Component, ...] = ()
        self._voxel_cid: dict[tuple[int, int, int], int] = {}

        if voxels:
            initial = _parse_batch(voxels, "voxels")
            _validate_within(
                initial, self._depth, self._height, self._width, "voxels"
            )
            self._voxels = set(initial)
            self._recompute()

    # ----- 基本属性 -------------------------------------------------------

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
    def num_voxels(self) -> int:
        return len(self._voxels)

    @property
    def components(self) -> tuple[Component, ...]:
        """最近一次提交后的组件元组（ID 1 起始，按最小坐标升序）。"""

        return self._components

    @property
    def voxels(self) -> frozenset[tuple[int, int, int]]:
        """当前全部体素坐标的只读快照。"""

        return frozenset(self._voxels)

    def component_id_at(self, z: int, y: int, x: int) -> int:
        """返回体素 ``(z, y, x)`` 当前所属组件 ID；该体素不存在则抛
        :class:`KeyError`。"""

        return self._voxel_cid[(z, y, x)]

    def label_result(self):
        """按当前状态产出一份与无状态 :func:`label` 同构的 LabelResult。"""

        vol = _encode_runs(set(self._voxels), self._depth, self._height)
        return label(
            vol, self._depth, self._height, self._width, self._connectivity
        )

    # ----- 容器风格查询 ---------------------------------------------------

    def __contains__(self, voxel: object) -> bool:
        try:
            return voxel in self._voxels
        except TypeError:
            return False

    def __len__(self) -> int:
        return len(self._voxels)

    # ----- 批量修改 -------------------------------------------------------

    def add(self, voxels: object) -> UpdateResult:
        """批量新增体素（已存在者幂等跳过），重算并返回事件。"""

        return self.update(add=voxels)

    def remove(self, voxels: object) -> UpdateResult:
        """批量删除体素（不存在者幂等跳过），重算并返回事件。"""

        return self.update(remove=voxels)

    def update(self, add: object = (), remove: object = ()) -> UpdateResult:
        """一次提交批量新增与删除，全量重算连通性并推导分裂/合并事件。

        同一提交中，同一体素不能既新增又删除。所有输入先完成全部校验，
        校验通过才写状态；写状态后重算若失败（如游程数超 ``MAX_RUNS``），
        回滚到提交前快照——任何失败路径都不会留下半更新状态。

        Returns:
            :class:`UpdateResult`。

        Raises:
            InvalidVolumeError: 批不可迭代、元素不是三元组/四元组、
                坐标非整数（含 bool/float）、坐标越界、空游程/反向游程、
                增删冲突，或提交后游程数超过 20000 等。
        """

        add_set = _parse_batch(add, "add")
        remove_set = _parse_batch(remove, "remove")
        _validate_within(
            add_set, self._depth, self._height, self._width, "add"
        )
        _validate_within(
            remove_set, self._depth, self._height, self._width, "remove"
        )

        conflict = add_set & remove_set
        if conflict:
            z, y, x = sorted(conflict)[0]
            raise InvalidVolumeError(
                "同一提交不能同时新增和删除同一体素，例如 "
                f"(z={z},y={y},x={x})"
            )

        return self._commit(add_set, remove_set)

    def reset(self, voxels: object = ()) -> None:
        """把模型重置为给定体素（默认清空，空输入合法）。

        非法输入不改变当前状态（先校验、后提交，失败回滚）。
        """

        parsed = _parse_batch(voxels, "voxels")
        _validate_within(
            parsed, self._depth, self._height, self._width, "voxels"
        )
        old_voxels = set(self._voxels)
        old_components = self._components
        old_cid = dict(self._voxel_cid)
        try:
            self._voxels = set(parsed)
            self._recompute()
        except BaseException:
            self._voxels = old_voxels
            self._components = old_components
            self._voxel_cid = old_cid
            raise

    # ----- 内部 -----------------------------------------------------------

    def _commit(
        self,
        add_set: set[tuple[int, int, int]],
        remove_set: set[tuple[int, int, int]],
    ) -> UpdateResult:
        # 提交前快照（事件推导与失败回滚都依赖它）
        old_voxels = set(self._voxels)
        old_components = self._components
        old_cid = dict(self._voxel_cid)

        actually_added = add_set - old_voxels
        actually_removed = remove_set & old_voxels

        self._voxels = (old_voxels - actually_removed) | actually_added
        try:
            self._recompute()
            result = self._diff(
                old_components,
                old_cid,
                actually_added,
                actually_removed,
            )
        except BaseException:
            # 重算/事件推导失败：恢复提交前状态，绝不半更新
            self._voxels = old_voxels
            self._components = old_components
            self._voxel_cid = old_cid
            raise
        return result

    def _recompute(self) -> None:
        """对当前体素集合全量重算（本题允许全量重算）。"""

        vol = _encode_runs(self._voxels, self._depth, self._height)
        num_runs = sum(len(row) for sl in vol for row in sl)
        if num_runs > MAX_RUNS:
            raise InvalidVolumeError(
                f"当前配置编码游程数 {num_runs} 超过上限 {MAX_RUNS}；"
                "提交已拒绝，模型状态不变"
            )
        result = label(
            vol,
            self._depth,
            self._height,
            self._width,
            self._connectivity,
        )
        cid_by_voxel: dict[tuple[int, int, int], int] = {}
        for z, sl in enumerate(result.run_labels):
            for y, row_labels in enumerate(sl):
                for cid, (s, e) in zip(row_labels, vol[z][y]):
                    for x in range(s, e):
                        cid_by_voxel[(z, y, x)] = cid
        self._components = tuple(result.components)
        self._voxel_cid = cid_by_voxel

    def _diff(
        self,
        old_components: tuple[Component, ...],
        old_cid: dict[tuple[int, int, int], int],
        added: set[tuple[int, int, int]],
        removed: set[tuple[int, int, int]],
    ) -> UpdateResult:
        new_components = self._components
        new_cid = self._voxel_cid
        old_by_id = {c.id: c for c in old_components}
        new_by_id = {c.id: c for c in new_components}

        # 存活体素上的旧组件 <-> 新组件对应关系
        old_to_new: dict[int, set[int]] = {}
        new_to_old: dict[int, set[int]] = {}
        for v in self._voxels:
            oc = old_cid.get(v)
            if oc is None:
                continue
            nc = new_cid[v]
            old_to_new.setdefault(oc, set()).add(nc)
            new_to_old.setdefault(nc, set()).add(oc)

        # ---- split：一个旧组件的存活体素落到 >=2 个新组件 ----
        splits: list[SplitEvent] = []
        for oc, ncs in old_to_new.items():
            if len(ncs) < 2:
                continue
            added_count = sum(
                1 for v in added if new_cid.get(v) in ncs
            )
            resulting = tuple(
                sorted(
                    (ComponentSnapshot.of(new_by_id[k]) for k in ncs),
                    key=_snapshot_key,
                )
            )
            splits.append(
                SplitEvent(
                    affected_components=(
                        ComponentSnapshot.of(old_by_id[oc]),
                    ),
                    resulting_components=resulting,
                    added_voxels=added_count,
                )
            )
        splits.sort(key=lambda ev: _snapshot_key(ev.affected_components[0]))

        # ---- merge：一个新组件的存活体素来自 >=2 个旧组件 ----
        merges: list[MergeEvent] = []
        for nc, ocs in new_to_old.items():
            if len(ocs) < 2:
                continue
            added_count = sum(
                1 for v in added if new_cid.get(v) == nc
            )
            removed_count = sum(
                1 for v in removed if old_cid.get(v) in ocs
            )
            merged = tuple(
                sorted(
                    (ComponentSnapshot.of(old_by_id[k]) for k in ocs),
                    key=_snapshot_key,
                )
            )
            merges.append(
                MergeEvent(
                    merged_components=merged,
                    resulting_component=ComponentSnapshot.of(new_by_id[nc]),
                    added_voxels=added_count,
                    removed_voxels=removed_count,
                )
            )
        merges.sort(key=lambda ev: _snapshot_key(ev.resulting_component))

        return UpdateResult(
            added=len(added),
            removed=len(removed),
            components=new_components,
            splits=tuple(splits),
            merges=tuple(merges),
            connectivity=self._connectivity,
            num_voxels=len(self._voxels),
        )
