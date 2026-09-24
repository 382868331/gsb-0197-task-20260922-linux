"""ComponentModel（批量增删 + split/merge 事件）的必要测试。

期望值来源彼此独立于被测动态层：

* 组件正确性逐字段比对 ``tests/reference.py`` 的体素展开 + BFS 参考
  （该模块不导入、不调用 rle_components）；
* split/merge 事件由本文件内一个直白的集合分区 BFS 独立推导，
  不使用 ComponentModel 的内部状态；
* 另含多个手算小例覆盖题面点名场景：桥断开、重新连接、空输入。
"""

import random
import unittest
from collections import deque

from rle_components import (
    ComponentModel,
    InvalidVolumeError,
)
from rle_components.core import MAX_RUNS

from reference import reference_label


# ---------------------------------------------------------------------------
# 测试辅助：体素集合 <-> RLE，独立 BFS 分区，独立事件推导
# ---------------------------------------------------------------------------


def encode_voxels(voxels, depth, height):
    """体素集合 -> [z][y] -> [(start, end), ...]。"""

    by_row = {}
    for z, y, x in voxels:
        by_row.setdefault((z, y), []).append(x)
    vol = []
    for z in range(depth):
        rows = []
        for y in range(height):
            xs = sorted(by_row.get((z, y), ()))
            row = []
            if xs:
                s = p = xs[0]
                for x in xs[1:]:
                    if x != p + 1:
                        row.append((s, p + 1))
                        s = x
                    p = x
                row.append((s, p + 1))
            rows.append(row)
        vol.append(rows)
    return vol


_NEIGHBORS = {
    6: [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if abs(dz) + abs(dy) + abs(dx) == 1
    ],
    18: [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if 0 < abs(dz) + abs(dy) + abs(dx) <= 2
    ],
    26: [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dz, dy, dx) != (0, 0, 0)
    ],
}


def bfs_partitions(voxels, connectivity):
    """独立 BFS：返回按最小体素升序的 frozenset 组件列表。"""

    voxels = set(voxels)
    nbrs = _NEIGHBORS[connectivity]
    parts = []
    seen = set()
    for start in sorted(voxels):
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        comp = set()
        while q:
            v = q.popleft()
            comp.add(v)
            z, y, x = v
            for dz, dy, dx in nbrs:
                w = (z + dz, y + dy, x + dx)
                if w in voxels and w not in seen:
                    seen.add(w)
                    q.append(w)
        parts.append(frozenset(comp))
    parts.sort(key=min)
    return parts


def expected_events(old_parts, new_parts, added, removed):
    """从旧/新分区独立推导规范化后的 (splits, merges) 事件列表。

    split: 旧组件存活体素落入 >=2 个新组件；
    merge: 新组件存活体素来自 >=2 个旧组件。
    组件用 (min_z,min_y,min_x,voxel_count) 标识。
    """

    old_of = {v: i for i, p in enumerate(old_parts) for v in p}
    new_of = {v: i for i, p in enumerate(new_parts) for v in p}

    def snap_part(p):
        mz, my, mx = min(p)
        return (mz, my, mx, len(p))

    splits = []
    for p in old_parts:
        ncs = {new_of[v] for v in p if v in new_of}
        if len(ncs) >= 2:
            targets = [new_parts[i] for i in ncs]
            union = set().union(*targets)
            splits.append(
                (
                    (snap_part(p),),
                    tuple(sorted(snap_part(q) for q in targets)),
                    sum(1 for v in added if v in union),
                )
            )
    splits.sort(key=lambda ev: ev[0][0])

    merges = []
    for p in new_parts:
        ocs = {old_of[v] for v in p if v in old_of}
        if len(ocs) >= 2:
            sources = [old_parts[i] for i in ocs]
            union = set().union(*sources)
            merges.append(
                (
                    tuple(sorted(snap_part(q) for q in sources)),
                    snap_part(p),
                    sum(1 for v in added if v in p),
                    sum(1 for v in removed if v in union),
                )
            )
    merges.sort(key=lambda ev: ev[1])
    return splits, merges


def normalize_actual(result):
    splits = [
        (
            tuple((s.min_z, s.min_y, s.min_x, s.voxel_count)
                  for s in ev.affected_components),
            tuple((s.min_z, s.min_y, s.min_x, s.voxel_count)
                  for s in ev.resulting_components),
            ev.added_voxels,
        )
        for ev in result.splits
    ]
    merges = [
        (
            tuple((s.min_z, s.min_y, s.min_x, s.voxel_count)
                  for s in ev.merged_components),
            (
                ev.resulting_component.min_z,
                ev.resulting_component.min_y,
                ev.resulting_component.min_x,
                ev.resulting_component.voxel_count,
            ),
            ev.added_voxels,
            ev.removed_voxels,
        )
        for ev in result.merges
    ]
    return splits, merges


def assert_state_matches_reference(testcase, model, voxels, d, h, w, conn):
    """当前模型状态必须逐字段等于独立 BFS 参考。"""

    vol = encode_voxels(voxels, d, h)
    ref_labels, ref_meta = reference_label(vol, d, h, w, conn)

    lr = model.label_result()
    testcase.assertEqual(lr.run_labels, ref_labels)
    testcase.assertEqual(lr.connectivity, conn)
    testcase.assertEqual(lr.num_runs,
                         sum(len(r) for sl in vol for r in sl))

    comps = model.components
    testcase.assertEqual(len(comps), len(ref_meta))
    testcase.assertEqual(model.num_voxels, len(voxels))
    testcase.assertEqual(len(model), len(voxels))
    for comp, m in zip(comps, ref_meta):
        testcase.assertEqual(comp.voxel_count, m["voxels"])
        testcase.assertEqual(tuple(comp.bbox), m["bbox"])
        testcase.assertEqual(comp.exposed_area, m["area"])
        testcase.assertEqual((comp.min_z, comp.min_y, comp.min_x), m["min"])

    cid_min = {(c.min_z, c.min_y, c.min_x): c.id for c in comps}
    for v in voxels:
        cid = model.component_id_at(*v)
        comp = comps[cid - 1]
        testcase.assertIn(v, model)
        # 组件 ID 与参考标签一致
        z, y, x = v
        testcase.assertEqual(cid, _ref_cid_at(ref_labels, vol, z, y, x))
        testcase.assertEqual(cid, cid_min[(comp.min_z, comp.min_y,
                                           comp.min_x)])


def _ref_cid_at(ref_labels, vol, z, y, x):
    for cid, (s, e) in zip(ref_labels[z][y], vol[z][y]):
        if s <= x < e:
            return cid
    raise KeyError((z, y, x))


# ---------------------------------------------------------------------------
# 手算小例：题面点名场景
# ---------------------------------------------------------------------------


class HandComputedDynamicTests(unittest.TestCase):
    def test_bridge_break_and_reconnect_inline(self):
        """一根 7 体素直棒：删掉中间桥体素 -> split；加回 -> merge。"""

        m = ComponentModel(1, 1, 7, 6, [(0, 0, 0, 7)])
        self.assertEqual(len(m.components), 1)

        r = m.remove([(0, 0, 3)])
        self.assertEqual(r.removed, 1)
        self.assertEqual(len(m.components), 2)
        self.assertEqual(len(r.splits), 1)
        self.assertEqual(r.merges, ())
        ev = r.splits[0]
        self.assertEqual(
            [(s.min_z, s.min_y, s.min_x, s.voxel_count)
             for s in ev.affected_components],
            [(0, 0, 0, 7)],
        )
        self.assertEqual(
            [(s.min_z, s.min_y, s.min_x, s.voxel_count)
             for s in ev.resulting_components],
            [(0, 0, 0, 3), (0, 0, 4, 3)],
        )
        self.assertEqual(ev.added_voxels, 0)
        self.assertEqual(r.num_voxels, 6)

        r = m.add([(0, 0, 3)])
        self.assertEqual(r.added, 1)
        self.assertEqual(len(m.components), 1)
        self.assertEqual(r.splits, ())
        self.assertEqual(len(r.merges), 1)
        me = r.merges[0]
        self.assertEqual(
            [(s.min_z, s.min_y, s.min_x) for s in me.merged_components],
            [(0, 0, 0), (0, 0, 4)],
        )
        self.assertEqual(
            (me.resulting_component.min_z,
             me.resulting_component.min_y,
             me.resulting_component.min_x,
             me.resulting_component.voxel_count),
            (0, 0, 0, 7),
        )
        self.assertEqual(me.added_voxels, 1)
        self.assertEqual(me.removed_voxels, 0)

    def test_ring_remains_one_after_breaking_one_link(self):
        """2x2 环有两条路径：删一个体素不分裂（无 split 事件）。"""

        # 2x3 矩形（两行各 3 体素），去掉中间 x=1 的一个体素
        m = ComponentModel(
            1, 2, 3, 6,
            [(0, 0, 0, 3), (0, 1, 0, 3)],
        )
        self.assertEqual(len(m.components), 1)
        r = m.remove([(0, 0, 1)])
        self.assertEqual(len(m.components), 1)
        self.assertEqual(r.splits, ())
        self.assertEqual(r.merges, ())
        # 剩下的 5 个体素仍经 y=1 行连通
        self.assertEqual(m.components[0].voxel_count, 5)

    def test_delete_whole_component_is_not_split(self):
        """整组件被删光（旧组件 -> 0 个新组件）不算分裂。"""

        m = ComponentModel(1, 1, 5, 6, [(0, 0, 0, 2), (0, 0, 3, 5)])
        self.assertEqual(len(m.components), 2)
        r = m.remove([(0, 0, 0), (0, 0, 1)])
        self.assertEqual(len(m.components), 1)
        self.assertEqual(r.splits, ())
        self.assertEqual(r.merges, ())
        self.assertEqual(m.components[0].voxel_count, 2)

    def test_internal_growth_emits_no_events(self):
        """在组件内部/旁边加体素使其长大：无 merge 事件。"""

        m = ComponentModel(1, 1, 4, 6, [(0, 0, 0)])
        r = m.add([(0, 0, 1), (0, 0, 2)])
        self.assertEqual(len(m.components), 1)
        self.assertEqual(r.splits, ())
        self.assertEqual(r.merges, ())
        self.assertEqual(m.components[0].voxel_count, 3)

    def test_empty_inputs_are_valid(self):
        """题面点名：空输入。"""

        m = ComponentModel(2, 2, 2, 6)
        self.assertEqual(m.components, ())
        self.assertEqual(m.num_voxels, 0)
        self.assertEqual(m.voxels, frozenset())

        r = m.add(())
        self.assertEqual((r.added, r.removed), (0, 0))
        self.assertEqual((r.splits, r.merges), ((), ()))
        self.assertEqual(r.components, ())

        r = m.update()
        self.assertEqual((r.added, r.removed, r.num_components), (0, 0, 0))

        r = m.remove([(0, 0, 0)])  # 删不存在的体素：幂等无事件
        self.assertEqual((r.added, r.removed), (0, 0))
        self.assertEqual((r.splits, r.merges), ((), ()))

        m.add([(0, 0, 0), (1, 1, 1)])
        self.assertEqual(len(m.components), 2)
        m.reset()
        self.assertEqual(m.components, ())
        self.assertEqual(m.num_voxels, 0)
        # 重置为空后 label_result 与无状态空体一致
        lr = m.label_result()
        self.assertEqual(lr.components, [])
        self.assertEqual(lr.run_labels, [[[], []], [[], []]])

    def test_run_form_batch(self):
        """四元组 (z,y,start,end) 游程形式的批量增删。"""

        m = ComponentModel(1, 1, 9, 6)
        r = m.add([(0, 0, 0, 2), (0, 0, 7, 9)])  # 两个分离的双体素块
        self.assertEqual(len(m.components), 2)
        self.assertEqual(r.added, 4)

        # 游程桥一次接上：merge 两个旧组件 + 3 个新桥体素
        r = m.add([(0, 0, 2, 7)])
        self.assertEqual(r.added, 5)
        self.assertEqual(len(m.components), 1)
        self.assertEqual(len(r.merges), 1)
        self.assertEqual(m.components[0].voxel_count, 9)

        # 游程批量删掉中段桥 x=2..6：剩下 x=0,1 与 x=7,8 两段
        r = m.remove([(0, 0, 2, 7)])
        self.assertEqual(r.removed, 5)
        self.assertEqual(len(m.components), 2)
        self.assertEqual(len(r.splits), 1)
        self.assertEqual(
            [(s.min_z, s.min_y, s.min_x, s.voxel_count)
             for s in r.splits[0].resulting_components],
            [(0, 0, 0, 2), (0, 0, 7, 2)],
        )

    def test_edge_bridge_18_and_corner_26(self):
        """跨 slice 边桥（18）断开/重连；角桥只在 26 下连通。"""

        # z0(y0,x0) -边- z1(y1,x0) -面- z2(y1,x0)
        init = [(0, 0, 0), (1, 1, 0), (2, 1, 0)]
        m18 = ComponentModel(3, 2, 1, 18, init)
        self.assertEqual(len(m18.components), 1)
        r = m18.remove([(1, 1, 0)])  # 删除边桥
        self.assertEqual(len(m18.components), 2)
        self.assertEqual(len(r.splits), 1)
        r = m18.add([(1, 1, 0)])
        self.assertEqual(len(m18.components), 1)
        self.assertEqual(len(r.merges), 1)

        # 同样布局在 6 下本来就不连通
        m6 = ComponentModel(3, 2, 1, 6, init)
        self.assertEqual(len(m6.components), 2)

        # 角接触只在 26 下合并两个既有组件：
        # A=(0,0,0)，B=(1,1,2)（26 下也不相邻：dx=2），加 C=(1,1,1)：
        # C 与 B 面接触、与 A 角接触 -> 26 下三者合一（merge），6 下 A 孤立
        for conn, expect in ((26, 1), (6, 2)):
            mc = ComponentModel(2, 2, 3, conn, [(0, 0, 0), (1, 1, 2)])
            self.assertEqual(len(mc.components), 2)
            r = mc.add([(1, 1, 1)])
            self.assertEqual(len(mc.components), expect)
            if conn == 26:
                self.assertEqual(len(r.merges), 1)

    def test_mixed_add_and_remove_one_commit(self):
        """同一提交既加又删：用新桥替换旧桥，组件始终为一。"""

        # y=0 行两段 x=0..2、x=4..6；旧桥是 y=1 的绕行 x=2,3,4（面连接）。
        # 同一提交删绕行中点 (0,1,3)、加直桥 (0,0,3)：始终一个组件。
        m = ComponentModel(
            1, 2, 7, 6,
            [(0, 0, 0, 3), (0, 0, 4, 7),
             (0, 1, 2, 5)],
        )
        self.assertEqual(len(m.components), 1)
        r = m.update(add=[(0, 0, 3)], remove=[(0, 1, 3)])
        self.assertEqual((r.added, r.removed), (1, 1))
        self.assertEqual(len(m.components), 1)
        self.assertEqual(r.splits, ())
        self.assertEqual(r.merges, ())
        self.assertEqual(m.components[0].voxel_count, 9)

    def test_idempotent_and_duplicate_spec(self):
        """重复坐标/游程按集合语义幂等；重复操作无事件。"""

        m = ComponentModel(1, 1, 3, 6,
                           [(0, 0, 0), (0, 0, 0), (0, 0, 0, 2),
                            (0, 0, 1)])
        self.assertEqual(m.num_voxels, 2)
        r = m.add([(0, 0, 0), (0, 0, 1)])
        self.assertEqual(r.added, 0)
        self.assertEqual((r.splits, r.merges), ((), ()))
        r = m.remove([(0, 0, 2)])  # 不存在，幂等
        self.assertEqual(r.removed, 0)
        self.assertEqual((r.splits, r.merges), ((), ()))

    def test_ids_reassigned_by_min_coordinate_after_update(self):
        """重算后 ID 始终按最小体素坐标重新分配。"""

        m = ComponentModel(
            2, 2, 1, 6,
            [(0, 1, 0), (1, 0, 0)],
        )
        # 两个组件：(0,1,0) 与 (1,0,0) 在 6 下仅边接触（dz=1,dy=1）
        self.assertEqual(
            [(c.min_z, c.min_y, c.min_x) for c in m.components],
            [(0, 1, 0), (1, 0, 0)],
        )
        self.assertEqual(m.component_id_at(0, 1, 0), 1)
        # 加 (0,0,0)：与两者都面接触，合并为一个，最小坐标变为 (0,0,0)
        m.add([(0, 0, 0)])
        self.assertEqual(len(m.components), 1)
        self.assertEqual(m.component_id_at(1, 0, 0), 1)
        self.assertEqual(
            (m.components[0].min_z, m.components[0].min_y), (0, 0)
        )

    def test_simultaneous_split_and_merge_in_one_commit(self):
        """同一提交既搭桥又断桥：merge 与 split 同时发生且计数正确。"""

        # A=x0..2, B=x4..6, C=x8（三个组件）；加桥 x3 连接 A、B，
        # 同时删 x5 使 B 的残端 x6 从合并体中掉出去。
        m = ComponentModel(1, 1, 10, 6,
                           [(0, 0, 0, 3), (0, 0, 4, 7), (0, 0, 8, 9)])
        r = m.update(add=[(0, 0, 3)], remove=[(0, 0, 5)])
        self.assertEqual(
            [(c.min_x, c.voxel_count) for c in m.components],
            [(0, 5), (6, 1), (8, 1)],
        )
        self.assertEqual(len(r.merges), 1)
        self.assertEqual(len(r.splits), 1)
        me = r.merges[0]
        self.assertEqual([s.min_x for s in me.merged_components], [0, 4])
        self.assertEqual(me.resulting_component.min_x, 0)
        self.assertEqual((me.added_voxels, me.removed_voxels), (1, 1))
        sp = r.splits[0]
        self.assertEqual(sp.affected_components[0].min_x, 4)
        self.assertEqual(
            [s.min_x for s in sp.resulting_components], [0, 6]
        )
        self.assertEqual(sp.added_voxels, 1)

    def test_batch_accepts_generator_and_other_iterables(self):
        m = ComponentModel(1, 1, 3, 6)
        r = m.add((z, y, x) for z, y, x in [(0, 0, 0), (0, 0, 1)])
        self.assertEqual(r.added, 2)
        r = m.remove(frozenset({(0, 0, 0)}))
        self.assertEqual(r.removed, 1)

    def test_reset_replaces_state(self):
        m = ComponentModel(1, 1, 4, 6, [(0, 0, 0, 4)])
        m.reset([(0, 0, 0)])
        self.assertEqual(m.num_voxels, 1)
        self.assertEqual(len(m.components), 1)
        # 非法 reset 不改变状态
        with self.assertRaises(InvalidVolumeError):
            m.reset([(0, 0, 9)])
        self.assertEqual(m.num_voxels, 1)
        self.assertEqual(m.components[0].voxel_count, 1)


# ---------------------------------------------------------------------------
# 随机差分：模型状态与事件都必须等于独立参考
# ---------------------------------------------------------------------------


class RandomizedDynamicTests(unittest.TestCase):
    def _run_sequence(self, rng, d, h, w, conn, steps):
        voxels = {
            (z, y, x)
            for z in range(d)
            for y in range(h)
            for x in range(w)
            if rng.random() < 0.35
        }
        model = ComponentModel(d, h, w, conn, sorted(voxels))
        old_parts = bfs_partitions(voxels, conn)
        assert_state_matches_reference(self, model, voxels, d, h, w, conn)

        for _ in range(steps):
            universe = [
                (z, y, x)
                for z in range(d)
                for y in range(h)
                for x in range(w)
            ]
            absent = [v for v in universe if v not in voxels]
            rng.shuffle(absent)
            present = sorted(voxels)
            rng.shuffle(present)

            add_batch = []
            na = rng.randrange(0, min(5, len(absent)) + 1)
            chosen_add = absent[:na]
            # 偶尔把同行相邻的一对改写为四元组游程
            i = 0
            while i < len(chosen_add):
                v = chosen_add[i]
                if (
                    rng.random() < 0.4
                    and i + 1 < len(chosen_add)
                ):
                    z, y, x = v
                    z2, y2, x2 = chosen_add[i + 1]
                    if (z, y) == (z2, y2) and abs(x - x2) == 1:
                        s, e = sorted((x, x2))
                        add_batch.append((z, y, s, e + 1))
                        i += 2
                        continue
                add_batch.append(v)
                i += 1

            nr = rng.randrange(0, min(5, len(present)) + 1)
            remove_batch = present[:nr]

            add_set = set()
            for it in add_batch:
                if len(it) == 4:
                    z, y, s, e = it
                    add_set.update((z, y, x) for x in range(s, e))
                else:
                    add_set.add(it)
            remove_set = set(remove_batch)
            assert not (add_set & remove_set)

            new_voxels = (voxels - remove_set) | add_set
            new_parts = bfs_partitions(new_voxels, conn)
            exp_splits, exp_merges = expected_events(
                old_parts, new_parts, add_set, remove_set
            )

            result = model.update(add=add_batch, remove=remove_batch)
            self.assertEqual(result.added, len(add_set))
            self.assertEqual(result.removed, len(remove_set))
            self.assertEqual(result.connectivity, conn)
            self.assertEqual(result.num_voxels, len(new_voxels))

            act_splits, act_merges = normalize_actual(result)
            self.assertEqual(act_splits, exp_splits)
            self.assertEqual(act_merges, exp_merges)

            voxels = new_voxels
            old_parts = new_parts
            assert_state_matches_reference(
                self, model, voxels, d, h, w, conn
            )

    def test_random_sequences_vs_reference(self):
        seed = 20260924
        for d, h, w in [
            (1, 1, 7),
            (1, 2, 5),
            (2, 2, 4),
            (3, 3, 3),
            (4, 3, 2),
        ]:
            for conn in (6, 18, 26):
                rng = random.Random(seed)
                seed += 1
                self._run_sequence(rng, d, h, w, conn, steps=15)

    def test_event_counts_on_random_long_bar(self):
        """单行棒上随机增删：split/merge 事件数与独立枚举一致。"""

        w = 12
        rng = random.Random(98765)
        voxels = {
            (0, 0, x) for x in range(w) if rng.random() < 0.6
        }
        model = ComponentModel(1, 1, w, 6, sorted(voxels))
        old_parts = bfs_partitions(voxels, 6)
        for _ in range(40):
            if rng.random() < 0.5 and voxels:
                v = rng.choice(sorted(voxels))
                batch_add, batch_remove = [], [v]
            else:
                absent = [
                    (0, 0, x) for x in range(w)
                    if (0, 0, x) not in voxels
                ]
                if not absent:
                    continue
                v = rng.choice(absent)
                batch_add, batch_remove = [v], []
            add_set = set(batch_add)
            remove_set = set(batch_remove)
            new_voxels = (voxels - remove_set) | add_set
            new_parts = bfs_partitions(new_voxels, 6)
            exp_s, exp_m = expected_events(
                old_parts, new_parts, add_set, remove_set
            )
            result = model.update(add=batch_add, remove=batch_remove)
            act_s, act_m = normalize_actual(result)
            self.assertEqual(act_s, exp_s)
            self.assertEqual(act_m, exp_m)
            voxels, old_parts = new_voxels, new_parts


# ---------------------------------------------------------------------------
# 错误语义：明确拒绝且绝不半更新
# ---------------------------------------------------------------------------


class DynamicErrorTests(unittest.TestCase):
    def setUp(self):
        self.model = ComponentModel(1, 2, 4, 6, [(0, 0, 0, 2)])

    def _assert_unchanged(self):
        self.assertEqual(self.model.num_voxels, 2)
        self.assertEqual(
            self.model.voxels, frozenset({(0, 0, 0), (0, 0, 1)})
        )
        self.assertEqual(len(self.model.components), 1)
        self.assertEqual(self.model.components[0].voxel_count, 2)

    def test_bad_batch_shapes(self):
        for bad in (
            "abc", 123, None, {"start": 0, "end": 1},
            [1], [(0, 0)], [(0, 0, 0, 0, 0)],
        ):
            with self.assertRaises(InvalidVolumeError):
                self.model.add(bad)
            self._assert_unchanged()

    def test_non_integer_bool_float_coords(self):
        for bad in [
            [(0, 0, 1.5)],
            [(0, 0, True)],
            [("0", 0, 0)],
            [(0, 0, None)],
            [(0, 0, 0, 2.0)],
        ]:
            with self.assertRaises(InvalidVolumeError):
                self.model.add(bad)
            self._assert_unchanged()

    def test_negative_and_out_of_bounds(self):
        for bad in [
            [(-1, 0, 0)],
            [(0, 5, 0)],
            [(0, 0, 4)],
            [(0, 0, -1)],
            [(0, 0, 0, 5)],   # 游程 end > width
            [(0, 0, 2, 2)],   # 空游程
            [(0, 0, 3, 1)],   # 反向游程
            [(0, 0, -1, 2)],  # 负 start
        ]:
            with self.assertRaises(InvalidVolumeError):
                self.model.add(bad)
            self._assert_unchanged()

    def test_add_remove_conflict(self):
        with self.assertRaises(InvalidVolumeError) as cm:
            self.model.update(add=[(0, 0, 2)], remove=[(0, 0, 2)])
        self.assertIn("不能同时新增和删除", str(cm.exception))
        self._assert_unchanged()

    def test_error_message_locatable(self):
        with self.assertRaises(InvalidVolumeError) as cm:
            self.model.add([(0, 0, 0), (0, 0, 99)])
        # 展开后的范围错误带批次名、坐标与尺寸边界
        self.assertIn("add", str(cm.exception))
        self.assertIn("99", str(cm.exception))
        self.assertIn("width", str(cm.exception))
        self._assert_unchanged()

    def test_error_message_has_batch_index_for_bad_item(self):
        with self.assertRaises(InvalidVolumeError) as cm:
            self.model.add([(0, 0, 0), (1, 2)])
        self.assertIn("add[1]", str(cm.exception))
        self._assert_unchanged()

    def test_bad_constructor_arguments(self):
        with self.assertRaises(InvalidVolumeError):
            ComponentModel(0, 1, 1)
        with self.assertRaises(InvalidVolumeError):
            ComponentModel(1, 1, 1, connectivity=True)
        with self.assertRaises(InvalidVolumeError):
            ComponentModel(1, 1, 1, connectivity=26.0)
        with self.assertRaises(InvalidVolumeError):
            ComponentModel(1, 1, 1, 6, voxels=[(0, 0, 1)])

    def test_remove_validation_failure_no_partial_state(self):
        # 删除批本身非法（含越界坐标）：合法部分也不能生效
        with self.assertRaises(InvalidVolumeError):
            self.model.remove([(0, 0, 0), (9, 0, 0)])
        self._assert_unchanged()

    def test_run_limit_commit_rejected_atomically(self):
        """提交使游程数超过 20000：整笔拒绝并回滚。"""

        d, h = 41, 489  # 共 20049 行
        coords = [
            (z, y, 0)
            for z in range(d)
            for y in range(h)
        ]
        model = ComponentModel(d, h, 2, 6, coords[:MAX_RUNS])
        self.assertEqual(model.label_result().num_runs, MAX_RUNS)

        model.remove([coords[0]])
        self.assertEqual(model.label_result().num_runs, MAX_RUNS - 1)
        # 在两个空行各加一个 -> 先回到 20000，接受
        model.add([coords[0]])
        self.assertEqual(model.label_result().num_runs, MAX_RUNS)
        # 再加一个新行 -> 20001，拒绝且状态不变
        with self.assertRaises(InvalidVolumeError) as cm:
            model.add([coords[MAX_RUNS]])
        self.assertIn(str(MAX_RUNS), str(cm.exception))
        self.assertEqual(model.label_result().num_runs, MAX_RUNS)
        self.assertEqual(model.num_voxels, MAX_RUNS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
