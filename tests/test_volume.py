"""VoxelVolume 批量增删与组件分裂/合并事件的测试。

期望值来源：
* 手算小例（桥断开/重连、一拆三、空输入等）；
* tests/reference.py 的独立体素展开 + BFS 参考（组件结果逐字段比对）；
* 本文件内 independent_events：基于参考标签与体素集合独立推导事件，
  不调用被测实现的事件逻辑。
"""

import random
import unittest

from rle_components import (
    ComponentEvent,
    InvalidVolumeError,
    VoxelVolume,
    label,
)

from reference import expand, reference_label


def voxel_cid_map(vol, depth, height, width, conn):
    """用独立参考实现得到 体素 -> 组件ID 映射。"""
    run_labels, _ = reference_label(vol, depth, height, width, conn)
    out = {}
    for z in range(depth):
        for y in range(height):
            for (s, e), cid in zip(vol[z][y], run_labels[z][y]):
                for x in range(s, e):
                    out[(z, y, x)] = cid
    return out


def independent_events(old_vol, new_vol, depth, height, width, conn):
    """独立推导两次快照之间的组件事件（集合语义，与被测实现无关）。"""
    old_map = voxel_cid_map(old_vol, depth, height, width, conn)
    new_map = voxel_cid_map(new_vol, depth, height, width, conn)
    old_n = max(old_map.values(), default=0)
    new_n = max(new_map.values(), default=0)

    forward = {oc: set() for oc in range(1, old_n + 1)}
    backward = {nc: set() for nc in range(1, new_n + 1)}
    for v, oc in old_map.items():
        nc = new_map.get(v)
        if nc is not None:
            forward[oc].add(nc)
            backward[nc].add(oc)

    events = []
    for oc in range(1, old_n + 1):
        if not forward[oc]:
            events.append(ComponentEvent("removed", (oc,), ()))
        elif len(forward[oc]) > 1:
            events.append(
                ComponentEvent("split", (oc,), tuple(sorted(forward[oc])))
            )
    merges = []
    created = []
    for nc in range(1, new_n + 1):
        if not backward[nc]:
            created.append(ComponentEvent("created", (), (nc,)))
        elif len(backward[nc]) > 1:
            merges.append(
                ComponentEvent("merge", tuple(sorted(backward[nc])), (nc,))
            )
    return events + merges + created


def assert_result_matches_reference(testcase, vol, conn):
    """当前 VoxelVolume 结果必须与独立 BFS 参考逐字段一致。"""
    snap = vol.runs_snapshot()
    result = vol.result()
    d, h, w = vol.depth, vol.height, vol.width
    ref_labels, ref_meta = reference_label(snap, d, h, w, conn)
    testcase.assertEqual(result.run_labels, ref_labels)
    testcase.assertEqual(len(result.components), len(ref_meta))
    testcase.assertEqual(vol.voxel_count, len(expand(snap, d, h, w)))
    testcase.assertEqual(vol.num_runs, result.num_runs)
    for comp, m in zip(result.components, ref_meta):
        testcase.assertEqual(comp.voxel_count, m["voxels"])
        testcase.assertEqual(tuple(comp.bbox), m["bbox"])
        testcase.assertEqual(comp.exposed_area, m["area"])
        testcase.assertEqual((comp.min_z, comp.min_y, comp.min_x), m["min"])


class BridgeScenarioTests(unittest.TestCase):
    """手算：一维桥接的断开与重连（6 邻接，1x1x5 体）。"""

    def setUp(self):
        self.vol = VoxelVolume(1, 1, 5, 6)

    def test_connect_split_reconnect(self):
        # 两段棒：x in [0,2) 与 x in [3,5)，中间 x=2 为桥位
        r = self.vol.add_voxels([(0, 0, 0), (0, 0, 1), (0, 0, 3), (0, 0, 4)])
        self.assertEqual(r.added, 4)
        self.assertEqual(r.removed, 0)
        self.assertEqual(len(r.result.components), 2)
        self.assertEqual(
            r.events,
            (ComponentEvent("created", (), (1,)),
             ComponentEvent("created", (), (2,))),
        )
        # ID 按最小体素坐标：组件 1 最小 x=0，组件 2 最小 x=3
        self.assertEqual(
            [(c.min_z, c.min_y, c.min_x) for c in r.result.components],
            [(0, 0, 0), (0, 0, 3)],
        )

        # 架桥：两个旧组件并入一个新组件
        r = self.vol.add_voxels([(0, 0, 2)])
        self.assertEqual(r.added, 1)
        self.assertEqual(len(r.result.components), 1)
        self.assertEqual(
            r.events, (ComponentEvent("merge", (1, 2), (1,)),)
        )
        c = r.result.components[0]
        self.assertEqual(c.voxel_count, 5)
        self.assertEqual(c.exposed_area, 6 * 5 - 2 * 4)  # 5 连体素棒

        # 拆桥：组件分裂为两个
        r = self.vol.remove_voxels([(0, 0, 2)])
        self.assertEqual(r.removed, 1)
        self.assertEqual(len(r.result.components), 2)
        self.assertEqual(
            r.events, (ComponentEvent("split", (1,), (1, 2)),)
        )
        areas = sorted(c.exposed_area for c in r.result.components)
        self.assertEqual(areas, [10, 10])  # 两段长度 2 的棒
        assert_result_matches_reference(self, self.vol, 6)

        # 重新连接：再次合并
        r = self.vol.add_voxels([(0, 0, 2)])
        self.assertEqual(
            r.events, (ComponentEvent("merge", (1, 2), (1,)),)
        )
        self.assertEqual(len(r.result.components), 1)
        assert_result_matches_reference(self, self.vol, 6)

    def test_batch_remove_splits_into_three(self):
        # x in [0,7) 整条棒，一批删除 x=2 与 x=4 → 三段
        vol = VoxelVolume(1, 1, 7, 6)
        vol.add_voxels([(0, 0, x) for x in range(7)])
        r = vol.remove_voxels([(0, 0, 2), (0, 0, 4)])
        self.assertEqual(r.removed, 2)
        self.assertEqual(len(r.result.components), 3)
        self.assertEqual(
            r.events, (ComponentEvent("split", (1,), (1, 2, 3)),)
        )
        sizes = [c.voxel_count for c in r.result.components]
        self.assertEqual(sizes, [2, 1, 2])  # 按最小 x 排序
        assert_result_matches_reference(self, vol, 6)

    def test_remove_last_voxels_emits_removed(self):
        vol = VoxelVolume(1, 1, 2, 6)
        vol.add_voxels([(0, 0, 0), (0, 0, 1)])
        r = vol.remove_voxels([(0, 0, 0), (0, 0, 1)])
        self.assertEqual(r.removed, 2)
        self.assertEqual(r.result.components, [])
        self.assertEqual(r.events, (ComponentEvent("removed", (1,), ()),))
        self.assertEqual(vol.voxel_count, 0)


class EmptyAndNoOpTests(unittest.TestCase):
    """题面点名：空输入。"""

    def test_empty_volume_initial_state(self):
        for conn in (6, 18, 26):
            vol = VoxelVolume(2, 2, 2, conn)
            r = vol.result()
            self.assertEqual(r.components, [])
            self.assertEqual(r.num_runs, 0)
            self.assertEqual(vol.voxel_count, 0)

    def test_empty_batch_is_noop(self):
        vol = VoxelVolume(1, 1, 3, 6)
        vol.add_voxels([(0, 0, 1)])
        before = vol.result()
        for r in (
            vol.apply(),
            vol.add_voxels([]),
            vol.remove_voxels([]),
            vol.apply(add_voxels=[], remove_voxels=[]),
        ):
            self.assertEqual(r.events, ())
            self.assertEqual(r.added, 0)
            self.assertEqual(r.removed, 0)
            self.assertIs(r.result, before)
        self.assertEqual(vol.voxel_count, 1)

    def test_idempotent_noops(self):
        vol = VoxelVolume(1, 1, 3, 6)
        vol.add_voxels([(0, 0, 0)])
        r = vol.add_voxels([(0, 0, 0), (0, 0, 0)])  # 已存在 + 批内重复
        self.assertEqual(r.added, 0)
        self.assertEqual(r.events, ())
        r = vol.remove_voxels([(0, 0, 2)])  # 不存在
        self.assertEqual(r.removed, 0)
        self.assertEqual(r.events, ())
        self.assertEqual(vol.voxel_count, 1)


class CombinedApplyTests(unittest.TestCase):
    def test_remove_and_add_in_one_batch(self):
        # 一删一增同批完成：桥从 x=2 移到 x=5（不经过中间态）
        vol = VoxelVolume(1, 1, 7, 6)
        vol.add_voxels([(0, 0, x) for x in (0, 1, 2, 3, 4, 5, 6)])
        r = vol.apply(add_voxels=[], remove_voxels=[(0, 0, 3)])
        self.assertEqual((r.added, r.removed), (0, 1))
        self.assertEqual(len(r.result.components), 2)
        r = vol.apply(add_voxels=[(0, 0, 3)], remove_voxels=[(0, 0, 5)])
        self.assertEqual((r.added, r.removed), (1, 1))
        # 结果：x in [0,5) 为组件 1，x=6 为组件 2。
        # 旧组件 2（x in [4,7)）的体素分流到新组件 1（x=4）与 2（x=6）；
        # 新组件 1 同时收到旧组件 1（x in [0,3)）与旧组件 2 的体素。
        self.assertEqual(len(r.result.components), 2)
        self.assertEqual(
            r.events,
            (ComponentEvent("split", (2,), (1, 2)),
             ComponentEvent("merge", (1, 2), (1,))),
        )
        assert_result_matches_reference(self, vol, 6)


class FromRunsTests(unittest.TestCase):
    def test_from_runs_roundtrip(self):
        runs = [[[(0, 2)], []], [[], [(1, 3)]]]
        vol = VoxelVolume.from_runs(runs, 2, 2, 3, 6)
        self.assertEqual(vol.voxel_count, 4)
        self.assertEqual(vol.result().run_labels,
                         label(runs, 2, 2, 3, 6).run_labels)
        self.assertEqual([tuple(t) for sl in vol.runs_snapshot()
                          for row in sl for t in row],
                         [(0, 2), (1, 3)])
        # 快照可直接作为 label 输入（沿用旧入口）
        again = label(vol.runs_snapshot(), 2, 2, 3, 6)
        self.assertEqual(again.run_labels, vol.result().run_labels)

    def test_from_runs_rejects_invalid(self):
        with self.assertRaises(InvalidVolumeError):
            VoxelVolume.from_runs([[[(0, 3), (2, 4)]]], 1, 1, 4)
        with self.assertRaises(InvalidVolumeError):
            VoxelVolume.from_runs([[[]]], 2, 1, 1)


class VolumeErrorTests(unittest.TestCase):
    def setUp(self):
        self.vol = VoxelVolume(2, 2, 4, 6)
        self.vol.add_voxels([(0, 0, 0)])

    def _assert_bad(self, *args, **kwargs):
        with self.assertRaises(InvalidVolumeError):
            self.vol.apply(*args, **kwargs)

    def test_constructor_validation(self):
        for bad in (
            lambda: VoxelVolume(0, 1, 1),
            lambda: VoxelVolume(1, 513, 1),
            lambda: VoxelVolume(1, 1, True),
            lambda: VoxelVolume(1, 1, 1, 8),
            lambda: VoxelVolume(1, 1, 1, 6.0),
            lambda: VoxelVolume(1, 1, 1, True),
        ):
            with self.assertRaises(InvalidVolumeError):
                bad()

    def test_malformed_voxels(self):
        self._assert_bad(add_voxels="(0,0,0)")
        self._assert_bad(add_voxels=5)
        self._assert_bad(add_voxels=[(0, 0)])
        self._assert_bad(add_voxels=[(0, 0, 0, 0)])
        self._assert_bad(add_voxels=[None])
        self._assert_bad(add_voxels=["0,0,0"])

    def test_non_integer_and_special_float_coords(self):
        self._assert_bad(add_voxels=[(True, 0, 0)])
        self._assert_bad(add_voxels=[(0, 0, 1.5)])
        self._assert_bad(add_voxels=[(0, 0, 2.0)])
        self._assert_bad(add_voxels=[(0, float("nan"), 0)])
        self._assert_bad(add_voxels=[(float("inf"), 0, 0)])
        self._assert_bad(add_voxels=[(0, 0, "1")])

    def test_out_of_range_coords(self):
        self._assert_bad(add_voxels=[(-1, 0, 0)])
        self._assert_bad(add_voxels=[(0, 0, 4)])   # x == width
        self._assert_bad(add_voxels=[(2, 0, 0)])   # z == depth
        self._assert_bad(remove_voxels=[(0, 2, 0)])

    def test_add_remove_conflict_same_batch(self):
        with self.assertRaises(InvalidVolumeError) as cm:
            self.vol.apply(add_voxels=[(0, 0, 1)], remove_voxels=[(0, 0, 1)])
        self.assertIn("同批", str(cm.exception))

    def test_failed_batch_leaves_state_unchanged(self):
        before_labels = self.vol.result().run_labels
        before_count = self.vol.voxel_count
        # 合法坐标 + 非法坐标混在一批：整体拒绝
        self._assert_bad(add_voxels=[(0, 0, 1), (0, 0, 99)])
        self._assert_bad(add_voxels=[(1, 1, 3), (0, 0, float("nan"))])
        self.assertEqual(self.vol.voxel_count, before_count)
        self.assertEqual(self.vol.result().run_labels, before_labels)
        # 状态仍可用
        r = self.vol.add_voxels([(0, 0, 1)])
        self.assertEqual(r.added, 1)

    def test_run_limit_enforced_atomically(self):
        # 512*40 = 20480 个可用行；先加到恰好 20000 条游程（接受）
        vol = VoxelVolume(512, 40, 1, 6)
        cells = [(z, y, 0) for z in range(512) for y in range(40)]
        r = vol.add_voxels(cells[:20000])
        self.assertEqual(r.added, 20000)
        self.assertEqual(vol.num_runs, 20000)
        # 再加 2 个 → 20002 条游程超上限：整批拒绝，状态不变
        with self.assertRaises(InvalidVolumeError) as cm:
            vol.add_voxels(cells[20000:20002])
        self.assertIn("20000", str(cm.exception))
        self.assertEqual(vol.num_runs, 20000)
        self.assertEqual(vol.voxel_count, 20000)


class RandomizedVolumeTests(unittest.TestCase):
    """随机批量增删：每步结果与事件都和独立参考比对。"""

    def test_random_batches_vs_reference(self):
        rng = random.Random(20260924)
        for conn in (6, 18, 26):
            d, h, w = 3, 4, 6
            vol = VoxelVolume(d, h, w, conn)
            prev_snap = vol.runs_snapshot()
            for _ in range(30):
                cells = [
                    (rng.randrange(d), rng.randrange(h), rng.randrange(w))
                    for _ in range(rng.randrange(0, 9))
                ]
                if rng.random() < 0.5:
                    r = vol.add_voxels(cells)
                else:
                    r = vol.remove_voxels(cells)
                # 结果与独立 BFS 参考一致
                assert_result_matches_reference(self, vol, conn)
                # 事件与独立推导一致
                expected = independent_events(
                    prev_snap, vol.runs_snapshot(), d, h, w, conn
                )
                self.assertEqual(sorted(r.events), sorted(expected))
                prev_snap = vol.runs_snapshot()

    def test_random_combined_apply_vs_reference(self):
        rng = random.Random(777)
        d, h, w = 4, 4, 5
        vol = VoxelVolume(d, h, w, 18)
        prev_snap = vol.runs_snapshot()
        for _ in range(25):
            adds = [
                (rng.randrange(d), rng.randrange(h), rng.randrange(w))
                for _ in range(rng.randrange(0, 6))
            ]
            rems = [
                (rng.randrange(d), rng.randrange(h), rng.randrange(w))
                for _ in range(rng.randrange(0, 6))
            ]
            rems = [c for c in rems if c not in set(adds)]  # 避免同批冲突
            r = vol.apply(add_voxels=adds, remove_voxels=rems)
            assert_result_matches_reference(self, vol, 18)
            expected = independent_events(
                prev_snap, vol.runs_snapshot(), d, h, w, 18
            )
            self.assertEqual(sorted(r.events), sorted(expected))
            prev_snap = vol.runs_snapshot()


if __name__ == "__main__":
    unittest.main(verbosity=2)
