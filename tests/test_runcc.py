"""runcc 的必要测试。

期望值由 tests/reference.py 中独立的体素展开 + BFS 参考实现给出，
不调用被测核心产生期望值；另含往返一致性补充断言。
"""

import os
import random
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runcc import Run, label_volume  # noqa: E402
from runcc.core import MAX_DIM, MAX_RUNS  # noqa: E402
from reference import reference_labeling  # noqa: E402


def voxels_to_runs(voxels):
    """把体素集合压缩为同行有序不重叠半开游程（确定性）。"""
    rows = {}
    for x, y, z in voxels:
        rows.setdefault((z, y), []).append(x)
    runs = []
    for (z, y), xs in sorted(rows.items()):
        xs.sort()
        start = prev = xs[0]
        for x in xs[1:]:
            if x == prev + 1:
                prev = x
                continue
            runs.append(Run(z, y, start, prev + 1))
            start = prev = x
        runs.append(Run(z, y, start, prev + 1))
    return runs


def assert_matches_reference(self, dims, runs, conn):
    comps, run_labels = reference_labeling(dims, runs, conn)
    result = label_volume(dims, runs, conn)

    self.assertEqual(result.num_components, len(comps))
    self.assertEqual(len(result.labels), len(runs))
    self.assertEqual(list(result.labels), run_labels)

    for cid, (label, ref) in enumerate(zip(result.components, comps), start=1):
        self.assertEqual(label.id, cid)
        self.assertEqual(label.origin, ref["origin"])
        self.assertEqual(label.voxel_count, ref["voxel_count"])
        self.assertEqual(tuple(label.bbox), ref["bbox"])
        self.assertEqual(label.exposed_area, ref["exposed_area"])
    # 补充：组件 id 连续且与 labels 中出现的集合一致
    self.assertEqual(
        sorted(set(l for l in result.labels if l is not None)),
        list(range(1, len(comps) + 1)),
    )
    return result


class ConnectivityGeometryTests(unittest.TestCase):
    def test_corner_only_contact(self):
        # 两个单体素仅在角 (dx,dy,dz)=(1,1,1) 处相触。
        runs = [Run(0, 0, 0, 1), Run(1, 1, 1, 2)]
        r6 = label_volume((3, 3, 3), runs, 6)
        r18 = label_volume((3, 3, 3), runs, 18)
        r26 = label_volume((3, 3, 3), runs, 26)
        self.assertEqual(r6.num_components, 2)
        self.assertEqual(r18.num_components, 2)
        self.assertEqual(r26.num_components, 1)
        # 角接触不共享面：合并后暴露面积仍为 6+6。
        self.assertEqual(r6.components[0].exposed_area, 6)
        self.assertEqual(r26.components[0].exposed_area, 12)
        assert_matches_reference(self, (3, 3, 3), runs, 6)
        assert_matches_reference(self, (3, 3, 3), runs, 18)
        assert_matches_reference(self, (3, 3, 3), runs, 26)

    def test_edge_only_contact(self):
        # 同切片内两条游程 (|dy|=1,|dx|=1)：边接触，非面。
        runs = [Run(0, 0, 0, 1), Run(0, 1, 1, 2)]
        r6 = label_volume((3, 3, 1), runs, 6)
        r18 = label_volume((3, 3, 1), runs, 18)
        self.assertEqual(r6.num_components, 2)
        self.assertEqual(r18.num_components, 1)
        # 边接触不减少面暴露面积。
        self.assertEqual(r18.components[0].exposed_area, 12)
        assert_matches_reference(self, (3, 3, 1), runs, 6)
        assert_matches_reference(self, (3, 3, 1), runs, 18)
        assert_matches_reference(self, (3, 3, 1), runs, 26)

    def test_edge_only_across_slices(self):
        # 跨切片 (|dz|=1,|dx|=1,dy=0)：边接触；18 合并、6 不合并。
        runs = [Run(0, 0, 0, 1), Run(1, 0, 1, 2)]
        self.assertEqual(label_volume((3, 1, 2), runs, 6).num_components, 2)
        self.assertEqual(label_volume((3, 1, 2), runs, 18).num_components, 1)
        assert_matches_reference(self, (3, 1, 2), runs, 18)

    def test_face_contact_exact_area(self):
        # 单体素面积 6；面邻接一对为 10；2x2x2 立方体为 24。
        one = label_volume((2, 2, 2), [Run(0, 0, 0, 1)], 6)
        self.assertEqual(one.components[0].exposed_area, 6)

        pair = label_volume((2, 1, 2), [Run(0, 0, 0, 2)], 6)
        self.assertEqual(pair.components[0].voxel_count, 2)
        self.assertEqual(pair.components[0].exposed_area, 10)

        cube = [
            Run(z, y, 0, 2) for z in range(2) for y in range(2)
        ]
        r = label_volume((2, 2, 2), cube, 6)
        self.assertEqual(r.num_components, 1)
        self.assertEqual(r.components[0].voxel_count, 8)
        self.assertEqual(r.components[0].exposed_area, 24)

    def test_cross_slice_bridge(self):
        # z=0 与 z=2 各有一段，6 邻接下只有 z=1 桥接存在时才连通。
        bottom = [Run(0, 0, 2, 4)]
        top = [Run(2, 0, 2, 4)]
        without_bridge = bottom + top
        with_bridge = bottom + [Run(1, 0, 2, 4)] + top
        self.assertEqual(label_volume((5, 2, 3), without_bridge, 6).num_components, 2)
        bridged = label_volume((5, 2, 3), with_bridge, 6)
        self.assertEqual(bridged.num_components, 1)
        self.assertEqual(bridged.components[0].voxel_count, 6)
        # 两个跨切片接缝各共享 2 个面：初始各 2 游程游程面积合计
        # 3 段 *10 = 30，扣 2 接缝 * 4 = 8 → 22。
        self.assertEqual(bridged.components[0].exposed_area, 22)
        assert_matches_reference(self, (5, 2, 3), with_bridge, 6)
        assert_matches_reference(self, (5, 2, 3), without_bridge, 6)

    def test_empty_rows_and_empty_runs(self):
        # y=0,2,4 有体素，中间行整行空缺；并混入长度 0 的空游程。
        runs = [
            Run(0, 0, 1, 3),
            Run(0, 1, 5, 5),    # 空游程
            Run(0, 2, 1, 3),
            Run(0, 4, 1, 3),
            Run(0, 3, 0, 0),    # 空游程
        ]
        r = label_volume((6, 5, 1), runs, 6)
        self.assertEqual(r.num_components, 3)
        self.assertIsNone(r.labels[1])
        self.assertIsNone(r.labels[4])
        # y=0、y=2、y=4 之间隔着空行（|dy|=2），6 邻接下互不连通。
        self.assertEqual(len({r.labels[0], r.labels[2], r.labels[3]}), 3)
        # 26 邻接（同切片等效 8 邻）同样不跨越 2 行的空缺。
        r26 = label_volume((6, 5, 1), runs, 26)
        self.assertEqual(r26.num_components, 3)
        assert_matches_reference(self, (6, 5, 1), runs, 6)
        assert_matches_reference(self, (6, 5, 1), runs, 18)
        assert_matches_reference(self, (6, 5, 1), runs, 26)

    def test_empty_volume(self):
        r = label_volume((4, 4, 4), [], 6)
        self.assertEqual(r.num_components, 0)
        self.assertEqual(r.labels, ())
        self.assertEqual(r.components, ())
        r2 = label_volume((4, 4, 4), [Run(0, 0, 2, 2)], 26)
        self.assertEqual(r2.num_components, 0)
        self.assertEqual(r2.labels, (None,))

    def test_id_order_by_min_zyx(self):
        # 输入乱序；id 必须按组件最小 (z,y,x) 排。
        runs = [
            Run(2, 3, 4, 5),  # 组件 B，origin (2,3,4)
            Run(0, 2, 1, 2),  # 组件 A，origin (0,2,1)
            Run(0, 0, 0, 1),  # 组件 C，origin (0,0,0)
        ]
        r = label_volume((6, 6, 4), runs, 6)
        self.assertEqual([c.origin for c in r.components],
                         [(0, 0, 0), (0, 2, 1), (2, 3, 4)])
        self.assertEqual(list(r.labels), [3, 2, 1])

    def test_runs_align_to_input_order(self):
        # 同行端点相接的两个游程，输入逆序给出：标签仍按下标对齐。
        runs_tuples = [(0, 0, 3, 5), (0, 0, 0, 3)]
        r = label_volume((6, 1, 1), runs_tuples, 6)
        self.assertEqual(r.num_components, 1)
        self.assertEqual(list(r.labels), [1, 1])
        self.assertEqual(r.components[0].voxel_count, 5)


class RandomizedReferenceTests(unittest.TestCase):
    """小体素随机用例：三种邻接下与独立 BFS 参考全量比对。"""

    def test_random_small_volumes(self):
        rng = random.Random(20260922)
        for case in range(80):
            nx = rng.randint(1, 6)
            ny = rng.randint(1, 6)
            nz = rng.randint(1, 5)
            density = rng.uniform(0.05, 0.45)
            voxels = {
                (x, y, z)
                for x in range(nx) for y in range(ny) for z in range(nz)
                if rng.random() < density
            }
            runs = voxels_to_runs(voxels)
            if case % 3 == 0:
                rng.shuffle(runs)  # 部分用例打乱输入顺序
            for conn in (6, 18, 26):
                assert_matches_reference(self, (nx, ny, nz), runs, conn)


class ErrorSemanticsTests(unittest.TestCase):
    def expect_error(self, exc, dims, runs, conn=6):
        with self.assertRaises(exc) as cm:
            label_volume(dims, runs, conn)
        return str(cm.exception)

    def test_reject_nan_infinity(self):
        msg = self.expect_error(ValueError, (2, 2, 2),
                                [(0, 0, float("nan"), 1)])
        self.assertIn("runs[0]", msg)
        msg = self.expect_error(ValueError, (2, 2, 2),
                                [(0, 0, 0, float("inf"))])
        self.assertIn("runs[0]", msg)

    def test_reject_bool_as_int(self):
        with self.assertRaises(TypeError):
            label_volume((True, 2, 2), [(0, 0, 0, 1)])
        with self.assertRaises(TypeError):
            label_volume((2, 2, 2), [(True, 0, 0, 1)])
        with self.assertRaises(TypeError):
            label_volume((2, 2, 2), [(0, 0, 0, 1)], connectivity=True)

    def test_reject_float_coords_and_dims(self):
        with self.assertRaises(TypeError):
            label_volume((2.0, 2, 2), [(0, 0, 0, 1)])
        with self.assertRaises(TypeError):
            label_volume((2, 2, 2), [(0, 0, 0, 1.0)])
        with self.assertRaises(TypeError):
            label_volume((2, 2, 2), [(0, 0, 0, 1)], connectivity=6.0)

    def test_reject_bad_connectivity(self):
        msg = self.expect_error(ValueError, (2, 2, 2),
                                [(0, 0, 0, 1)], conn=12)
        self.assertIn("connectivity", msg)

    def test_reject_out_of_range(self):
        with self.assertRaises(ValueError):
            label_volume((2, 2, 2), [(2, 0, 0, 1)])
        with self.assertRaises(ValueError):
            label_volume((2, 2, 2), [(0, 2, 0, 1)])
        with self.assertRaises(ValueError):
            label_volume((2, 2, 2), [(0, 0, -1, 1)])
        with self.assertRaises(ValueError):
            label_volume((2, 2, 2), [(0, 0, 0, 3)])
        with self.assertRaises(ValueError):
            label_volume((2, 2, 2), [(0, 0, 1, 0)])  # x0 > x1

    def test_reject_bad_dims(self):
        with self.assertRaises(ValueError):
            label_volume((2, 2), [(0, 0, 0, 1)])
        with self.assertRaises(ValueError):
            label_volume((0, 2, 2), [(0, 0, 0, 1)])
        with self.assertRaises(ValueError):
            label_volume((MAX_DIM + 1, 2, 2), [])
        with self.assertRaises(TypeError):
            label_volume("222", [])

    def test_reject_overlap_locatable(self):
        with self.assertRaises(ValueError) as cm:
            label_volume((8, 2, 1), [(0, 0, 0, 3), (0, 0, 2, 5)])
        self.assertIn("z=0", str(cm.exception))
        self.assertIn("重叠", str(cm.exception))
        # 端点相接合法
        r = label_volume((8, 2, 1), [(0, 0, 0, 3), (0, 0, 3, 5)])
        self.assertEqual(r.components[0].voxel_count, 5)

    def test_reject_malformed_run(self):
        with self.assertRaises(ValueError):
            label_volume((4, 4, 4), [(0, 0, 1)])
        with self.assertRaises(TypeError):
            label_volume((4, 4, 4), [1234])

    def test_too_many_runs(self):
        runs = [(z, y, 0, 1) for z in range(512) for y in range(40)]
        self.assertEqual(len(runs), 20480)
        with self.assertRaises(ValueError) as cm:
            label_volume((8, 40, 512), runs)
        self.assertIn(str(MAX_RUNS), str(cm.exception))

    def test_failure_leaves_no_partial_result(self):
        # 有状态对象（DSU/聚合）全部封装在函数内部；
        # 非法调用抛错后，下一次合法调用结果必须完整正确。
        with self.assertRaises(ValueError):
            label_volume((2, 2, 2), [(0, 0, 0, 3)])
        r = label_volume((2, 2, 2), [(0, 0, 0, 1)], 26)
        self.assertEqual(r.num_components, 1)
        self.assertEqual(r.components[0].exposed_area, 6)


class DemoSmokeTest(unittest.TestCase):
    def test_demo_runs_and_reports(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        proc = subprocess.run(
            [sys.executable, "demo.py"],
            cwd=root, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("connectivity", proc.stdout.lower())
        self.assertIn("error", proc.stdout.lower())  # 实际触发的拒绝边界


if __name__ == "__main__":
    unittest.main(verbosity=2)
