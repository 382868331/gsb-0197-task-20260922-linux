"""find_cavities 的必要测试。

参考期望值来自 tests/reference.py 的 reference_cavities：独立的逐体素
BFS（前景 6 邻接、背景 26 邻接、盒边界即外部），不导入、不调用被测
核心。另含手算小例（空心壳、角点孔道、嵌套壳、开放凹槽）、同一 RLE
不同分段的不变性，以及错误语义测试。
"""

import random
import unittest

from rle_components import (
    BoundingBox,
    InvalidVolumeError,
    find_cavities,
    label,
)

from reference import reference_cavities


def encode_dense(dense, depth, height):
    vol = []
    for z in range(depth):
        rows = []
        for y in range(height):
            row, s = [], None
            for x, on in enumerate(dense[z][y]):
                if on and s is None:
                    s = x
                if not on and s is not None:
                    row.append((s, x))
                    s = None
            if s is not None:
                row.append((s, len(dense[z][y])))
            rows.append(row)
        vol.append(rows)
    return vol


def encode_voxels(voxels, depth, height, width):
    dense = [
        [[(z, y, x) in voxels for x in range(width)] for y in range(height)]
        for z in range(depth)
    ]
    return encode_dense(dense, depth, height)


def shell_voxels(lo, hi):
    """[lo, hi) 立方体的表面体素集合。"""
    out = set()
    for z in range(lo, hi):
        for y in range(lo, hi):
            for x in range(lo, hi):
                if (z in (lo, hi - 1) or y in (lo, hi - 1)
                        or x in (lo, hi - 1)):
                    out.add((z, y, x))
    return out


def assert_matches_reference(testcase, vol, depth, height, width):
    result = find_cavities(vol, depth, height, width)
    ref = reference_cavities(vol, depth, height, width)

    testcase.assertEqual(len(result.cavities), len(ref))
    testcase.assertEqual(
        [c.id for c in result.cavities],
        list(range(1, len(ref) + 1)),
    )
    mins = [(c.min_z, c.min_y, c.min_x) for c in result.cavities]
    testcase.assertEqual(mins, sorted(mins))

    for cav, m in zip(result.cavities, ref):
        testcase.assertEqual(cav.voxel_count, m["voxels"])
        testcase.assertEqual(tuple(cav.bbox), m["bbox"])
        testcase.assertEqual(cav.touching_components, m["touching"])
        testcase.assertEqual((cav.min_z, cav.min_y, cav.min_x), m["min"])

    # 前景口径与 label(..., 6) 一致
    testcase.assertEqual(result.foreground.connectivity, 6)
    fg6 = label(vol, depth, height, width, 6)
    testcase.assertEqual(result.foreground.components, fg6.components)
    # 接触 ID 必然引用真实存在的前景组件
    ncomp = len(fg6.components)
    for cav in result.cavities:
        for cid in cav.touching_components:
            testcase.assertTrue(1 <= cid <= ncomp)
    return result


class HandComputedCavityTests(unittest.TestCase):
    def test_hollow_shell_single_voxel_cavity(self):
        # 5^3 盒内放 3^3 壳（坐标 1..3），中心 (2,2,2) 为唯一空腔体素
        vol = encode_voxels(shell_voxels(1, 4), 5, 5, 5)
        r = find_cavities(vol, 5, 5, 5)
        self.assertEqual(len(r.cavities), 1)
        cav = r.cavities[0]
        self.assertEqual(cav.voxel_count, 1)
        self.assertEqual(cav.bbox, BoundingBox(2, 2, 2, 3, 3, 3))
        self.assertEqual(cav.touching_components, (1,))
        self.assertEqual(r.background_components, 2)  # 外部 + 空腔

    def test_shell_with_corner_pinhole_is_open(self):
        # 只去掉壳的一个角点体素：背景 26 邻接沿角对角泄漏，不是空腔
        voxels = shell_voxels(1, 4)
        voxels.discard((1, 1, 1))
        vol = encode_voxels(voxels, 5, 5, 5)
        r = find_cavities(vol, 5, 5, 5)
        self.assertEqual(r.cavities, [])
        self.assertEqual(r.background_components, 1)

    def test_shell_with_face_pinhole_is_open(self):
        voxels = shell_voxels(1, 4)
        voxels.discard((1, 2, 2))  # 面中心开孔
        vol = encode_voxels(voxels, 5, 5, 5)
        self.assertEqual(find_cavities(vol, 5, 5, 5).cavities, [])

    def test_nested_shells_two_cavities(self):
        # 11^3 盒：外壳 7^3（2..8），内壳 3^3（4..6），中心 (5,5,5) 空
        outer = shell_voxels(2, 9)
        inner = shell_voxels(4, 7)
        vol = encode_voxels(outer | inner, 11, 11, 11)
        r = find_cavities(vol, 11, 11, 11)
        self.assertEqual(len(r.cavities), 2)
        gap, core = r.cavities
        # 壳间夹层：5^3 - 3^3 = 98，同时接触外、内两个壳组件
        self.assertEqual(gap.voxel_count, 98)
        self.assertEqual(gap.bbox, BoundingBox(3, 3, 3, 8, 8, 8))
        self.assertEqual(gap.touching_components, (1, 2))
        # 最内层：单体素，只接触内壳
        self.assertEqual(core.voxel_count, 1)
        self.assertEqual(core.bbox, BoundingBox(5, 5, 5, 6, 6, 6))
        self.assertEqual(core.touching_components, (2,))
        self.assertEqual(r.background_components, 3)

    def test_open_groove_at_boundary_not_cavity(self):
        # 前景块贴盒底，顶面开一条通到盒边界的凹槽：背景始终通外部
        voxels = set()
        for z in range(0, 3):
            for y in range(0, 5):
                for x in range(0, 5):
                    voxels.add((z, y, x))
        # 凹槽：z=2 层挖掉 y=0..4, x=2 一整列，直通 y 边界
        for y in range(5):
            voxels.discard((2, y, 2))
        vol = encode_voxels(voxels, 5, 5, 5)
        self.assertEqual(find_cavities(vol, 5, 5, 5).cavities, [])

    def test_foreground_bbox_flush_with_boundary_still_open(self):
        # 前景包围盒恰好贴边：凹槽开在向盒外的面上，不得误判为闭腔
        voxels = set()
        for z in range(3):
            for y in range(3):
                for x in range(3):
                    voxels.add((z, y, x))
        # 从 z=2 顶面向下挖 1 体素深坑（开放）
        voxels.discard((2, 1, 1))
        vol = encode_voxels(voxels, 3, 3, 3)
        self.assertEqual(find_cavities(vol, 3, 3, 3).cavities, [])
        # 对照：把坑口用顶面体素封死（z=2 整层填满）才是空腔
        voxels2 = set(voxels)
        for y in range(3):
            for x in range(3):
                voxels2.add((2, y, x))
        voxels2.discard((1, 1, 1))  # 留下被封闭的单体素腔
        vol2 = encode_voxels(voxels2, 3, 3, 3)
        r2 = find_cavities(vol2, 3, 3, 3)
        self.assertEqual(len(r2.cavities), 1)
        self.assertEqual(r2.cavities[0].voxel_count, 1)

    def test_same_volume_different_run_segmentation(self):
        # 同一前景：整行一条游程 vs 拆成相接的多条游程，空腔结果一致
        shell = shell_voxels(1, 4)
        vol_a = encode_voxels(shell, 5, 5, 5)
        # 把每条长度 >= 2 的游程拆成两条相接游程
        vol_b = []
        for sl in vol_a:
            rows = []
            for row in sl:
                new_row = []
                for s, e in row:
                    if e - s >= 2:
                        mid = (s + e) // 2
                        new_row.append((s, mid))
                        new_row.append((mid, e))
                    else:
                        new_row.append((s, e))
                rows.append(new_row)
            vol_b.append(rows)
        ra = find_cavities(vol_a, 5, 5, 5)
        rb = find_cavities(vol_b, 5, 5, 5)
        self.assertEqual(ra.cavities, rb.cavities)
        self.assertEqual(len(ra.cavities), 1)

    def test_empty_and_full_volume(self):
        empty = [[[] for _ in range(3)] for _ in range(3)]
        r = find_cavities(empty, 3, 3, 3)
        self.assertEqual(r.cavities, [])
        self.assertEqual(r.background_components, 1)
        self.assertEqual(r.foreground.components, [])

        full = [[[(0, 3)] for _ in range(3)] for _ in range(3)]
        r = find_cavities(full, 3, 3, 3)
        self.assertEqual(r.cavities, [])
        self.assertEqual(r.background_components, 0)

    def test_single_voxel_box(self):
        r = find_cavities([[[]]], 1, 1, 1)
        self.assertEqual(r.cavities, [])
        r = find_cavities([[[(0, 1)]]], 1, 1, 1)
        self.assertEqual(r.cavities, [])


class RandomCavityTests(unittest.TestCase):
    def test_random_against_reference(self):
        rng = random.Random(20260926)
        for trial in range(60):
            depth = rng.randint(1, 5)
            height = rng.randint(1, 5)
            width = rng.randint(1, 5)
            p = rng.choice([0.2, 0.4, 0.6, 0.8])
            dense = [
                [[rng.random() < p for _ in range(width)]
                 for _ in range(height)]
                for _ in range(depth)
            ]
            vol = encode_dense(dense, depth, height)
            with self.subTest(trial=trial, dims=(depth, height, width), p=p):
                assert_matches_reference(self, vol, depth, height, width)

    def test_structured_random_shells(self):
        # 随机打洞的壳：只有完全封闭的壳才产生空腔
        rng = random.Random(7)
        for trial in range(20):
            voxels = shell_voxels(1, 6)
            for v in list(voxels):
                if rng.random() < 0.05:
                    voxels.discard(v)
            vol = encode_voxels(voxels, 7, 7, 7)
            with self.subTest(trial=trial):
                assert_matches_reference(self, vol, 7, 7, 7)


class CavityErrorTests(unittest.TestCase):
    def test_invalid_dimensions(self):
        for bad in (0, -1, 513, 1.5, True, "3"):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidVolumeError):
                    find_cavities([[]], bad, 1, 1)

    def test_nan_and_infinity_coords(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(InvalidVolumeError):
                    find_cavities([[[(0, bad)]]], 1, 1, 4)

    def test_overlapping_and_unordered_runs(self):
        with self.assertRaises(InvalidVolumeError):
            find_cavities([[[(0, 3), (2, 4)]]], 1, 1, 4)
        with self.assertRaises(InvalidVolumeError):
            find_cavities([[[(2, 4), (0, 1)]]], 1, 1, 4)

    def test_structure_mismatch(self):
        with self.assertRaises(InvalidVolumeError):
            find_cavities([[], []], 1, 1, 4)  # 切片数不符
        with self.assertRaises(InvalidVolumeError):
            find_cavities([[], []], 2, 2, 4)  # 行数不符

    def test_out_of_range_and_empty_run(self):
        with self.assertRaises(InvalidVolumeError):
            find_cavities([[[(0, 5)]]], 1, 1, 4)
        with self.assertRaises(InvalidVolumeError):
            find_cavities([[[(2, 2)]]], 1, 1, 4)

    def test_failure_leaves_no_partial_state(self):
        # 纯函数语义：失败后合法调用仍正常
        with self.assertRaises(InvalidVolumeError):
            find_cavities([[[(0, float("nan"))]]], 1, 1, 4)
        r = find_cavities([[[(0, 1)]]], 1, 1, 4)
        self.assertEqual(r.cavities, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
