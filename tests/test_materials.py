"""label_materials 多材料入口的必要测试。

参考期望值来自 tests/reference.py 的 reference_label_materials：独立的
体素展开 + 同材料 BFS + 逐体素六方向数面实现，不导入、不调用被测核心。
另含手算小例与题面点名场景：不同材料交错、跨切片相贴、仅角接触、
分段等价、空体、不同连通方式下编号变化、单个长游程（防逐体素退化）。
"""

import random
import time
import unittest

from rle_components import (
    BoundingBox,
    InvalidVolumeError,
    MaterialRun,
    label_materials,
)

from reference import expand_materials, reference_label_materials


def encode_dense_materials(dense, depth, height):
    """dense[z][y][x] -> None 或材料整数；编码为 (s, e, m) 游程。"""
    vol = []
    for z in range(depth):
        rows = []
        for y in range(height):
            row, s, cur = [], None, None
            for x, m in enumerate(dense[z][y]):
                if m is not None and s is None:
                    s, cur = x, m
                elif m is not None and m != cur:
                    row.append((s, x, cur))
                    s, cur = x, m
                elif m is None and s is not None:
                    row.append((s, x, cur))
                    s, cur = None, None
            if s is not None:
                row.append((s, len(dense[z][y]), cur))
            rows.append(row)
        vol.append(rows)
    return vol


def random_material_volume(rng, depth, height, width, p, n_materials):
    dense = []
    for z in range(depth):
        sl = []
        for y in range(height):
            row = []
            for x in range(width):
                if rng.random() < p:
                    row.append(rng.randrange(n_materials))
                else:
                    row.append(None)
            sl.append(row)
        dense.append(sl)
    return encode_dense_materials(dense, depth, height)


def assert_matches_reference(testcase, vol, depth, height, width, conn):
    result = label_materials(vol, depth, height, width, conn)
    ref_labels, ref_meta, ref_contacts = reference_label_materials(
        vol, depth, height, width, conn
    )

    testcase.assertEqual(result.run_labels, ref_labels)
    testcase.assertEqual(len(result.components), len(ref_meta))
    testcase.assertEqual(
        result.num_runs, sum(len(r) for sl in vol for r in sl)
    )

    # ID 连续且按最小 (z, y, x) 升序
    mins = [(c.min_z, c.min_y, c.min_x) for c in result.components]
    testcase.assertEqual(mins, sorted(mins))
    testcase.assertEqual([c.id for c in result.components],
                         list(range(1, len(mins) + 1)))

    occupied = len(expand_materials(vol, depth, height, width))
    testcase.assertEqual(sum(c.voxel_count for c in result.components),
                         occupied)

    for comp, m in zip(result.components, ref_meta):
        testcase.assertEqual(comp.material, m["material"])
        testcase.assertEqual(comp.voxel_count, m["voxels"])
        testcase.assertEqual(tuple(comp.bbox), m["bbox"])
        testcase.assertEqual(comp.exposed_area, m["area"])
        testcase.assertEqual(
            (comp.min_z, comp.min_y, comp.min_x), m["min"]
        )
        testcase.assertEqual(comp.min_z, comp.bbox.min_z)
        testcase.assertGreaterEqual(comp.exposed_area, 0)

    # 接触表：每对仅一次、较小 ID 在前、面积与参考一致
    got = [(c.component_a, c.component_b, c.area) for c in result.contacts]
    testcase.assertEqual(got, ref_contacts)
    for a, b, area in got:
        testcase.assertLess(a, b)
        testcase.assertGreater(area, 0)
        # 接触对必为不同材料（同材料面相邻必然已合并）
        testcase.assertNotEqual(result.components[a - 1].material,
                                result.components[b - 1].material)

    # 面积守恒：6V = 暴露 + 2*内部 + 接触；接触面两侧对称
    for comp in result.components:
        contact_sum = sum(
            c.area for c in result.contacts
            if comp.id in (c.component_a, c.component_b)
        )
        internal = (6 * comp.voxel_count - comp.exposed_area - contact_sum)
        testcase.assertGreaterEqual(internal, 0)
        testcase.assertEqual(internal % 2, 0)

    return result


class HandComputedMaterialTests(unittest.TestCase):
    def test_interleaved_materials_same_row(self):
        """同一行内不同材料端点相接：分开，接触面积各计 1。"""
        vol = [[[(0, 2, 1), (2, 4, 2)]]]
        for conn in (6, 18, 26):
            r = label_materials(vol, 1, 1, 4, conn)
            self.assertEqual(len(r.components), 2)
            self.assertEqual([c.material for c in r.components], [1, 2])
            self.assertEqual([c.voxel_count for c in r.components], [2, 2])
            # 每个组件：6*2 - 2*1(内部) - 1(接触) = 9
            self.assertEqual([c.exposed_area for c in r.components], [9, 9])
            self.assertEqual(
                [(c.component_a, c.component_b, c.area) for c in r.contacts],
                [(1, 2, 1)],
            )

    def test_same_material_adjacent_runs_merge(self):
        """同材料端点相接：合并为一个组件，无接触对。"""
        vol = [[[(0, 2, 3), (2, 4, 3)]]]
        r = label_materials(vol, 1, 1, 4, 6)
        self.assertEqual(len(r.components), 1)
        c = r.components[0]
        self.assertEqual(c.material, 3)
        self.assertEqual(c.voxel_count, 4)
        self.assertEqual(c.exposed_area, 18)
        self.assertEqual(r.contacts, [])

    def test_segmentation_equivalence(self):
        """题面点名：长游程切成几段（同材料）结果必须一致。"""
        whole = [
            [[(0, 4, 1), (4, 6, 2)], [(0, 2, 2)]],
            [[(1, 3, 1)], []],
        ]
        split = [
            [[(0, 2, 1), (2, 4, 1), (4, 5, 2), (5, 6, 2)],
             [(0, 1, 2), (1, 2, 2)]],
            [[(1, 2, 1), (2, 3, 1)], []],
        ]
        for conn in (6, 18, 26):
            a = label_materials(whole, 2, 2, 6, conn)
            b = label_materials(split, 2, 2, 6, conn)
            self.assertEqual(a.components, b.components)
            self.assertEqual(a.contacts, b.contacts)

    def test_cross_slice_touching_different_materials(self):
        """题面点名：跨切片相贴——两切片整面贴合但材料不同。"""
        vol = [
            [[(0, 2, 1)], []],
            [[(0, 2, 2)], []],
        ]
        r = label_materials(vol, 2, 2, 2, 6)
        self.assertEqual(len(r.components), 2)
        # 共享 2 个体素面
        self.assertEqual(
            [(c.component_a, c.component_b, c.area) for c in r.contacts],
            [(1, 2, 2)],
        )
        # 每个组件：6*2 - 2*1(内部) - 2(接触) = 8
        self.assertEqual([c.exposed_area for c in r.components], [8, 8])

    def test_corner_only_contact_zero_area(self):
        """题面点名：仅角接触——不同材料角对角，任何连通性下都无接触面积。"""
        vol = [[[] for _ in range(2)] for _ in range(2)]
        vol[0][0] = [(0, 1, 1)]
        vol[1][1] = [(1, 2, 2)]
        for conn in (6, 18, 26):
            r = label_materials(vol, 2, 2, 2, conn)
            self.assertEqual(len(r.components), 2)
            self.assertEqual(r.contacts, [])
            self.assertEqual([c.exposed_area for c in r.components], [6, 6])

    def test_corner_only_same_material_connectivity(self):
        """同材料对角链：6/18 下 3 个组件，26 下 1 个；面积始终不含角。"""
        vol = [[[] for _ in range(3)] for _ in range(3)]
        for z, y, x in [(0, 0, 0), (1, 1, 1), (2, 2, 2)]:
            vol[z][y] = [(x, x + 1, 7)]
        self.assertEqual(len(label_materials(vol, 3, 3, 3, 6).components), 3)
        self.assertEqual(len(label_materials(vol, 3, 3, 3, 18).components), 3)
        r = label_materials(vol, 3, 3, 3, 26)
        self.assertEqual(len(r.components), 1)
        self.assertEqual(r.components[0].exposed_area, 18)  # 角接触不算面
        self.assertEqual(r.contacts, [])

    def test_empty_volume(self):
        """题面点名：空体。"""
        empty = [[[] for _ in range(2)] for _ in range(2)]
        for conn in (6, 18, 26):
            r = label_materials(empty, 2, 2, 2, conn)
            self.assertEqual(r.components, [])
            self.assertEqual(r.contacts, [])
            self.assertEqual(r.num_runs, 0)
            self.assertEqual(r.run_labels, [[[], []], [[], []]])

    def test_connectivity_changes_ids_and_contacts(self):
        """题面点名：换连通模式使组件合并，接触表与编号同步更新。

        体素：mat1@(0,0,0)，mat2@(0,1,0)，mat1@(0,1,1)。
        6 邻接：3 个组件，接触 (1,2)=1、(2,3)=1；
        18/26 邻接：两个 mat1 经边合并，组件数 2，接触 (1,2)=2。
        """
        vol = [[[(0, 1, 1)], [(0, 1, 2), (1, 2, 1)]]]
        r6 = label_materials(vol, 1, 2, 2, 6)
        self.assertEqual([c.material for c in r6.components], [1, 2, 1])
        self.assertEqual(
            [(c.component_a, c.component_b, c.area) for c in r6.contacts],
            [(1, 2, 1), (2, 3, 1)],
        )
        for conn in (18, 26):
            r = label_materials(vol, 1, 2, 2, conn)
            self.assertEqual(len(r.components), 2)
            self.assertEqual([c.material for c in r.components], [1, 2])
            self.assertEqual(r.components[0].voxel_count, 2)
            self.assertEqual(
                [(c.component_a, c.component_b, c.area) for c in r.contacts],
                [(1, 2, 2)],
            )
            # 合并不改变总暴露面积：mat1 组件 12-2=10，mat2 6-2=4
            self.assertEqual([c.exposed_area for c in r.components], [10, 4])

    def test_same_material_disconnected_components_separate(self):
        """同材料但不连通：分别统计，互不出现在接触表。"""
        vol = [[[(0, 1, 5), (3, 4, 5)]]]
        r = label_materials(vol, 1, 1, 4, 6)
        self.assertEqual(len(r.components), 2)
        self.assertEqual([c.material for c in r.components], [5, 5])
        self.assertEqual([c.exposed_area for c in r.components], [6, 6])
        self.assertEqual(r.contacts, [])

    def test_single_long_run(self):
        """题面点名：单个长游程——结果正确且不按体素逐点工作。"""
        t0 = time.perf_counter()
        r = label_materials([[[(0, 512, 1)]]], 1, 1, 512, 6)
        elapsed = time.perf_counter() - t0
        self.assertEqual(len(r.components), 1)
        c = r.components[0]
        self.assertEqual(c.voxel_count, 512)
        self.assertEqual(c.exposed_area, 6 * 512 - 2 * 511)
        self.assertEqual(c.bbox, BoundingBox(0, 0, 0, 1, 1, 512))
        self.assertLess(elapsed, 5.0)

    def test_large_slab_single_component_fast(self):
        """19968 条游程的实心板（39x512x512）：聚合成一个组件。

        若核心退化为逐体素工作（千万级体素），纯 Python 无法在该
        时间预算内完成；游程级实现应在约 1 秒内返回。
        """
        d, h, w = 39, 512, 512
        vol = [[[(0, w, 1)] for _ in range(h)] for _ in range(d)]
        t0 = time.perf_counter()
        r = label_materials(vol, d, h, w, 26)
        elapsed = time.perf_counter() - t0
        self.assertEqual(r.num_runs, d * h)
        self.assertEqual(len(r.components), 1)
        c = r.components[0]
        self.assertEqual(c.voxel_count, d * h * w)
        self.assertEqual(
            c.exposed_area, 2 * (d * h + d * w + h * w)
        )
        self.assertEqual(r.contacts, [])
        self.assertLess(elapsed, 5.0)

    def test_materialrun_and_dict_forms_accepted(self):
        vol_t = [[[MaterialRun(0, 2, 1)]]]
        vol_d = [[[{"start": 0, "end": 2, "material": 1}]]]
        for vol in (vol_t, vol_d):
            r = label_materials(vol, 1, 1, 2, 6)
            self.assertEqual(r.components[0].voxel_count, 2)
            self.assertEqual(r.components[0].material, 1)

    def test_component_at_and_contact_area_helpers(self):
        vol = [[[(0, 1, 1), (1, 2, 2)]]]
        r = label_materials(vol, 1, 1, 2, 6)
        self.assertIs(r.component_at(0, 0, 0), r.components[0])
        self.assertEqual(r.contact_area(1, 2), 1)
        self.assertEqual(r.contact_area(2, 1), 1)
        self.assertEqual(r.contact_area(1, 1), 0)


class RandomizedMaterialReferenceTests(unittest.TestCase):
    """随机多材料小体：核心结果必须逐字段等于独立 BFS 参考。"""

    def test_random_vs_bfs_reference(self):
        cases = [(1, 1, 7), (2, 3, 5), (4, 4, 4), (3, 5, 6), (2, 2, 8)]
        seed = 20260925
        for d, h, w in cases:
            for p in (0.35, 0.6):
                for n_mat in (1, 2, 4):
                    rng = random.Random(seed)
                    seed += 1
                    vol = random_material_volume(rng, d, h, w, p, n_mat)
                    for conn in (6, 18, 26):
                        assert_matches_reference(self, vol, d, h, w, conn)

    def test_wider_rows_vs_reference(self):
        rng = random.Random(777)
        for _ in range(6):
            vol = random_material_volume(
                rng, 3, 3, 40, rng.choice((0.3, 0.55)), 3
            )
            for conn in (6, 18, 26):
                assert_matches_reference(self, vol, 3, 3, 40, conn)


class MaterialErrorSemanticsTests(unittest.TestCase):
    def _assert_bad(self, *args, **kwargs):
        with self.assertRaises(InvalidVolumeError):
            label_materials(*args, **kwargs)

    def test_bad_material_values(self):
        self._assert_bad([[[(0, 1, -1)]]], 1, 1, 1)          # 负材料
        self._assert_bad([[[(0, 1, True)]]], 1, 1, 1)        # bool
        self._assert_bad([[[(0, 1, 1.0)]]], 1, 1, 1)         # float
        self._assert_bad([[[(0, 1, "1")]]], 1, 1, 1)         # 字符串
        self._assert_bad([[[(0, 1, None)]]], 1, 1, 1)

    def test_missing_material_and_malformed_runs(self):
        self._assert_bad([[[(0, 1)]]], 1, 1, 1)              # 缺 material
        self._assert_bad([[[(0, 1, 1, 2)]]], 1, 1, 1)        # 四元组
        self._assert_bad(
            [[[{"start": 0, "end": 1}]]], 1, 1, 1            # 字典缺键
        )
        self._assert_bad(
            [[[{"start": 0, "end": 1, "material": 1, "x": 0}]]], 1, 1, 1
        )

    def test_overlap_and_unsorted(self):
        with self.assertRaises(InvalidVolumeError) as cm:
            label_materials([[[(0, 3, 1), (2, 4, 2)]]], 1, 1, 4)
        self.assertIn("run=1", str(cm.exception))
        self.assertIn("重叠", str(cm.exception))
        with self.assertRaises(InvalidVolumeError):
            label_materials([[[(2, 3, 1), (0, 1, 1)]]], 1, 1, 3)

    def test_coordinate_and_dim_violations(self):
        self._assert_bad([[[(0, 2, 1)]]], 1, 1, 1)           # end > width
        self._assert_bad([[[(1, 1, 1)]]], 1, 1, 2)           # 空游程
        self._assert_bad([[[(0, float("nan"), 1)]]], 1, 1, 1)
        self._assert_bad([], 0, 1, 1)
        self._assert_bad([[[]]], 1, 1, 1, 8)                 # 连通性非法
        self._assert_bad([[[]]], 1, 1, 1, True)

    def test_too_many_runs(self):
        d, h = 41, 489
        big = [[[] for _ in range(h)] for _ in range(d)]
        count = 0
        for z in range(d):
            for y in range(h):
                if count < 20001:
                    big[z][y] = [(0, 1, 1)]
                    count += 1
        with self.assertRaises(InvalidVolumeError) as cm:
            label_materials(big, d, h, 1, 6)
        self.assertIn("20000", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
