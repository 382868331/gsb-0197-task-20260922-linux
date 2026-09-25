"""rle_components.label_materials 的必要测试。

参考期望值来自 tests/reference.py 中独立的体素展开 + BFS + 逐面计数
实现（不导入、不调用被测核心）。覆盖题面点名场景：不同材料交错、
跨切片相贴、仅角接触、分段等价、空体、不同连通方式下编号变化，
以及单个长游程（防止核心退化为逐体素工作）。
"""

import random
import unittest

from rle_components import (
    BoundingBox,
    InvalidVolumeError,
    MaterialRun,
    label_materials,
)

from reference import expand_materials, reference_label_materials


def encode_material_dense(dense, depth, height):
    """把 [[[material x]*width]*height]*depth（0=空白）编码为 RLE 输入。"""
    vol = []
    for z in range(depth):
        rows = []
        for y in range(height):
            row, s, cur = [], None, 0
            for x, m in enumerate(dense[z][y]):
                if m and s is None:
                    s, cur = x, m
                elif m and m != cur:
                    row.append((s, x, cur))
                    s, cur = x, m
                elif not m and s is not None:
                    row.append((s, x, cur))
                    s = None
            if s is not None:
                row.append((s, len(dense[z][y]), cur))
            rows.append(row)
        vol.append(rows)
    return vol


def random_material_volume(rng, depth, height, width, p, n_materials):
    dense = [
        [
            [
                rng.randint(1, n_materials) if rng.random() < p else 0
                for _ in range(width)
            ]
            for _ in range(height)
        ]
        for _ in range(depth)
    ]
    return encode_material_dense(dense, depth, height)


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
    testcase.assertEqual(
        [c.id for c in result.components], list(range(1, len(mins) + 1))
    )

    occupied = len(expand_materials(vol, depth, height, width))
    testcase.assertEqual(
        sum(c.voxel_count for c in result.components), occupied
    )

    for comp, m in zip(result.components, ref_meta):
        testcase.assertEqual(comp.material, m["material"])
        testcase.assertEqual(comp.voxel_count, m["voxels"])
        testcase.assertEqual(tuple(comp.bbox), m["bbox"])
        testcase.assertEqual(comp.exposed_area, m["area"])
        testcase.assertEqual((comp.min_z, comp.min_y, comp.min_x), m["min"])
        testcase.assertEqual(comp.min_z, comp.bbox.min_z)
        testcase.assertGreaterEqual(comp.exposed_area, 0)

    # 接触表：与参考逐面对照；较小 ID 在前、每对一次、整体有序
    got = {(c.component_a, c.component_b): c.area for c in result.contacts}
    testcase.assertEqual(got, ref_contacts)
    keys = [(c.component_a, c.component_b) for c in result.contacts]
    testcase.assertEqual(keys, sorted(keys))
    for a, b in keys:
        testcase.assertLess(a, b)
        testcase.assertGreater(got[(a, b)], 0)

    return result


class HandComputedMaterialTests(unittest.TestCase):
    def test_interleaved_materials_in_row(self):
        """同一行内材料交错：1-2-1，相邻异材料各共享 1 个面。"""
        vol = [[[(0, 2, 1), (2, 4, 2), (4, 6, 1)]]]
        for conn in (6, 18, 26):
            r = label_materials(vol, 1, 1, 6, conn)
            self.assertEqual(len(r.components), 3)
            self.assertEqual([c.material for c in r.components], [1, 2, 1])
            # 每段 2 体素：内部面 1；首尾段接触面 1 → 暴露 12-2-1 = 9，
            # 中间段两侧各 1 个接触面 → 暴露 12-2-2 = 8
            self.assertEqual(
                [c.exposed_area for c in r.components], [9, 8, 9]
            )
            self.assertEqual(
                [(c.component_a, c.component_b, c.area) for c in r.contacts],
                [(1, 2, 1), (2, 3, 1)],
            )
            # 材料 1 的两段不相连，各自独立统计
            self.assertEqual(r.run_labels[0][0], [1, 2, 3])

    def test_cross_slice_touching(self):
        """跨切片相贴：同 y 行、相邻 z、异材料 → 接触面积 = x 重叠列数。"""
        vol = [
            [[(0, 2, 1)]],
            [[(0, 2, 2)]],
        ]
        for conn in (6, 18, 26):
            r = label_materials(vol, 2, 1, 2, conn)
            self.assertEqual(len(r.components), 2)
            self.assertEqual(
                [(c.component_a, c.component_b, c.area) for c in r.contacts],
                [(1, 2, 2)],
            )
            # 暴露 = 12 - 2*1(内部) - 2(接触) = 8
            self.assertEqual(
                [c.exposed_area for c in r.components], [8, 8]
            )

    def test_contact_area_ignores_connectivity(self):
        """2x2 切片相贴：接触面积 4，在 6/18/26 下完全一致（不多算）。"""
        slice_a = [[(0, 2, 1)], [(0, 2, 1)]]
        slice_b = [[(0, 2, 2)], [(0, 2, 2)]]
        vol = [slice_a, slice_b]
        for conn in (6, 18, 26):
            r = label_materials(vol, 2, 2, 2, conn)
            self.assertEqual(len(r.components), 2)
            self.assertEqual(len(r.contacts), 1)
            self.assertEqual(r.contacts[0].area, 4)
            # 每组件 4 体素、内部面 4、接触面 4 → 暴露 = 24-8-4 = 12
            self.assertEqual(
                [c.exposed_area for c in r.components], [12, 12]
            )

    def test_corner_only_different_materials(self):
        """仅角接触的异材料：任何连通性下都不合并，也无接触面积。"""
        vol = [[[] for _ in range(2)] for _ in range(2)]
        vol[0][0] = [(0, 1, 1)]
        vol[1][1] = [(1, 2, 2)]
        for conn in (6, 18, 26):
            r = label_materials(vol, 2, 2, 2, conn)
            self.assertEqual(len(r.components), 2)
            self.assertEqual(r.contacts, [])
            self.assertEqual(
                [c.exposed_area for c in r.components], [6, 6]
            )

    def test_corner_only_same_material(self):
        """仅角接触的同材料：6/18 下分开，26 下合并；接触表始终为空。"""
        vol = [[[] for _ in range(2)] for _ in range(2)]
        vol[0][0] = [(0, 1, 5)]
        vol[1][1] = [(1, 2, 5)]
        self.assertEqual(len(label_materials(vol, 2, 2, 2, 6).components), 2)
        self.assertEqual(len(label_materials(vol, 2, 2, 2, 18).components), 2)
        r26 = label_materials(vol, 2, 2, 2, 26)
        self.assertEqual(len(r26.components), 1)
        self.assertEqual(r26.components[0].voxel_count, 2)
        self.assertEqual(r26.components[0].exposed_area, 12)
        for conn in (6, 18, 26):
            self.assertEqual(label_materials(vol, 2, 2, 2, conn).contacts, [])

    def test_edge_only_different_materials_no_contact(self):
        """仅边接触的异材料：26 邻接也不多算面积，接触表为空。"""
        vol = [[[] for _ in range(2)]]
        vol[0][0] = [(0, 1, 1)]
        vol[0][1] = [(1, 2, 2)]
        for conn in (6, 18, 26):
            r = label_materials(vol, 1, 2, 2, conn)
            self.assertEqual(len(r.components), 2)
            self.assertEqual(r.contacts, [])

    def test_segmentation_equivalence(self):
        """同材料长游程切成几段，结果必须与单条游程完全一致。"""
        neighbor = (6, 8, 2)  # 异材料邻段，检验接触面积也不受分段影响
        whole = [[[(0, 6, 3), neighbor]]]
        split2 = [[[(0, 2, 3), (2, 6, 3), neighbor]]]
        split3 = [[[(0, 1, 3), (1, 4, 3), (4, 6, 3), neighbor]]]
        expected = label_materials(whole, 1, 1, 8, 6)
        for vol in (split2, split3):
            for conn in (6, 18, 26):
                r = label_materials(vol, 1, 1, 8, conn)
                e = label_materials(whole, 1, 1, 8, conn)
                self.assertEqual(r.components, e.components)
                self.assertEqual(r.contacts, e.contacts)
        # 单条基准：材料 3 的 6 体素棒 + 材料 2 的 2 体素棒，接触面 1
        self.assertEqual(len(expected.components), 2)
        self.assertEqual(expected.components[0].voxel_count, 6)
        self.assertEqual(expected.components[0].exposed_area, 6 * 6 - 2 * 5 - 1)
        self.assertEqual(
            [(c.component_a, c.component_b, c.area) for c in expected.contacts],
            [(1, 2, 1)],
        )

    def test_empty_volume(self):
        """空体：无组件、无接触。"""
        empty = [[[] for _ in range(3)] for _ in range(2)]
        for conn in (6, 18, 26):
            r = label_materials(empty, 2, 3, 4, conn)
            self.assertEqual(r.components, [])
            self.assertEqual(r.contacts, [])
            self.assertEqual(r.num_runs, 0)
            self.assertEqual(r.run_labels, [[[], [], []], [[], [], []]])

    def test_same_material_disconnected(self):
        """同材料但不连通：分别统计，接触表为空。"""
        vol = [[[(0, 1, 4), (3, 4, 4)]]]
        r = label_materials(vol, 1, 1, 4, 6)
        self.assertEqual(len(r.components), 2)
        self.assertEqual([c.material for c in r.components], [4, 4])
        self.assertEqual(r.contacts, [])
        self.assertEqual([c.exposed_area for c in r.components], [6, 6])

    def test_numbering_changes_with_connectivity(self):
        """连通方式改变使组件合并时，编号与接触表同步更新。

        A=(0,0,0) 材料1，B=(0,1,1) 材料1（与 A 仅边接触），
        C=(0,2,2) 材料2（与 B 仅边接触）。
        6 邻接：{A},{B},{C} 三个组件；18/26：{A,B},{C} 两个，
        C 的 ID 由 3 变 2，且任何情况下都没有共享面（接触表为空）。
        """
        vol = [[[(0, 1, 1)], [(1, 2, 1)], [(2, 3, 2)]]]
        r6 = label_materials(vol, 1, 3, 3, 6)
        self.assertEqual(len(r6.components), 3)
        self.assertEqual(r6.run_labels[0][2], [3])
        self.assertEqual(r6.contacts, [])
        for conn in (18, 26):
            r = label_materials(vol, 1, 3, 3, conn)
            self.assertEqual(len(r.components), 2)
            self.assertEqual(r.run_labels[0][0], [1])
            self.assertEqual(r.run_labels[0][1], [1])
            self.assertEqual(r.run_labels[0][2], [2])
            self.assertEqual(r.components[0].voxel_count, 2)
            self.assertEqual(r.components[0].exposed_area, 12)
            self.assertEqual(r.contacts, [])

    def test_face_contact_pair_merged_by_26_updates_contacts(self):
        """26 邻接把同材料角接触体素并入组件后，接触表按新组件重算。

        材料1：A=(0,0,0) 与 B=(1,1,1)（仅角接触）；
        材料2：C1=(1,0,0)、C2=(1,1,0)，彼此面贴合成一个组件，
        且 C1 与 A、C2 与 B 各共享 1 个面。
        6 邻接：材料1 两个组件 → 接触记录 (1,2,1)、(2,3,1)；
        26 邻接：材料1 并为一个组件 → 接触记录 (1,2,2)。
        """
        vol = [
            [[(0, 1, 1)], []],
            [[(0, 1, 2)], [(0, 1, 2), (1, 2, 1)]],
        ]
        r6 = label_materials(vol, 2, 2, 2, 6)
        self.assertEqual(len(r6.components), 3)
        self.assertEqual(
            [(c.component_a, c.component_b, c.area) for c in r6.contacts],
            [(1, 2, 1), (2, 3, 1)],
        )
        r26 = label_materials(vol, 2, 2, 2, 26)
        self.assertEqual(len(r26.components), 2)
        self.assertEqual(
            [(c.component_a, c.component_b, c.area) for c in r26.contacts],
            [(1, 2, 2)],
        )

    def test_single_long_run(self):
        """单个长游程：一条 (0,512) 游程直接成组件，不逐体素展开。"""
        vol = [[[(0, 512, 7)]]]
        for conn in (6, 18, 26):
            r = label_materials(vol, 1, 1, 512, conn)
            self.assertEqual(r.num_runs, 1)
            self.assertEqual(len(r.components), 1)
            c = r.components[0]
            self.assertEqual(c.material, 7)
            self.assertEqual(c.voxel_count, 512)
            self.assertEqual(c.exposed_area, 6 * 512 - 2 * 511)
            self.assertEqual(c.bbox, BoundingBox(0, 0, 0, 1, 1, 512))
            self.assertEqual(r.contacts, [])

    def test_many_long_runs_stay_run_based(self):
        """20000 条长游程（上限）：按游程计算，结果按切片材料交替。

        40 切片 × 500 行 × (0,512) 游程，材料按 z 奇偶交替：
        每切片内部 500 行同材料面贴合成 1 个组件，异材料切片间分离，
        共 40 个组件、39 条接触记录，每条面积 500*512。
        若核心退化为逐体素工作（千万级体素）将无法在此规模下完成。
        """
        d, h, w = 40, 500, 512
        vol = [
            [[(0, w, 1 + z % 2)] for _ in range(h)]
            for z in range(d)
        ]
        r = label_materials(vol, d, h, w, 6)
        self.assertEqual(r.num_runs, 20000)
        self.assertEqual(len(r.components), d)
        self.assertEqual(
            sum(c.voxel_count for c in r.components), d * h * w
        )
        for c in r.components:
            self.assertEqual(c.voxel_count, h * w)
            self.assertEqual(c.material, 1 + (c.min_z % 2))
        self.assertEqual(len(r.contacts), d - 1)
        for c in r.contacts:
            self.assertEqual(c.area, h * w)
            self.assertEqual(c.component_b, c.component_a + 1)

    def test_material_run_and_dict_forms_accepted(self):
        vol_t = [[[MaterialRun(0, 2, 3)]]]
        vol_d = [[[{"start": 0, "end": 2, "material": 3}]]]
        for vol in (vol_t, vol_d):
            r = label_materials(vol, 1, 1, 2, 6)
            self.assertEqual(r.components[0].material, 3)
            self.assertEqual(r.components[0].voxel_count, 2)

    def test_result_helpers(self):
        vol = [
            [[(0, 1, 1)]],
            [[(0, 1, 2)]],
        ]
        r = label_materials(vol, 2, 1, 1, 6)
        self.assertIs(r.component_at(0, 0, 0), r.components[0])
        self.assertIs(r.component_at(1, 0, 0), r.components[1])
        self.assertEqual(r.contact_between(1, 2), 1)
        self.assertEqual(r.contact_between(2, 1), 1)
        self.assertEqual(r.contact_between(1, 1), 0)


class RandomizedMaterialReferenceTests(unittest.TestCase):
    """随机多材料小体：核心结果必须逐字段等于独立 BFS 参考。"""

    def test_random_vs_bfs_reference(self):
        cases = [
            (1, 1, 1), (1, 1, 7), (2, 3, 5), (4, 4, 4),
            (3, 5, 6), (5, 3, 2), (6, 6, 3), (2, 2, 8),
        ]
        seed = 20260925
        for d, h, w in cases:
            for p in (0.3, 0.55, 0.8):
                for n_mat in (1, 2, 4):
                    rng = random.Random(seed)
                    seed += 1
                    vol = random_material_volume(rng, d, h, w, p, n_mat)
                    for conn in (6, 18, 26):
                        assert_matches_reference(
                            self, vol, d, h, w, conn
                        )

    def test_wider_rows_vs_reference(self):
        rng = random.Random(909090)
        for _ in range(8):
            vol = random_material_volume(
                rng, 3, 3, 40, rng.choice((0.35, 0.6)), 3
            )
            for conn in (6, 18, 26):
                assert_matches_reference(self, vol, 3, 3, 40, conn)


class MaterialErrorSemanticsTests(unittest.TestCase):
    def _assert_bad(self, *args, **kwargs):
        with self.assertRaises(InvalidVolumeError):
            label_materials(*args, **kwargs)

    def test_bad_material_values(self):
        self._assert_bad([[[(0, 1, -1)]]], 1, 1, 1)          # 负材料
        self._assert_bad([[[(0, 1, True)]]], 1, 1, 1)         # bool
        self._assert_bad([[[(0, 1, 1.0)]]], 1, 1, 1)          # 整数值 float
        self._assert_bad([[[(0, 1, 1.5)]]], 1, 1, 2)          # 非整 float
        self._assert_bad([[[(0, 1, float("nan"))]]], 1, 1, 1)  # NaN
        self._assert_bad([[[(0, 1, float("inf"))]]], 1, 1, 1)  # Infinity
        self._assert_bad([[[(0, 1, "1")]]], 1, 1, 1)          # 字符串

    def test_material_error_is_locatable(self):
        try:
            label_materials([[[(0, 2, -3)]]], 1, 1, 2)
        except InvalidVolumeError as exc:
            msg = str(exc)
            self.assertIn("z=0", msg)
            self.assertIn("y=0", msg)
            self.assertIn("run=0", msg)
            self.assertIn("material", msg)
        else:  # pragma: no cover
            self.fail("应当抛出 InvalidVolumeError")

    def test_two_tuple_run_rejected(self):
        # 二值接口的 (start, end) 游程在多材料入口不合法
        self._assert_bad([[[(0, 1)]]], 1, 1, 1)
        self._assert_bad([[[{"start": 0, "end": 1}]]], 1, 1, 1)
        self._assert_bad(
            [[[{"start": 0, "end": 1, "material": 1, "x": 2}]]], 1, 1, 1
        )

    def test_overlap_and_unsorted(self):
        with self.assertRaises(InvalidVolumeError) as cm:
            label_materials([[[(0, 3, 1), (2, 4, 2)]]], 1, 1, 4)
        self.assertIn("run=1", str(cm.exception))
        with self.assertRaises(InvalidVolumeError):
            label_materials([[[(2, 3, 1), (0, 1, 1)]]], 1, 1, 3)

    def test_coordinate_and_structure_errors(self):
        self._assert_bad([[[(-1, 1, 0)]]], 1, 1, 1)
        self._assert_bad([[[(0, 2, 0)]]], 1, 1, 1)
        self._assert_bad([[[(1, 1, 0)]]], 1, 1, 2)
        self._assert_bad([[[]]], 2, 1, 1)
        self._assert_bad([[[]]], 1, 2, 1)
        self._assert_bad([], 0, 1, 1)
        self._assert_bad([[[]]], 1, 1, 1, 8)
        self._assert_bad([[[]]], 1, 1, 1, connectivity=True)

    def test_too_many_runs_boundary(self):
        # 恰好 20000 条：接受
        d, h, w = 40, 500, 1
        vol = [[[(0, 1, 2)] for _ in range(h)] for _ in range(d)]
        r = label_materials(vol, d, h, w, 6)
        self.assertEqual(r.num_runs, 20000)

        # 20001 条：拒绝
        d2, h2 = 41, 489
        big = [[[] for _ in range(h2)] for _ in range(d2)]
        count = 0
        for z in range(d2):
            for y in range(h2):
                if count < 20001:
                    big[z][y] = [(0, 1, 0)]
                    count += 1
        with self.assertRaises(InvalidVolumeError) as cm:
            label_materials(big, d2, h2, 1, 6)
        self.assertIn("20000", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
