"""find_cavities 的必要测试。

参考期望值来自 tests/reference.py 的 reference_cavities：独立的体素
展开 + 背景 BFS 实现，不导入、不调用被测核心。另含题面点名场景：
空心壳、只开一个角点的孔道、嵌套壳、分段方式不同的同一 RLE、
边界贴边的开放凹槽，以及 box 的错误语义。
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


def encode_voxels(voxels, depth, height, width):
    """把体素坐标集合编码为 [z][y] -> [(start, end), ...]。"""
    grid = [[set() for _ in range(height)] for _ in range(depth)]
    for z, y, x in voxels:
        grid[z][y].add(x)
    vol = []
    for z in range(depth):
        rows = []
        for y in range(height):
            xs = sorted(grid[z][y])
            row, s, prev = [], None, None
            for x in xs:
                if s is None:
                    s = x
                elif x != prev + 1:
                    row.append((s, prev + 1))
                    s = x
                prev = x
            if s is not None:
                row.append((s, prev + 1))
            rows.append(row)
        vol.append(rows)
    return vol


def shell_voxels(n, offset=(0, 0, 0)):
    """n^3 立方体的表面体素（空心壳）。"""
    oz, oy, ox = offset
    vox = set()
    for z in range(n):
        for y in range(n):
            for x in range(n):
                if (z in (0, n - 1) or y in (0, n - 1) or x in (0, n - 1)):
                    vox.add((oz + z, oy + y, ox + x))
    return vox


def split_runs(vol):
    """同一体素的另一种合法分段：长度 >= 2 的游程从中间拆开。"""
    out = []
    for sl in vol:
        out_sl = []
        for row in sl:
            out_row = []
            for s, e in row:
                if e - s >= 2:
                    mid = (s + e) // 2
                    out_row.append((s, mid))
                    out_row.append((mid, e))
                else:
                    out_row.append((s, e))
            out_sl.append(out_row)
        out.append(out_sl)
    return out


def assert_matches_reference(testcase, vol, depth, height, width, box):
    result = find_cavities(vol, depth, height, width, box)
    ref = reference_cavities(vol, depth, height, width, box)
    testcase.assertEqual(len(result.cavities), len(ref))
    testcase.assertEqual(
        [c.id for c in result.cavities],
        list(range(1, len(ref) + 1)),
    )
    for cav, m in zip(result.cavities, ref):
        testcase.assertEqual(cav.voxel_count, m["voxels"])
        testcase.assertEqual(tuple(cav.bbox), m["bbox"])
        testcase.assertEqual(set(cav.touches), m["touches"])
        testcase.assertEqual(cav.touches, tuple(sorted(cav.touches)))
    return result


FULL = lambda n: (0, 0, 0, n, n, n)  # noqa: E731


class TestHollowShell(unittest.TestCase):
    def test_closed_shell_has_one_cavity(self):
        n = 6
        vol = encode_voxels(shell_voxels(n), n, n, n)
        result = find_cavities(vol, n, n, n, FULL(n))
        self.assertEqual(len(result.cavities), 1)
        cav = result.cavities[0]
        self.assertEqual(cav.voxel_count, (n - 2) ** 3)
        self.assertEqual(tuple(cav.bbox), (1, 1, 1, n - 1, n - 1, n - 1))
        # 整个壳是单一 6 邻接前景组件
        self.assertEqual(cav.touches, (1,))
        self.assertEqual(len(result.foreground.components), 1)

    def test_shell_with_face_hole_is_open(self):
        # 壳面挖掉一个面体素 -> 内部与外部面连通 -> 无空腔
        n = 6
        vox = shell_voxels(n)
        vox.discard((0, 3, 3))
        vol = encode_voxels(vox, n, n, n)
        result = find_cavities(vol, n, n, n, FULL(n))
        self.assertEqual(result.cavities, [])

    def test_shell_with_corner_channel_only_is_open(self):
        # 只去掉壳的一个角点：6 邻接前景仍是一个组件，但 26 邻接背景
        # 能穿过角点孔道 -> 内部不再是空腔
        n = 6
        vox = shell_voxels(n)
        vox.discard((0, 0, 0))
        vol = encode_voxels(vox, n, n, n)
        result = find_cavities(vol, n, n, n, FULL(n))
        self.assertEqual(result.cavities, [])
        self.assertEqual(len(result.foreground.components), 1)
        # 对照：角点保留时确为空腔
        vol2 = encode_voxels(shell_voxels(n), n, n, n)
        self.assertEqual(len(find_cavities(vol2, n, n, n, FULL(n)).cavities), 1)


class TestNestedShells(unittest.TestCase):
    def test_nested_shells_two_cavities(self):
        n = 9
        vox = shell_voxels(9) | shell_voxels(3, offset=(3, 3, 3))
        vol = encode_voxels(vox, n, n, n)
        result = find_cavities(vol, n, n, n, FULL(n))
        self.assertEqual(len(result.foreground.components), 2)
        self.assertEqual(len(result.cavities), 2)
        middle, inner = result.cavities  # 按最小 (z,y,x) 排序
        # 内外壳之间：7^3 - 3^3，最小体素 (1,1,1)，同时接触两个组件
        self.assertEqual(middle.voxel_count, 7 ** 3 - 3 ** 3)
        self.assertEqual(tuple(middle.bbox), (1, 1, 1, 8, 8, 8))
        self.assertEqual(middle.touches, (1, 2))
        # 内壳内部：1^3，只接触内壳（组件 2，最小坐标 (3,3,3)）
        self.assertEqual(inner.voxel_count, 1)
        self.assertEqual(tuple(inner.bbox), (4, 4, 4, 5, 5, 5))
        self.assertEqual(inner.touches, (2,))


class TestSegmentationInvariance(unittest.TestCase):
    def test_same_volume_different_run_splitting(self):
        n = 7
        vox = shell_voxels(7) | shell_voxels(3, offset=(2, 2, 2))
        vol_a = encode_voxels(vox, n, n, n)
        vol_b = split_runs(vol_a)
        self.assertNotEqual(
            sum(len(r) for sl in vol_a for r in sl),
            sum(len(r) for sl in vol_b for r in sl),
        )
        ra = find_cavities(vol_a, n, n, n, FULL(n))
        rb = find_cavities(vol_b, n, n, n, FULL(n))
        self.assertEqual(ra.cavities, rb.cavities)
        # 前景组件 ID 也不受分段影响
        self.assertEqual(
            [c.id for c in ra.foreground.components],
            [c.id for c in rb.foreground.components],
        )
        self.assertEqual(
            [c.voxel_count for c in ra.foreground.components],
            [c.voxel_count for c in rb.foreground.components],
        )


class TestBoundary(unittest.TestCase):
    def test_open_groove_at_box_face_is_not_cavity(self):
        # 凹槽：5^3 壳去掉整个顶面 -> 开口触达盒边界面 -> 无空腔
        n = 5
        vox = {v for v in shell_voxels(n) if v[0] != 0}
        vol = encode_voxels(vox, n, n, n)
        result = find_cavities(vol, n, n, n, FULL(n))
        self.assertEqual(result.cavities, [])

    def test_box_tighter_than_volume_keeps_boundary_hole_open(self):
        # 前景包围盒恰好贴住查询盒边：盒外还有前景，但盒内凹槽
        # 开口于盒边界面，不能误判为闭腔
        n = 7
        vox = shell_voxels(5, offset=(1, 1, 1))
        vox.discard((1, 3, 3))  # 在 z=1 面开洞，洞贴着查询盒边界
        vol = encode_voxels(vox, n, n, n)
        # 查询盒恰好包住壳（壳占 1..5）
        result = find_cavities(vol, n, n, n, (1, 1, 1, 6, 6, 6))
        self.assertEqual(result.cavities, [])
        # 对照：不开洞时同一查询盒内确为空腔
        vox2 = shell_voxels(5, offset=(1, 1, 1))
        vol2 = encode_voxels(vox2, n, n, n)
        r2 = find_cavities(vol2, n, n, n, (1, 1, 1, 6, 6, 6))
        self.assertEqual(len(r2.cavities), 1)
        self.assertEqual(r2.cavities[0].voxel_count, 27)

    def test_box_excludes_foreground_from_touches(self):
        # 盒外前景不参与 touches：壳外再放一个独立前景块，
        # 查询盒只罩住壳时 touches 不含外部块
        n = 8
        vox = shell_voxels(5) | {(7, 7, 7)}
        vol = encode_voxels(vox, n, n, n)
        result = find_cavities(vol, n, n, n, (0, 0, 0, 5, 5, 5))
        self.assertEqual(len(result.cavities), 1)
        self.assertEqual(result.cavities[0].touches, (1,))
        self.assertEqual(len(result.foreground.components), 2)

    def test_empty_volume_has_no_cavity(self):
        vol = [[[] for _ in range(4)] for _ in range(4)]
        result = find_cavities(vol, 4, 4, 4, FULL(4))
        self.assertEqual(result.cavities, [])
        self.assertEqual(result.foreground.components, [])


class TestRandomAgainstReference(unittest.TestCase):
    def test_random_volumes(self):
        rng = random.Random(20260926)
        for trial in range(30):
            depth = rng.randint(2, 6)
            height = rng.randint(2, 6)
            width = rng.randint(2, 6)
            p = rng.choice([0.2, 0.35, 0.5, 0.7])
            dense = [
                [[rng.random() < p for _ in range(width)]
                 for _ in range(height)]
                for _ in range(depth)
            ]
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
                        row.append((s, width))
                    rows.append(row)
                vol.append(rows)
            # 随机子盒（含整盒）
            z0 = rng.randint(0, depth - 1)
            y0 = rng.randint(0, height - 1)
            x0 = rng.randint(0, width - 1)
            box = (z0, y0, x0,
                   rng.randint(z0 + 1, depth),
                   rng.randint(y0 + 1, height),
                   rng.randint(x0 + 1, width))
            with self.subTest(trial=trial, box=box, p=p):
                assert_matches_reference(self, vol, depth, height, width, box)


class TestBoxErrors(unittest.TestCase):
    def setUp(self):
        self.vol = encode_voxels(shell_voxels(4), 4, 4, 4)

    def test_valid_box_forms(self):
        r1 = find_cavities(self.vol, 4, 4, 4, BoundingBox(0, 0, 0, 4, 4, 4))
        r2 = find_cavities(self.vol, 4, 4, 4, (0, 0, 0, 4, 4, 4))
        r3 = find_cavities(self.vol, 4, 4, 4, [0, 0, 0, 4, 4, 4])
        r4 = find_cavities(self.vol, 4, 4, 4, {
            "min_z": 0, "min_y": 0, "min_x": 0,
            "max_z": 4, "max_y": 4, "max_x": 4,
        })
        self.assertEqual(r1.cavities, r2.cavities)
        self.assertEqual(r2.cavities, r3.cavities)
        self.assertEqual(r3.cavities, r4.cavities)
        self.assertEqual(len(r1.cavities), 1)

    def test_rejects_bad_box(self):
        bad_boxes = [
            (0, 0, 0, 4, 4),              # 长度不足
            (0, 0, 0, 4, 4, 4, 4),        # 长度过长
            (2, 0, 0, 2, 4, 4),           # min == max（空区间）
            (3, 0, 0, 2, 4, 4),           # min > max
            (-1, 0, 0, 4, 4, 4),          # 负坐标
            (0, 0, 0, 5, 4, 4),           # 超出 depth
            (0, 0, 0, 4, 4, 9),           # 超出 width
            (True, 0, 0, 4, 4, 4),        # bool 冒称整数
            (0.0, 0, 0, 4, 4, 4),         # float 坐标
            (0, 0, 0, 4, 4, None),        # 非数值
            "000444",                     # 字符串不是合法 box
            {"min_z": 0},                 # 字典缺键
        ]
        for box in bad_boxes:
            with self.subTest(box=box):
                with self.assertRaises(InvalidVolumeError):
                    find_cavities(self.vol, 4, 4, 4, box)

    def test_rejects_bad_volume_and_dims(self):
        with self.assertRaises(InvalidVolumeError):
            find_cavities(self.vol, 4, 4, 3, (0, 0, 0, 4, 4, 4))  # width 不符
        with self.assertRaises(InvalidVolumeError):
            find_cavities(self.vol, 0, 4, 4, (0, 0, 0, 4, 4, 4))
        with self.assertRaises(InvalidVolumeError):
            find_cavities([[[(0, float("nan"))]]], 1, 1, 1, (0, 0, 0, 1, 1, 1))

    def test_foreground_ids_match_label(self):
        # touches 引用的前景 ID 与 label(connectivity=6) 完全一致
        n = 6
        vox = shell_voxels(6) | {(0, 0, 3), (5, 5, 0)}
        vol = encode_voxels(vox, n, n, n)
        result = find_cavities(vol, n, n, n, FULL(n))
        direct = label(vol, n, n, n, connectivity=6)
        self.assertEqual(result.foreground.components, direct.components)
        self.assertEqual(result.foreground.run_labels, direct.run_labels)


if __name__ == "__main__":
    unittest.main()
