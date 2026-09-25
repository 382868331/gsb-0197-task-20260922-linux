"""rle_components 的必要测试。

参考期望值来自 tests/reference.py：独立的体素展开 + BFS + 逐面计数实现，
不导入、不调用被测核心。与核心的一致性对比是主要验收手段；
另含手算小例、题面点名场景（仅角接触、空行、跨 slice 桥接）与
错误语义测试。
"""

import random
import unittest

from rle_components import (
    BoundingBox,
    InvalidVolumeError,
    Run,
    StreamLabeler,
    label,
)

from reference import expand, reference_label


def encode_dense(dense, depth, height):
    """把 [[[bool x]*width]*height]*depth 编码为 RLE 输入（列表）。"""
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


def random_volume(rng, depth, height, width, p):
    dense = [
        [[rng.random() < p for _ in range(width)] for _ in range(height)]
        for _ in range(depth)
    ]
    return encode_dense(dense, depth, height)


def assert_matches_reference(testcase, vol, depth, height, width, conn):
    result = label(vol, depth, height, width, conn)
    ref_labels, ref_meta = reference_label(vol, depth, height, width, conn)

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

    occupied = len(expand(vol, depth, height, width))
    testcase.assertEqual(sum(c.voxel_count for c in result.components),
                         occupied)

    for comp, m in zip(result.components, ref_meta):
        testcase.assertEqual(comp.voxel_count, m["voxels"])
        testcase.assertEqual(tuple(comp.bbox), m["bbox"])
        testcase.assertEqual(comp.exposed_area, m["area"])
        testcase.assertEqual(
            (comp.min_z, comp.min_y, comp.min_x), m["min"]
        )
        # 字典序最小体素的 z 必等于 bbox.min_z；y/x 可能比几何 min 大
        # （更小 y/x 的体素可能只出现在更大的 z 上，如 26 邻接对角链）
        testcase.assertEqual(comp.min_z, comp.bbox.min_z)
        testcase.assertGreaterEqual(comp.min_y, comp.bbox.min_y)
        testcase.assertGreaterEqual(comp.min_x, comp.bbox.min_x)
        # 暴露面积非负且与参考逐面值相等（上面已断言 area == m["area"]）
        testcase.assertGreaterEqual(comp.exposed_area, 0)

    return result


class HandComputedTests(unittest.TestCase):
    def test_single_run_bar(self):
        # 一行 4 个体素：面积 = 6*4 - 2*3 = 18 = 4*4+2
        vol = [[[(0, 4)]]]
        for conn in (6, 18, 26):
            r = label(vol, 1, 1, 4, conn)
            self.assertEqual(len(r.components), 1)
            c = r.components[0]
            self.assertEqual(c.voxel_count, 4)
            self.assertEqual(c.exposed_area, 18)
            self.assertEqual(c.bbox, BoundingBox(0, 0, 0, 1, 1, 4))

    def test_full_cube_2x2x2(self):
        row = [(0, 2)]
        vol = [[row, row], [row, row]]
        r = label(vol, 2, 2, 2, 6)
        self.assertEqual(len(r.components), 1)
        c = r.components[0]
        self.assertEqual(c.voxel_count, 8)
        self.assertEqual(c.exposed_area, 24)  # 2x2x2 外表面积
        self.assertEqual(c.bbox, BoundingBox(0, 0, 0, 2, 2, 2))

    def test_face_edge_corner_pair(self):
        """两个单像素：面/边/角三种相对位置。"""
        def with_voxel(z, y, x, d, h, w):
            vol = [[[] for _ in range(h)] for _ in range(d)]
            vol[z][y] = [(x, x + 1)]
            return vol

        base = with_voxel(0, 0, 0, 2, 2, 2)
        # 面接触 (0,0,0)-(0,0,1)
        v = [sl[:] for sl in base]
        v[0][0] = [(0, 2)]
        self.assertEqual(len(label(v, 2, 2, 2, 6).components), 1)

        # 边接触 (0,0,0)-(0,1,1)
        v = with_voxel(0, 1, 1, 2, 2, 2)
        v[0][0] = [(0, 1)]
        self.assertEqual(len(label(v, 2, 2, 2, 6).components), 2)
        self.assertEqual(len(label(v, 2, 2, 2, 18).components), 1)
        self.assertEqual(len(label(v, 2, 2, 2, 26).components), 1)
        # 边接触不共享面：合并后面积仍为 12
        c18 = label(v, 2, 2, 2, 18).components[0]
        self.assertEqual(c18.voxel_count, 2)
        self.assertEqual(c18.exposed_area, 12)

        # 角接触 (0,0,0)-(1,1,1)
        v = with_voxel(1, 1, 1, 2, 2, 2)
        v[0][0] = [(0, 1)]
        self.assertEqual(len(label(v, 2, 2, 2, 6).components), 2)
        self.assertEqual(len(label(v, 2, 2, 2, 18).components), 2)
        r26 = label(v, 2, 2, 2, 26)
        self.assertEqual(len(r26.components), 1)
        # 角接触不共享面：合并后面积仍为 12，包围盒跨全角
        self.assertEqual(r26.components[0].exposed_area, 12)
        self.assertEqual(r26.components[0].bbox,
                         BoundingBox(0, 0, 0, 2, 2, 2))

    def test_corner_only_required_scenario(self):
        """题面点名：仅角接触——三个体素沿 (1,1,1) 对角链。"""
        d = h = w = 3
        vol = [[[] for _ in range(h)] for _ in range(d)]
        for z, y, x in [(0, 0, 0), (1, 1, 1), (2, 2, 2)]:
            vol[z][y] = [(x, x + 1)]
        self.assertEqual(len(label(vol, d, h, w, 6).components), 3)
        self.assertEqual(len(label(vol, d, h, w, 18).components), 3)
        self.assertEqual(len(label(vol, d, h, w, 26).components), 1)

    def test_empty_rows_and_slices(self):
        """题面点名：空行、空切片、整体空体。"""
        # 两个体素中间隔着空行与空切片，互不相连
        vol = [
            [[] for _ in range(3)],
            [[] for _ in range(3)],
            [[] for _ in range(3)],
        ]
        vol[0][0] = [(0, 1)]
        vol[2][2] = [(2, 3)]
        for conn in (6, 18, 26):
            r = label(vol, 3, 3, 3, conn)
            self.assertEqual(len(r.components), 2)
            self.assertEqual(r.run_labels[0][0], [1])
            self.assertEqual(r.run_labels[2][2], [2])

        empty = [[[] for _ in range(2)] for _ in range(2)]
        for conn in (6, 18, 26):
            r = label(empty, 2, 2, 2, conn)
            self.assertEqual(r.components, [])
            self.assertEqual(r.num_runs, 0)
            self.assertEqual(r.run_labels, [[[], []], [[], []]])

    def test_cross_slice_bridge(self):
        """题面点名：跨 slice 桥接。

        z=0: (y=0,x=0)；z=1: (y=1,x=0)（与下方仅边接触）；z=2: (y=1,x=0)
        6 邻接下 z=0 孤立；18/26 下经边桥全部连通。
        """
        vol = [
            [[(0, 1)], []],
            [[], [(0, 1)]],
            [[], [(0, 1)]],
        ]
        self.assertEqual(len(label(vol, 3, 2, 1, 6).components), 2)
        self.assertEqual(len(label(vol, 3, 2, 1, 18).components), 1)
        self.assertEqual(len(label(vol, 3, 2, 1, 26).components), 1)
        r18 = label(vol, 3, 2, 1, 18)
        self.assertEqual(r18.components[0].voxel_count, 3)
        # 一条面 (z1-z2 同 yx) + 一条边：面积 = 18 - 2 = 16
        self.assertEqual(r18.components[0].exposed_area, 16)

    def test_row_runs_adjacent_vs_gap(self):
        # 端点相接：同属一组件，共享 1 个面
        r = label([[[(0, 2), (2, 4)]]], 1, 1, 4, 6)
        self.assertEqual(r.run_labels[0][0], [1, 1])
        self.assertEqual(r.components[0].voxel_count, 4)
        self.assertEqual(r.components[0].exposed_area, 18)

        # 隔一个空 x：两个组件（长度 2 的棒暴露面积 = 6*2-2*1 = 10）
        r = label([[[(0, 2), (3, 5)]]], 1, 1, 5, 6)
        self.assertEqual(r.run_labels[0][0], [1, 2])
        self.assertEqual([c.exposed_area for c in r.components], [10, 10])

    def test_id_ordering_by_min_coordinate(self):
        # 两个不相连组件，最小坐标分别在 (0,2,..) 与 (1,0,..)
        vol = [
            [[], [], [(1, 2)]],
            [[(0, 1)], [], []],
        ]
        r = label(vol, 2, 3, 3, 6)
        self.assertEqual([(c.min_z, c.min_y, c.min_x) for c in r.components],
                         [(0, 2, 1), (1, 0, 0)])
        self.assertEqual(r.run_labels[0][2], [1])
        self.assertEqual(r.run_labels[1][0], [2])

    def test_namedtuple_and_dict_runs_accepted(self):
        vol_t = [[[Run(0, 2)]]]
        vol_d = [[[{"start": 0, "end": 2}]]]
        for vol in (vol_t, vol_d):
            r = label(vol, 1, 1, 2, 6)
            self.assertEqual(r.components[0].voxel_count, 2)

    def test_component_at_helper(self):
        r = label([[[(0, 1)], [(0, 1)]]], 1, 2, 1, 6)
        self.assertIs(r.component_at(0, 1, 0), r.components[0])


class RandomizedReferenceTests(unittest.TestCase):
    """对随机小体素：核心结果必须逐字段等于独立 BFS 参考。"""

    def test_random_vs_bfs_reference(self):
        cases = [
            (1, 1, 1), (1, 1, 7), (2, 3, 5), (4, 4, 4),
            (3, 5, 6), (5, 3, 2), (6, 6, 3), (2, 2, 8),
        ]
        seed = 20260922
        for d, h, w in cases:
            for p in (0.25, 0.5, 0.7):
                rng = random.Random(seed)
                seed += 1
                vol = random_volume(rng, d, h, w, p)
                for conn in (6, 18, 26):
                    assert_matches_reference(
                        self, vol, d, h, w, conn
                    )

    def test_wider_rows_vs_reference(self):
        # 更宽的行，保证单行多游程与各种触达情形
        rng = random.Random(424242)
        for _ in range(8):
            vol = random_volume(rng, 3, 3, 40, rng.choice((0.3, 0.55)))
            for conn in (6, 18, 26):
                assert_matches_reference(self, vol, 3, 3, 40, conn)


class StreamingTests(unittest.TestCase):
    def _assert_stream_matches_reference(
        self, vol, depth, height, width, conn
    ):
        stream = StreamLabeler(depth, height, width, conn)
        streamed_labels = []
        closed_by_tid = {}
        active_events = []
        for z in range(depth):
            result = stream.feed_slice(z, vol[z])
            streamed_labels.append(result.run_labels)
            active_events.append(
                (stream.num_active_components, stream.num_boundary_runs)
            )
            for closed in result.closed:
                self.assertNotIn(closed.temporary_id, closed_by_tid)
                closed_by_tid[closed.temporary_id] = closed
        finished = stream.finish()
        for closed in finished.closed:
            closed_by_tid.setdefault(closed.temporary_id, closed)

        ref_labels, ref_meta = reference_label(
            vol, depth, height, width, conn
        )
        final_labels = [
            [[finished.id_map[tid] for tid in row] for row in sl]
            for sl in streamed_labels
        ]
        self.assertEqual(final_labels, ref_labels)
        self.assertEqual(
            len(closed_by_tid), len(finished.components)
        )
        for comp in finished.components:
            tid = next(
                tid
                for tid in closed_by_tid
                if finished.id_map[tid] == comp.id
            )
            closed = closed_by_tid[tid]
            self.assertEqual(closed.voxel_count, comp.voxel_count)
            self.assertEqual(closed.bbox, comp.bbox)
            self.assertEqual(closed.exposed_area, comp.exposed_area)
            self.assertEqual(
                (closed.min_z, closed.min_y, closed.min_x),
                (comp.min_z, comp.min_y, comp.min_x),
            )
        self.assertEqual(
            active_events[-1][1],
            sum(len(row) for row in vol[-1]),
        )
        self.assertEqual(stream.num_active_components, 0)
        self.assertEqual(stream.num_boundary_runs, 0)
        return finished

    def test_random_streaming_matches_bfs(self):
        rng = random.Random(77103)
        for _ in range(12):
            d, h, w = (
                rng.randint(1, 6),
                rng.randint(1, 6),
                rng.randint(1, 18),
            )
            vol = random_volume(rng, d, h, w, rng.choice((0.2, 0.45, 0.7)))
            for conn in (6, 18, 26):
                self._assert_stream_matches_reference(
                    vol, d, h, w, conn
                )

    def test_explicit_empty_slices_are_required(self):
        stream = StreamLabeler(3, 1, 2, 6)
        first = stream.feed_slice(0, [[(0, 1)]])
        self.assertEqual(stream.num_active_components, 1)
        empty = stream.feed_slice(1, [[]])
        self.assertEqual(empty.run_labels, [[]])
        self.assertEqual(len(empty.closed), 1)
        self.assertEqual(stream.num_active_components, 0)
        self.assertEqual(stream.num_boundary_runs, 0)
        with self.assertRaises(InvalidVolumeError) as cm:
            stream.feed_slice(3, [[(1, 2)]])
        self.assertIn("z=2", str(cm.exception))
        third = stream.feed_slice(2, [[(1, 2)]])
        finished = stream.finish()
        self.assertEqual(len(finished.components), 2)
        self.assertEqual(finished.components[0].id, 1)
        self.assertEqual(finished.components[1].id, 2)
        self.assertEqual(first.closed, [])

    def test_invalid_feed_leaves_stream_state_unchanged(self):
        stream = StreamLabeler(2, 1, 2, 6)
        stream.feed_slice(0, [[(0, 1)]])
        with self.assertRaises(InvalidVolumeError):
            stream.feed_slice(2, [[(0, 1)]])
        with self.assertRaises(InvalidVolumeError):
            stream.feed_slice(1, [[(2, 3)]])
        result = stream.feed_slice(1, [[(0, 1)]])
        self.assertEqual(result.closed, [])
        self.assertEqual(stream.num_active_components, 1)
        finished = stream.finish()
        self.assertEqual(len(finished.components), 1)
        self.assertEqual(finished.components[0].voxel_count, 2)

    def test_long_spans_short_then_blank_slice_and_final_ids(self):
        # 长条从 z0 延伸到 z2，短条夹在 z1 但不接触；z3 整层空白。
        vol = [
            [[(0, 4)], [], []],
            [[(0, 4)], [], [(0, 1)]],
            [[(0, 4)], [], []],
            [[], [], []],
        ]
        stream = StreamLabeler(4, 3, 4, 6)
        r0 = stream.feed_slice(0, vol[0])
        r1 = stream.feed_slice(1, vol[1])
        r2 = stream.feed_slice(2, vol[2])
        r3 = stream.feed_slice(3, vol[3])
        self.assertEqual(r0.closed, [])
        self.assertEqual(r1.closed, [])
        self.assertEqual(len(r2.closed), 1)
        self.assertEqual(r2.closed[0].voxel_count, 1)
        self.assertEqual(len(r3.closed), 1)
        self.assertEqual(r3.closed[0].voxel_count, 12)
        finished = stream.finish()
        self.assertEqual(
            [(c.min_z, c.min_y, c.min_x) for c in finished.components],
            [(0, 0, 0), (1, 2, 0)],
        )
        self.assertEqual(
            [c.voxel_count for c in finished.components], [12, 1]
        )
        self.assertEqual(finished.id_map[r3.closed[0].temporary_id], 1)
        self.assertEqual(finished.id_map[r2.closed[0].temporary_id], 2)

    def test_diagonal_cross_slice_depends_on_connectivity(self):
        vol = [
            [[(0, 1)], [], []],
            [[], [(0, 1)], []],
            [[], [], [(1, 2)]],
        ]
        for conn, count, closed_after_first, closed_after_second in (
            (6, 3, 1, 1),
            (18, 2, 0, 1),
            (26, 1, 0, 0),
        ):
            stream = StreamLabeler(3, 3, 5, conn)
            stream.feed_slice(0, vol[0])
            second = stream.feed_slice(1, vol[1])
            third = stream.feed_slice(2, vol[2])
            finished = stream.finish()
            self.assertEqual(len(second.closed), closed_after_first)
            self.assertEqual(len(third.closed), closed_after_second)
            self.assertEqual(len(finished.components), count)

    def test_long_run_state_is_boundary_runs_not_expanded_voxels(self):
        stream = StreamLabeler(3, 1, 512, 6)
        stream.feed_slice(0, [[(0, 512)]])
        self.assertEqual(stream.num_boundary_runs, 1)
        self.assertEqual(stream.num_active_components, 1)
        stream.feed_slice(1, [[]])
        self.assertEqual(stream.num_boundary_runs, 0)
        self.assertEqual(stream.num_active_components, 0)
        stream.feed_slice(2, [[]])
        self.assertEqual(stream.num_active_components, 0)
        with self.assertRaises(InvalidVolumeError):
            stream.feed_slice(2, [[]])
        finished = stream.finish()
        self.assertEqual(finished.components[0].voxel_count, 512)
        self.assertEqual(finished.components[0].exposed_area, 2050)
        with self.assertRaises(InvalidVolumeError):
            stream.finish()


class ErrorSemanticsTests(unittest.TestCase):
    def _assert_bad(self, *args, **kwargs):
        with self.assertRaises(InvalidVolumeError):
            label(*args, **kwargs)

    def test_bool_rejected_as_integer(self):
        self._assert_bad([], True, 1, 1)
        self._assert_bad([], 1, False, 1)
        self._assert_bad([], 1, 1, 1, connectivity=True)
        self._assert_bad([[[Run(0, 1)]]], 1, 1, True)

    def test_dim_out_of_range(self):
        self._assert_bad([], 0, 1, 1)
        self._assert_bad([], 1, 513, 1)
        self._assert_bad([], 1, 1, -1)
        self._assert_bad("x", 1, 1, 1)

    def test_bad_connectivity(self):
        self._assert_bad([[[]]], 1, 1, 1, 8)
        self._assert_bad([[[]]], 1, 1, 1, "6")
        self._assert_bad([[[]]], 1, 1, 1, 6.0)

    def test_nan_infinity_rejected(self):
        self._assert_bad([[[(0, float("nan"))]]], 1, 1, 1)
        self._assert_bad([[[(float("inf"), 1)]]], 1, 1, 2)
        self._assert_bad(
            [[[{"start": 0, "end": float("-inf")}]]], 1, 1, 1
        )

    def test_non_integer_coords_and_malformed_runs(self):
        self._assert_bad([[[(1.5, 2)]]], 1, 1, 2)        # 非整数 float
        self._assert_bad([[[("0", 1)]]], 1, 1, 1)        # 字符串
        self._assert_bad([[[(1, 2, 3)]]], 1, 1, 3)       # 三元组
        self._assert_bad([[[None]]], 1, 1, 1)
        self._assert_bad(
            [[[{"start": 0, "end": 1, "x": 2}]]], 1, 1, 1
        )

    def test_coordinate_range_violations(self):
        self._assert_bad([[[(-1, 1)]]], 1, 1, 1)         # 越界
        self._assert_bad([[[(0, 2)]]], 1, 1, 1)          # end > width
        self._assert_bad([[[(1, 1)]]], 1, 1, 2)          # 空游程
        self._assert_bad([[[(1, 0)]]], 1, 1, 2)          # start >= end

    def test_overlap_and_unsorted(self):
        with self.assertRaises(InvalidVolumeError) as cm:
            label([[[(0, 3), (2, 4)]]], 1, 1, 4)
        self.assertIn("run=1", str(cm.exception))
        self.assertIn("重叠", str(cm.exception))
        with self.assertRaises(InvalidVolumeError):
            label([[[(2, 3), (0, 1)]]], 1, 1, 3)

    def test_structure_mismatch(self):
        with self.assertRaises(InvalidVolumeError) as cm:
            label([[[]]], 2, 1, 1)  # 切片数不符
        self.assertIn("depth", str(cm.exception))
        with self.assertRaises(InvalidVolumeError):
            label([[[]]], 1, 2, 1)  # 行数不符
        with self.assertRaises(InvalidVolumeError):
            label([[123]], 1, 1, 1)  # 行不是序列
        with self.assertRaises(InvalidVolumeError):
            label(123, 1, 1, 1)

    def test_error_is_value_error_and_locatable(self):
        try:
            label([[[(0, float("nan"))]]], 1, 1, 1)
        except ValueError as exc:
            self.assertIn("z=0", str(exc))
            self.assertIn("y=0", str(exc))
            self.assertIn("run=0", str(exc))
        else:  # pragma: no cover
            self.fail("应当抛出 InvalidVolumeError")

    def test_too_many_runs_boundary(self):
        # 恰好 20000 条：接受（40*500 = 20000，尺寸均 <= 512）
        d, h, w = 40, 500, 1
        vol = [[[(0, 1)] for _ in range(h)] for _ in range(d)]
        r = label(vol, d, h, w, 6)
        self.assertEqual(r.num_runs, 20000)
        self.assertEqual(sum(c.voxel_count for c in r.components), 20000)

        # 20001 条：拒绝，消息含上限与定位（41*489 = 20049 行）
        d2, h2 = 41, 489
        big = [[[] for _ in range(h2)] for _ in range(d2)]
        count = 0
        for z in range(d2):
            for y in range(h2):
                if count < 20001:
                    big[z][y] = [(0, 1)]
                    count += 1
        with self.assertRaises(InvalidVolumeError) as cm:
            label(big, d2, h2, 1, 6)
        self.assertIn("20000", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
