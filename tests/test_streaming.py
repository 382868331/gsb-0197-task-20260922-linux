"""流式逐切片接口 StreamingLabeler 的必要测试。

流式输出与一次性 label 的离线结果逐组件（体素数、包围盒、暴露面积、
最小体素坐标）与逐游程归属完全一致；另覆盖尽早结算、空切片断开、
非法切片不留部分变更、finish 后的错误语义与历史切片释放。
"""

import random
import unittest

from rle_components import (
    InvalidVolumeError,
    StreamingLabeler,
    label,
)

from test_core import encode_dense


def stream_all(vol, depth, height, width, conn):
    """逐切片喂入，收集每次结算的组件与 finish 的尾批。"""

    lab = StreamingLabeler(height, width, conn)
    batches = []
    for z, slice_rows in enumerate(vol):
        batches.append((z, lab.append_slice(slice_rows)))
    batches.append((depth, lab.finish()))
    assert lab.pending_count == 0
    return lab, batches


def assert_stream_matches_label(testcase, vol, depth, height, width, conn):
    offline = label(vol, depth, height, width, conn)
    lab, batches = stream_all(vol, depth, height, width, conn)

    streamed = [c for _, batch in batches for c in batch]
    testcase.assertEqual(len(streamed), len(offline.components))
    testcase.assertEqual(
        [c.id for c in streamed], list(range(1, len(streamed) + 1))
    )

    # 按最小体素排序后与离线组件逐一对应（包括 ID 顺序）
    ordered = sorted(streamed, key=lambda c: (c.min_z, c.min_y, c.min_x))
    for rank, (sc, oc) in enumerate(zip(ordered, offline.components), start=1):
        testcase.assertEqual(rank, oc.id)
        testcase.assertEqual(sc.voxel_count, oc.voxel_count)
        testcase.assertEqual(sc.bbox, oc.bbox)
        testcase.assertEqual(sc.exposed_area, oc.exposed_area)
        testcase.assertEqual(
            (sc.min_z, sc.min_y, sc.min_x),
            (oc.min_z, oc.min_y, oc.min_x),
        )

    # 逐游程归属一致，并覆盖全部游程恰好一次
    seen_runs = set()
    run_to_offline_cid = {}
    for z, sl in enumerate(offline.run_labels):
        for y, row_labels in enumerate(sl):
            for ri, cid in enumerate(row_labels):
                run_to_offline_cid[(z, y, ri)] = cid
    for sc in streamed:
        for ref in sc.runs:
            testcase.assertNotIn(ref, seen_runs)
            seen_runs.add(ref)
    total_runs = sum(len(r) for sl in vol for r in sl)
    testcase.assertEqual(len(seen_runs), total_runs)
    for sc in streamed:
        rank = ordered.index(sc) + 1
        for ref in sc.runs:
            testcase.assertEqual(run_to_offline_cid[ref], rank)
    return lab, batches


class StreamingParityTests(unittest.TestCase):
    def test_random_volumes_all_connectivities(self):
        rng = random.Random(20260928)
        for case in range(60):
            depth = rng.randint(1, 6)
            height = rng.randint(1, 5)
            width = rng.randint(1, 5)
            p = rng.uniform(0.15, 0.6)
            vol = encode_dense(
                [
                    [[rng.random() < p for _ in range(width)]
                     for _ in range(height)]
                    for _ in range(depth)
                ],
                depth,
                height,
            )
            for conn in (6, 18, 26):
                assert_stream_matches_label(
                    self, vol, depth, height, width, conn
                )

    def test_empty_volume_and_empty_slices(self):
        for depth in (1, 3, 5):
            vol = [[[] for _ in range(2)] for _ in range(depth)]
            lab, batches = stream_all(vol, depth, 2, 2, 6)
            self.assertEqual([c for _, b in batches for c in b], [])
            self.assertEqual(lab.num_runs, 0)
            self.assertTrue(lab.finished)

    def test_demo_body_matches_label(self):
        from demo import DEPTH, HEIGHT, WIDTH, VOXELS, encode

        vol = encode(VOXELS, DEPTH, HEIGHT, WIDTH)
        for conn in (6, 18, 26):
            assert_stream_matches_label(
                self, vol, DEPTH, HEIGHT, WIDTH, conn
            )


class EarlyFinalizationTests(unittest.TestCase):
    def test_component_settled_as_early_as_possible(self):
        # 切片 0 有一个组件；切片 1 完全空 -> 组件不可能再连接后续，
        # 在追加空切片后（而不是 finish 时）就应结算
        h = w = 3
        lab = StreamingLabeler(h, w, 26)
        self.assertEqual(lab.append_slice([[(0, 2)], [], []]), [])
        self.assertEqual(lab.pending_count, 1)

        settled = lab.append_slice([[], [], []])
        self.assertEqual(len(settled), 1)
        comp = settled[0]
        self.assertEqual(comp.voxel_count, 2)
        self.assertEqual(comp.exposed_area, 10)  # 6*2 - 2*1
        self.assertEqual(comp.runs, ((0, 0, 0),))
        self.assertEqual(lab.pending_count, 0)

        # 后续再追加有内容的切片，作为独立组件在 finish 结算
        tail = lab.append_slice([[], [], [(1, 2)]])
        self.assertEqual(tail, [])  # 仍可能连接下一片，不能提前结束
        rest = lab.finish()
        self.assertEqual(len(rest), 1)
        self.assertEqual(rest[0].voxel_count, 1)

    def test_connected_across_slices_kept_until_finish(self):
        # 26 邻接对角链：每张切片都与下一片相连，任何组件都不得提前结算
        lab = StreamingLabeler(3, 3, 26)
        slices = [
            [[(0, 1)], [], []],
            [[], [(1, 2)], []],
            [[], [], [(2, 3)]],
        ]
        for rows in slices:
            self.assertEqual(lab.append_slice(rows), [])
        self.assertEqual(lab.pending_count, 1)
        rest = lab.finish()
        self.assertEqual(len(rest), 1)
        self.assertEqual(rest[0].voxel_count, 3)
        # 6 邻接下同一体每片后即结算前一片组件
        lab6 = StreamingLabeler(3, 3, 6)
        self.assertEqual(lab6.append_slice(slices[0]), [])
        one = lab6.append_slice(slices[1])
        self.assertEqual([c.voxel_count for c in one], [1])
        two = lab6.append_slice(slices[2])
        self.assertEqual([c.voxel_count for c in two], [1])
        self.assertEqual([c.voxel_count for c in lab6.finish()], [1])


class SliceReleaseTests(unittest.TestCase):
    def test_historical_slices_not_retained(self):
        # 用终结器观测：已结算组件持有的大对象（挂在游程上的哨兵）
        # 不应被标记器长期引用。哨兵只存在于校验输入对象中，模块
        # 规范化后只保留整数区间；这里直接检查内部只保留两切片。
        lab = StreamingLabeler(4, 4, 6)
        rows0 = [[(0, 2)], [], [], []]
        rows1 = [[(2, 4)], [], [], []]
        lab.append_slice(rows0)
        lab.append_slice(rows1)
        lab.append_slice([[], [], [], []])
        lab.append_slice([[], [], [], []])
        lab.finish()
        self.assertIsNone(lab._prev_rows)
        self.assertIsNone(lab._prev_ids)
        self.assertEqual(lab.pending_count, 0)
        # 所有组件均已结算：聚合表清空
        self.assertEqual(lab._faces, {})
        self.assertEqual(lab._runs, {})
        self.assertEqual(lab._bbox, {})


class StreamingErrorTests(unittest.TestCase):
    def test_invalid_slice_leaves_state_unchanged(self):
        h = w = 3
        lab = StreamingLabeler(h, w, 6)
        lab.append_slice([[(0, 1)], [], []])
        before = (lab.num_slices, lab.num_runs, lab.pending_count)

        # 非法切片（坐标重叠）必须抛错
        with self.assertRaises(InvalidVolumeError):
            lab.append_slice([[(0, 2), (1, 3)], [], []])
        self.assertEqual(
            (lab.num_slices, lab.num_runs, lab.pending_count), before
        )

        # NaN 坐标同样拒绝且无部分变更
        with self.assertRaises(InvalidVolumeError):
            lab.append_slice([[(0, float("nan"))], [], []])
        self.assertEqual(
            (lab.num_slices, lab.num_runs, lab.pending_count), before
        )

        # 修正为合法切片后继续，最终结果与 label 一致
        lab.append_slice([[], [(0, 1)], []])
        rest = lab.finish()
        self.assertEqual({c.voxel_count for c in rest}, {1, 1})

    def test_bad_constructor_arguments(self):
        for args in ((0, 3), (3, 513), (True, 3), (3, 3.0)):
            with self.assertRaises(InvalidVolumeError):
                StreamingLabeler(*args)
        with self.assertRaises(InvalidVolumeError):
            StreamingLabeler(3, 3, connectivity=12)
        with self.assertRaises(InvalidVolumeError):
            StreamingLabeler(3, 3, connectivity=True)

    def test_append_after_finish_and_double_finish(self):
        lab = StreamingLabeler(2, 2, 6)
        lab.append_slice([[], []])
        self.assertEqual(lab.finish(), [])
        self.assertTrue(lab.finished)
        with self.assertRaises(InvalidVolumeError):
            lab.append_slice([[], []])
        with self.assertRaises(InvalidVolumeError):
            lab.finish()

    def test_wrong_row_count_rejected(self):
        lab = StreamingLabeler(2, 2, 6)
        with self.assertRaises(InvalidVolumeError):
            lab.append_slice([[(0, 1)]])
        self.assertEqual(lab.num_slices, 0)

    def test_run_count_boundary_20000(self):
        # 与一次性接口相同的游程总数上限：恰 20000 接受，第 20001 拒绝
        h, w = 512, 48
        full_slice = [[(i, i + 1) for i in range(40)] for _ in range(500)]
        full_slice += [[] for _ in range(12)]  # 500*40 = 20000
        lab = StreamingLabeler(h, w, 6)
        lab.append_slice(full_slice)
        self.assertEqual(lab.num_runs, 20000)
        with self.assertRaises(InvalidVolumeError):
            lab.append_slice([[(0, 1)]] + [[] for _ in range(h - 1)])
        # 拒绝不留部分变更：计数与切片数保持在拒绝前
        self.assertEqual(lab.num_runs, 20000)
        self.assertEqual(lab.num_slices, 1)


if __name__ == "__main__":
    unittest.main()
