"""StreamingLabeler 的必要测试。

核心验收：逐切片追加 + finish 拼接出的组件序列，必须与离线 label()
（以及独立 BFS 参考）逐字段一致；同时覆盖提前结算时机、错误语义与
状态误用。
"""

import random
import unittest

from rle_components import (
    InvalidVolumeError,
    StreamingLabeler,
    label,
)

from reference import reference_label
from test_core import encode_dense, random_volume


def stream_components(vol, height, width, conn):
    """逐切片追加，拼接全部已结算组件。"""
    streamer = StreamingLabeler(height, width, connectivity=conn)
    out = []
    for rows in vol:
        out.extend(streamer.append_slice(rows))
    out.extend(streamer.finish())
    return streamer, out


def comp_tuple(c):
    return (c.id, c.voxel_count, tuple(c.bbox), c.exposed_area,
            (c.min_z, c.min_y, c.min_x))


class StreamingEquivalenceTests(unittest.TestCase):
    def assert_stream_matches_offline(self, vol, d, h, w, conn):
        offline = label(vol, d, h, w, conn)
        streamer, comps = stream_components(vol, h, w, conn)
        self.assertEqual([comp_tuple(c) for c in comps],
                         [comp_tuple(c) for c in offline.components])
        self.assertEqual(streamer.num_slices, d)
        self.assertEqual(streamer.num_runs, offline.num_runs)
        self.assertEqual(streamer.finalized_count, len(offline.components))
        return comps

    def test_hand_volumes_all_connectivities(self):
        cases = [
            ([[[(0, 4)]]], 1, 1, 4),
            ([[[(0, 2)], [(0, 2)]]], 1, 2, 2),
            ([[[(0, 1)], []], [[], [(0, 1)]], [[], [(0, 1)]]], 3, 2, 1),
            ([[[], [], [(1, 2)]], [[(0, 1)], [], []]], 2, 3, 3),
        ]
        for vol, d, h, w in cases:
            for conn in (6, 18, 26):
                self.assert_stream_matches_offline(vol, d, h, w, conn)

    def test_corner_chain_all_connectivities(self):
        d = h = w = 3
        vol = [[[] for _ in range(h)] for _ in range(d)]
        for z, y, x in [(0, 0, 0), (1, 1, 1), (2, 2, 2)]:
            vol[z][y] = [(x, x + 1)]
        for conn in (6, 18, 26):
            self.assert_stream_matches_offline(vol, d, h, w, conn)

    def test_random_volumes_match_offline_and_reference(self):
        rng = random.Random(77001)
        for _ in range(12):
            d = rng.randint(1, 6)
            h = rng.randint(1, 5)
            w = rng.randint(1, 9)
            vol = random_volume(rng, d, h, w, rng.choice((0.2, 0.45, 0.7)))
            for conn in (6, 18, 26):
                comps = self.assert_stream_matches_offline(vol, d, h, w, conn)
                _, ref_meta = reference_label(vol, d, h, w, conn)
                self.assertEqual(
                    [(c.voxel_count, tuple(c.bbox), c.exposed_area,
                      (c.min_z, c.min_y, c.min_x)) for c in comps],
                    [(m["voxels"], m["bbox"], m["area"], m["min"])
                     for m in ref_meta],
                )

    def test_empty_volume(self):
        vol = [[[] for _ in range(2)] for _ in range(2)]
        for conn in (6, 18, 26):
            _, comps = stream_components(vol, 2, 2, conn)
            self.assertEqual(comps, [])


class StreamingEarlyFinalizationTests(unittest.TestCase):
    def test_closed_component_returned_early(self):
        # z=0 的孤立体素在追加 z=1 后即可确定封闭（连接只跨相邻切片）
        streamer = StreamingLabeler(1, 2, connectivity=6)
        self.assertEqual(streamer.append_slice([[(0, 1)]]), [])
        done = streamer.append_slice([[]])
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0].id, 1)
        self.assertEqual(done[0].voxel_count, 1)
        self.assertEqual(done[0].exposed_area, 6)
        self.assertEqual(streamer.append_slice([[(1, 2)]]), [])
        rest = streamer.finish()
        self.assertEqual(len(rest), 1)
        self.assertEqual(rest[0].id, 2)
        self.assertEqual(rest[0].min_z, 2)

    def test_active_component_not_finalized_early(self):
        # 组件触及最新切片时必须保留，不能在 append 中提前结束
        streamer = StreamingLabeler(1, 1, connectivity=6)
        for _ in range(5):
            self.assertEqual(streamer.append_slice([[(0, 1)]]), [])
        rest = streamer.finish()
        self.assertEqual(len(rest), 1)
        self.assertEqual(rest[0].voxel_count, 5)
        self.assertEqual(rest[0].exposed_area, 6 * 5 - 2 * 4)

    def test_id_order_waits_for_smaller_key(self):
        # z=0 有两个组件：key 较小者 (y=0) 在 z=1 仍活跃，key 较大者 (y=2)
        # 已封闭但必须等活跃者结算后才发出，保证 ID 与离线一致。
        vol = [
            [[(0, 1)], [], [(0, 1)]],   # z=0: y=0 与 y=2 各一个体素
            [[(0, 1)], [], []],         # z=1: 只延续 y=0 的组件
        ]
        streamer = StreamingLabeler(3, 1, connectivity=6)
        self.assertEqual(streamer.append_slice(vol[0]), [])
        # y=2 组件已封闭但排序键更大，此时不得提前发出
        self.assertEqual(streamer.append_slice(vol[1]), [])
        rest = streamer.finish()
        ids = [c.id for c in rest]
        self.assertEqual(ids, [1, 2])
        self.assertEqual(rest[0].voxel_count, 2)   # y=0 跨两片，ID 1
        self.assertEqual(rest[1].voxel_count, 1)   # y=2 孤立，ID 2
        offline = label(vol, 2, 3, 1, 6)
        self.assertEqual([comp_tuple(c) for c in rest],
                         [comp_tuple(c) for c in offline.components])


class StreamingErrorTests(unittest.TestCase):
    def test_bad_constructor_args(self):
        for args in [(0, 1), (1, 513), (True, 1)]:
            with self.assertRaises(InvalidVolumeError):
                StreamingLabeler(*args)
        with self.assertRaises(InvalidVolumeError):
            StreamingLabeler(1, 1, connectivity=8)
        with self.assertRaises(InvalidVolumeError):
            StreamingLabeler(1, 1, connectivity=True)

    def test_bad_slice_data(self):
        streamer = StreamingLabeler(2, 2, connectivity=6)
        with self.assertRaises(InvalidVolumeError):
            streamer.append_slice([[(0, 1)]])            # 行数不符
        with self.assertRaises(InvalidVolumeError):
            streamer.append_slice([[(0, float("nan"))], []])
        with self.assertRaises(InvalidVolumeError):
            streamer.append_slice([[(0, 3)], []])        # end > width
        with self.assertRaises(InvalidVolumeError):
            streamer.append_slice([[(0, 2), (1, 2)], []])  # 重叠
        # 失败后对象仍可用（校验在合并前完成，无部分变更）
        done = streamer.append_slice([[(0, 1)], []])
        self.assertEqual(done, [])
        rest = streamer.finish()
        self.assertEqual(len(rest), 1)

    def test_state_misuse(self):
        streamer = StreamingLabeler(1, 1)
        streamer.append_slice([[(0, 1)]])
        streamer.finish()
        with self.assertRaises(RuntimeError):
            streamer.append_slice([[(0, 1)]])
        with self.assertRaises(RuntimeError):
            streamer.finish()

    def test_depth_limit(self):
        streamer = StreamingLabeler(1, 1)
        for _ in range(512):
            streamer.append_slice([[]])
        with self.assertRaises(InvalidVolumeError):
            streamer.append_slice([[]])

    def test_run_count_limit(self):
        streamer = StreamingLabeler(500, 1)
        for _ in range(40):                      # 40*500 = 20000，接受
            streamer.append_slice([[(0, 1)] for _ in range(500)])
        with self.assertRaises(InvalidVolumeError):
            streamer.append_slice([[(0, 1)]])    # 第 20001 条，拒绝


if __name__ == "__main__":
    unittest.main(verbosity=2)
