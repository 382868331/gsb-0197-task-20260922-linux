"""流式接口（StreamLabeler.feed_slice / finish）的必要测试。

期望值来源：
  * tests/reference.py 的独立体素展开 + BFS 参考（不调用被测核心）；
  * 批式 label() 与流式 finish 的最终组件必须一致（同一模块两种入口
    对同一输入的交叉验证，参考 BFS 仍是最终仲裁）。
另覆盖题面点名场景：长组件夹短组件、整层空白、斜角跨片、超长游程，
以及活跃状态只依赖边界与未封闭组件的内存性质。
"""

import random
import unittest

from rle_components import (
    BoundingBox,
    InvalidVolumeError,
    StreamLabeler,
    label,
)

from reference import expand, reference_label
from test_core import encode_dense, random_volume


def stream(vol, height, width, conn, collector=None):
    """逐切片喂入，返回 (SliceResult 列表, StreamResult)。"""
    st = StreamLabeler(height, width, conn)
    outs = []
    for slice_rows in vol:
        out = st.feed_slice(slice_rows)
        outs.append(out)
        if collector is not None:
            collector(st, out)
    return outs, st.finish()


def assert_stream_matches_batch(testcase, vol, depth, height, width, conn):
    """流式最终结果必须与批式 label 及独立 BFS 参考一致。"""
    outs, final = stream(vol, height, width, conn)
    batch = label(vol, depth, height, width, conn)
    ref_labels, ref_meta = reference_label(vol, depth, height, width, conn)

    # 最终组件与批式逐字段一致（最终 ID 口径相同）
    testcase.assertEqual(len(final.components), len(batch.components))
    for c_stream, c_batch in zip(final.components, batch.components):
        testcase.assertEqual(c_stream, c_batch)

    # 与独立 BFS 参考逐字段一致
    testcase.assertEqual(len(final.components), len(ref_meta))
    for comp, m in zip(final.components, ref_meta):
        testcase.assertEqual(comp.voxel_count, m["voxels"])
        testcase.assertEqual(tuple(comp.bbox), m["bbox"])
        testcase.assertEqual(comp.exposed_area, m["area"])
        testcase.assertEqual((comp.min_z, comp.min_y, comp.min_x), m["min"])

    # 每游程：临时 ID 经 finish 映射后等于批式/参考最终 ID
    # （id_of_temp 覆盖全部发布过的临时 ID，含被合并吸收的别名）
    all_temps = {
        t for out in outs for row in out.run_labels for t in row
    }
    testcase.assertTrue(all_temps <= set(final.id_of_temp))
    for z in range(depth):
        for y in range(height):
            temps = outs[z].run_labels[y]
            mapped = [final.id_of_temp[t] for t in temps]
            testcase.assertEqual(mapped, batch.run_labels[z][y])
            testcase.assertEqual(mapped, ref_labels[z][y])

    # 中途封闭记录与最终组件一致（面积在封闭时已最终确定）；
    # 其余组件由 finish 结算，不出现在任何 SliceResult.closed 中
    closed_recs = [rec for out in outs for rec in out.closed]
    testcase.assertLessEqual(len(closed_recs), len(final.components))
    by_temp = {c.id: c for c in final.components}
    for rec in closed_recs:
        comp = by_temp[final.id_of_temp[rec.temp_id]]
        testcase.assertEqual(rec.voxel_count, comp.voxel_count)
        testcase.assertEqual(rec.bbox, comp.bbox)
        testcase.assertEqual(rec.exposed_area, comp.exposed_area)
        testcase.assertEqual(
            (rec.min_z, rec.min_y, rec.min_x),
            (comp.min_z, comp.min_y, comp.min_x),
        )
    return outs, final


class StreamScenarioTests(unittest.TestCase):
    def test_long_component_sandwiches_short_ones(self):
        """长组件贯穿全部切片，短组件在中间切片封闭并提前交付。"""
        depth, height, width = 6, 3, 3
        vol = [[[] for _ in range(height)] for _ in range(depth)]
        for z in range(depth):
            vol[z][0] = [(0, 1)]          # 长组件：z=0..5 的竖柱
        vol[2][2] = [(2, 3)]              # 短组件 A：只在 z=2
        vol[4][1] = [(1, 2)]              # 短组件 B：只在 z=4

        st = StreamLabeler(height, width, 6)
        closed_at = {}
        for z in range(depth):
            out = st.feed_slice(vol[z])
            for rec in out.closed:
                closed_at[rec.temp_id] = (z, rec)
        final = st.finish()

        self.assertEqual(len(final.components), 3)
        # 短组件 A 在喂入 z=3 后封闭，B 在 z=5 后封闭；长组件由 finish 结算
        closed_zs = sorted(z for z, _ in closed_at.values())
        self.assertEqual(closed_zs, [3, 5])
        rec_a = next(r for z, r in closed_at.values() if z == 3)
        self.assertEqual(rec_a.voxel_count, 1)
        self.assertEqual(rec_a.bbox, BoundingBox(2, 2, 2, 3, 3, 3))
        self.assertEqual(rec_a.exposed_area, 6)
        # 提前结束的临时 ID 不得冒充按最小坐标排序的最终 ID：
        # 长组件最小坐标 (0,0,0) 必须是最终 1 号
        self.assertEqual(final.components[0].min_z, 0)
        self.assertEqual(final.components[0].voxel_count, depth)
        temp_of_long = 1  # 第一个被创建的组件
        self.assertEqual(final.id_of_temp[temp_of_long], 1)
        # 与批式一致
        batch = label(vol, depth, height, width, 6)
        self.assertEqual(final.components, batch.components)

    def test_empty_slice_must_be_explicit_and_severs(self):
        """整层空白显式传入：6 邻接下上下两段被空切片断开。"""
        height = width = 2
        vol = [
            [[(0, 1)], []],
            [[], []],                      # 整层空白，必须显式传入
            [[(0, 1)], []],
        ]
        outs, final = stream(vol, height, width, 6)
        self.assertEqual(len(final.components), 2)
        # 下段在空切片喂入后即封闭（不可能再与未来切片相连）
        self.assertEqual(len(outs[1].closed), 1)
        self.assertEqual(outs[1].closed[0].bbox, BoundingBox(0, 0, 0, 1, 1, 1))
        # 26 邻接也跨不过整层空白（dz=2）
        _, final26 = stream(vol, height, width, 26)
        self.assertEqual(len(final26.components), 2)
        # 少传切片（隐式跳过空层）违背契约：库无法察觉跳过，
        # 会把两段错并——因此契约要求空切片显式传入
        st = StreamLabeler(height, width, 6)
        st.feed_slice(vol[0])
        st.feed_slice(vol[2])
        final_bad = st.finish()
        self.assertEqual(len(final_bad.components), 1)

    def test_diagonal_corner_chain_across_slices(self):
        """斜角跨片：(1,1,1) 对角链，仅 26 邻接连通。"""
        d = h = w = 4
        vol = [[[] for _ in range(h)] for _ in range(d)]
        for i in range(4):
            vol[i][i] = [(i, i + 1)]
        for conn, expect in ((6, 4), (18, 4), (26, 1)):
            outs, final = stream(vol, h, w, conn)
            self.assertEqual(len(final.components), expect)
            if conn == 26:
                # 链上组件始终接触最新切片：中途不封闭，finish 才结算
                self.assertEqual(
                    [rec for out in outs for rec in out.closed], []
                )
                c = final.components[0]
                self.assertEqual(c.voxel_count, 4)
                self.assertEqual(c.exposed_area, 24)  # 角接触不共享面
                self.assertEqual(c.bbox, BoundingBox(0, 0, 0, 4, 4, 4))

    def test_single_very_long_run(self):
        """一条游程很长：单行 500 体素的游程跨切片形成厚板。"""
        depth, height, width = 3, 1, 500
        vol = [[[(0, 500)]] for _ in range(depth)]
        outs, final = stream(vol, height, width, 6)
        self.assertEqual(len(final.components), 1)
        c = final.components[0]
        self.assertEqual(c.voxel_count, 1500)
        # 3*1*500 长方体外表面积 = 2*(3*1 + 1*500 + 3*500) = 4006
        self.assertEqual(c.exposed_area, 4006)
        self.assertEqual(c.bbox, BoundingBox(0, 0, 0, 3, 1, 500))
        # 每切片只有一条活跃游程，活跃状态与游程长度无关
        for out in outs:
            self.assertEqual(out.closed, [])

    def test_active_state_bounded_by_boundary(self):
        """活跃状态只依赖边界切片与未封闭组件，不保留全部历史。"""
        depth, height, width = 12, 4, 4
        rng = random.Random(777)
        vol = random_volume(rng, depth, height, width, 0.4)
        max_slice_runs = max(
            sum(len(row) for row in sl) for sl in vol
        )
        seen = []

        def watch(st, out):
            seen.append((st.active_node_count(), st.active_component_count()))

        outs, final = stream(vol, height, width, 6, collector=watch)
        # 全部历史游程数远大于边界：活跃节点数始终不超过单切片游程数
        total_runs = sum(len(r) for sl in vol for r in sl)
        self.assertGreater(total_runs, 2 * max_slice_runs)
        for nodes, comps in seen:
            self.assertLessEqual(nodes, max_slice_runs)
            self.assertLessEqual(comps, nodes)
        # finish 后活跃状态清空
        st = StreamLabeler(height, width, 6)
        for sl in vol:
            st.feed_slice(sl)
        st.finish()
        self.assertEqual(st.active_node_count(), 0)
        self.assertEqual(st.active_component_count(), 0)

    def test_closed_state_released_for_short_components(self):
        """短组件封闭后其合并状态被释放，不随深度累积。"""
        depth, height, width = 30, 2, 2
        vol = [[[] for _ in range(height)] for _ in range(depth)]
        for z in range(depth):
            vol[z][0] = [(0, 1)]           # 每片一个孤立体素（6 邻接下相连）
        # 6 邻接下整柱是一个组件：活跃组件恒为 1
        st = StreamLabeler(height, width, 6)
        for z in range(depth):
            st.feed_slice(vol[z])
            self.assertEqual(st.active_component_count(), 1)
            self.assertEqual(st.active_node_count(), 1)
        # 换成互不相连的散点（26 邻接也够不到）：每片封闭一片
        vol2 = [[[] for _ in range(3)] for _ in range(depth)]
        for z in range(depth):
            vol2[z][0] = [(0, 1)] if z % 2 == 0 else []
            vol2[z][2] = [] if z % 2 == 0 else [(2, 3)]
        st2 = StreamLabeler(3, 3, 26)
        for z in range(depth):
            st2.feed_slice(vol2[z])
            self.assertLessEqual(st2.active_component_count(), 1)
            self.assertLessEqual(st2.active_node_count(), 1)
        self.assertEqual(len(st2.finish().components), depth)


class StreamVsReferenceTests(unittest.TestCase):
    def test_random_small_volumes(self):
        seed = 991199
        for d, h, w in [(1, 1, 5), (2, 3, 5), (4, 4, 4), (5, 3, 6), (6, 6, 3)]:
            for p in (0.3, 0.55, 0.75):
                rng = random.Random(seed)
                seed += 1
                vol = random_volume(rng, d, h, w, p)
                for conn in (6, 18, 26):
                    assert_stream_matches_batch(self, vol, d, h, w, conn)

    def test_empty_volume(self):
        vol = [[[], []], [[], []], [[], []]]
        outs, final = stream(vol, 2, 2, 26)
        self.assertEqual(final.components, [])
        self.assertEqual(final.id_of_temp, {})
        self.assertEqual(final.num_runs, 0)
        for out in outs:
            self.assertEqual(out.closed, [])

    def test_z_indices_and_temp_ids_stable(self):
        vol = [
            [[(0, 1)]],
            [[(2, 3)]],                    # 新组件（与 z=0 不相连）
            [[(0, 1)]],
        ]
        outs, final = stream(vol, 1, 3, 6)
        self.assertEqual([o.z for o in outs], [0, 1, 2])
        temps = [t for o in outs for row in o.run_labels for t in row]
        self.assertEqual(temps, [1, 2, 3])  # 稳定临时身份按创建次序
        # 最终 ID 按最小坐标排序，与临时 ID 次序无关
        self.assertEqual(final.id_of_temp, {1: 1, 2: 2, 3: 3})
        # 封闭次序 != 最终 ID 次序：提前封闭的短组件不能先拿最终 ID
        vol2 = [
            [[(0, 1)], []],     # 长组件起点（最终 ID 1）
            [[(0, 1)], [(1, 2)]],
            [[(0, 1)], []],     # 短组件在此切片后封闭
        ]
        outs2, final2 = stream(vol2, 2, 3, 6)
        closed_first = outs2[2].closed[0]
        self.assertEqual((closed_first.min_z, closed_first.min_y,
                          closed_first.min_x), (1, 1, 1))
        # 先封闭的是短组件，但最终 1 号属于最小坐标 (0,0,0) 的长组件
        self.assertEqual(final2.id_of_temp[closed_first.temp_id], 2)
        self.assertEqual(final2.components[0].voxel_count, 3)


class StreamErrorTests(unittest.TestCase):
    def test_feed_after_finish_rejected(self):
        st = StreamLabeler(1, 1, 6)
        st.feed_slice([[(0, 1)]])
        st.finish()
        with self.assertRaises(InvalidVolumeError):
            st.feed_slice([[(0, 1)]])
        with self.assertRaises(InvalidVolumeError):
            st.finish()

    def test_bad_constructor_args(self):
        for args in [(True, 1), (1, False), (0, 1), (1, 513)]:
            with self.assertRaises(InvalidVolumeError):
                StreamLabeler(*args)
        with self.assertRaises(InvalidVolumeError):
            StreamLabeler(1, 1, 7)
        with self.assertRaises(InvalidVolumeError):
            StreamLabeler(1, 1, 6.0)

    def test_bad_slice_structure(self):
        st = StreamLabeler(2, 2, 6)
        with self.assertRaises(InvalidVolumeError):
            st.feed_slice([[(0, 1)]])            # 行数与 height 不符
        with self.assertRaises(InvalidVolumeError):
            st.feed_slice("xx")
        with self.assertRaises(InvalidVolumeError):
            st.feed_slice([[(0, 3)], []])        # end > width
        with self.assertRaises(InvalidVolumeError):
            st.feed_slice([[(0, float("nan"))], []])
        with self.assertRaises(InvalidVolumeError):
            st.feed_slice([[(True, 1)], []])     # bool 坐标
        # 失败不留下部分变更：之后正常喂入仍从 z=0 继续
        out = st.feed_slice([[(0, 1)], []])
        self.assertEqual(out.z, 0)
        final = st.finish()
        self.assertEqual(len(final.components), 1)

    def test_depth_limit(self):
        st = StreamLabeler(1, 1, 6)
        for _ in range(512):
            st.feed_slice([[(0, 1)]])
        with self.assertRaises(InvalidVolumeError):
            st.feed_slice([[(0, 1)]])

    def test_run_count_limit(self):
        st = StreamLabeler(500, 1, 6)
        for _ in range(40):                      # 40*500 = 20000，接受
            st.feed_slice([[(0, 1)] for _ in range(500)])
        with self.assertRaises(InvalidVolumeError) as cm:
            st.feed_slice([[(0, 1)]] + [[]] * 499)
        self.assertIn("20000", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
