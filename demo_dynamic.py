"""固定输入演示：有状态体素模型的批量增删与分裂/合并事件。

运行：python demo_dynamic.py（Python 3.14，标准库，离线，无需任何参数）。

演示序列（6 邻接，全部结果现场计算，手算常数仅用于结尾 assert 核对）：

  1) 正常：一根 7 体素直棒上删除桥体素 -> split 事件；
     把桥体素加回 -> merge 事件；
  2) 边界：空模型 / 空批次 / 整组件删光 / 冗余环路删一个连接不分裂；
  3) 实际错误：同一提交既增又删同一体素被拒绝，且状态保持不变；
     越界游程同样被拒绝。

不 sleep、不读取外部数据、不使用预录结果。
"""

from rle_components import ComponentModel, InvalidVolumeError

WIDTH = 7


def show_components(model):
    if not model.components:
        print("    （无组件，空体）")
        return
    for c in model.components:
        b = c.bbox
        print(
            f"    组件 {c.id}: 体素={c.voxel_count}  "
            f"包围盒 z[{b.min_z},{b.max_z}) y[{b.min_y},{b.max_y}) "
            f"x[{b.min_x},{b.max_x})  最小体素=({c.min_z},{c.min_y},"
            f"{c.min_x})  暴露面积={c.exposed_area}"
        )


def show_events(result):
    if not result.splits and not result.merges:
        print("    事件：无（拓扑未变）")
        return
    for ev in result.splits:
        old = ev.affected_components[0]
        new = ", ".join(
            f"({s.min_z},{s.min_y},{s.min_x})/{s.voxel_count}体素"
            for s in ev.resulting_components
        )
        print(
            f"    SPLIT: 旧组件 ({old.min_z},{old.min_y},{old.min_x})/"
            f"{old.voxel_count}体素 -> [{new}]（区域内新增 "
            f"{ev.added_voxels}）"
        )
    for ev in result.merges:
        src = ", ".join(
            f"({s.min_z},{s.min_y},{s.min_x})/{s.voxel_count}体素"
            for s in ev.merged_components
        )
        r = ev.resulting_component
        print(
            f"    MERGE: [{src}] -> 新组件 ({r.min_z},{r.min_y},"
            f"{r.min_x})/{r.voxel_count}体素（新增 {ev.added_voxels}，"
            f"删除 {ev.removed_voxels}）"
        )


def section(title):
    print(f"\n=== {title} ===")


def main():
    # ------------------------------------------------------------------
    # 1) 正常路径：桥断开（split）与重新连接（merge）
    # ------------------------------------------------------------------
    section("初始：一根 7 体素直棒 x=0..6（6 邻接）")
    model = ComponentModel(1, 1, WIDTH, 6, [(0, 0, 0, WIDTH)])
    show_components(model)
    assert len(model.components) == 1

    print("\n提交 remove=[(0,0,3)]：删除唯一桥体素")
    r = model.remove([(0, 0, 3)])
    print(f"  生效删除={r.removed}  剩余体素={r.num_voxels}  "
          f"组件数={r.num_components}")
    show_components(model)
    show_events(r)
    # 手算核对：7 体素棒删中点 -> 两个 3 体素组件，恰好一个 split
    assert r.removed == 1 and r.num_voxels == 6
    assert r.num_components == 2 and len(r.splits) == 1
    assert [s.voxel_count for s in r.splits[0].resulting_components] == [3, 3]
    assert [(s.min_x) for s in r.splits[0].resulting_components] == [0, 4]

    print("\n提交 add=[(0,0,3)]：桥重新接上")
    r = model.add([(0, 0, 3)])
    print(f"  生效新增={r.added}  体素={r.num_voxels}  "
          f"组件数={r.num_components}")
    show_components(model)
    show_events(r)
    # 手算核对：两个 3 体素组件 + 1 桥体素 -> 7 体素单组件，一个 merge
    assert r.added == 1 and r.num_components == 1
    assert len(r.merges) == 1
    assert [s.min_x for s in r.merges[0].merged_components] == [0, 4]
    assert r.merges[0].resulting_component.voxel_count == 7

    print("\n再次 add 同一桥体素（幂等）：")
    r = model.add([(0, 0, 3)])
    print(f"  生效新增={r.added}  组件数={r.num_components}")
    show_events(r)
    assert r.added == 0 and not r.splits and not r.merges

    # ------------------------------------------------------------------
    # 2) 边界：空输入、整组件删光、冗余路径不分裂、游程批量
    # ------------------------------------------------------------------
    section("边界 1：空模型与空批次")
    empty = ComponentModel(2, 2, 2, 6)
    print(f"  初始体素数={empty.num_voxels}  组件数={len(empty.components)}")
    show_components(empty)
    r = empty.update()
    print(f"  空 update：新增={r.added} 删除={r.removed} "
          f"组件数={r.num_components}")
    r = empty.remove([(1, 1, 1)])
    print(f"  从空模型删不存在体素：生效删除={r.removed}（幂等，无事件）")
    show_events(r)
    assert empty.num_voxels == 0 and r.components == ()
    empty.add([(0, 0, 0, 2)])  # 游程形式：两个体素
    empty.reset()
    assert empty.num_voxels == 0 and empty.components == ()
    print("  add 游程后再 reset()：回到空模型")

    section("边界 2：整组件删光不算 split")
    two = ComponentModel(1, 1, 5, 6, [(0, 0, 0, 2), (0, 0, 3, 5)])
    print("  初始：两个分离双体素块")
    r = two.remove([(0, 0, 0), (0, 0, 1)])
    print(f"  删除左块全部体素：组件数={r.num_components}")
    show_events(r)
    assert r.splits == () and r.merges == () and r.num_components == 1

    section("边界 3：冗余环路删一个连接不分裂")
    # 2x3 实心矩形：上下两行各 3 体素，存在两条绕行路径
    ring = ComponentModel(1, 2, 3, 6,
                          [(0, 0, 0, 3), (0, 1, 0, 3)])
    r = ring.remove([(0, 0, 1)])
    print("  2x3 矩形删除上排中体素后：")
    show_components(ring)
    show_events(r)
    assert r.num_components == 1 and r.splits == ()
    assert ring.components[0].voxel_count == 5

    section("边界 4：游程批量一次桥接多个组件（18 邻接边桥）")
    edge = ComponentModel(3, 2, 1, 18)
    edge.add([(0, 0, 0), (2, 1, 0)])  # 两个远离的单体素
    r = edge.add([(1, 1, 0)])          # 边接触桥（dz=1,dy=1 / dz=1,dy=0）
    print("  加边桥体素 (1,1,0) 后：")
    show_components(edge)
    show_events(r)
    assert edge.num_voxels == 3 and r.num_components == 1
    assert len(r.merges) == 1
    r = edge.remove([(1, 1, 0)])
    print("  删除边桥后：")
    show_events(r)
    assert r.num_components == 2 and len(r.splits) == 1

    # ------------------------------------------------------------------
    # 3) 实际错误：必须明确拒绝且不留半更新状态
    # ------------------------------------------------------------------
    section("错误 1：同一提交既新增又删除同一体素")
    print(f"  当前体素={sorted(model.voxels)} 组件数={len(model.components)}")
    try:
        model.update(add=[(0, 0, 3)], remove=[(0, 0, 3)])
    except InvalidVolumeError as exc:
        print(f"  如预期拒绝：{type(exc).__name__}: {exc}")
    else:  # pragma: no cover
        raise AssertionError("增删冲突未被拒绝")
    assert model.num_voxels == 7 and len(model.components) == 1
    print("  拒绝后状态不变：体素=7，组件数=1")

    section("错误 2：越界游程（end 超过 width）")
    before = sorted(model.voxels)
    try:
        model.add([(0, 0, 0, 99)])
    except InvalidVolumeError as exc:
        print(f"  如预期拒绝：{type(exc).__name__}: {exc}")
    else:  # pragma: no cover
        raise AssertionError("越界游程未被拒绝")
    assert sorted(model.voxels) == before
    print(f"  拒绝后状态不变：体素={model.num_voxels}")

    print("\n全部演示与手算核对通过。")


if __name__ == "__main__":
    main()
