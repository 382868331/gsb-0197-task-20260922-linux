"""固定输入演示：稀疏 RLE 体的游程连通组件标记与批量增删。

运行：python demo.py（Python 3.14，标准库，离线，无需任何参数）。

本脚本真实计算并打印：
  1) 一个固定小体素体在 6 / 18 / 26 三种邻接下的组件结果
     （三者组件数刻意不同：3 / 2 / 1）；
  2) VoxelVolume 批量增删演示：架桥（merge 事件）、拆桥（split 事件）、
     空批次（无事件）；
  3) 两个被实际触发并拒绝的非法输入边界（NaN 游程坐标、越界体素），
     并展示失败批次不留下半更新状态。

不 sleep、不读取外部数据、不使用预录结果。
"""

from rle_components import InvalidVolumeError, VoxelVolume, label

DEPTH = HEIGHT = WIDTH = 4

# 固定体素（坐标 z,y,x），由确定性代码编码为每行有序半开 x 游程：
#   A: (0,0,0)-(0,0,1)         面接触对
#   B: (0,1,2)                 相对 A 的 (0,0,1) 为边接触(dy=1,dx=1)
#   C: (1,2,3)                 相对 B 为角接触(dz=1,dy=1,dx=1)
# 于是 6 邻接：{A},{B},{C} 三个；18 邻接：{A,B},{C} 两个；
# 26 邻接：{A,B,C} 一个。其余行均为空行。
VOXELS = [
    (0, 0, 0), (0, 0, 1),
    (0, 1, 2),
    (1, 2, 3),
]


def encode(voxels, depth, height, width):
    """把固定体素坐标集合编码为 [z][y] -> [(start, end), ...]。"""
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


def show(title, result):
    print(title)
    print(f"  游程总数={result.num_runs}  组件数={len(result.components)}")
    for c in result.components:
        b = c.bbox

        print(
            f"  组件 {c.id}: 体素={c.voxel_count}  "
            f"包围盒 z[{b.min_z},{b.max_z}) y[{b.min_y},{b.max_y}) "
            f"x[{b.min_x},{b.max_x})  暴露面积={c.exposed_area}  "
            f"最小体素=({c.min_z},{c.min_y},{c.min_x})"
        )
    print("  每游程组件标签（仅列非空行）:")
    for z, sl in enumerate(result.run_labels):
        for y, row_labels in enumerate(sl):
            if row_labels:
                print(f"    z={z} y={y}: {row_labels}")
    print()


def main():
    vol = encode(VOXELS, DEPTH, HEIGHT, WIDTH)
    print("固定输入体素：", VOXELS)
    print(f"体尺寸 depth={DEPTH} height={HEIGHT} width={WIDTH}\n")

    results = {}
    for conn in (6, 18, 26):
        result = label(vol, DEPTH, HEIGHT, WIDTH, conn)
        results[conn] = result
        show(f"=== connectivity={conn} ===", result)

    # 真实验算：三种邻接组件数必须不同，且体素数守恒、面积恒等式成立
    counts = {conn: len(results[conn].components) for conn in (6, 18, 26)}
    assert counts == {6: 3, 18: 2, 26: 1}, counts
    for conn, result in results.items():
        total = sum(c.voxel_count for c in result.components)
        assert total == len(VOXELS) == 4
        for c in result.components:
            assert c.exposed_area >= 0
            assert c.exposed_area % 2 == 0  # 任何组件暴露面数均为偶数
    print("验算通过：三种邻接组件数分别为 3/2/1，体素守恒，面积非负偶数值。\n")

    # --- 实际触发的拒绝边界：NaN 坐标必须被拒绝（不静默纠正） ---
    print("=== 非法输入边界（NaN 游程坐标）===")
    bad = [[[(0, float("nan"))]]]
    try:
        label(bad, 1, 1, 1, 6)
    except InvalidVolumeError as exc:
        print(f"  如预期拒绝：{type(exc).__name__}: {exc}")
    else:  # pragma: no cover
        raise AssertionError("NaN 输入未被拒绝")

    # ================= 批量增删与组件分裂/合并事件 =================
    print("\n=== VoxelVolume 批量增删：架桥 → 拆桥 → 重连 ===")
    vol = VoxelVolume(1, 1, 5, connectivity=6)

    def show_update(title, r):
        comps = r.result.components
        print(f"{title}: 组件数={len(comps)} 体素数={vol.voxel_count}")
        for c in comps:
            print(f"    组件 {c.id}: 体素={c.voxel_count} "
                  f"最小体素=({c.min_z},{c.min_y},{c.min_x}) "
                  f"暴露面积={c.exposed_area}")
        for ev in r.events:
            print(f"    事件 {ev.kind}: 旧组件{ev.old_ids or '∅'} -> "
                  f"新组件{ev.new_ids or '∅'}")

    # 两段棒 x in [0,2) 与 [3,5)，一批加入 → 两个组件各自 created
    r = vol.add_voxels([(0, 0, 0), (0, 0, 1), (0, 0, 3), (0, 0, 4)])
    show_update("批量加入两段棒（各 2 体素）", r)
    assert [e.kind for e in r.events] == ["created", "created"]

    # 架桥：加入 x=2 → merge
    r = vol.add_voxels([(0, 0, 2)])
    show_update("架桥：加入桥接体素 x=2", r)
    assert r.events[0].kind == "merge" and r.events[0].old_ids == (1, 2)
    assert len(r.result.components) == 1

    # 拆桥：删除 x=2 → split
    r = vol.remove_voxels([(0, 0, 2)])
    show_update("拆桥：删除桥接体素 x=2", r)
    assert r.events[0].kind == "split" and r.events[0].new_ids == (1, 2)
    assert len(r.result.components) == 2

    # 重连：再次加入 x=2 → merge
    r = vol.add_voxels([(0, 0, 2)])
    show_update("重连：再次加入 x=2", r)
    assert r.events[0].kind == "merge"
    assert len(r.result.components) == 1

    # 空批次：无变化、无事件
    r = vol.apply()
    print(f"空批次 apply(): 事件数={len(r.events)} "
          f"added={r.added} removed={r.removed} 组件数="
          f"{len(r.result.components)}")
    assert r.events == () and len(r.result.components) == 1

    # --- 实际触发的拒绝边界：越界体素，且失败批次不留半更新 ---
    print("\n=== 非法输入边界（批量体素坐标越界）===")
    before = vol.voxel_count
    try:
        vol.add_voxels([(0, 0, 0), (0, 0, 99)])  # 一个合法 + 一个越界
    except InvalidVolumeError as exc:
        print(f"  如预期拒绝：{type(exc).__name__}: {exc}")
    else:  # pragma: no cover
        raise AssertionError("越界体素未被拒绝")
    assert vol.voxel_count == before, "失败批次不得改变体素数"
    print(f"  状态未变：体素数仍为 {vol.voxel_count}（合法体素也未被部分应用）")
    print("\n演示完成：全部断言通过。")


if __name__ == "__main__":
    main()
