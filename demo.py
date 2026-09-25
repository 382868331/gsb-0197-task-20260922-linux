"""固定输入演示：稀疏 RLE 体的游程连通组件标记（批式 + 流式）。

运行：python demo.py（Python 3.14，标准库，离线，无需任何参数）。

本脚本真实计算并打印：
  1) 一个固定小体素体在 6 / 18 / 26 三种邻接下的组件结果
     （三者组件数刻意不同：3 / 2 / 1）；
  2) 同一体数据改用 StreamLabeler 逐切片喂入：组件封闭即输出终结记录，
     finish 给出临时 ID -> 最终 ID 映射，结果与批式完全一致；
  3) 一个被实际触发并拒绝的非法输入边界（NaN 游程坐标）。

不 sleep、不读取外部数据、不使用预录结果。
"""

from rle_components import InvalidVolumeError, StreamLabeler, label

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

    # --- 流式接口：逐切片喂入同一体数据，结果必须与批式一致 ---
    print("=== 流式接口（StreamLabeler，connectivity=26）===")
    st = StreamLabeler(HEIGHT, WIDTH, 26)
    for z, slice_rows in enumerate(vol):
        out = st.feed_slice(slice_rows)
        nonempty = [t for row in out.run_labels for t in row]
        print(
            f"  z={z}: 本切片游程临时 ID={nonempty or '无'}  "
            f"新封闭组件={len(out.closed)}  "
            f"活跃组件={st.active_component_count()}"
        )
        for rec in out.closed:
            print(
                f"    封闭 temp={rec.temp_id}: 体素={rec.voxel_count}  "
                f"暴露面积={rec.exposed_area}"
            )
    final = st.finish()
    print(f"  finish: 临时->最终 ID 映射 {final.id_of_temp}")
    for c in final.components:
        print(
            f"  组件 {c.id}: 体素={c.voxel_count}  "
            f"暴露面积={c.exposed_area}  "
            f"最小体素=({c.min_z},{c.min_y},{c.min_x})"
        )
    # 流式最终结果与批式 label 逐字段一致
    assert final.components == results[26].components
    print("验算通过：流式 finish 结果与批式 label(connectivity=26) 完全一致。\n")

    # --- 实际触发的拒绝边界：NaN 坐标必须被拒绝（不静默纠正） ---
    print("=== 非法输入边界（NaN 游程坐标）===")
    bad = [[[(0, float("nan"))]]]
    try:
        label(bad, 1, 1, 1, 6)
    except InvalidVolumeError as exc:
        print(f"  如预期拒绝：{type(exc).__name__}: {exc}")
    else:  # pragma: no cover
        raise AssertionError("NaN 输入未被拒绝")


if __name__ == "__main__":
    main()
