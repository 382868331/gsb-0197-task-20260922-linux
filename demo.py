"""固定输入演示：稀疏 RLE 体的游程连通组件标记（二值 + 多材料）。

运行：python demo.py（Python 3.14，标准库，离线，无需任何参数）。

本脚本真实计算并打印：
  1) 二值入口 label：固定小体在 6 / 18 / 26 三种邻接下的组件结果
     （三者组件数刻意不同：3 / 2 / 1）；
  2) 多材料入口 label_materials：一个三材料小体的分材料组件、
     对空白暴露面积与组件间接触面积（26 邻接下材料 3 的角接触
     体素合并，接触表同步更新）；
  3) 两个被实际触发并拒绝的非法输入边界（NaN 游程坐标、
     多材料输入的游程重叠）。

不 sleep、不读取外部数据、不使用预录结果。
"""

from rle_components import InvalidVolumeError, label, label_materials

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

# 多材料固定体素（z, y, x, material）：
#   材料 1：z=0 的 2x2 板；材料 2：z=1 的 2x2 板（与材料 1 面贴，
#           接触面积 4）；材料 3：(2,2,2)、(3,3,3) 仅角接触的对角对，
#           6/18 下为两个组件，26 下并为一个（角接触面积恒为 0）。
MATERIAL_VOXELS = [
    (0, 0, 0, 1), (0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 1),
    (1, 0, 0, 2), (1, 0, 1, 2), (1, 1, 0, 2), (1, 1, 1, 2),
    (2, 2, 2, 3),
    (3, 3, 3, 3),
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


def encode_materials(voxels, depth, height, width):
    """把 (z, y, x, material) 集合编码为 [z][y] -> [(s, e, material), ...]。"""
    grid = [[{} for _ in range(height)] for _ in range(depth)]
    for z, y, x, m in voxels:
        grid[z][y][x] = m
    vol = []
    for z in range(depth):
        rows = []
        for y in range(height):
            row, s, prev, cur = [], None, None, None
            for x in sorted(grid[z][y]):
                m = grid[z][y][x]
                if s is None:
                    s, cur = x, m
                elif x != prev + 1 or m != cur:
                    row.append((s, prev + 1, cur))
                    s, cur = x, m
                prev = x
            if s is not None:
                row.append((s, prev + 1, cur))
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


def show_materials(title, result):
    print(title)
    print(f"  游程总数={result.num_runs}  组件数={len(result.components)}")
    for c in result.components:
        b = c.bbox
        print(
            f"  组件 {c.id}: 材料={c.material}  体素={c.voxel_count}  "
            f"包围盒 z[{b.min_z},{b.max_z}) y[{b.min_y},{b.max_y}) "
            f"x[{b.min_x},{b.max_x})  对空白暴露面积={c.exposed_area}"
        )
    if result.contacts:
        print("  组件间接触面积（仅共享方形面）:")
        for ct in result.contacts:
            print(
                f"    组件 {ct.component_a} <-> 组件 {ct.component_b}: "
                f"{ct.area}"
            )
    else:
        print("  组件间接触面积：无")
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

    # --- 多材料入口：现场计算一个三材料小体 ---
    mvol = encode_materials(MATERIAL_VOXELS, DEPTH, HEIGHT, WIDTH)
    print("多材料固定体素 (z,y,x,material)：", MATERIAL_VOXELS, "\n")
    mresults = {}
    for conn in (6, 18, 26):
        mr = label_materials(mvol, DEPTH, HEIGHT, WIDTH, conn)
        mresults[conn] = mr
        show_materials(f"=== 多材料 connectivity={conn} ===", mr)

    # 真实验算：材料 1/2 板间接触面积恒为 4（与连通性无关）；
    # 材料 3 的角接触对在 6/18 下为两个组件、26 下并为一个；
    # 角接触不产生接触面积，接触表始终只有 (1,2) 一条。
    for conn, mr in mresults.items():
        assert len(mr.contacts) == 1
        ct = mr.contacts[0]
        assert (ct.component_a, ct.component_b, ct.area) == (1, 2, 4)
        total = sum(c.voxel_count for c in mr.components)
        assert total == len(MATERIAL_VOXELS) == 10
        for c in mr.components:
            assert c.exposed_area >= 0
    assert len(mresults[6].components) == 4
    assert len(mresults[18].components) == 4
    assert len(mresults[26].components) == 3
    print("验算通过：接触面积恒为 4，材料 3 组件数 6/18/26 = 2/2/1，"
          "体素守恒。\n")

    # --- 实际触发的拒绝边界：NaN 坐标必须被拒绝（不静默纠正） ---
    print("=== 非法输入边界（NaN 游程坐标）===")
    bad = [[[(0, float("nan"))]]]
    try:
        label(bad, 1, 1, 1, 6)
    except InvalidVolumeError as exc:
        print(f"  如预期拒绝：{type(exc).__name__}: {exc}")
    else:  # pragma: no cover
        raise AssertionError("NaN 输入未被拒绝")

    # --- 实际触发的拒绝边界：多材料输入游程重叠 ---
    print("=== 非法输入边界（多材料游程重叠）===")
    bad_overlap = [[[(0, 3, 1), (2, 4, 2)]]]
    try:
        label_materials(bad_overlap, 1, 1, 4, 6)
    except InvalidVolumeError as exc:
        print(f"  如预期拒绝：{type(exc).__name__}: {exc}")
    else:  # pragma: no cover
        raise AssertionError("重叠输入未被拒绝")


if __name__ == "__main__":
    main()
