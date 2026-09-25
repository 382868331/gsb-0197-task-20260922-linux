"""固定输入演示：稀疏 RLE 体的游程连通组件标记（二值 + 多材料）。

运行：python demo.py（Python 3.14，标准库，离线，无需任何参数）。

本脚本真实计算并打印：
  1) 二值入口 label：一个固定小体在 6 / 18 / 26 三种邻接下的组件结果
     （三者组件数刻意不同：3 / 2 / 1）；
  2) 多材料入口 label_materials：一个两种材料交错的小体在 6 / 18
     邻接下的组件、材料、对空白暴露面积与组件间接触面积表
     （18 邻接下同材料经边合并，组件数与接触表同步变化）；
  3) 两个被实际触发并拒绝的非法输入边界（NaN 游程坐标、重叠游程）。

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


# ---------------------------------------------------------------------------
# 多材料入口演示
# ---------------------------------------------------------------------------

# 固定多材料体素（z, y, x, material），由确定性代码编码为游程：
#   mat1 横条 (0,0,0)-(0,0,1)；mat2 横条 (0,1,0)-(0,1,1) 与其整面贴合
#   （共享 2 个面）；mat1 孤点 (1,1,1) 与 mat2 面贴（共享 1 个面），
#   与 mat1 横条仅边接触（dz=1, dy=1）。
# 6 邻接：mat1 分成 2 个组件（共 3 组件）；18 邻接：mat1 经边合并
# （共 2 组件），接触表同步从 {(1,2):2, (2,3):1} 变为 {(1,2):3}。
MATERIAL_VOXELS = [
    (0, 0, 0, 1), (0, 0, 1, 1),
    (0, 1, 0, 2), (0, 1, 1, 2),
    (1, 1, 1, 1),
]


def encode_materials(voxels, depth, height, width):
    """把固定 (z, y, x, material) 体素编码为 [z][y] -> [(s, e, m), ...]。"""
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


def show_materials(title, result):
    print(title)
    print(f"  游程总数={result.num_runs}  组件数={len(result.components)}")
    for c in result.components:
        b = c.bbox
        print(
            f"  组件 {c.id}: 材料={c.material}  体素={c.voxel_count}  "
            f"包围盒 z[{b.min_z},{b.max_z}) y[{b.min_y},{b.max_y}) "
            f"x[{b.min_x},{b.max_x})  对空白暴露面积={c.exposed_area}  "
            f"最小体素=({c.min_z},{c.min_y},{c.min_x})"
        )
    if result.contacts:
        print("  组件间接触面积（仅共享面，较小 ID 在前）:")
        for c in result.contacts:
            print(f"    组件 {c.component_a} <-> 组件 {c.component_b}: "
                  f"{c.area}")
    else:
        print("  组件间接触面积：无")
    print()


def main_materials():
    vol = encode_materials(MATERIAL_VOXELS, DEPTH, HEIGHT, WIDTH)
    print("固定多材料体素（z,y,x,material）：", MATERIAL_VOXELS)
    print(f"体尺寸 depth={DEPTH} height={HEIGHT} width={WIDTH}\n")

    r6 = label_materials(vol, DEPTH, HEIGHT, WIDTH, 6)
    show_materials("=== 多材料 connectivity=6 ===", r6)
    r18 = label_materials(vol, DEPTH, HEIGHT, WIDTH, 18)
    show_materials("=== 多材料 connectivity=18 ===", r18)

    # 真实验算：6 邻接 3 组件、18 邻接 2 组件；接触面积与暴露面积手算值
    assert len(r6.components) == 3 and len(r18.components) == 2
    assert [(c.component_a, c.component_b, c.area) for c in r6.contacts] == [
        (1, 2, 2), (2, 3, 1),
    ]
    assert [(c.component_a, c.component_b, c.area) for c in r18.contacts] == [
        (1, 2, 3),
    ]
    assert [c.exposed_area for c in r18.components] == [13, 7]
    for r in (r6, r18):
        assert sum(c.voxel_count for c in r.components) == len(MATERIAL_VOXELS)
    print("验算通过：6/18 邻接组件数 3/2，接触表随合并同步更新，"
          "体素守恒。\n")

    # --- 实际触发的拒绝边界：重叠游程（即使材料不同也非法） ---
    print("=== 非法输入边界（重叠游程）===")
    bad = [[[(0, 3, 1), (2, 4, 2)]]]
    try:
        label_materials(bad, 1, 1, 4, 6)
    except InvalidVolumeError as exc:
        print(f"  如预期拒绝：{type(exc).__name__}: {exc}")
    else:  # pragma: no cover
        raise AssertionError("重叠输入未被拒绝")


if __name__ == "__main__":
    main()
    print()
    main_materials()
