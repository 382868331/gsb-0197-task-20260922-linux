"""固定输入演示：RLE 体数据的封闭空腔查询（本题增量）。

运行：python demo_iteration.py（Python 3.14，标准库，离线，无参数）。

真实计算并打印：
  1) 空心壳：一个封闭空腔（体积 / 包围盒 / 接触前景组件）；
  2) 同一壳只开一个角点孔道：26 邻接背景穿角泄漏 -> 空腔消失；
  3) 嵌套壳：两个空腔，touches 分别引用 6 邻接前景组件 ID；
  4) 同一 RLE 换分段方式：空腔结果与前景组件 ID 完全不变；
  5) 一个被实际触发并拒绝的非法 box（min > max）。

不 sleep、不读外部数据、不使用预录结果；实测远低于 90 秒。
"""

import time

from rle_components import InvalidVolumeError, find_cavities


def encode(voxels, depth, height, width):
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


def shell(n, offset=(0, 0, 0)):
    """n^3 立方体表面体素（空心壳）。"""
    oz, oy, ox = offset
    return {
        (oz + z, oy + y, ox + x)
        for z in range(n) for y in range(n) for x in range(n)
        if z in (0, n - 1) or y in (0, n - 1) or x in (0, n - 1)
    }


def split_runs(vol):
    """同一体素的另一种合法分段：长度 >= 2 的游程从中间拆开。"""
    return [
        [
            [piece for s, e in row
             for piece in ([(s, (s + e) // 2), ((s + e) // 2, e)]
                           if e - s >= 2 else [(s, e)])]
            for row in sl
        ]
        for sl in vol
    ]


def show(title, result):
    print(title)
    fg = result.foreground
    print(f"  前景组件数={len(fg.components)}  空腔数={len(result.cavities)}")
    for c in result.cavities:
        b = c.bbox
        print(
            f"  空腔 {c.id}: 体积={c.voxel_count}  "
            f"包围盒 z[{b.min_z},{b.max_z}) y[{b.min_y},{b.max_y}) "
            f"x[{b.min_x},{b.max_x})  接触前景组件={list(c.touches)}"
        )
    print()


def main():
    t0 = time.perf_counter()

    # 1) 空心壳：8^3 壳，内部 6^3 = 216 一个封闭空腔
    n = 8
    vol = encode(shell(n), n, n, n)
    box = (0, 0, 0, n, n, n)
    r1 = find_cavities(vol, n, n, n, box)
    show("=== 1) 8^3 空心壳 ===", r1)
    assert len(r1.cavities) == 1
    assert r1.cavities[0].voxel_count == 216
    assert r1.cavities[0].touches == (1,)

    # 2) 同一壳只开一个角点：背景 26 邻接穿角 -> 不再是空腔
    vox2 = shell(n) - {(0, 0, 0)}
    vol2 = encode(vox2, n, n, n)
    r2 = find_cavities(vol2, n, n, n, box)
    show("=== 2) 同一壳只开一个角点孔道 ===", r2)
    assert r2.cavities == []
    assert len(r2.foreground.components) == 1  # 前景 6 邻接仍为一体

    # 3) 嵌套壳：11^3 外壳套 3^3 内壳 -> 两个空腔
    n3 = 11
    vox3 = shell(11) | shell(3, offset=(4, 4, 4))
    vol3 = encode(vox3, n3, n3, n3)
    r3 = find_cavities(vol3, n3, n3, n3, (0, 0, 0, n3, n3, n3))
    show("=== 3) 嵌套壳（11^3 套 3^3）===", r3)
    assert len(r3.cavities) == 2
    assert r3.cavities[0].touches == (1, 2)   # 壳间区域接触两个组件
    assert r3.cavities[1].touches == (2,)     # 内壳内部只接触内壳

    # 4) 同一 RLE 换分段：结果必须完全不变
    vol3_split = split_runs(vol3)
    r4 = find_cavities(vol3_split, n3, n3, n3, (0, 0, 0, n3, n3, n3))
    print("=== 4) 同一 RLE 换分段方式 ===")
    print(f"  游程数 {r3.foreground.num_runs} -> {r4.foreground.num_runs}，"
          f"空腔结果一致：{r3.cavities == r4.cavities}\n")
    assert r3.cavities == r4.cavities
    assert r3.foreground.components == r4.foreground.components

    # 5) 实际触发的拒绝：box 的 min > max
    print("=== 5) 非法 box（min > max）===")
    try:
        find_cavities(vol, n, n, n, (3, 0, 0, 2, n, n))
    except InvalidVolumeError as exc:
        print(f"  如预期拒绝：{type(exc).__name__}: {exc}")
    else:
        raise AssertionError("非法 box 未被拒绝")

    dt = time.perf_counter() - t0
    print(f"\n全部验算通过，现场计算耗时 {dt:.3f} 秒（上限 90 秒）。")


if __name__ == "__main__":
    main()
