"""固定输入演示：RLE 体数据的封闭空腔查询（本题增量）。

运行：python demo_iteration.py（Python 3.14，标准库，离线，无参数）。

真实计算并打印：
  1) 空心壳：1 个封闭空腔；
  2) 只开一个角点的壳：背景 26 邻接沿角泄漏，0 个空腔；
  3) 嵌套壳：2 个空腔，夹层同时接触两个前景组件；
  4) 贴边开放凹槽：前景包围盒贴盒边界，凹槽仍正确判为开放；
  5) 一个被实际触发并拒绝的非法输入（重叠游程）。

不 sleep、不读外部数据、不使用预录结果；全部输入由确定性代码生成。
"""

import time

from rle_components import InvalidVolumeError, find_cavities


def encode_voxels(voxels, depth, height, width):
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


def shell(lo, hi):
    """[lo, hi) 立方体表面的体素集合。"""
    out = set()
    for z in range(lo, hi):
        for y in range(lo, hi):
            for x in range(lo, hi):
                if z in (lo, hi - 1) or y in (lo, hi - 1) or x in (lo, hi - 1):
                    out.add((z, y, x))
    return out


def show(title, result):
    print(title)
    print(f"  前景组件数={len(result.foreground.components)}  "
          f"背景成分数={result.background_components}  "
          f"封闭空腔数={len(result.cavities)}")
    for c in result.cavities:
        b = c.bbox
        print(
            f"  空腔 {c.id}: 体积={c.voxel_count}  "
            f"包围盒 z[{b.min_z},{b.max_z}) y[{b.min_y},{b.max_y}) "
            f"x[{b.min_x},{b.max_x})  接触前景组件={list(c.touching_components)}"
        )
    print()


def main():
    t0 = time.perf_counter()

    # 1) 空心壳：7^3 盒内 5^3 壳（坐标 1..5），内部 3^3=27 体素封闭空腔
    vol = encode_voxels(shell(1, 6), 7, 7, 7)
    r1 = find_cavities(vol, 7, 7, 7)
    show("=== 1) 空心壳（封闭） ===", r1)
    assert len(r1.cavities) == 1 and r1.cavities[0].voxel_count == 27
    assert r1.cavities[0].touching_components == (1,)

    # 2) 同一只壳只去掉一个角点体素：26 邻接背景经角对角通道泄漏
    voxels = shell(1, 6)
    voxels.discard((1, 1, 1))
    vol = encode_voxels(voxels, 7, 7, 7)
    r2 = find_cavities(vol, 7, 7, 7)
    show("=== 2) 只开一个角点的壳（泄漏，非空腔） ===", r2)
    assert r2.cavities == []

    # 3) 嵌套壳：11^3 盒内 7^3 壳（2..8）套 3^3 壳（4..6）
    vol = encode_voxels(shell(2, 9) | shell(4, 7), 11, 11, 11)
    r3 = find_cavities(vol, 11, 11, 11)
    show("=== 3) 嵌套壳（两个空腔） ===", r3)
    assert [c.voxel_count for c in r3.cavities] == [98, 1]
    assert r3.cavities[0].touching_components == (1, 2)
    assert r3.cavities[1].touching_components == (2,)

    # 4) 前景包围盒贴盒边界的开放凹槽：不得误判为闭腔
    block = {(z, y, x) for z in range(3) for y in range(3) for x in range(3)}
    block.discard((2, 1, 1))  # 顶面（贴盒边界）向下开 1 体素深坑
    vol = encode_voxels(block, 3, 3, 3)
    r4 = find_cavities(vol, 3, 3, 3)
    show("=== 4) 贴边开放凹槽（非空腔） ===", r4)
    assert r4.cavities == []

    # 5) 实际触发的拒绝：行内游程重叠
    print("=== 5) 非法输入边界（行内游程重叠） ===")
    try:
        find_cavities([[[(0, 3), (2, 4)]]], 1, 1, 4)
    except InvalidVolumeError as exc:
        print(f"  如预期拒绝：{type(exc).__name__}: {exc}")
    else:
        raise AssertionError("重叠游程未被拒绝")

    dt = time.perf_counter() - t0
    print()
    print(f"验算通过：1/0/2/0 空腔与预期一致。总耗时 {dt:.3f} 秒（现场计算）。")


if __name__ == "__main__":
    main()
