#!/usr/bin/env python3
"""runcc 固定输入演示：真实计算，不使用 sleep 或预录结果。

数据（确定性硬编码，nx=10 ny=10 nz=6）：
  核心游程   (z=0,y=2,x∈[2,7))            5 体素，面内连通
  边接触体素 (z=0,y=1,x=1)  与核心仅 |dx|=|dy|=1  → 18/26 连通、6 不连通
  角接触体素 (z=1,y=1,x=7)  与核心 |dx|=|dy|=|dz|=1 → 仅 26 连通
  孤立体素   (z=3,y=4,x=9)
因此组件数：6 邻接=4，18 邻接=3，26 邻接=2。
"""

import time

from runcc import Run, label_volume

DIMS = (10, 10, 6)
RUNS = [
    Run(0, 2, 2, 7),   # 核心游程
    Run(0, 1, 1, 2),   # 边接触体素
    Run(1, 1, 7, 8),   # 角接触体素
    Run(3, 4, 9, 10),  # 孤立体素
]


def main() -> None:
    print("=== runcc 游程连通组件演示 ===")
    print(f"dims (nx,ny,nz) = {DIMS}, runs = {[tuple(r) for r in RUNS]}")
    print()

    t0 = time.perf_counter()
    results = {c: label_volume(DIMS, RUNS, c) for c in (6, 18, 26)}
    elapsed = time.perf_counter() - t0

    for conn, result in results.items():
        kind = {6: "仅面", 18: "面+边（不含角）", 26: "面+边+角"}[conn]
        print(f"--- connectivity={conn}（{kind}）---")
        print(f"components: {result.num_components}")
        for comp in result.components:
            print(
                f"  id={comp.id} origin(z,y,x)={comp.origin} "
                f"voxels={comp.voxel_count} bbox={tuple(comp.bbox)} "
                f"exposed_area={comp.exposed_area}"
            )
        print(f"labels per run: {list(result.labels)}")
        print()
    print(f"三种邻接实际计算耗时: {elapsed * 1000:.2f} ms")
    print()

    print("=== 错误边界：NaN 坐标必须被拒绝（不静默纠正）===")
    bad = [Run(0, 2, 2, 7), Run(0, 1, float("nan"), 2)]
    try:
        label_volume(DIMS, bad, 6)
    except ValueError as exc:
        print(f"实际捕获 ValueError: {exc}")
    else:  # pragma: no cover - 演示用，测试单独保证该行为
        raise SystemExit("错误：NaN 输入未被拒绝")


if __name__ == "__main__":
    main()
