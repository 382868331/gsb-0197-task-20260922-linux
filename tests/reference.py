"""独立参考实现：仅用于测试，自己展开小体素做 BFS 与逐面面积统计。

本模块刻意不 import runcc，所有期望值独立计算；
与被测核心的往返一致只作为补充断言。
"""

from collections import deque


def _neighbor_offsets(connectivity):
    offs = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                manhattan = abs(dx) + abs(dy) + abs(dz)
                if connectivity == 6:
                    if manhattan > 1:
                        continue
                elif connectivity == 18:
                    # 18 = 6 面 + 12 边（恰两个坐标非零），不含角。
                    if manhattan > 2:
                        continue
                # 26: 全部
                offs.append((dx, dy, dz))
    return offs


FACE_6 = _neighbor_offsets(6)


def reference_labeling(dims, runs, connectivity):
    """返回 (voxel_components, run_labels)。

    voxel_components: list of dict(voxels=frozenset, origin, voxel_count,
        bbox=(x0,y0,z0,x1,y1,z1), exposed_area)，按 origin 排序。
    run_labels: 与 runs 等长，空游程为 None，否则为 1 起始组件号。
    """
    nx, ny, nz = dims
    occupied: set[tuple[int, int, int]] = set()
    for z, y, x0, x1 in runs:
        for x in range(x0, x1):
            occupied.add((x, y, z))

    offsets = _neighbor_offsets(connectivity)
    seen: set[tuple[int, int, int]] = set()
    comps = []
    for start in occupied:
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        voxels = []
        while queue:
            x, y, z = queue.popleft()
            voxels.append((x, y, z))
            for dx, dy, dz in offsets:
                nb = (x + dx, y + dy, z + dz)
                if nb in occupied and nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        vset = frozenset(voxels)
        xs = [v[0] for v in voxels]
        ys = [v[1] for v in voxels]
        zs = [v[2] for v in voxels]
        # 逐面统计暴露面积：每个体素的 6 个面中朝向空体素/体边界的数量。
        area = 0
        for x, y, z in voxels:
            for dx, dy, dz in FACE_6:
                if (x + dx, y + dy, z + dz) not in occupied:
                    area += 1
        origin = min((z, y, x) for x, y, z in voxels)
        comps.append({
            "voxels": vset,
            "origin": origin,
            "voxel_count": len(voxels),
            "bbox": (min(xs), min(ys), min(zs),
                     max(xs) + 1, max(ys) + 1, max(zs) + 1),
            "exposed_area": area,
        })

    comps.sort(key=lambda c: c["origin"])
    voxel_to_comp = {}
    for cid, c in enumerate(comps, start=1):
        for v in c["voxels"]:
            voxel_to_comp[v] = cid

    run_labels = []
    for z, y, x0, x1 in runs:
        if x0 == x1:
            run_labels.append(None)
            continue
        run_labels.append(voxel_to_comp[(x0, y, z)])
    return comps, run_labels
