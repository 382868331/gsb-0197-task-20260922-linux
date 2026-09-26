"""测试辅助：完全独立的参考实现。

不导入也不调用 rle_components 的任何代码。对小规模体数据：
  * 把游程展开成体素集合，用 collections.deque 做 BFS 求连通组件；
  * 逐体素逐方向数暴露面得到面积；
  * 组件按最小 (z, y, x) 排序赋 ID。
仅用于测试（输入规模很小），故意写得直白。
"""

from collections import deque


def expand(vol, depth, height, width):
    """vol: [z][y] -> [(s, e), ...]；返回体素 set。"""
    voxels = set()
    for z in range(depth):
        for y in range(height):
            for s, e in vol[z][y]:
                for x in range(s, e):
                    voxels.add((z, y, x))
    return voxels


_NEIGHBORS = {
    6: [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if abs(dz) + abs(dy) + abs(dx) == 1],
    18: [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1)
         for dx in (-1, 0, 1)
         if 0 < abs(dz) + abs(dy) + abs(dx) <= 2],
    26: [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1)
         for dx in (-1, 0, 1)
         if not (dz == 0 and dy == 0 and dx == 0)],
}


def reference_label(vol, depth, height, width, connectivity):
    """返回 (run_labels, components_dict_by_mincoord)。

    components: dict (minz,miny,minx) -> dict(voxels, bbox, area)
    run_labels: [z][y] -> [cid,...]，cid 按 (minz,miny,minx) 升序，1 起始。
    """
    voxels = expand(vol, depth, height, width)
    nbrs = _NEIGHBORS[connectivity]
    comps = []  # list of sets
    seen = set()
    for start in sorted(voxels):
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        comp = set()
        while q:
            v = q.popleft()
            comp.add(v)
            z, y, x = v
            for dz, dy, dx in nbrs:
                w = (z + dz, y + dy, x + dx)
                if w in voxels and w not in seen:
                    seen.add(w)
                    q.append(w)
        comps.append(comp)

    face_nbrs = _NEIGHBORS[6]
    items = []
    for comp in comps:
        mn = min(comp)
        exposed = 0
        for (z, y, x) in comp:
            for dz, dy, dx in face_nbrs:
                if (z + dz, y + dy, x + dx) not in comp:
                    exposed += 1
        zs = [v[0] for v in comp]
        ys = [v[1] for v in comp]
        xs = [v[2] for v in comp]
        items.append((comp, {
            "min": mn,
            "voxels": len(comp),
            "bbox": (min(zs), min(ys), min(xs),
                     max(zs) + 1, max(ys) + 1, max(xs) + 1),
            "area": exposed,
        }))

    items.sort(key=lambda item: item[1]["min"])
    meta = [m for _, m in items]

    voxel_to_cid = {}
    for cid, (comp, m) in enumerate(items, start=1):
        for v in comp:
            voxel_to_cid[v] = cid

    run_labels = []
    for z in range(depth):
        sl = []
        for y in range(height):
            row = []
            for s, e in vol[z][y]:
                row.append(voxel_to_cid[(z, y, s)])
            sl.append(row)
        run_labels.append(sl)

    return run_labels, meta


def reference_cavities(vol, depth, height, width, box):
    """独立空腔参考：体素展开 + BFS，不调用被测核心。

    box: (min_z, min_y, min_x, max_z, max_y, max_x) 半开区间。
    返回 list of dict(min, voxels, bbox, touches)，按最小 (z,y,x) 升序。
    touches 为盒内 6 邻接前景组件 ID（ID 规则：全 volume 前景 6 邻接
    BFS 后按最小体素字典序 1 起始，与 label 的排序键一致）。
    """
    voxels = expand(vol, depth, height, width)

    # 前景组件（6 邻接，全体数据范围）
    face_nbrs = _NEIGHBORS[6]
    comps = []
    seen = set()
    for start in sorted(voxels):
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        comp = set()
        while q:
            v = q.popleft()
            comp.add(v)
            z, y, x = v
            for dz, dy, dx in face_nbrs:
                w = (z + dz, y + dy, x + dx)
                if w in voxels and w not in seen:
                    seen.add(w)
                    q.append(w)
        comps.append(comp)
    comps.sort(key=min)
    voxel_to_cid = {}
    for cid, comp in enumerate(comps, start=1):
        for v in comp:
            voxel_to_cid[v] = cid

    z0, y0, x0, z1, y1, x1 = box
    bg_nbrs = _NEIGHBORS[26]

    def in_box(v):
        z, y, x = v
        return z0 <= z < z1 and y0 <= y < y1 and x0 <= x < x1

    cavities = []
    seen_bg = set()
    for z in range(z0, z1):
        for y in range(y0, y1):
            for x in range(x0, x1):
                seed = (z, y, x)
                if seed in voxels or seed in seen_bg:
                    continue
                q = deque([seed])
                seen_bg.add(seed)
                region = set()
                outside = False
                touches = set()
                while q:
                    v = q.popleft()
                    region.add(v)
                    vz, vy, vx = v
                    if (vz in (z0, z1 - 1) or vy in (y0, y1 - 1)
                            or vx in (x0, x1 - 1)):
                        outside = True
                    for dz, dy, dx in bg_nbrs:
                        w = (vz + dz, vy + dy, vx + dx)
                        if not in_box(w) or w in voxels or w in seen_bg:
                            continue
                        seen_bg.add(w)
                        q.append(w)
                    for dz, dy, dx in face_nbrs:
                        w = (vz + dz, vy + dy, vx + dx)
                        if in_box(w) and w in voxel_to_cid:
                            touches.add(voxel_to_cid[w])
                if not outside:
                    zs = [v[0] for v in region]
                    ys = [v[1] for v in region]
                    xs = [v[2] for v in region]
                    cavities.append({
                        "min": min(region),
                        "voxels": len(region),
                        "bbox": (min(zs), min(ys), min(xs),
                                 max(zs) + 1, max(ys) + 1, max(xs) + 1),
                        "touches": touches,
                    })
    cavities.sort(key=lambda c: c["min"])
    return cavities
