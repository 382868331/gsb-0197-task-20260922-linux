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


def reference_cavities(vol, depth, height, width):
    """独立空腔参考：前景 6 邻接、背景 26 邻接，逐体素 BFS。

    返回 list of dict：keys = min, voxels, bbox, touching（升序元组），
    按 min 升序。与盒外连通的背景（触及盒任一边界面）不算空腔。
    """
    voxels = expand(vol, depth, height, width)

    # 前景组件（6 邻接）：体素 -> 组件 ID（与 reference_label 同口径）
    _, fg_meta = reference_label(vol, depth, height, width, 6)
    fg_id = {}
    # 重新逐体素求前景组件 ID（直接 BFS，不复用 run_labels 输出）
    fg_seen = set()
    ordered = []
    for start in sorted(voxels):
        if start in fg_seen:
            continue
        q = deque([start])
        fg_seen.add(start)
        comp = set()
        while q:
            v = q.popleft()
            comp.add(v)
            z, y, x = v
            for dz, dy, dx in _NEIGHBORS[6]:
                w = (z + dz, y + dy, x + dx)
                if w in voxels and w not in fg_seen:
                    fg_seen.add(w)
                    q.append(w)
        ordered.append(comp)
    ordered.sort(key=min)
    for cid, comp in enumerate(ordered, start=1):
        for v in comp:
            fg_id[v] = cid
    assert len(ordered) == len(fg_meta)

    # 背景成分（26 邻接，限制在盒内）
    bg_seen = set()
    cavities = []
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                seed = (z, y, x)
                if seed in voxels or seed in bg_seen:
                    continue
                q = deque([seed])
                bg_seen.add(seed)
                comp = set()
                touches_outside = False
                while q:
                    v = q.popleft()
                    comp.add(v)
                    vz, vy, vx = v
                    if (vz == 0 or vz == depth - 1 or vy == 0
                            or vy == height - 1 or vx == 0 or vx == width - 1):
                        touches_outside = True
                    for dz, dy, dx in _NEIGHBORS[26]:
                        w = (vz + dz, vy + dy, vx + dx)
                        if (w in bg_seen or w in voxels
                                or not (0 <= w[0] < depth
                                        and 0 <= w[1] < height
                                        and 0 <= w[2] < width)):
                            continue
                        bg_seen.add(w)
                        q.append(w)
                if touches_outside:
                    continue
                touching = set()
                for (vz, vy, vx) in comp:
                    for dz, dy, dx in _NEIGHBORS[6]:
                        w = (vz + dz, vy + dy, vx + dx)
                        if w in fg_id:
                            touching.add(fg_id[w])
                zs = [v[0] for v in comp]
                ys = [v[1] for v in comp]
                xs = [v[2] for v in comp]
                cavities.append({
                    "min": min(comp),
                    "voxels": len(comp),
                    "bbox": (min(zs), min(ys), min(xs),
                             max(zs) + 1, max(ys) + 1, max(xs) + 1),
                    "touching": tuple(sorted(touching)),
                })
    cavities.sort(key=lambda c: c["min"])
    return cavities
