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


def expand_materials(vol, depth, height, width):
    """vol: [z][y] -> [(s, e, material), ...]；返回 {(z,y,x): material}。"""
    voxels = {}
    for z in range(depth):
        for y in range(height):
            for s, e, m in vol[z][y]:
                for x in range(s, e):
                    voxels[(z, y, x)] = m
    return voxels


def reference_label_materials(vol, depth, height, width, connectivity):
    """多材料参考：BFS 只走同材料邻居；逐体素六方向数面。

    返回 (run_labels, meta, contacts)：
      * meta: 按 (minz,miny,minx) 升序的 dict(material, voxels, bbox, area, min)
      * contacts: [(cid_a, cid_b, area), ...]，cid_a < cid_b，按对升序
    """
    voxels = expand_materials(vol, depth, height, width)
    nbrs = _NEIGHBORS[connectivity]
    comps = []
    seen = set()
    for start in sorted(voxels):
        if start in seen:
            continue
        mat = voxels[start]
        q = deque([start])
        seen.add(start)
        comp = set()
        while q:
            v = q.popleft()
            comp.add(v)
            z, y, x = v
            for dz, dy, dx in nbrs:
                w = (z + dz, y + dy, x + dx)
                if w in voxels and w not in seen and voxels[w] == mat:
                    seen.add(w)
                    q.append(w)
        comps.append(comp)

    comp_of = {}
    for idx, comp in enumerate(comps):
        for v in comp:
            comp_of[v] = idx

    face_nbrs = _NEIGHBORS[6]
    contact = {}
    items = []
    for idx, comp in enumerate(comps):
        exposed = 0
        for (z, y, x) in comp:
            for dz, dy, dx in face_nbrs:
                w = (z + dz, y + dy, x + dx)
                j = comp_of.get(w)
                if j is None:
                    exposed += 1
                elif j != idx:
                    key = (idx, j) if idx < j else (j, idx)
                    contact[key] = contact.get(key, 0) + 1
        zs = [v[0] for v in comp]
        ys = [v[1] for v in comp]
        xs = [v[2] for v in comp]
        items.append((comp, {
            "min": min(comp),
            "material": voxels[next(iter(comp))],
            "voxels": len(comp),
            "bbox": (min(zs), min(ys), min(xs),
                     max(zs) + 1, max(ys) + 1, max(xs) + 1),
            "area": exposed,
        }))

    # 每个共享面从两侧各计一次
    contact = {k: c // 2 for k, c in contact.items()}

    order = sorted(range(len(items)), key=lambda i: items[i][1]["min"])
    old_to_cid = {old: cid for cid, old in enumerate(order, start=1)}
    meta = [items[old][1] for old in order]

    contacts = sorted(
        (old_to_cid[a], old_to_cid[b], area)
        for (a, b), area in contact.items()
    )

    voxel_to_cid = {}
    for old, comp in enumerate(comps):
        for v in comp:
            voxel_to_cid[v] = old_to_cid[old]

    run_labels = []
    for z in range(depth):
        sl = []
        for y in range(height):
            row = []
            for s, e, m in vol[z][y]:
                row.append(voxel_to_cid[(z, y, s)])
            sl.append(row)
        run_labels.append(sl)

    return run_labels, meta, contacts
