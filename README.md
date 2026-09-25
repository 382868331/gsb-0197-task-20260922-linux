# 稀疏体数据的游程连通组件（RLE Connected Components）

离线、纯标准库的可复用 Python 库：对**三维二值体按行压缩的半开 x 游程**做
连通组件标记（connected-component labeling），输出每条游程的组件 ID 以及
每组件的体素数、轴对齐包围盒、暴露面面积。不展开整个体素：

* 批式 `label()`：任何时刻只保留当前切片与前一切片的游程；
* 流式 `StreamLabeler`：逐切片 `feed_slice` 喂入，组件一旦不接触最新
  切片即封闭——输出终结记录并释放其活跃合并状态，活跃内存只与边界切片
  及未封闭组件相关；`finish` 结算尾部并给出最终 ID 映射。

## 运行环境

* Python **3.14.7**（仅使用 3.14 标准库；不依赖任何第三方包，无需网络）。
* Linux 原生运行；装好 Python 后无需 `pip install` 任何东西。
  仓库里的 `requirements.txt` 无实际依赖内容。

## 准确命令

在仓库根目录执行：

```bash
# 必要测试（独立 BFS 参考 + 手算小例 + 流式场景 + 错误语义）
./.venv/bin/python -m unittest discover -s tests -v

# 固定输入演示：三种邻接的不同结果 + 流式逐切片封闭 + 非法输入拒绝
./.venv/bin/python demo.py
```

demo 只做真实小规模计算，实测约 0.07 秒（远小于 90 秒），不 sleep、
不读外部数据、不打印预录结果。

## 公开接口

```python
from rle_components import label, Run, BoundingBox, Component, LabelResult

result = label(runs, depth, height, width, connectivity=6)
```

* `runs`：`[z][y] -> [(start, end), ...]`。固定 (z, y) 的一行内，游程为
  沿 x 轴的**半开区间 `[start, end)`**；每行游程必须按 `start` 升序、
  互不重叠（端点恰好相接合法，重叠非法）。游程元素可为
  `(start, end)` 元组/列表、`Run(start, end)` 或
  `{"start": int, "end": int}` 字典。空行用 `[]` 表示。
* `depth / height / width`：z / y / x 方向尺寸，整数，均在 `[1, 512]`。
* `connectivity`：
  * `6`：仅面接触（坐标差 L1 = 1）；
  * `18`：面 + 边（恰两个坐标差为 1），**不含角**；
  * `26`：面 + 边 + 角（三个坐标差均为 1）。

返回 `LabelResult`（frozen dataclass）：

| 字段 | 含义 |
| --- | --- |
| `run_labels` | 与输入同形 `[z][y] -> [component_id, ...]`，顺序对应该行游程 |
| `components` | `Component` 列表，ID 为 1 起始的连续整数 |
| `connectivity` / `num_runs` | 实际连通性 / 输入游程总数 |

`Component` 字段：`id`、`voxel_count`（体素数）、`bbox`（半开
`BoundingBox(min_z,min_y,min_x,max_z,max_y,max_x)`，各区间 `[min,max)`）、
`exposed_area`（暴露面面积，单位为体素面）、`min_z/min_y/min_x`（组件按
字典序最小的体素坐标，即 ID 排序键）。

组件 ID 按组件最小体素坐标 **(z, y, x) 字典序**升序分配。注意该排序键不
必然等于几何包围盒的 `(min_z,min_y,min_x)`：在 18/26 邻接下，几何上
`y`/`x` 更小的体素可能只出现在更大的 `z` 切片（测试中明确覆盖此区别）。

## 流式接口

    from rle_components import StreamLabeler

    st = StreamLabeler(height, width, connectivity=6)
    for slice_rows in slices:               # z 从 0 开始连续递增，一次一片
        out = st.feed_slice(slice_rows)     # 空切片也必须显式传 [[]] * height
        for rec in out.closed:              # 本切片处理后被判定封闭的组件
            ...                             # ClosedComponent（临时 ID + 最终统计）
    final = st.finish()                     # 结算最后切片，只能调一次

* `feed_slice(slice_rows) -> SliceResult`：
  * `z`：本次切片下标（0 起始，每次调用隐式 +1，不可跳层——空切片必须
    显式传入，否则“不接触最新切片即封闭”的判定会被破坏）；
  * `run_labels`：本切片每行游程所属组件的**稳定临时 ID**；
  * `closed`：本切片封闭的 `ClosedComponent` 列表。组件不再接触最新
    切片时，它不可能再与未来任何切片相连，此刻体素数 / 包围盒 / 暴露面积
    即最终确定，随终结记录交付，活跃合并状态立即释放。
* `finish() -> StreamResult`：封闭全部剩余组件（最后切片的结算），返回
  * `components`：最终 `Component` 列表，ID 1 起始、按最小 (z,y,x)
    升序，**与批式 `label` 的编号口径完全一致**；
  * `id_of_temp`：临时 ID -> 最终 ID 的完整映射（含被合并吸收的临时
    身份别名）。临时 ID 只按创建次序分配、终身稳定，**不按提前结束的
    次序冒充最终 ID**；
  * `connectivity` / `num_runs`。
* 监控用：`active_node_count()` / `active_component_count()` 返回当前
  活跃并查集节点数 / 未封闭组件数，可验证活跃状态只随边界与未封闭组件
  增长，而不随已喂切片总数增长。

流式阶段的限制与批式相同：`height`/`width` ∈ [1, 512]，切片数
`<= 512`，累计游程 `<= 20000`；`finish` 后再 `feed_slice` 或重复
`finish` 均抛 `InvalidVolumeError`。单次 `feed_slice` 校验失败时不
产生部分变更（z 计数不前进），修正后可继续喂入。

## 输入上限（接口边界）

* `depth`、`height`、`width` 各 `<= 512`；
* 游程总数 `<= 20000`；
* 坐标必须为整数，`0 <= start < end <= width`。

这些是硬接口边界，超限一律拒绝；演示与测试不穷举上限规模，但包含
恰为 20000（接受）与 20001（拒绝）的边界测试。

## 错误语义

所有契约违背抛 `rle_components.InvalidVolumeError`（`ValueError` 子类），
消息带可定位前缀（如 `z=0 y=1 run=2.start`），**不静默纠正、不截断**：

* 数值坐标拒绝 `NaN` / `+Infinity` / `-Infinity`；
* 整数参数（尺寸、坐标、connectivity）拒绝 `bool`（`True/False` 不充当
  1/0），也拒绝 `6.0` 这样的 float——必须是真正的 `int`；
* 尺寸越界、connectivity 不是 6/18/26、空游程、坐标越界、行内游程重叠或
  乱序、嵌套结构尺寸与 depth/height 不符、游程数超 20000 等均拒绝。

`label` 是无状态纯函数：输入要么完整通过校验后一次性产出结果，要么抛异常，
不存在部分结果或部分变更。

## 算法与设计取舍

1. **扫描 + 并查集**。按 z 递增扫描；每条游程在 DSU 中建一个节点，与
   * 同行前一条游程（仅端点相接时合并，x 方向面接触）、
   * 同切片 `y-1` 行（面/边）、
   * **前一切片的 `y-1`、`y`、`y+1` 三行**（面/边/角）
   做半开区间触达判定后合并。跨切片必须扫三行：(当前 y, 前片 y+1) 这类
   游程对无法靠处理其他行覆盖（两行分属不同切片的不同行）；同切片 y 方向
   则只需扫 y-1，行对 `(y-1,y)` 在处理 y 时恰好计一次。
2. **区间扫描线性化**。行内游程有序，配对时用单调游标跳过不可能触达的
   游程，单行配对为 `O(len(a)+len(b)+触达对数)`，整体近似
   `O(R + 触达游程对)`，`R` 为游程数；区间重叠长度直接给出共享面的
   体素列数。
3. **不展开体素**。内存只与游程数和两个切片相关（DSU 每节点常数个数组：
   父指针、体素数、z/y/x0/x1）。20000 游程量级下实测 26 邻接全稀疏体
   约 0.14 秒。
4. **暴露面积精确计数**。扫描时在 x/y/z 三个面方向各把每个面邻居计一次
   （同游程内部 `length-1` 个面、行内相邻、行间/切片间重叠列），
   `exposed_area = 6 * voxel_count - 2 * internal_face_count`。
   边/角接触不共享体素面，故不影响面积（三种连通性下面积定义一致）。
5. **排序键与包围盒分离聚合**。节点按 (z,y,行内序) 创建，DSU 根第一次
   被遇到的位置即组件字典序最小体素；几何包围盒独立取各轴 min/max。
6. **流式封闭即释放**。流式路径用 dict 版并查集，节点只保留最近切片的
   边界游程；每片处理完先交付封闭组件的终结记录，再把幸存组件的根重挂
   到最新切片节点上并丢弃旧节点。面积在封闭时结算
   （`6*voxels - 2*faces`），最终 ID 统一由 `finish` 按最小坐标排序
   映射，提前结束的组件不会抢占小编号。

## 数值口径 / 误差

模块的全部输出（体素数、包围盒、面积、ID）都是**精确整数运算，不产生任何
浮点结果**，因此不存在浮点误差或容差问题。唯一接触浮点的地方是输入校验：
显式拒绝 NaN/Infinity，且 float 坐标（即使是 `6.0` 这种整数值 float）
不被当作合法整数坐标。

## 测试策略与独立性

`tests/reference.py` 是**完全独立**的参考实现：把游程展开成体素集合后用
BFS 求组件、逐体素逐方向数暴露面；它不导入、不调用 `rle_components` 的
任何代码。`tests/test_core.py` 的随机对照用例以参考实现的输出为期望值
（组件标签、体素数、包围盒、面积逐字段比对），另含手算小例和题面点名场景：

* **仅角接触**：`(0,0,0)-(1,1,1)-(2,2,2)` 对角链，6/18 下 3 个组件、
  26 下 1 个；
* **空行/空切片/空体**；
* **跨 slice 桥接**：经边接触（dz=1,dy=1）在 18/26 下连通、6 下断开。

`tests/test_stream.py` 覆盖流式接口：随机小体与批式 `label` 及独立 BFS
参考逐字段对照（含每游程临时 ID 经 `id_of_temp` 映射后等于最终 ID）、
长组件夹短组件（短组件提前封闭交付）、整层空白显式传入、斜角跨片对角链、
单条 500 体素长游程、活跃状态不随深度累积（`active_node_count` 恒有界）、
以及 finish 后喂入 / 重复 finish / 超限等错误语义。

“核心结果自身往返一致”不作为正确性依据，只做补充断言（体素守恒、ID 连续
有序、面积非负偶数值等）。

## 已知限制（按本题契约的有限子集）

* 只实现本题规定的 6/18/26 三维栅格连通性与按行 x 游程输入，不宣称兼容
  ITK/VTK/`ndimage` 等行业库的全部标签选项、像素类型或流式协议。
* 不做体素展开后的形态学操作、不输出体素级标签体（设计上避免展开）。
* 流式接口只沿 z 轴单方向前进：切片必须按 z 连续递增喂入（空切片显式
  传入），不支持乱序、回退或跳层；批式 `label` 仍需一次性给出全部切片。
* 尺寸与游程数上限为题面给定的 512 / 20000；超过即拒绝而非自动外存化。
