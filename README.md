# 稀疏体数据的游程连通组件（RLE Connected Components）

离线、纯标准库的可复用 Python 库：对**三维二值体按行压缩的半开 x 游程**做
连通组件标记（connected-component labeling），输出每条游程的组件 ID 以及
每组件的体素数、轴对齐包围盒、暴露面面积。不展开整个体素，任何时刻只保留
当前切片与前一切片的游程。

在无状态一次性标记之上，另提供**有状态模型** `ComponentModel`：支持
批量增删体素、删除桥接后自动重新计算连通性，并报告组件的**分裂（split）/
合并（merge）事件**（见下文“有状态动态层”一节）。

## 运行环境

* Python **3.14.7**（仅使用 3.14 标准库；不依赖任何第三方包，无需网络）。
* Linux 原生运行；装好 Python 后无需 `pip install` 任何东西。
  仓库里的 `requirements.txt` 无实际依赖内容。

## 准确命令

在仓库根目录执行：

```bash
# 必要测试（独立 BFS 参考 + 手算小例 + 错误语义 + 动态层随机差分）
python -m unittest discover -s tests -v

# 固定输入演示：三种邻接的不同结果 + 一个被实际触发的非法输入拒绝
python demo.py

# 动态层演示：桥断开（split）、重新连接（merge）、空输入/边界与实际错误
python demo_dynamic.py
```

demo 只做真实小规模计算，两个演示实测均约 0.04 秒（远小于 90 秒预算），
不 sleep、不读外部数据、不打印预录结果。

## 公开接口

```python
# 无状态一次性标记
from rle_components import label, Run, BoundingBox, Component, LabelResult
result = label(runs, depth, height, width, connectivity=6)

# 有状态动态层：批量增删 + 分裂/合并事件
from rle_components import (
    ComponentModel, UpdateResult, SplitEvent, MergeEvent,
    ComponentSnapshot,
)
model = ComponentModel(depth, height, width, connectivity=6, voxels=())
result = model.add([(0, 0, 3)])            # 批量新增
result = model.remove([(0, 0, 3)])         # 批量删除
result = model.update(add=[...], remove=[...])  # 一次提交既增又删
model.reset()                              # 清空（空输入合法）
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

## 有状态动态层（ComponentModel）

`ComponentModel(depth, height, width, connectivity=6, voxels=())` 在固定
尺寸与连通性下保存当前体素集合，提供批量增删。`connectivity` 在模型生命
周期内固定，语义与 `label` 完全一致（6/18/26）。

**批次格式**：一个可迭代批，每个元素是

* `(z, y, x)`：单个体素；或
* `(z, y, start, end)`：固定行的半开 x 游程 `[start, end)`（同 `label`
  的区间口径，`0 <= start < end <= width`）。

同批中重复指定同一体素按集合语义幂等去重，不报错；`add` 已存在的体素、
`remove` 不存在的体素都幂等跳过（不计入 `UpdateResult.added/removed`）。

**提交方法**：

| 方法 | 行为 |
| --- | --- |
| `add(batch)` | 批量新增后重算，等价于 `update(add=batch)` |
| `remove(batch)` | 批量删除后重算，等价于 `update(remove=batch)` |
| `update(add=(), remove=())` | 一次提交混合增删；同一体素不能既增又删 |
| `reset(voxels=())` | 整体替换为给定体素，默认清空；空输入合法 |

每次提交后**全量重算**连通组件（本题明确允许），组件 ID 始终按组件最小
体素坐标 (z, y, x) 字典序 1 起始重新分配——ID 不保证跨更新稳定。

**查询**：`model.components`（`Component` 元组，与 `label` 返回同构）、
`model.voxels`（只读 `frozenset`）、`model.num_voxels`、`len(model)`、
`(z,y,x) in model`、`model.component_id_at(z,y,x)`（空体素抛
`KeyError`）、`model.label_result()`（按当前状态产出一份完整
`LabelResult`）。

### UpdateResult 与分裂/合并事件

`update/add/remove` 返回 `UpdateResult`：

| 字段 | 含义 |
| --- | --- |
| `added` / `removed` | 本次实际生效的新增 / 删除体素数 |
| `components` | 重算后的组件元组（与 `model.components` 同一对象） |
| `splits` | `SplitEvent` 元组，按受影响旧组件最小坐标升序 |
| `merges` | `MergeEvent` 元组，按合并后新组件最小坐标升序 |
| `connectivity` / `num_voxels` / `num_components` | 连通性 / 提交后体素数 / 组件数 |

事件按“旧组件集合 → 新组件集合”的体素归属变化定义：

* **split**：某个提交前组件的**存活**体素在提交后落到 ≥2 个新组件
  （删除桥接的典型情形）。事件含 `affected_components`（旧组件摘要）、
  `resulting_components`（新组件摘要列表）与 `added_voxels`
  （落入这些新组件的本次新增体素数；纯删除为 0，混合提交可非 0）。
* **merge**：某个提交后组件的**存活**体素来自 ≥2 个提交前组件（新增
  桥接的典型情形）。事件含 `merged_components`、`resulting_component`、
  `added_voxels` 与 `removed_voxels`（相关旧组件区域内本次删除数；
  纯新增为 0）。

不产生事件的情形：组件内部长大（在组件上贴体素）、从端点缩减、冗余环路
删一个连接后仍连通、幂等空操作，以及整个组件被一次性删光（旧组件 →
0 个新组件不是“一分为多”，故不算 split）。

`ComponentSnapshot` 是不可变组件摘要 `(min_z, min_y, min_x,
voxel_count)`，用于在 ID 重排后跨更新跟随同一组件；本层**不维护持久
组件谱系**（不在本题范围内）。

### 动态层错误语义与原子性

所有契约违背同样抛 `InvalidVolumeError`，消息带批次与下标前缀
（如 `add[2].start`、`remove: z=3 超出 depth=2`）：批不可迭代、元素
不是三元组/四元组、坐标非整数（bool/float 一律拒绝）、负坐标或越界、
空游程或反向游程、同一提交增删同一体素等。

**不留下半更新状态**：每笔提交先把两个批完整解析并校验（尺寸、冲突），
通过后才写体素集合；写入后若重算失败（例如提交后游程数超过 20000），
恢复提交前的体素、组件与体素→ID 映射快照后再把异常抛出。`reset`
同样先校验、失败保持现状。

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

动态层测试 `tests/test_dynamic.py` 同样不拿被测实现生成期望值：

* 每个随机序列步都把模型状态编码回 RLE，逐字段比对独立 BFS 参考
  （标签、体素数、包围盒、面积、ID）；
* split/merge 事件由测试文件内另一个直白的集合分区比较独立推导，
  再与模型返回的事件逐条比对；
* 手算小例覆盖题面点名场景：**桥断开 → split、重新连接 → merge、
  空输入**，另含整组件删光不算 split、冗余环路删一连不分裂、
  增删混合提交、18 边桥与 26 角桥、ID 重排、幂等等边界；
* 失败路径断言状态不变（形状/类型/越界/增删冲突、游程数 20000
  接受、20001 整笔拒绝并回滚）。

“核心结果自身往返一致”不作为正确性依据，只做补充断言（体素守恒、ID 连续
有序、面积非负偶数值等）。

## 已知限制（按本题契约的有限子集）

* 只实现本题规定的 6/18/26 三维栅格连通性与按行 x 游程输入，不宣称兼容
  ITK/VTK/`ndimage` 等行业库的全部标签选项、像素类型或流式协议。
* 不做体素展开后的形态学操作；无状态 `label` 不输出体素级标签体
  （设计上避免展开），动态层以体素坐标集合为修改单位。
* `label` 仍为一次性纯函数；增量能力由 `ComponentModel` 承担，但
  **每次提交都是全量重算**（本题允许），不保证局部更新复杂度。
* **不维护持久组件谱系**：组件 ID 每次提交后按最小坐标重排，跨更新
  跟随组件需用 `ComponentSnapshot`（最小坐标 + 体素数），不提供
  谱系 DAG、洞证据或跨代追踪。
* 尺寸与游程数上限为题面给定的 512 / 20000；动态层提交后游程数超限
  同样整笔拒绝、状态回滚，而非自动外存化。
* 动态层内部以体素坐标集合为权威状态（便于按坐标批量增删），其内存为
  O(当前体素数) 而非无状态 `label` 的 O(游程数)；无状态 `label`
  本身的流式低内存特性不受影响。
