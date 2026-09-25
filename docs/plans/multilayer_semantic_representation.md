# 面部动态多层表示设计讨论纪要

*基于 Facial Dynamics Foundation Model Research Roadmap 与 facial-dynamics 仓库（63维张量契约）*

> **状态：L1 IMPLEMENTED（2026-09-23），L2/L3 PLANNED。** 2026-09-16 定为下一步计划。
> L1 的落地取舍与实测数字见 `docs/decisions/l1_text_codec.md`；L2/L3 的分步方案见
> `docs/plans/l2_l3_rollout.md`。
>
> **定位**：本文描述的三层结构**建立在**本仓库现有四阶段流水线产出的 63 维张量**之上**，
> 不修改 Observation / Geometry / Features / Dynamics 任何一层，也不改动
> `schemas/geometry.schema.json`。L1 的输入就是 `features.npz` 的 target 张量。
>
> **文档类型**：前瞻性计划，故不放在 `decisions/`（那里只记已做出的技术取舍）
> 或 `engineering_log/`（那里只记症状→根因→修复）。落地过程中产生的具体取舍
> 应另立 `decisions/` 记录，本文只保留计划本身。

---

## 0. 核心目标

最终要达到的效果：**用户只需调整 L3（语义层），就能改变面部表情的输出**，且调整某一个语义维度时，其余维度和无关的几何细节不应被意外扰动。

要证明的核心跳跃：

```
tensor(63维) → 多层人类可读/可编辑文本 token → 重建 tensor
```

这个跳跃要证明两件事：
1. 中间态是**多层、人类可理解、可调整**的；
2. 往返（encode→decode）在允许的信息损失范围内是**可控、可归因**的。

---

## 1. 三层结构的定位与区别

### 1.1 两种"多层"的本质区别

- **L1 / L2 的多层 = 空间并行（spatial/parallel）**
  同一时间轴上，不同身体部位/通道各自独立演化——眉毛、嘴部、头部姿态、注视，各是一条独立轨道。这天然对应仓库里已有的三条 track（landmarks+blendshapes、headpose、gaze）。

- **L3 的多层 = 功能并行（functional/semantic）**
  不是身体部位的划分，而是**对同一个脸部事件，从不同解释维度同时描述**：情绪、对话场景、表情含义、动作幅度。这四个是看待同一动作的正交视角，不是脸的不同部位。

### 1.2 各层定义

| 层级 | 内容 | 性质 |
|---|---|---|
| **L1** 原子层 | 每个通道离散化为强度分档（none/slight/moderate/strong）+ 时间分段 | 规则驱动，接近确定性可逆，**无需训练** |
| **L2** 动作短语层 | 识别常见共现通道组合，命名为动作（微笑、点头、眨眼），带时长/强度参数 | 组合但仍可解析的结构化文本，需要轻量分割/聚类 |
| **L3** 交流意图层 | 情绪 / 对话场景 / 表情含义 / 动作幅度，多条平行语义轨道 | 多对一压缩，天然有损，解码需生成模型 |

---

## 2. L1、L2 对 L3 的意义

1. **提供可验证锚点**：L3→tensor 单独看无法判断"对不对"（多对一映射）。经过 L2 中转后，可以分别检验 L3→L2（意图到动作是否合理）和 L2→L1（动作到几何是否精确），把主观的"看起来自然吗"拆成可打分的子问题。

2. **构成 disentanglement 的证据链**：核心主张"调整 L3 只影响可预测的 L1 子集"需要显式、可审计的中间层才能验证，而不是端到端黑箱里事后猜相关性。

3. **提供误差基线（下限）**：L1 的往返误差接近确定性下限。L2 的额外误差 = 动作聚类/分割引入的损失；L3 的额外误差 = 语义压缩本身的代价。三层误差可以分别归因，而不是笼统的"生成质量差"。

4. **作为 L3 训练的弱监督来源**：用 L1/L2 自动生成大量 `(tensor, 动作短语)` 标注对，训练 L3 的意图↔短语映射，摆脱人工语义标注的覆盖度瓶颈。

**风险提示**：如果 L2 的动作短语词表定义不好（例如"微笑"混杂了几种几何上差异很大的模式），L3 会继承这种模糊性。上 L3 之前应先做 L2 内部一致性检查（同一短语标签对应的 L1 几何模式是否足够一致）。

---

## 3. L3 多轴设计的关键问题（待验证假设）

目标函数形式：

```
tensor(t) = Decode( L2( emotion(t), scenario(t), meaning(t), amplitude(t) ) )
```

理想情况：固定其他轴，只调 amplitude，输出中**只有幅度相关通道被缩放**，动作内容、参与通道、时序结构不变。

三个需要提前验证、而非假设成立的问题：

1. **轴间是否真正正交？**
   情绪和幅度可能不是独立可加的——"愤怒轻微"和"愉悦轻微"在 L1 层面激活的通道子集可能完全不同。amplitude 更可能是对其他轴的**调制算子**，而非平级独立维度。scenario 和 meaning 也可能存在条件依赖（同一"赞同"在倾听/说话时几何实现不同）。

2. **各轨道时间粒度天然不同**
   情绪变化慢（秒~十几秒级），对话场景在话轮边界跳变，表情含义随语音韵律变化（更短窗口），动作幅度可能逐帧连续调制。若强行统一到同一采样窗口，会导致粗粒度轴冗余、细粒度轴欠采样。可能需要**分层时间轴**：每条语义轨道用原生时间分辨率，解码到 L2 时再插值对齐。

3. **amplitude 是全局标量还是附着在具体语义单元上？**
   用户可能想要"点头夸张、微笑不变"这种局部调节。若是这样，amplitude 不应是独立的第五条轨道，而应作为 meaning 轨道上每个语义单元自带的属性参数。

**建议**：在定死四个轴之前，先用现有/弱标注数据做**轴间独立性探索分析**——检查"含义相同、场景不同"或"情绪相同、幅度不同"切片下，L1 通道变化模式是否表现出预期的"只变一部分、其余不变"结构。若观测到强耦合，应相应调整轴设计（如把 amplitude 建模为条件参数），而非先定轴再发现纠缠。

---

## 4. L1 的具体实现方案

### 4.1 本质

一对**确定性规则映射**（无学习成分，这是与 L2/L3 的本质区别，也是它能作为"无损基线"的前提）：

```
encode: features.npz 的一个 window（30帧 × 62维有效通道）→ 结构化文本
decode: 结构化文本 → 重建的 window（30帧 × 62维）
```

### 4.2 中间格式：结构化分段文本（非自由语言）

按通道独立分段，每段标离散等级，示例：

```json
{
  "window_id": "...",
  "channels": {
    "browInnerUp": [
      {"t": [0.0, 0.3], "level": "slight",   "value_hint": 0.18},
      {"t": [0.3, 0.7], "level": "moderate", "value_hint": 0.45},
      {"t": [0.7, 1.0], "level": "none",     "value_hint": 0.02}
    ],
    "jawOpen": [
      {"t": [0.0, 1.0], "level": "moderate", "value_hint": 0.38}
    ],
    "headpose_rotation_6d": [ "..." ],
    "gaze": [ "..." ]
  }
}
```

选择结构化分段文本而非自由语言句子，是因为它同时满足：人可读、可编辑（改 level 或改分段起止即可重新生成）、完全确定性可解码。

### 4.3 必须明确的设计决策

1. **分段算法**：把连续曲线切成"近似恒定"段的方法（固定阈值触发变化点 / 滑窗均值+变化点检测）需要写死、可复现，不含随机性。

2. **量化等级到数值的映射**：decode 时应**主要依据 level**（离散标签）还原数值，`value_hint`（段内均值）更多是审计/调试信息——因为"可调整"针对的是用户能改的离散等级，而不是精确数值本身。

3. **旋转通道特殊处理**：`headpose_rotation_6d` 不能走 blendshape 那套逐元素数值分档逻辑。离散化和重建需走四元数 nlerp 路径，与仓库既有不变量一致（旋转永远不做 Euler/element-wise 插值）。

4. **只在 TARGET 空间操作**：encode 从 `geometry.parquet` / `features.npz` 的 target 值出发，decode 也重建到 target 空间；blendshape 的 `[-1,1]` 对称化映射只在下游训练消费 `x_input` 时应用，不进入 L1 往返流程。

5. **Mask 信号不进文本**：`is_padding`、`is_gap_filled` 帧不生成文本描述，对应时间区间在文本中直接缺省；decode 时用仓库既有的 `loss_weight()` 规则重新标记，不在文本层重新发明 mask 语义。

### 4.4 验收标准

L1 完成的判定不是"代码跑起来"，而是：

- encode→decode 的重建误差（用仓库已有的 velocity/accel/jerk 校验逻辑）应**只等于量化分档带来的误差**，不应引入分段算法本身的系统性偏差；
- 对比"仅数值分档、不做时间分段"（逐帧输出）与"分档+分段"两种方案的误差，量化分段步骤单独引入的额外损失；
- 抽查若干通道的文本描述是否符合直觉（如明显眨眼动作，`eyeBlinkLeft/Right` 的文本是否呈现"骤降-恢复"的分段结构）。

**交付物**：`encode.py` + `decode.py` 两个脚本、一份分档/分段规则的 schema 说明、一份往返误差报告。这三者齐备即视为 L1 完成，而非模糊的"感觉能用"。

---

## 5. 待推进的下一步

- [x] 在 `data/talkvid_bench/age/` 验证 clip 上跑通 L1 encode/decode 原型
  （`scripts/l1_encode.py` / `l1_decode.py`，规则 `schemas/l1_rules.json`）
- [x] 产出 L1 往返误差报告（分段 vs. 不分段对比）（`scripts/l1_report.py`，
  run `talkvid-age__20260923T182430Z__24ffdd5/l1/`）
- [ ] 若有粗略的情绪/场景标注数据，做 L3 轴间独立性的探索性分析
- [ ] 视 L1 结果调整 L2 的动作短语词表设计

---

## 附：本文引用到的仓库锚点

落地时对照这些位置，避免重新发明已有的约定。

| 本文提到的 | 仓库里的实际位置 |
|---|---|
| 63 维张量契约、通道切片 | `schemas/geometry.schema.json`（唯一真源）、`src/schema/feature_vector.py` |
| 三条 track | `src/geometry/landmarks.py`（A）、`headpose.py`（B）、`gaze.py`（C） |
| 30 帧 window、`is_padding` / `is_gap_filled` | `src/sequence/window.py`、`src/sequence/resample.py` |
| `loss_weight()` 规则 | `src/sequence/contract.py` |
| 四元数 nlerp（旋转插值） | `src/sequence/resample.py:interpolate_rotation_6d()` |
| velocity / accel / jerk 校验逻辑 | `src/dynamics/derive.py` |
| 52 个 blendshape 的索引对应表 | `CLAUDE.md`（HPC 本地笔记，未入库） |
| 验证 clip 路径 | 相对本仓库是 `../data/talkvid_bench/age/`（`data/` 是仓库的同级目录，且被 gitignore） |

两个与本文 §4 直接相关的既有事实：

- **"62 维有效通道"** 指的是 63 维中扣掉索引 0 的 `_neutral` —— 它是 MediaPipe 的占位类别，
  实测恒为 0（mean 1.3e-6, max 6.0e-6）。注意 `channel_loss_normalizer()` 目前仍按 52 计
  blendshape 的维数，这个不一致在动 L1 之前值得先定掉。
- **52 个 blendshape 的列名没有落盘**。`src/geometry/pipeline.py` 采集了 `blendshape_names`
  但 `geometry_metadata.json` 里没有这个字段，所以 parquet 的 52 列目前是纯位置索引。
  L1 的文本层要按通道名输出（`browInnerUp`、`jawOpen` …），这一项是前置依赖，需要先补上。
  **（2026-09-23 已补上：`geometry_metadata.json` → `tracks.A_landmarks_blendshapes.blendshape_names`。）**
