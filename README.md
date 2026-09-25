# facial-dynamics：表情的语义化 token 表示

> 让表情调整从"改参数"变成"改语义"。

## 我们要做什么

今天调一个表情，无论对象是 3D 数字人、动漫角色、游戏 NPC 还是机器人面部，做法几乎都一样：
去改一串只有专业人员看得懂的参数。比如 `jawOpen = 0.37`、`browInnerUp = 0.12`、
舵机 7 转到 42°，而且每种载体的参数体系各不相同。

我们想要的是一套**与载体无关、人类可解释、可直接编辑的表情语义表示**：

```
任意载体的表情 ──编码──▶ 语义 token（人可读、可修改） ──解码──▶ 表情参数 ──▶ 任意载体
 真人视频 / 3D / 动漫            "嘴 moderate 张开 0.3 s，                     
 游戏 / 机器人                    随后一次 strong 眨眼"                        
```

它要满足三件事：

1. **可解释**：中间表示是人能读懂的分段文本，不是一个隐向量。
2. **可修改**：直接改文本（例如把 `slight` 改成 `strong`，把眨眼拉长 0.1 s），解码后
   只有对应的通道和时间段跟着变，其他部分不受影响（解耦）。
3. **可逆且误差可归因**：编码后再解码，往返误差是被测量出来的，并且能拆分到每一层
   各自丢了多少信息。

最终目标是让 3D、动漫、游戏、机器人表情的调整都在语义层完成。改动先落到统一的表情参数上，
再重定向到具体载体。

## 总体架构

```
            ┌─────────────── 载体适配（输入） ───────────────┐
 真人视频 ──▶ Observation → Geometry → Features               │  ← 本仓库已实现
 3D / 游戏 ──▶ ARKit 风格 blendshape 直接映射                  │  ← 规划
 动漫 / 机器人 ──▶ 各自的参数 → 统一表情参数                   │  ← 规划
            └────────────────────┬──────────────────────────┘
                                 ▼
               统一表情参数：63 维 / 帧（schemas/geometry.schema.json）
               52 blendshape · 头部平移 3 · 头部旋转 6D · 注视 2
                                 ▼
   ┌──────────────────── 语义层（多层 token） ────────────────────┐
   │ L1 原子层    每个通道：强度档位 + 时间分段     规则，确定性    │ ✅ 已实现
   │ L2 动作短语  眨眼 / 微笑 / 点头 …（带时长、强度） 词表 + 模板   │ 📋 计划中
   │ L3 交流意图  情绪 · 场景 · 含义 · 幅度（并行轨道） 生成模型    │ 📋 计划中
   └──────────────────────────────┬───────────────────────────────┘
                                  ▼  解码（每一层都能完整回到 L1）
               统一表情参数 ──▶ 载体重定向（3D / 动漫 / 游戏 / 机器人） ← 规划
```

为什么要分三层：L3 到参数是多对一映射，单独看无法判断"对不对"。L1 和 L2 是可审计的
中间锚点：L3→L2 检验"意图到动作是否合理"，L2→L1 检验"动作到几何是否精确"，
L1→参数给出量化误差的下限。"改一个语义轴只影响预期的那部分通道"这一解耦主张，
也要靠这条证据链来验证。详见 [`docs/plans/multilayer_semantic_representation.md`](docs/plans/multilayer_semantic_representation.md)。

统一参数选用 ARKit 命名的 52 个 blendshape（MediaPipe 输出与之同名），是因为这套命名
在 3D 和游戏管线里已被广泛使用。后续接入这些载体时，可以直接复用同一套语义。

## L1 长什么样

一个 1 秒窗口（30 帧 @ 30 Hz）里一次眨眼的真实编码结果（验证 clip，窗口 @60）：

```json
"eyeBlinkLeft": [
  {"t": [0.0, 0.6],    "level": "slight",   "value_hint": 0.1261},
  {"t": [0.6, 0.8333], "level": "strong",   "value_hint": 0.563},
  {"t": [0.8333, 1.0], "level": "moderate", "value_hint": 0.2758}
]
```

- 档位固定为 `none / slight / moderate / strong`；有符号通道写作 `+slight`、`-moderate`。
- `level` 是编辑入口。改了档位，解码结果就跟着档位走；`value_hint` 只负责在档内细化强度。
- 头部旋转用相对于窗口参考姿态的旋转向量（x/y/z 联合分档），不用欧拉角。
- 所有阈值只写在 [`schemas/l1_rules.json`](schemas/l1_rules.json)，不从数据拟合，同一窗口
  永远得到同一段文本。

## 当前进展

| 模块 | 状态 | 说明 |
|---|---|---|
| 真人视频 → 63 维参数 | ✅ | 五阶段 pipeline，带质量掩码与逐帧置信度 |
| 音频伴随特征 | ✅ | 80 维 log-mel + 波形，按真实时间与视频窗口逐行对齐（不进 63 维） |
| L1 编码 / 解码 / 评测 | ✅ | 57 个文本通道，结构检查 7 项全过 |
| 多说话人数据 | 🟡 | TalkVid 206 位说话人、6245 个窗口（约 50 分钟），目标约 1500 个 clip |
| L2 动作短语 | 📋 | 规则种子词表（blink / jaw_open / smile / brow_raise / nod）→ 数据驱动扩充 |
| L3 交流意图 | 📋 | 先用弱标注验证轴间独立性，再定轴；生成模型在建模仓库训练 |
| 其他载体的输入适配与输出重定向 | 📋 | 3D / 动漫 / 游戏 / 机器人 |

L1 在 206 个 clip 上的往返误差（只统计被文本描述的帧）：blendshape 平均误差 0.0136，
头部旋转平均 0.92°，每通道每窗口平均 2.0 个分段。已知问题和待定事项见
[`docs/milestones/2026-09-23_l1_audio_talkvid.md`](docs/milestones/2026-09-23_l1_audio_talkvid.md)。
分阶段推进计划和每一步的闸门见 [`docs/plans/l2_l3_rollout.md`](docs/plans/l2_l3_rollout.md)。

## 仓库边界

本仓库产出**数据、语义层与评测**，不放模型训练代码。L3 生成模型、细节补全网络
（L1 解码结果 + 音频 → 原始参数）在建模仓库中训练，本仓库负责导出训练对和评测脚本。
`schemas/geometry.schema.json` 是和建模组之间的跨团队契约。

## 快速开始

环境仅锁定 linux-64（AICR HPC），见 `environment/env-v0.1/NOTES.md`。
在 HPC 上**不要在登录节点计算**，先用 `srun` 申请计算节点。

```bash
conda-lock install --name talkvid environment/env-v0.1/conda-lock.yml

# 视频 → 63 维参数 + 音频伴随特征（唯一入口，按阶段断点续跑）
python scripts/run_pipeline.py \
    --input_dir ../data/talkvid_bench \
    --model_path ../models/face_landmarker.task \
    --output_root ../runs

# 参数 → L1 文本 → 参数，并生成往返误差报告
python scripts/l1_encode.py --run_dir ../runs/<id> --output ../runs/<id>/l1/l1_text.json
python scripts/l1_decode.py --text ../runs/<id>/l1/l1_text.json --output ../runs/<id>/l1/l1_decoded.npz
python scripts/l1_report.py --run_dir ../runs/<id> --output_dir ../runs/<id>/l1 --text ../runs/<id>/l1/l1_text.json

# 测试（不需要视频和模型，< 1 s）
python -m pytest tests -q
```

## 目录

```
schemas/        跨团队契约与规则：geometry.schema.json（63 维布局）、l1_rules.json、audio_features.json
src/observation 时间轴解码与可信度判定（不读像素）
src/geometry    三条轨道：landmarks+blendshapes、头部姿态（6D 旋转）、注视
src/sequence    重采样、QC、切窗、训练契约（掩码与 loss 权重）
src/dynamics    基于真实 PTS 的速度 / 加速度校验
src/audio       音频伴随特征
src/semantic    L1 语义编解码与评测
scripts/        各阶段 CLI、run_pipeline.py、l1_*.py、fetch_talkvid.py
docs/           decisions/（为什么这样选）· engineering_log/（故障与根因）
                plans/（前瞻计划）· milestones/（阶段快照）；索引见 docs/README.md
```

## 设计原则（摘要）

每一条都来自一次真实的静默数据损坏，理由记录在 `docs/decisions/`：

- 时间一律来自真实 PTS，不用 `frame_idx / fps`；损坏的时间轴只报告、不修补。
- 旋转用 6D 表示，插值走流形，不用欧拉角或逐元素运算。
- 注视是独立轨道，不从 `eyeLook*` blendshape 推出。
- 四种掩码信号分开存储、使用时再组合；语义层只描述 `loss_weight() > 0` 的帧。
- 所有布局和阈值只来自 JSON，代码不重复写任何一个索引或阈值。
