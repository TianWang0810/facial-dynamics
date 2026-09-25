# 里程碑 2026-09-23：L1 文本层、音频接入、首批多说话人数据

> 一份带日期的快照，记录这一天做成了什么、验证到什么程度、数据在哪、还有什么没解决。
> 各项取舍的理由不在这里重复，见文中链接的 `decisions/` 与 `engineering_log/`。
> 代码全部**未提交**（分支 `Shuran`，HEAD 仍是 `24ffdd5`）。

---

## 1. 做成了什么

| 模块 | 内容 | 入口 | 理由记录 |
|---|---|---|---|
| **L1 文本层** | 63 维张量 ↔ 人可读、可编辑的分段文本；规则唯一真源 `schemas/l1_rules.json` | `scripts/l1_encode.py` / `l1_decode.py` / `l1_report.py` | `decisions/l1_text_codec.md` |
| **L1 解码后处理** | 段均值（落在档位区间内时）+ Whittaker 平滑，文本格式不变 | 同上（默认） | 同上 §3 |
| **音频接入** | 新阶段 `04_audio`：80 维 log-mel @100 Hz + 16 kHz 波形，按真实时间与视频窗口逐行对齐；不进 63 维 | `scripts/run_audio.py`（pipeline 默认包含） | `decisions/audio_conditioning_sidecar.md` |
| **数据获取** | TalkVid 抽样与下载（每说话人 1 段、视频+音频分别选 https 流） | `scripts/fetch_talkvid.py` | `engineering_log/talkvid_audio_download.md` |
| **前置修复** | blendshape 名字写入 `geometry_metadata.json`；`tail_frame_gap` 单列并只排除末帧；dynamics 对含小数点 clip id 的 bug；youtu.be 链接解析 | — | 同上 |

测试：**107 个全部通过**（原 64 + L1 31 + 音频 11 + 时间线 1），每个测试文件都可 pytest 或直接运行。

## 2. 数据现状

| 项 | 值 |
|---|---|
| 位置 | `/scratch/zhao_shur_neu/talkvid/`（**scratch 可能被自动清理**，小体积报告已备份到 `~/facial_dynamic/runs/_reports/talkvid-train207/`） |
| 已下载 | 207 个 clip / 207 位说话人，全部带音轨（AAC 44.1 kHz），817 MB |
| pipeline run | `runs/talkvid-train207__20260923T200526Z__24ffdd5`，5 个阶段全部 ok |
| 保留 | 206 / 207 个 clip（1 个人脸检测率不足），**6245 个窗口**，约 50 分钟，按说话人可划分（`--speaker_from parent_dir`） |
| 张量 | `features` (6245, 30, 63)；`audio.npz`: `mel` (6245, 100, 80)、`wave` (6245, 16000)、`audio_valid` (6245, 100) |
| 音频可用 | 206 / 206，有效率 99.2%（其余为 padding），无音画时长不一致 |
| 源帧率 | 30 fps 138 个、24 fps 39 个、25 fps 29 个 |
| 目标 | 约 1500 个 clip；**因 YouTube 机器人检测暂停**（见 §5） |

## 3. 验证到什么程度（实测数字）

### 3.1 音画同步（诊断，不作门槛）
jawOpen 与音频能量的互相关，±0.5 s 范围：

| | 16 个 clip 试下 | 206 个 clip |
|---|---|---|
| 池化峰值偏移（正 = 声音晚于嘴） | +110 ms | **+100 ms** |
| 峰值相关 / 0 偏移相关 | 0.137 / 0.004 | 0.098 / 0.033 |
| 错配对照组 | ≤ 0.04 | — |

单 clip 估计噪声较大（四分位 −62 ~ +160 ms），只看池化结果。

### 3.2 L1 往返误差（仅统计被文本描述的帧）

| 指标 | 1 个 clip（验证 clip） | **206 个 clip** |
|---|---|---|
| 文本量（段/通道/窗口） | 1.66 | 2.00 |
| blendshape 平均误差（旧阶梯解码 → 默认解码） | 0.023 → 0.0091 | 0.027 → **0.0136** |
| blendshape 最大误差 | 0.53 → 0.18 | 0.63 → **0.65** |
| 头部平移平均误差 | 0.092 → 0.044 | 0.336 → **0.140** |
| 旋转 平均 / 最大 | 1.36° / 4.43° → 0.90° / 2.33° | 2.00° / 18.5° → **0.92° / 11.6°** |
| 7 项结构检查 | 全部通过 | **全部通过** |

结构检查包括：只做同档连段时完全无损；分段后任何帧偏离不超过一档；段均值都落在所属档位区间内；四种 mask 往返精确；`loss_weight()` 在解码结果上复现原描述范围；跨进程编码结果完全一致。

### 3.3 自然度（仅验证 clip，blendshape/旋转空间）
以"去掉逐帧抖动后的原始数据"为 100%：默认解码保留嘴部运动 57%、眼 62%、眉 62%、头部 56%，峰值 80%，左右同步 0.54（目标 0.63）。
曲线图：`runs/talkvid-age__20260923T182430Z__24ffdd5/l1/naturalness/curves.png`。
代理 rig 渲染已作废：单 clip 拟合的 rig 在留出帧上只能解释约 14% 的方差，不可信。

## 4. 没解决的问题

1. **L1 规则在多说话人上泛化不够**：blendshape 最大误差 0.65 没有被后处理压下来；平移误差是单 clip 的 3 倍，平移分档阈值是按单 clip 幅度定的。需要先定位最大误差窗口再调规则。
2. **自然度**：说话时嘴部小幅开合（jawOpen 0.01–0.03）全落在 none 档；峰值被削平；小幅点头丢失；左右不同步。已测试的改进（加 subtle 档 + 2 帧合并阈值 + 左右共享边界：嘴 75%、头 83%、峰值 89%、同步 0.62，文本量 1.9 倍）**尚未采用**，需决定。
3. **速度信息**：约一半原始速度是跟踪抖动（平滑原始数据后的速度误差下限约 0.07）。其余差距是段内真实细节，文本没有记录，需要更细的文本或残差/细节网络。
4. **没有可信的 avatar 渲染**，"看起来自然"目前只能靠曲线和指标判断。
5. `channel_loss_normalizer()` 仍把 `_neutral` 计入（52 而非 51），属于与建模组的训练契约，未改。

## 5. 待你决定的事项

| 事项 | 选项 | 建议 |
|---|---|---|
| 继续获取数据 | ① 单线程限速慢慢下；② 你的 YouTube 账号 cookies（有服务条款/封号风险）；③ 换 HDTF、MEAD 等可直接下载的数据集；④ 先用现有 206 个 | ④ 起步 + ① 在后台补 |
| L1 文本细化 | 采用 subtle 档 + 左右共享边界（文本量约 1.9 倍）/ 维持 4 档 | 采用 |
| 细节补全网络 | 放建模组仓库 / 本仓库破例 / 新仓库 | 建模组仓库；本仓库负责导出训练对与评测 |
| 代码提交 | 目前全部未提交 | 分批提交（pipeline 基础、L1、音频、数据获取） |

## 6. 建议的下一步

1. 定位 206 个 clip 上 L1 最大误差与平移误差的来源，修正规则后重跑报告。
2. 导出细节补全网络的训练对：输入 `l1_decoded.npz` + `audio.npz`，目标 `features.npz`，按说话人划分训练/验证/测试。
3. 把自然度指标（运动保留率、峰值、左右同步、重新编码一致率、编辑局部性）做成正式评测脚本。
4. 按确定的方式补齐数据到约 1500 个 clip。

## 附：本次涉及的文件

- 新增：`schemas/l1_rules.json`、`schemas/audio_features.json`、`src/semantic/`、`src/audio/`、`scripts/l1_*.py`、`scripts/run_audio.py`、`scripts/fetch_talkvid.py`、`tests/test_l1_codec.py`、`tests/test_audio_features.py`、`docs/decisions/l1_text_codec.md`、`docs/decisions/audio_conditioning_sidecar.md`、`docs/engineering_log/talkvid_audio_download.md`、`docs/plans/l2_l3_rollout.md`、本文件
- 修改：`src/observation/decode.py`、`src/observation/merge.py`、`src/geometry/summarize.py`、`src/pipeline/layout.py`、`scripts/run_features.py`、`scripts/run_geometry.py`、`scripts/run_dynamics.py`、`scripts/run_pipeline.py`、`tests/test_observation_schema.py`、`tests/test_timeline_and_ids.py`
- 实验脚本（未入库）：`~/facial_dynamic/runs/_scratch/`（解码方案对比、自然度分析、sync 检验等）
