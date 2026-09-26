# 参考文献索引（L2 层与知识库）

> 本项目在 2026-09-26 的 L2 设计、知识库第 ② 层和检索算法中实际用到的文章。每条写明**用在哪里**和**怎么核实的**；
> 数值型先验的完整条目（与本语料的对照）在 `priors.json`。只列真正查过的来源，未核实的数字不写进先验。

## 数值先验（知识库第 ② 层，`priors.json`）

| # | 文献 | 用在哪里 | 核实方式 |
|---|---|---|---|
| 1 | Bentivoglio AR, Bressman SB, Cassetta E, Carretta D, Tonali P, Albanese A (1997). *Analysis of blink rate patterns in normal subjects.* Movement Disorders 12(6):1028–1034. [PubMed](https://pubmed.ncbi.nlm.nih.gov/9399231/) | `blink` 频率先验：静息 17 次/分，对话 26 次/分，阅读 4.5 次/分。本语料中位 28.4 次/分，与对话值一致 | 摘要（网络检索） |
| 2 | Baker RS, Abou-Jaoude ES, Napier SM (2005). *Kinematic Comparison of Spontaneously Generated Blinks and Voluntary Blinks in Normal Adult Subjects.* [SAGE](https://journals.sagepub.com/doi/10.1177/074880680502200105) | `blink` 时长：只作记录——各研究对"时长"的定义不同（该文自发眨眼 77.6 ± 10.0 ms），**不作检查标准** | 摘要；77.6 ms 对应哪个阶段未从全文确认 |
| 3 | Ekman P, Davidson RJ, Friesen WV (1990). *The Duchenne smile: emotional expression and brain physiology. II.* Journal of Personality and Social Psychology. [Semantic Scholar](https://www.semanticscholar.org/paper/The-Duchenne-smile:-emotional-expression-and-brain-Ekman-Davidson/2d3fa735944906bf59e0406ff3988062d13d2db6) | `smile` 的 FACS 定义：真笑 = AU6 + AU12。本语料 cheekSquint（≈AU6）从未亮起 → 按 D5 以语料为准，smile 只用 AU12 通道 | 标题与 AU6+12 定义（网络检索） |
| 4 | *Head Nodding and Hand Coordination Across Dyads in Different Conversational Contexts* (2023). Research Square 预印本. [链接](https://www.researchsquare.com/article/rs-3526068/v1) | `head_move`：快点头 > 1.5 Hz（多为倾听时的反馈）与慢点头 < 1.5 Hz 之分；将来给 head_move 加重复次数/频率参数时参考 | 摘要级（网络检索）；**未经同行评审** |
| 5 | Hadar U, Steiner TJ, Rose FC (1985). *Head movement during listening turns in conversation.* Journal of Nonverbal Behavior. | `head_move` 上下文：说话时头动多于倾听时；TalkVid 多为说话方 | 存在性与摘要（网络检索）；全文未读 |

## 算法设计来源（实例检索，计划 §6）

| # | 文献 | 借鉴了什么 | 核实方式 |
|---|---|---|---|
| 6 | Hunt AJ, Black AW (1996). *Unit selection in a concatenative speech synthesis system using a large speech database.* Proc. ICASSP-96, Atlanta, pp. 373–376. [Semantic Scholar](https://www.semanticscholar.org/paper/Unit-selection-in-a-concatenative-speech-synthesis-Hunt-Black/1dd0140d51e870a713340ae30734c8438b03d1a3) | 目标代价 + 衔接代价、在候选网络上最小化总代价（我们的 §6.3 目标代价与 §6.4 Viterbi） | 出处与方法描述（网络检索） |
| 7 | Clavet S (2016). *Motion Matching and The Road to Next-Gen Animation.* GDC 2016, Ubisoft Montréal. [GDC Vault](https://gdcvault.com/play/1023280/Motion-Matching-and-The-Road) · [slides](https://media.gdcvault.com/gdc2016/Presentations/Clavet_Simon_MotionMatching.pdf) | 在长段真实数据上做"特征向量 + 最近邻"检索并短过渡拼接；在长数据上打少量结构化标注（对应我们在 L1 轨道上标短语） | 出处与方法描述（网络检索） |

## 仓库里已有的其他引用

- Zhou Y, Barnes C, Lu J, Yang J, Li H (2019). *On the Continuity of Rotation Representations in Neural Networks.* CVPR —— 6D 旋转表示（`src/geometry/headpose.py`，早于本轮）。
- FACS 动作单元编号（AU1/2/12/41/42/43/45/51–56/61–64）写在 `schemas/l2_vocabulary.json` 各短语的 `facs` 字段。
