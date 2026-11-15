# ETPNav-MID360 项目结项报告

**项目状态：已结项 / 停止推进（2026-11-16）**

本项目曾探索将 ETPNav 的 dense RGB-D waypoint proposal 适配到 MID360-inspired sparse LiDAR geometry。经过 Habitat 受控实验、显式几何候选方案、zero-shot failure forensics 和 feature-alignment 检查后，决定不再继续 Stage 5 learned adapter、Isaac Sim 或 MID360 实机路线；后续主线转向 Room-Sign-Guided Navigation。

本结项保留已有代码、分支和实验记录，不把未完成的 adapter / Isaac / 实机计划描述成已交付能力。仓库保持公开，不删除历史分支，不将仓库标记为 GitHub Archived。

## 1. 项目问题与边界

研究问题：在尽量冻结 Original ETPNav 的 GraphMap、language/topological planner 和 execution 的前提下，研究 dense Depth 到 MID360-like sparse geometry 的 waypoint interface adaptation。

Habitat 中的 sparse depth 是由 dense depth 确定性构造的几何近似，不是 Livox MID360 原生扫描仿真；不包含完整非重复扫描时序、真实噪声、运动畸变、FAST-LIO2 定位误差或真实机器人约束。

## 2. 已形成的工程与实验成果

### 2.1 Original ETPNav baseline 与接口审计

- 梳理 12-view RGB/depth、CLIP 与 DD-PPO depth encoder、waypoint predictor、GraphMap、language-conditioned graph planner、Habitat executor 的职责边界。
- 将原始 learned waypoint proposal 与 MID360 geometry provider 分离，保留 downstream ten-field waypoint contract。
- 在五个固定公开场景 pose 上做 Original provider 与 refactored provider 对照；首次观测的候选数量、角度、距离、图像索引、候选特征和 panorama 特征最大绝对差均为 0.0。
- 这属于接口/数值一致性工程验证，不是导航 benchmark。

### 2.2 Explicit geometry waypoint experiments

实现并比较 Geometry V0、BNMS、MRU 与 G-Min。得到的关键工程观察：

- Geometry V0 的 long-range preference 产生明显 3.0 m 距离偏置；约 64% candidates 位于 3.0 m。
- BNMS 修正角度 circular / 距离 bounded 的 NMS 拓扑；Geometry V0 + BNMS 的 frozen-20 development-panel SR 从 0.50 到 0.55，但仍未恢复 Original baseline。
- MRU 将候选距离拉回中距离，却没有改善导航 SR；这说明只调整 radial distance prior 不足以恢复导航。
- G-Min 改善了部分 SPL、NE、OS、nDTW 和 SDTW，但 SR 仍为 0.50。
- GraphMap interface audit 显示 raw proposal diversity 不等于 graph action diversity：部分近距离 candidates 被 visited nodes 吸收。

### 2.3 MID360-like sparse-depth zero-shot 与 failure forensics

- 直接将 MID360-like sparse depth 输入 Original learned waypoint proposer，20/20 development episodes 失败；记录 SR=0、SPL=0、NE=9.108 m，candidate distance collapse 到 0.75 m。
- activation tracing 定位到 `vis_classifier` 后 ReLU：9216 个 activation 中为 0；depth encoder 前段仍保留约 23–24% activation。
- channel-wise mean/std Fixed Feature Alignment 在 5 个 held-out static observations 上未恢复 classifier post-ReLU activation 或 waypoint radial diversity。
- 这些结果是受控 development / static mechanism evidence，不是正式 R2R-CE benchmark，也不证明真实 MID360 传感器表现。

### 2.4 Isaac adapter design

仓库保留 RGBPanoramaProvider、GeometryProvider、PoseProvider、WaypointProvider、Executor 等接口契约，以及外部 VLN-CE-Isaac / Isaac Lab 的 source compatibility audit。**Isaac Sim runtime 没有安装或运行，原生 Isaac LiDAR、FAST-LIO2 和 MID360 实机均未完成。**

## 3. Frozen-20 development panel（非正式 benchmark）

以下数字来自 README 中同一组 frozen 20 val_unseen development episodes 的受控比较，不应写成论文复现结果或正式 benchmark：

| 方法 | SR | SPL | NE (m) | OS | nDTW | SDTW |
|---|---:|---:|---:|---:|---:|---:|
| Original ETPNav | 0.800 | 0.698 | 2.922 | 0.800 | 0.714 | 0.651 |
| Geometry V0 | 0.500 | 0.321 | 5.055 | 0.500 | 0.448 | 0.305 |
| Geometry V0 + BNMS | 0.550 | 0.364 | 4.526 | 0.600 | 0.467 | 0.356 |
| MRU | 0.450 | 0.365 | 4.730 | 0.550 | 0.534 | 0.351 |
| G-Min | 0.500 | 0.428 | 4.274 | 0.650 | 0.582 | 0.399 |
| Sparse Depth Zero-shot | 0.000 | 0.000 | 9.108 | 0.000 | 0.281 | 0.000 |

## 4. Portfolio 工程展示素材（历史证据后处理）

为个人 Portfolio 准备了可追溯的展示素材，见 [Portfolio Evidence Pack](portfolio-evidence/README.md)：

- episode 53 同一观测的 Original dense-depth vs MID360-like sparse-depth waypoint proposals；
- G-Min 595 个候选经过 GraphMap 关联后的聚合审计；
- 5 个 held-out static observations 的 depth encoder / waypoint classifier post-ReLU activity 对照；
- Frozen-20 development panel 指标汇总。

本轮只从已保存的实验报告与历史日志提取数据、整理 CSV 并生成 SVG；**没有重新启动 Habitat、重新加载 checkpoint、运行 navigation、训练或产生新实验数据**。原始的 matched dense/sparse depth 图像对未能从现有公开材料中恢复：历史演示文稿中的 Habitat dense-depth 截图与 Livox 官方扫描可视化并非同一视角，因此没有拼接成虚假的 matched pair。episode 53 候选图作为诚实替代，并明确标注它展示的是 proposer 输出，不是原始传感器图像。

原始 `outputs/` 目录为历史本地 ignored artifacts，不在公开仓库；当前提交的 CSV 是图表使用的汇总证据，不等于恢复了原始 NPZ / JSON / telemetry 文件。

## 4. Stage 5 Learned Adapter：保留为未完成原型

Stage 5 M0 / M1 形成了 adapter contract、单 pair smoke、feature cache / scene-level split / A1 training pipeline 代码，以及 static evaluator 分支和 CPU tests。但这条线没有完成可接受的端到端研究验收：

- PR #3（由协作者 RtimesC 发起）未合并；审查中记录了 collector 临时目录与 cache no-overwrite 冲突、reset episode/scene identity 校验缺失、provenance contract 尚需补齐等真实 host execution 前阻塞。
- Issue #1 的多 observation 数据采集、真实 cache collection 和 A1 训练没有形成已审查并合并的最终 checkpoint 交付。
- Issue #2 的 static evaluator 有 nonformal single-pair overfit smoke，但正式 held-out-5 gate 需要唯一冻结 checkpoint；该正式 gate 未完成。
- 因此不能声称 Learned Sparse→Dense Adapter 已泛化、已恢复 waypoint diversity，或已提升导航成功率。

相关 feature branches 保留作历史研究记录；PR #3 关闭但不合并，Issues #1/#2 以项目路线终止为由关闭。关闭不代表协作者代码被否定，也不把未完成工作算作交付。

## 5. 明确未完成 / 不纳入本项目成果

- Learned adapter 的真实多 observation feature collection、正式训练与 frozen held-out validation。
- 正式完整 R2R-CE / val_unseen benchmark、多场景统计和与论文可比的正式结果。
- Isaac Sim runtime、原生 LiDAR sensor simulation、VLN-CE-Isaac 端到端运行。
- 真实 Livox MID360、FAST-LIO2、camera/IMU 融合、ROS 2 / Nav2 / MPPI 部署。
- 任何真实机器人上的安全、定位、控制或导航验证。

## 6. 仓库维护与分支处置

- `main` 保留稳定、可解释的实验记录和负结果；README 不再标注 Learned Adapter 为当前开发方向。
- `mid360-sparse-depth-learned-adapter`、`stage5-m1-data-training`、`stage5-m1-static-eval`、`mid360-sparse-depth-zero-shot`、`mid360-sparse-depth-feature-alignment`、`isaac-integration`、`etp-mid360` 等分支保留，不继续开发、不删除。
- PR #3 关闭不合并；Issues #1/#2 关闭为 not planned；已关闭未合并的 PR #4 保留历史状态。
- 不下载/提交 checkpoint、Matterport3D 数据集、完整训练 cache 或运行产物。
- 不将仓库归档（GitHub Archived）；项目结项与仓库只读归档是不同决定。

## 7. 个人主页 / 简历的准确表述

> Explored MID360-inspired sparse-depth adaptation for ETPNav in Habitat, implementing explicit geometric waypoint proposals (Geometry V0, BNMS, MRU, G-Min), provider-interface regression checks, and sparse-depth failure forensics. Identified waypoint collapse at the frozen `vis_classifier` ReLU and showed that simple feature-statistics alignment was insufficient. The research line was discontinued before learned-adapter generalization, formal VLN benchmarking, Isaac Sim runtime, or physical MID360 deployment.

该表述强调实际工程工作和负结果，不声称完成真实 LiDAR 导航。

## 8. 上游引用

ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments, TPAMI 2024. See the original [ETPNav repository](https://github.com/MarSaKi/ETPNav) and [paper](https://arxiv.org/abs/2304.03047).

**最终结论：ETPNav-MID360 是一条已停止推进的研究/工程探索路线，保留了可复查的算法对照、几何前端实现、接口审计和负结果；它没有交付可在真实 MID360 上运行的导航系统。**