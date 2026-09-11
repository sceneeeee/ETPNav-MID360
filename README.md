# ETPNav-MID360

基于 ETPNav 的稀疏 LiDAR / MID360 视觉语言导航适配研究。

本项目关注一个核心问题：

> 原本依赖稠密 RGB-D depth 的 ETPNav waypoint proposal，如何适配 MID360 一类稀疏 LiDAR 观测，同时尽量保持后续 GraphMap、语言规划器和导航框架不变？

当前主要在 Habitat / R2R-CE / Matterport3D 中进行受控实验，后续计划迁移到 Isaac Sim 和真实机器人。

## 当前架构

```text
RGB -------------------------------> ETPNav downstream
                                      GraphMap / Planner
Dense Depth
    ↓
MID360-like sparsification
    ↓
Sparse Depth
    ↓
Frozen Depth Encoder
    ↓
Waypoint interface adaptation
    ↓
Frozen ETPNav Waypoint Predictor
```

当前研究重点是 **waypoint interface adaptation**，而不是重新训练整个 ETPNav。

## 项目阶段

| 阶段 | 内容 | 状态 | 主要结果 |
|---|---|---|---|
| 1 | Original ETPNav baseline | ✅ 完成 | frozen 20 episodes：SR 0.80，SPL 0.698 |
| 2 | Sparse-depth zero-shot | ✅ 完成 | SR 0；waypoint collapse 到固定 0.75 m pattern |
| 3 | Failure forensic | ✅ 完成 | 定位首个 complete collapse：`vis_classifier` ReLU = 0 |
| 4 | Fixed feature alignment | ✅ 完成 | 5/5 Gate C；mean/std alignment 无法恢复 classifier activity |
| 5 | Learned sparse→dense adapter | 🚧 当前阶段 | 训练轻量非线性 feature adapter |
| 6 | Navigation validation | ⏳ 计划 | frozen 20 episodes，比较 SR / SPL / NE / nDTW / SDTW |
| 7 | Full benchmark & geometry comparison | ⏳ 计划 | learned adaptation vs explicit LiDAR geometry proposer |
| 8 | Isaac Sim / real robot | ⏳ 计划 | simulated LiDAR → GT pose → FAST-LIO2 → MID360 实机 |

## 已确认的 failure mechanism

Sparse depth zero-shot 并不是在 depth encoder 后立即完全失效：

```text
Sparse Depth
    ↓
Depth Encoder
    ↓
visual_fc_depth ReLU
约 23–24% activation
    ↓
Transformer
    ↓
vis_classifier Linear
全部落到负区间
    ↓
vis_classifier ReLU
0 / 9216
    ↓
bias-only logits
    ↓
固定 0.75 m waypoint pattern
```

简单的 channel-wise mean/std feature alignment 虽然改变了上游 feature statistics，但在 5 个 held-out observations 上均未恢复 `vis_classifier` activation，因此下一阶段转向 learned adapter。

## 当前实验原则

- Depth Encoder 冻结
- ETPNav Waypoint Predictor 冻结
- GraphMap / Planner 在静态机制实验中不运行
- 先验证 waypoint interface，再进行 navigation benchmark
- frozen 20 episodes 目前仅作为开发集，不代表完整论文 benchmark
- 不直接 fine-tune 整个 ETPNav

## 后续两条路线

**Learned adaptation**

```text
Sparse Depth
→ Frozen Depth Encoder
→ Lightweight Adapter
→ Original Waypoint Predictor
```

**Explicit geometry**

```text
MID360 / LiDAR
→ Occupancy / Traversability
→ Geometric Waypoint Proposal
→ Original GraphMap / Planner
```

最终将在相同 downstream 条件下比较两种方案。

## 上游项目

本仓库基于 ETPNav：

**ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments**

Dong An, Hanqing Wang, Wenguan Wang, Zun Wang, Yan Huang, Keji He, Liang Wang  
IEEE TPAMI 2024

原项目：
https://github.com/MarSaKi/ETPNav

论文：
https://arxiv.org/abs/2304.03047

如果使用 ETPNav 原始方法或代码，请引用原论文：

```bibtex
@article{an2024etpnav,
  title={ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments},
  author={An, Dong and Wang, Hanqing and Wang, Wenguan and Wang, Zun and Huang, Yan and He, Keji and Wang, Liang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024}
}
```
