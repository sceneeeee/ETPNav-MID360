# ETPNav-MID360

基于 ETPNav 的稀疏 LiDAR / MID360 视觉语言导航适配研究。

本项目关注一个核心问题：

> 原本依赖稠密 RGB-D depth 的 ETPNav waypoint proposal，如何适配 MID360 一类稀疏 LiDAR 观测，同时尽量保持后续 GraphMap、语言/拓扑规划器和执行框架不变？

当前工作以 Habitat / R2R-CE / Matterport3D 的受控实验为主，后续计划迁移到 Isaac Sim，并最终接入真实 MID360、FAST-LIO2 和机器人导航栈。

**当前阶段：Learned Sparse→Dense Feature Adapter。**

---

## 1. 为什么从 NaVIDA 转向 ETPNav

项目早期尝试过 NaVIDA。NaVIDA 更接近端到端 VLN：视觉输入、模型推理和动作输出耦合较紧，在 Jetson 一类边缘平台上继续做部署优化时，很容易把模型速度、显存、控制频率和导航性能混在一起。

因此后续将主要研究平台切换为 ETPNav。ETPNav 的模块边界更清晰：

```text
Perception
→ Waypoint Proposal
→ GraphMap
→ Language / Topological Planner
→ Execution
```

这使我们可以尽量保持 downstream 不变，只研究“局部 waypoint 是如何由新的几何传感器产生或适配出来的”。

---

## 2. 当前研究边界

当前研究重点是 **waypoint interface adaptation**，不是重新训练整个 ETPNav。

```text
RGB ----------------------------------------------------┐
                                                       ↓
Dense Depth → MID360-like sparse observation → adaptation
                                                       ↓
                                      Original Waypoint Predictor
                                                       ↓
                                                   GraphMap
                                                       ↓
                                      Language / Topological Planner
                                                       ↓
                                                   Execution
```

当前原则：

- Original ETPNav downstream 尽量保持冻结；
- 先做 static / mechanism experiment，再做 navigation benchmark；
- 不因为单个结果不好就同时修改 proposer、GraphMap、planner 和 executor；
- frozen 20 episodes 只作为受控 development panel，不代表完整论文 benchmark；
- Habitat 中的 MID360-like 输入目前仍是从 dense Depth 构造的确定性稀疏几何近似，并不等同于真实 Livox 扫描物理；
- Isaac Sim 阶段再切换到原生 LiDAR sensor simulation，并逐步加入 localization error 和真实机器人约束。

---

## 3. 项目研究历程

```text
NaVIDA
  ↓
转向模块化 ETPNav
  ↓
Original ETPNav baseline
  ↓
┌──────────────── Explicit Geometry Route ────────────────┐
│ Geometry V0                                             │
│   ↓                                                     │
│ NMS topology audit → BNMS                              │
│   ↓                                                     │
│ MRU distance-distribution ablation                      │
│   ↓                                                     │
│ G-Min angular/radial coverage                           │
│   ↓                                                     │
│ GraphMap interface audit                                │
│   ↓                                                     │
│ Geometry V1 exploration → scope drift / 暂缓            │
└─────────────────────────────────────────────────────────┘
  ↓
重新考虑保留 Original learned waypoint prior
  ↓
MID360-like Sparse Depth Zero-shot
  ↓
Waypoint Predictor Failure Forensics
  ↓
Fixed Feature Alignment
  ↓
Learned Sparse→Dense Adapter   ← 当前
  ↓
Navigation Validation
  ↓
Full Benchmark / Geometry Comparison
  ↓
Isaac Sim → FAST-LIO2 → MID360 实机
```

---

## 4. Frozen-20 development panel

以下方法使用同一组 frozen 20 val_unseen episodes 做受控比较。

| 方法 | SR ↑ | SPL ↑ | NE ↓ | OS ↑ | nDTW ↑ | SDTW ↑ |
|---|---:|---:|---:|---:|---:|---:|
| **Original ETPNav** | **0.800** | **0.698** | **2.922** | **0.800** | **0.714** | **0.651** |
| Geometry V0 | 0.500 | 0.321 | 5.055 | 0.500 | 0.448 | 0.305 |
| Geometry V0 + BNMS | 0.550 | 0.364 | 4.526 | 0.600 | 0.467 | 0.356 |
| MRU | 0.450 | 0.365 | 4.730 | 0.550 | 0.534 | 0.351 |
| G-Min | 0.500 | 0.428 | 4.274 | 0.650 | 0.582 | 0.399 |
| **Sparse Depth Zero-shot** | **0.000** | **0.000** | **9.108** | **0.000** | **0.281** | **0.000** |

这组结果形成了两条重要结论：

1. 显式几何路线并不是完全不可行，但目前所有 geometry variants 都没有恢复 Original ETPNav 的成功率；
2. 直接把 MID360-like sparse depth zero-shot 输入 Original learned proposer 会发生更严重的 domain failure，因此保留 learned prior 仍需要专门的 adaptation。

---

## 5. Explicit Geometry Route

### 5.1 Geometry V0：直接替换 learned proposal

第一条路线尝试完全替换 Original learned waypoint proposal：

```text
12-view Depth
→ metric reconstruction
→ 3D point cloud
→ MID360-like filtering
→ local occupancy / traversability
→ explicit waypoint proposal
→ frozen GraphMap / planner
```

Geometry V0 对 traversable candidate 使用简单的 long-range preference：

```text
S(a, d) = d / 3
```

结果出现明显的远距离偏置：

- candidate mean ≈ **2.583 m**；
- median = **3.0 m**；
- P95 = **3.0 m**；
- 约 **64%** candidates 恰好位于 3.0 m；
- navigation：SR = **0.50**，SPL = **0.321**。

这说明“可通行区域中越远越优先”是一个过强的先验。

### 5.2 NMS topology audit → BNMS

Geometry V0 的二维 candidate map 可以写成：

```text
angle × distance
```

其中两维的拓扑不同：

- angle 是 circular topology；
- distance / radius 是 bounded topology。

Geometry 分支早期的 suppression 逻辑把 radial dimension 也当成了环形邻域，因此我们为 geometry proposer 引入 **BNMS（Bounded NMS）**：角度继续环绕，距离不环绕。

这里不修改 Original ETPNav released NMS；BNMS 只用于 geometry branch，保证 Original baseline 不受污染。

修正后的 Geometry baseline：

```text
SR : 0.50  → 0.55
SPL: 0.321 → 0.364
NE : 5.06  → 4.53 m
OS : 0.50  → 0.60
```

但 long-range bias 仍然存在。

### 5.3 MRU：中距离是不是关键？

MRU（Middle-distance Radial Utility）用于验证一个单独假设：

> Geometry 性能差，是否主要因为 V0 过度偏向远距离？

使用 radial utility：

```text
S(d) = x · exp(1 - x)
x = 2d / Dmax
```

其峰值约在 1.5 m。

MRU 的候选分布确实被拉回中距离：

- candidate mean ≈ **1.449 m**；
- **89.25%** candidates 恰好为 1.5 m；
- radial entropy = **0.685 bits**。

但导航没有恢复：

```text
BNMS SR = 0.55
MRU  SR = 0.45
```

因此得到一个明确的 negative result：

> 仅仅把平均 waypoint distance 调回 Original 附近，并不足以恢复导航性能。

### 5.4 G-Min：不再规定“最佳距离”

G-Min（Geometry-Minimal Proposal）进一步取消单一 radial score 对 candidate 重要性的控制，改为：

```text
geometry admissibility
+
angular coverage
+
radial coverage
```

目标不是猜测“哪个距离最好”，而是尽量覆盖多个方向和多个距离。

Frozen-20 结果：

```text
SR   = 0.500
SPL  = 0.428
NE   = 4.274 m
OS   = 0.650
nDTW = 0.582
SDTW = 0.399
```

G-Min 没有提升 SR，但相对 BNMS 改善了 SPL、NE、OS、nDTW 和 SDTW。

因此问题不能简单归结为“显式几何本身失败”或“candidate 距离分布不对”。

### 5.5 Raw proposal diversity ≠ Graph action diversity

进一步检查 G-Min candidate 进入 frozen GraphMap 后发生了什么：

```text
G-Min raw candidates: 595
        ↓ GraphMap association
411 new ghosts
25 existing-ghost merges
159 visited-node absorptions
```

其中：

- 0.25 m candidates：**119 / 119 全部被 visited nodes 吸收**；
- G-Min 实际 candidate-associated actions：平均 **3.664 / nonempty decision**；
- BNMS：平均 **4.339 / nonempty decision**。

这说明：

> Raw proposal diversity 不等于 Graph action diversity。

几何上不同的 waypoint，在 Original GraphMap 的拓扑语义中可能是冗余的。该结果支持 **proposal–GraphMap interface coupling** 的存在，但并不意味着 GraphMap 本身“有 bug”。

### 5.6 Geometry V1：为什么暂缓

之后曾探索更复杂的 Geometry V1，包括：

- height-aware 3D ray carving；
- FREE / HIT / UNKNOWN / RANGE-CENSOR；
- clearance-ridge structural candidates；
- progress fallback；
- 更严格的 local safety / traversability 约束。

这条路线逐渐从“替换 waypoint proposal”扩展成了“局部感知 + epistemic mapping + safety-certified local planning”。真实 provider smoke 中甚至出现过 safe polar cells = 0、candidate count = 0 的过度保守情况。

因此 V1 在进入正式 frozen-20 之前被暂缓。当前认为执行安全更应该由后续 Nav2 / MPPI / collision layer 承担，而不是全部塞进 VLN waypoint proposer。

---

## 6. 回到 learned waypoint prior：Sparse Depth Zero-shot

Geometry route 暴露了两个问题：

1. 手工几何 candidate 很难自动恢复 Original ETPNav 学到的 waypoint prior；
2. 新 proposal 还需要适配 frozen GraphMap 的拓扑语义。

因此项目转向另一条路线：

> 保留 Original learned waypoint predictor，只解决 dense Depth → MID360 / sparse LiDAR 的 sensor-domain gap。

当前 Habitat sparse-depth abstraction：

```text
Dense normalized Depth
→ metric point cloud
→ deterministic MID360-like angular sparsification
→ nearest-return z-buffer projection
→ Sparse normalized Depth
→ Original frozen waypoint proposer
```

这是一种确定性的 engineering approximation，不包含完整 Livox 扫描噪声、运动畸变或真实时序采样。

### Zero-shot 结果

```text
Original ETPNav
SR  = 0.80
SPL = 0.698

Sparse Depth Zero-shot
SR  = 0.00
SPL = 0.000
NE  = 9.108 m
```

20/20 episodes 全部失败，而且生成的 candidate distance 全部塌缩到固定 **0.75 m**。

这证明 Original proposer 不能 zero-shot 接受当前 MID360-like sparse depth。

---

## 7. Failure Forensics：collapse 到底发生在哪里？

后续没有直接重新训练，而是先追踪 network activation。

结果发现 sparse depth 并不是在 depth encoder 后立即完全失去结构：

```text
Sparse Depth
    ↓
Frozen Depth Encoder
    ↓
visual_fc_depth
    ↓ ReLU
仍保留约 23–24% activation
    ↓
Transformer
    ↓
vis_classifier Linear
几乎全部落到负区间
    ↓ ReLU
0 / 9216 activation
    ↓
bias-only logits
    ↓
固定 0.75 m waypoint pattern
```

关键发现：

> 第一个 complete collapse point 是 `vis_classifier` 的 ReLU，而不是 depth encoder 本身。

在 held-out forensic observations 中，Sparse 与 All-Far 条件还表现出相同的 collapse pattern，说明稀疏输入已经把 predictor 推入类似“无有效几何信息”的工作区间。

---

## 8. Fixed Feature Alignment：简单统计对齐够不够？

为了验证 failure 是否主要来自 encoder feature 的一二阶统计偏移，进一步做了 channel-wise mean/std alignment：

```text
F_sparse
→ (F_sparse - μ_sparse) / σ_sparse
→ × σ_dense + μ_dense
→ Frozen Original Waypoint Predictor
```

实验流程：

- 使用 **53 observations** 完成 calibration；
- calibration 只需要 Depth，因此运行时只实例化 12 个 Depth sensors；
- 已验证正常场景下 24-sensor 与 12-Depth-only 产生的 Depth tensor bit-exact；
- 使用 5 个 held-out observations 做 static mechanism test；
- 不运行 navigation、GraphMap rollout 或 training。

最终：

```text
5 / 5 observations = Gate C
Sparse-Aligned classifier_post_relu = 0
waypoint radial diversity 未恢复
```

因此得到第二个明确的 negative result：

> 简单的一二阶 channel-wise feature-statistics alignment 不足以解决 sparse-depth domain shift。

这也是当前转向 learned nonlinear adapter 的直接原因。

---

## 9. 当前阶段：Learned Sparse→Dense Feature Adapter

当前开发分支：

```text
mid360-sparse-depth-learned-adapter
```

下一步不直接 fine-tune 整个 ETPNav，而是先验证一个轻量非线性 adapter：

```text
Dense Depth
→ Frozen Depth Encoder
→ F_dense ----------------------┐
                               │ teacher target
Sparse Depth                   │
→ Frozen Depth Encoder         │
→ F_sparse                     │
→ Lightweight Adapter          │
→ F_adapt ---------------------┘
→ Frozen Original Waypoint Predictor
```

第一阶段要回答的不是“SR 能不能立刻恢复”，而是：

1. `F_adapt` 是否更接近 dense feature domain；
2. `vis_classifier` activation 是否从 0 恢复；
3. waypoint angular / radial diversity 是否恢复；
4. static mechanism test 通过后，再进入 frozen-20 navigation evaluation。

如果 adapter 有效，再讨论是否需要 partial fine-tuning；如果无效，再重新评估 waypoint front-end adaptation 的位置和目标函数。

---

## 10. 后续计划

```text
Stage A  Learned Adapter static validation
   ↓
Stage B  Frozen-20 navigation validation
   ↓
Stage C  Full val_unseen benchmark
   ↓
Stage D  Learned adaptation vs explicit geometry comparison
   ↓
Stage E  Isaac Sim native LiDAR
   ↓
Stage F  GT pose → FAST-LIO2 localization
   ↓
Stage G  MID360 + camera + IMU real robot
   ↓
Stage H  Nav2 / MPPI execution and safety layer
```

真实机器人阶段计划将 MID360 分成两条用途：

```text
MID360 + IMU
→ FAST-LIO2 / localization

MID360 geometry
→ VLN waypoint interface
```

高层 VLN 只负责目标 waypoint / topological decision，实际局部路径跟踪、动态避障和执行安全交给 Nav2 / MPPI 等下游模块。

---

## 11. Branch 结构

```text
main
├── 已验证的项目稳定状态
│
├── mid360-sparse-depth-zero-shot
│   └── sparse-depth zero-shot + failure evidence
│
├── mid360-sparse-depth-feature-alignment
│   └── fixed feature alignment negative result
│
└── mid360-sparse-depth-learned-adapter
    └── 当前开发分支

etp-mid360
└── 早期 MID360 / ETPNav base 分支

isaac-integration
└── Isaac integration / geometry development history
```

`main` 作为默认分支，保留已经验证、可以稳定解释的项目状态；新实验在独立 feature branch 中推进。

---

## 12. 相关文档

- [ETPNav baseline architecture](docs/etp_baseline_architecture.md)
- [MID360 sparse-depth zero-shot](docs/mid360_sparse_depth_zero_shot.md)
- [MID360 example-scene results](docs/mid360_example_scene_results.md)
- [Isaac adapter contract](docs/isaac_adapter_contract.md)
- [Isaac integration audit](docs/isaac_integration_audit.md)
- [Renderer bootstrap](docs/renderer_bootstrap.md)
- [Setup log](docs/setup_log.md)

大量运行产物、checkpoint、raw logs 和正式 evaluation evidence 默认不作为 README 内容；README 只记录会影响研究决策的结果。

---

## 13. 上游项目

本仓库基于：

**ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments**  
Dong An, Hanqing Wang, Wenguan Wang, Zun Wang, Yan Huang, Keji He, Liang Wang  
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024

原项目：<https://github.com/MarSaKi/ETPNav>  
论文：<https://arxiv.org/abs/2304.03047>

如果使用 ETPNav 原始方法或代码，请引用原论文：

```bibtex
@article{an2024etpnav,
  title={ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments},
  author={An, Dong and Wang, Hanqing and Wang, Wenguan and Wang, Zun and Huang, Yan and He, Keji and Wang, Liang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024}
}
```
