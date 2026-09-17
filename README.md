# ETPNav-MID360

Research on adapting **ETPNav** from dense RGB-D observations to **MID360-like sparse LiDAR geometry** for Vision-and-Language Navigation (VLN).

> Core question: how can the original ETPNav waypoint proposal interface be adapted to sparse LiDAR observations while keeping the downstream GraphMap, language/topological planner, and execution pipeline as unchanged as possible?

The current work is conducted primarily in **Habitat / R2R-CE / Matterport3D** under controlled experiments. The longer-term path is **Habitat -> Isaac Sim -> real Livox MID360 + localization + robot navigation stack**.

**Current research stage:** learned sparse-to-dense feature adaptation before navigation-level validation.

[Chinese README](README_CN.md)

---

## Research Scope

The project focuses on **waypoint-interface adaptation**, not full retraining of ETPNav.

```text
RGB ----------------------------------------------------┐
                                                       v
Dense Depth -> MID360-like sparse observation -> adaptation
                                                       v
                                      Original Waypoint Predictor
                                                       v
                                                   GraphMap
                                                       v
                                      Language / Topological Planner
                                                       v
                                                   Execution
```

Research constraints:

- keep the original ETPNav downstream stack frozen whenever possible;
- separate static/mechanism experiments from navigation evaluation;
- avoid changing proposer, GraphMap, planner, and executor simultaneously;
- use a frozen 20-episode panel only as a controlled development set, not as a full paper benchmark;
- treat the current Habitat MID360-like observation as a deterministic sparse-geometry approximation, **not** a physically complete Livox sensor model;
- defer native LiDAR simulation, localization noise, and real-robot constraints to later Isaac Sim / hardware stages.

---

## Research Progression

```text
Original ETPNav baseline
        |
        +--> Explicit geometry route
        |      Geometry V0
        |        -> bounded-NMS audit
        |        -> MRU radial ablation
        |        -> G-Min coverage proposal
        |        -> proposal/GraphMap interface analysis
        |        -> Geometry V1 exploration (paused)
        |
        +--> Preserve learned waypoint prior
               MID360-like sparse-depth zero-shot
                 -> waypoint-predictor failure forensics
                 -> fixed feature alignment
                 -> learned sparse-to-dense adapter  [current]
                 -> navigation validation
                 -> full benchmark / geometry comparison
                 -> Isaac Sim native LiDAR
                 -> FAST-LIO2 / real MID360 integration
```

---

## Controlled Frozen-20 Results

All methods below were evaluated on the same frozen 20 `val_unseen` development episodes.

| Method | SR ↑ | SPL ↑ | NE ↓ | OS ↑ | nDTW ↑ | SDTW ↑ |
|---|---:|---:|---:|---:|---:|---:|
| **Original ETPNav** | **0.800** | **0.698** | **2.922** | **0.800** | **0.714** | **0.651** |
| Geometry V0 | 0.500 | 0.321 | 5.055 | 0.500 | 0.448 | 0.305 |
| Geometry V0 + BNMS | 0.550 | 0.364 | 4.526 | 0.600 | 0.467 | 0.356 |
| MRU | 0.450 | 0.365 | 4.730 | 0.550 | 0.534 | 0.351 |
| G-Min | 0.500 | 0.428 | 4.274 | 0.650 | 0.582 | 0.399 |
| **Sparse Depth Zero-shot** | **0.000** | **0.000** | **9.108** | **0.000** | **0.281** | **0.000** |

These runs are used for controlled development and mechanism analysis. They should not be interpreted as a complete benchmark or paper-level final result.

---

## Route 1: Explicit Geometry Waypoint Proposal

### Geometry V0

The first route replaces the learned waypoint proposer with an explicit geometric frontend:

```text
12-view Depth
-> metric reconstruction
-> 3D point cloud
-> MID360-like filtering
-> local occupancy / traversability
-> explicit waypoint proposal
-> frozen GraphMap / planner
```

The first scoring rule used a simple long-range preference:

```text
S(a, d) = d / 3
```

This created a strong distance bias: approximately 64% of candidates were placed at exactly 3.0 m. Navigation reached `SR = 0.50`, `SPL = 0.321`.

### BNMS: bounded radial topology

The geometry candidate map has two different topologies:

- angle: circular;
- distance/radius: bounded.

The initial suppression logic incorrectly treated the radial dimension as periodic. A geometry-specific **Bounded NMS (BNMS)** keeps angular wraparound while preventing radial wraparound.

This improved the controlled run from:

```text
SR : 0.50  -> 0.55
SPL: 0.321 -> 0.364
NE : 5.06  -> 4.53 m
OS : 0.50  -> 0.60
```

The original ETPNav NMS remains unchanged.

### MRU: radial-distribution ablation

MRU tests whether the geometry route fails mainly because candidates are too far away:

```text
S(d) = x * exp(1 - x)
x = 2d / Dmax
```

The candidate distribution moved toward approximately 1.5 m, but navigation did not recover (`SR = 0.45`). This produced a useful negative result:

> Matching the average waypoint distance to the original model is not sufficient to recover navigation performance.

### G-Min: coverage instead of a single preferred distance

G-Min removes a single radial preference and emphasizes:

```text
geometry admissibility
+
angular coverage
+
radial coverage
```

It did not improve SR over the best earlier geometry baseline, but improved several secondary trajectory metrics.

A subsequent interface audit showed that proposal diversity is not preserved one-to-one by the frozen GraphMap. In one controlled analysis:

```text
595 raw G-Min candidates
-> 411 new ghosts
-> 25 existing-ghost merges
-> 159 visited-node absorptions
```

This supports the existence of **proposal-GraphMap interface coupling** without implying that GraphMap itself is defective.

### Geometry V1: paused

A more complex geometry route explored ray carving, FREE/HIT/UNKNOWN states, clearance-aware structural candidates, and stricter local safety constraints. The scope began drifting toward a local planner rather than a waypoint-interface adapter, and some real-provider smoke tests became overly conservative.

That route is currently paused. Local safety and collision handling are intended to remain downstream responsibilities of Nav2 / MPPI or an equivalent execution layer.

---

## Route 2: Preserve the Learned Waypoint Prior

The second route keeps the original learned waypoint predictor and focuses on the **sensor-domain gap**.

Current Habitat abstraction:

```text
Dense normalized Depth
-> metric point cloud
-> deterministic MID360-like angular sparsification
-> nearest-return z-buffer projection
-> Sparse normalized Depth
-> Original frozen waypoint proposer
```

This approximation does **not** model the complete Livox sampling pattern, motion distortion, temporal accumulation, or sensor noise.

### Sparse-depth zero-shot failure

Directly feeding the MID360-like sparse depth into the original proposer failed completely on the frozen development panel:

```text
Original ETPNav
SR  = 0.80
SPL = 0.698

Sparse Depth Zero-shot
SR  = 0.00
SPL = 0.000
NE  = 9.108 m
```

The candidate-distance distribution collapsed to a fixed 0.75 m pattern.

### Failure forensics

Activation tracing showed that sparse depth did not immediately destroy all structure at the depth encoder output.

```text
Sparse Depth
  -> Frozen Depth Encoder
  -> visual_fc_depth + ReLU       (partially active)
  -> Transformer
  -> vis_classifier Linear
  -> ReLU                         (complete collapse)
  -> bias-dominated logits
  -> fixed waypoint pattern
```

The first complete collapse point was observed at the `vis_classifier` ReLU rather than directly at the depth encoder.

### Fixed feature alignment: negative result

A channel-wise mean/std alignment was tested:

```text
F_sparse
-> normalize with sparse statistics
-> re-scale with dense statistics
-> Frozen Original Waypoint Predictor
```

The transformation improved some feature-domain statistics but did not reactivate the collapsed classifier or restore waypoint radial diversity on held-out mechanism tests.

This produced a second negative result:

> First- and second-order channel-wise feature-statistics alignment is insufficient for the observed sparse-depth domain shift.

---

## Current Stage: Learned Sparse-to-Dense Feature Adapter

Active development uses a lightweight residual nonlinear adapter:

```text
Dense Depth
-> Frozen Depth Encoder
-> F_dense ----------------------┐
                               | teacher target
Sparse Depth                    |
-> Frozen Depth Encoder         |
-> F_sparse                     |
-> Lightweight Adapter          |
-> F_adapt ---------------------┘
-> Frozen Original Waypoint Predictor
```

The current experimental sequence is deliberately mechanism-first:

1. determine whether `F_adapt` moves toward the dense feature domain;
2. test whether `vis_classifier` activation recovers;
3. test whether angular/radial waypoint diversity recovers;
4. only then proceed to frozen-20 navigation evaluation.

The repository currently contains separate development branches for adapter implementation, data/training, and static held-out evaluation. These branches are intentionally not merged directly into `main` until the corresponding research gates are satisfied.

---

## Planned Path

```text
Stage A  Learned adapter static validation
   ->
Stage B  Frozen-20 navigation validation
   ->
Stage C  Full val_unseen benchmark
   ->
Stage D  Learned adaptation vs explicit geometry comparison
   ->
Stage E  Isaac Sim native LiDAR
   ->
Stage F  GT pose -> FAST-LIO2 localization
   ->
Stage G  MID360 + camera + IMU real robot
   ->
Stage H  Nav2 / MPPI execution and safety layer
```

The intended real-robot separation is:

```text
MID360 + IMU
-> FAST-LIO2 / localization

MID360 geometry
-> VLN waypoint interface
```

High-level VLN should produce goal waypoints or topological decisions; local path following, dynamic obstacle avoidance, and execution safety remain downstream responsibilities.

---

## Repository Structure

Key areas include:

- `vlnce_baselines/waypoint_provider/` — original and MID360-oriented waypoint-provider interfaces;
- `vlnce_baselines/perception/` — sparse-depth, geometry, and ray-based perception utilities;
- `scripts/` — controlled experiments, diagnostics, replay, analysis, and adapter tooling;
- `tests/` — regression and mechanism-level tests;
- `docs/` — experiment contracts, architecture notes, setup notes, and result summaries;
- `configs/` — geometry and sparse-depth experiment configurations.

### Branch model

```text
main
├── validated and explainable project state
│
├── mid360-sparse-depth-zero-shot
├── mid360-sparse-depth-feature-alignment
├── mid360-sparse-depth-learned-adapter
├── stage5-m1-data-training
└── stage5-m1-static-eval

etp-mid360
└── early MID360 / ETPNav development history

isaac-integration
└── Isaac / geometry integration history
```

`main` is kept as the stable, interpretable research state. New experiments progress on dedicated branches and are integrated only after their experimental contract is satisfied.

---

## Data, Checkpoints, and Reproducibility

This repository does **not** redistribute restricted Matterport3D scene assets, private dataset credentials, or local experiment outputs.

Expected external assets include, depending on the experiment:

- Matterport3D / R2R-CE / RxR data obtained from their official sources under their applicable terms;
- original ETPNav checkpoints obtained from the upstream project or other officially provided sources;
- locally generated feature caches, checkpoints, logs, and evaluation outputs.

Local data, checkpoints, logs, caches, and generated outputs are intentionally excluded from version control. Users are responsible for obtaining third-party datasets and pretrained weights under their respective licenses and access conditions.

---

## Upstream Attribution and License

This repository is derived from the original **ETPNav** implementation:

**ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments**  
Dong An, Hanqing Wang, Wenguan Wang, Zun Wang, Yan Huang, Keji He, Liang Wang  
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024

Upstream repository: <https://github.com/MarSaKi/ETPNav>  
Paper: <https://arxiv.org/abs/2304.03047>

The upstream code is distributed under the MIT License. This repository retains the upstream MIT license notice. The MID360 adaptation, geometry baselines, sparse-depth experiments, diagnostics, experiment contracts, and related modifications in this repository are ongoing research work built on top of that codebase.

If you use the original ETPNav method or code, please cite the original paper:

```bibtex
@article{an2024etpnav,
  title={ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments},
  author={An, Dong and Wang, Hanqing and Wang, Wenguan and Wang, Zun and Huang, Yan and He, Keji and Wang, Liang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024}
}
```

---

## Status

This is an active research repository. Results, branch structure, and experiment protocols may evolve as the MID360 adaptation progresses. Current reported numbers are development-stage results unless explicitly stated otherwise.
