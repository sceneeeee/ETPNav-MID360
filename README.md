<div align="center">

# ETPNav-MID360

### Toward Resource-Efficient Geometry-Based Vision-Language Navigation for Edge Robots

**Explicit LiDAR-like Waypoint Proposal for ETPNav**

</div>

---

## Overview

This repository is a research extension of [ETPNav](https://github.com/MarSaKi/ETPNav) that investigates a focused question:

> **Can explicit LiDAR-like geometry replace ETPNav's learned waypoint proposal mechanism while preserving its downstream topological and language-conditioned navigation planner?**

The project does **not** assume that geometric waypoint proposal is superior to the learned ETPNav waypoint predictor.

Instead, the goal is to study the interaction between:

- learned waypoint priors,
- explicit local geometry,
- topological GraphMap construction,
- downstream language-conditioned planning,
- computational cost,
- and eventual real-robot deployability.

The current work uses a controlled Habitat / R2R development environment and progressively moves toward a real-robot stack based on a **Livox MID360, RGB camera, Jetson Orin NX, FAST-LIO2 / PGO, Nav2, and MPPI**.

> **Current status:** research prototype.  
> Real-robot deployment and end-to-end resource savings have **not yet been demonstrated**.

---

## Research Scope

The central intervention is deliberately narrow.

### Original ETPNav

```text
RGB / Depth observations
        ↓
Learned waypoint predictor
        ↓
Candidate waypoints
        ↓
Panorama representation
        ↓
GraphMap
        ↓
Language-conditioned navigation planner
        ↓
Habitat execution
```

### Geometry variants

```text
RGB / Depth observations
        ↓
Metric local geometry
        ↓
Explicit waypoint proposal
        ↓
Existing ETPNav candidate representation
        ↓
Existing panorama representation
        ↓
Existing GraphMap
        ↓
Existing language-conditioned navigation planner
        ↓
Habitat execution
```

The controlled comparison therefore attempts to keep the downstream ETPNav stack fixed while changing the **waypoint proposal mechanism**.

---

## What the Geometry Module Does — and Does Not Do

The geometry waypoint proposer is intended to provide:

> **Obstacle-aware geometric admissibility and candidate coverage.**

It is **not** intended to replace:

- full 3D mapping,
- SLAM,
- local motion planning,
- Nav2,
- MPPI,
- execution-time collision avoidance,
- or a certified real-robot safety system.

This separation became an important design decision during the project.

An earlier Geometry V1 direction explored a substantially stronger epistemic representation with `FREE / HIT / UNKNOWN / RANGE-CENSOR`, full-height free-space reasoning, and structural candidate extraction.

That work is preserved in this repository as an **experimental secondary direction**, but it is no longer the primary architecture because it imposed a substantially stronger perception and safety requirement than the original ETPNav waypoint proposer.

---

# Geometry Variants

## Original ETPNav

The original released ETPNav waypoint predictor serves as the principal reference.

The downstream GraphMap, panorama processing, navigation planner, and Habitat execution remain the reference architecture throughout the controlled comparisons.

---

## Geometry V0

The first explicit geometry implementation used:

```text
12 depth views
        ↓
metric reconstruction
        ↓
3D point cloud
        ↓
MID360-like filtering
        ↓
height-filtered local obstacle occupancy
        ↓
robot-footprint inflation
        ↓
polar traversability
        ↓
waypoint candidate extraction
```

Its candidate score was approximately:

```text
S(a, d) = d / Dmax
```

for traversable cells.

This produced a strong long-distance bias, with many proposals collapsing to the maximum 3 m range.

---

## Geometry V0-BNMS

A correctness issue was identified in the original geometry candidate suppression.

Angular topology is circular:

```text
0° ≈ 360°
```

but radial topology is bounded:

```text
0.25 m != 3.0 m
```

The corrected Geometry baseline therefore uses:

> **circular angular suppression + bounded radial suppression**

without changing the released Original ETPNav NMS behavior.

This is currently retained as the **corrected Geometry baseline**.

---

## MRU-BNMS

MRU was a controlled experiment designed to remove the long-range saturation of Geometry V0-BNMS.

Its radial utility was constructed with a maximum near 1.5 m.

Although this corrected the mean candidate distance, it produced a new pathology: proposals strongly collapsed around a single radial bin.

This experiment provided an important negative result:

> **Matching the mean waypoint distance of Original ETPNav is not sufficient to recover its navigation performance.**

MRU is preserved as an ablation, not as the current baseline.

---

## G-Min

G-Min is the current deliberately simplified geometry proposal experiment.

It discards radial score magnitudes and starts from the existing Boolean obstacle-admissibility structure.

Conceptually:

```text
metric local geometry
        ↓
obstacle-aware admissibility mask
        ↓
deterministic angular / radial coverage
        ↓
≤ 5 waypoint candidates
        ↓
unchanged ETPNav downstream stack
```

G-Min uses:

- bounded radial topology,
- circular angular topology,
- deterministic tie breaking,
- angular sector coverage,
- explicit radial diversity,
- no language information inside the proposer,
- no ground-truth goal information,
- no navmesh oracle,
- no benchmark-fitted distance target.

It intentionally does **not** attempt to provide a complete 3D safety certificate.

---

# Controlled Development Results

The following results are from the same fixed **20-episode R2R `val_unseen` development panel**.

They are intended for controlled development comparisons and **must not be interpreted as final full-dataset benchmark results**.

| Method | SR ↑ | SPL ↑ | NE ↓ | OS ↑ | nDTW ↑ | SDTW ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Original ETPNav | **0.800** | **0.698** | **2.922** | **0.800** | **0.714** | **0.651** |
| Geometry V0 | 0.500 | 0.321 | 5.055 | — | — | — |
| Geometry V0-BNMS | **0.550** | 0.364 | 4.526 | 0.600 | 0.467 | 0.356 |
| MRU-BNMS | 0.450 | 0.365 | 4.730 | 0.550 | 0.534 | 0.351 |
| G-Min | 0.500 | **0.428** | **4.274** | **0.650** | **0.582** | **0.399** |

Bold values among the geometry methods are not intended to imply a statistically established winner; the panel contains only 20 development episodes.

---

## Current Interpretation

G-Min does **not** outperform the corrected BNMS baseline in Success Rate:

```text
BNMS SR  = 0.55
G-Min SR = 0.50
```

However, compared with BNMS, G-Min improves several trajectory-level metrics:

```text
SPL   0.364 → 0.428
NE    4.526 → 4.274
OS    0.600 → 0.650
nDTW  0.467 → 0.582
SDTW  0.356 → 0.399
```

Therefore the first G-Min result is treated as a **mixed result**, not a new winning baseline.

The current disposition is:

> **Keep Geometry V0-BNMS as the corrected Geometry baseline and analyze why G-Min's improved proposal coverage does not translate directly into higher SR.**

---

# Key Findings So Far

## 1. Longer waypoints are not automatically better

Geometry V0 strongly favored the longest available candidates and performed substantially below Original ETPNav.

---

## 2. Matching Original ETPNav's mean waypoint distance is not enough

MRU moved the mean candidate distance close to Original ETPNav but collapsed proposals around approximately 1.5 m and did not improve navigation.

---

## 3. Radial diversity alone is also not sufficient

G-Min produces substantially more structured radial coverage.

Across candidate-bearing decisions in the first frozen G-Min development run, its five candidate slots used five distinct radial bins.

Despite this, SR remained below the corrected BNMS baseline.

This suggests that raw waypoint-distance statistics alone cannot explain the Original-vs-Geometry performance gap.

---

## 4. Proposal diversity is not necessarily GraphMap diversity

A particularly important observation from the first G-Min run is that all observed 0.25 m proposals were merged into already visited GraphMap nodes under the unchanged GraphMap association behavior.

Therefore:

> **A geometrically distinct candidate does not necessarily create a distinct topological action for the downstream planner.**

The next analysis focuses on the complete interface:

```text
Proposal
    ↓
GraphMap update
    ↓
Ghost creation / merging
    ↓
Planner input
    ↓
STOP / navigation decision
```

---

## 5. STOP behavior may be part of the remaining gap

In the first G-Min 20-episode run:

- all final terminations were planner-selected STOP actions,
- none were caused by exhaustion of graph ghosts,
- none were caused by the step limit.

In two changed-outcome episodes, the agent entered the success radius and later navigated away before eventually stopping.

This does **not** prove that the planner is wrong — the planner does not observe ground-truth success radius — but it motivates further analysis of the interaction between proposal distribution, GraphMap structure, and the frozen ETPNav planner.

---

# Current Resource Results

The original motivation of this project includes resource-efficient edge deployment.

However, the current implementation does **not yet demonstrate end-to-end resource savings**.

Measured in the first G-Min development evaluation:

| Component | Approx. mean latency |
|---|---:|
| G-Min candidate extraction | ~2.08 ms |
| Geometry frontend | ~52.91 ms |
| G-Min synchronized decision | ~95.50 ms |
| Original ETPNav synchronized decision | ~38.67 ms |

The candidate selector itself is inexpensive.

The current bottleneck is primarily the preceding geometry frontend rather than G-Min extraction.

In addition, the learned waypoint predictor is currently bypassed during Geometry inference but is still instantiated and CUDA-resident.

Therefore:

> **Avoided waypoint-predictor inference is not yet equivalent to released model memory or lower end-to-end latency.**

This will be addressed separately from the navigation comparison so that algorithmic and systems changes are not confounded.

---

# Current Research Questions

The project is currently organized around three distinct levels.

## Primary research question

> Can explicit local geometry replace ETPNav's learned waypoint proposal while keeping the downstream topological and language-conditioned planner fixed?

## Mechanism questions

- How do different geometric candidates affect GraphMap construction?
- How often do raw proposals create genuinely new ghosts?
- How much proposal diversity survives graph merging?
- Does the frozen planner depend on structural properties of the learned ETP waypoint distribution?
- How does waypoint proposal affect STOP behavior?

## Deployment questions

- How should the Habitat geometry abstraction map to actual MID360 measurements?
- How should RGB / depth panorama requirements be provided on the real robot?
- Which Geometry operations should be moved or optimized for Jetson?
- How much memory can actually be released when the learned waypoint predictor is removed from initialization?
- How should the high-level VLN planner interface with FAST-LIO2, Nav2, MPPI, and execution safety?

These questions are intentionally evaluated separately.

---

# Target Real-Robot Architecture

The intended deployment direction is:

```text
                    RGB observation
                          │
                          ▼
                   ETP representation
                          │
Livox MID360 ──────► Geometry proposal
                          │
                          ▼
                      GraphMap
                          │
                          ▼
             language-conditioned planner
                          │
                          ▼
                    PlannerOutput
                          │
                          ▼
                target / path adapter
                          │
                          ▼
FAST-LIO2 + PGO ───► Nav2 planning / costmap
                          │
                          ▼
                         MPPI
                          │
                          ▼
                       cmd_vel
                          │
                          ▼
                 low-level controller
```

Target platform:

- Jetson Orin NX 16 GB
- Ubuntu 22.04
- ROS 2 Humble
- Livox MID360
- RGB camera
- FAST-LIO2
- pose-graph optimization
- Nav2
- MPPI

This stack represents the **deployment target**, not the current experimentally validated state.

---

# Repository Structure

Relevant extensions to the upstream ETPNav codebase include approximately:

```text
configs/
├── mid360_waypoint*.yaml
└── ...

vlnce_baselines/
├── perception/
│   ├── local_traversability.py
│   ├── depth_to_pointcloud.py
│   ├── mid360_like_filter.py
│   ├── gmin_candidates.py
│   ├── depth_to_rays.py
│   ├── ray_carving_representation.py
│   └── geometry_v1_candidates.py
│
├── waypoint_provider/
│   ├── original_etp.py
│   ├── mid360_geometry.py
│   └── mid360_geometry_v1.py
│
├── observation_provider/
│   └── isaac.py
│
└── ...

scripts/
├── run_geometry_*.bash
├── analyze_geometry_*.py
├── check_gmin_saved_observation.py
├── run_gmin_frozen20.bash
└── ...

tests/
├── test_geometry_provider.py
├── test_gmin_stage1.py
├── test_geometry_v1_*.py
└── ...
```

The exact structure may evolve as experimental branches are consolidated.

---

# Reproducibility Strategy

Controlled development experiments use:

- the same R2R `val_unseen` dataset,
- the same released navigation checkpoint,
- the same downstream ETPNav planner,
- the same 20 episode IDs and order,
- the same Habitat execution controls,
- explicit run manifests,
- candidate-level telemetry,
- trajectory telemetry,
- graph-state diagnostics,
- asset hashes where available.

The fixed 20-episode panel is intentionally treated as a **development panel**.

Repeated use of this panel for method design means it is not considered an independent final evaluation set.

Future confirmatory results should use data not involved in architecture and hyperparameter development.

---

# Setup

This repository is based on the original ETPNav Habitat stack.

The upstream implementation uses:

- Python 3.6
- Habitat-Lab 0.1.7
- Habitat-Sim 0.1.7
- VLN-CE-compatible environments

Create the upstream environment:

```bash
conda env create -f environment.yaml
conda activate vlnce
```

Install Habitat-Sim headless:

```bash
conda install -c aihabitat -c conda-forge habitat-sim=0.1.7 headless
```

Clone and install the matching Habitat-Lab version:

```bash
git clone --branch v0.1.7 git@github.com:facebookresearch/habitat-lab.git
cd habitat-lab

python -m pip install -r requirements.txt
python -m pip install -r habitat_baselines/rl/requirements.txt
python -m pip install -r habitat_baselines/rl/ddppo/requirements.txt
python setup.py develop --all
```

Install the remaining upstream dependencies:

```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html

pip install git+https://github.com/openai/CLIP.git
pip install gym==0.21.0
```

The environment remains based on the legacy ETPNav / Habitat stack; reproducing it on modern systems may require additional compatibility handling.

---

# Data and Trained Weights

ETPNav model weights are maintained by the original authors on ModelScope.

```bash
pip install modelscope
modelscope download --model admagic/ETPNav --local_dir ./data
```

Expected structure:

```text
data/
├── connectivity_graphs.pkl
├── datasets
├── ddppo-models
├── logs
├── pretrained
└── wp_pred
```

---

# Matterport3D

Matterport3D scene reconstructions are required for the current Habitat evaluation.

Follow the official Matterport3D access procedure and use the authorized download script.

The expected layout is:

```text
data/
└── scene_datasets/
    └── mp3d/
        └── {scene}/
            └── {scene}.glb
```

This repository does not redistribute Matterport3D data.

---

# Running

## Original ETPNav

The upstream training / evaluation entry points remain available under `run_r2r/` and `run_rxr/`.

For example:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_r2r/main.bash eval 2333
```

Exact commands used for controlled project experiments are stored in the corresponding research scripts and run manifests.

---

## Geometry Experiments

Geometry experiments are configured through the corresponding MID360 waypoint YAML files and research scripts.

Current experimental families include:

- Geometry V0-BNMS
- MRU-BNMS
- G-Min
- Geometry V1 experimental path

For G-Min:

```bash
bash scripts/run_gmin_frozen20.bash
```

Review the script and local dataset/checkpoint paths before executing.

Existing experiment outputs should not be overwritten.

---

# Current Development Status

## Completed

- [x] Separate Original ETP waypoint provider from Geometry provider
- [x] MID360-like local geometric frontend
- [x] Explicit local obstacle occupancy and footprint inflation
- [x] Geometry V0 controlled evaluation
- [x] Bounded-radial NMS correction
- [x] Corrected Geometry V0-BNMS baseline
- [x] MRU controlled negative ablation
- [x] Geometry V1 epistemic representation study
- [x] Project architecture redesign / simplification
- [x] G-Min deterministic coverage proposer
- [x] G-Min regression and same-state validation
- [x] First G-Min frozen-20 development evaluation
- [x] Evaluation telemetry and experiment provenance improvements
- [x] Initial Isaac observation-provider / compatibility work

## Current

- [ ] Proposal → GraphMap → Planner mechanism analysis
- [ ] Explain G-Min vs BNMS changed-outcome episodes
- [ ] Analyze GraphMap candidate / ghost survival
- [ ] Analyze STOP behavior under geometry proposals

## Next

- [ ] Decide whether frozen-planner adaptation is scientifically justified
- [ ] Profile and simplify the Geometry frontend
- [ ] Remove unused learned waypoint predictor residency in Geometry-only deployment
- [ ] Define the real RGB / depth / MID360 observation contract
- [ ] Validate the Isaac simulation bridge
- [ ] Integrate the high-level planner with the real ROS 2 navigation stack
- [ ] Real-robot controlled evaluation
- [ ] Independent final navigation evaluation

---

# Limitations

Current limitations include:

1. The principal navigation numbers above use only a 20-episode development panel.
2. The MID360-like Habitat frontend is an engineering abstraction, not a complete Livox scan-pattern simulation.
3. Current Geometry variants still use ETPNav's RGB and depth panorama encoders.
4. The learned waypoint predictor remains resident in GPU memory during current Geometry evaluation.
5. Current Geometry processing is slower end-to-end than Original ETPNav in the measured Habitat implementation.
6. Real-robot sensor synchronization and panorama construction are not yet finalized.
7. Habitat navigation performance does not establish real-robot safety.
8. Geometry V1's stronger epistemic/safety formulation remains experimental and is not the current primary method.

---

# Upstream ETPNav

This project is built on:

**ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments**

Dong An, Hanqing Wang, Wenguan Wang, Zun Wang, Yan Huang, Keji He, Liang Wang

IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

Original repository:

https://github.com/MarSaKi/ETPNav

Paper:

https://arxiv.org/abs/2304.03047

The original ETPNav implementation, model architecture, released weights, and benchmark contributions belong to the original authors.

---

# Acknowledgements

This project builds directly on ETPNav and its upstream dependencies, including Habitat, VLN-CE, CWP, Sim2Sim, and DUET.

Matterport3D data is used under its respective access terms and is not redistributed by this repository.

---

# Citation

If you use the original ETPNav implementation or model, please cite the original paper:

```bibtex
@article{an2024etpnav,
  title={ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments},
  author={An, Dong and Wang, Hanqing and Wang, Wenguan and Wang, Zun and Huang, Yan and He, Keji and Wang, Liang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024}
}
```

This repository is currently an ongoing research extension and does not yet define a separate publication citation.
