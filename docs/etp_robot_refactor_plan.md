# ETPNav robot-boundary refactor plan

## Objective and non-goals

The objective is to preserve the verified original ETP behavior while moving every
simulator dependency behind four narrow interfaces. This document is design only.
It does not implement a MID360 waypoint generator, alter GraphMap/ETP weights, add
ROS 2, connect Isaac Sim, or change the baseline evaluator.

The target package shape is:

```text
etp_robot/
├── models/
│   ├── etp_policy.py
│   ├── language_encoder.py
│   ├── panorama_encoder.py
│   └── graph_planner.py
├── graph/
│   ├── graph_map.py
│   └── geometry.py
├── perception/
│   ├── rgb_adapter.py
│   └── waypoint_provider.py
├── interfaces/
│   ├── observation_provider.py
│   ├── pose_provider.py
│   └── executor.py
└── configs/
```

Files should move only after characterization tests lock down the released path.
Initial work should add adapters around existing implementations, not duplicate the
299M-parameter policy or rewrite it.

## Canonical data contracts

All contracts must carry timestamps and named coordinate frames. No provider may
silently assume Habitat's y-up, negative-z-forward convention.

### ObservationProvider

Input to the application is an instruction or instruction ID. Output is an
observation batch containing:

- monotonically ordered capture timestamp;
- one or more calibrated RGB frames and their camera-frame names;
- depth/range data when the selected waypoint provider needs it;
- intrinsics and extrinsics version/identifier;
- validity, staleness, and synchronization status.

The provider owns acquisition and transport. `RGBAdapter` owns resizing,
normalization, panorama ordering, and conversion to ETP tensors.

### PoseProvider

For a requested timestamp, return:

- `T_map_base` as translation plus normalized quaternion;
- source and target frame IDs;
- timestamp and maximum interpolation/extrapolation age;
- covariance or an equivalent confidence/health state;
- localization state such as initializing, tracking, degraded, or lost.

`graph/geometry.py` is the only component allowed to convert between this canonical
frame and the legacy Habitat convention. FAST-LIO2 becomes one implementation, not
a dependency of GraphMap.

### WaypointProvider

Given a synchronized observation and pose context, return zero or more candidates:

- stable candidate ID for the current frame;
- relative SE(2) target `(x_forward, y_left, yaw)` or an explicitly named equivalent;
- distance, bearing, confidence, and traversability/validity;
- ETP-ready candidate RGB/depth-or-geometry feature vectors;
- source panorama/ray association when available;
- capture timestamp and frame ID.

The provider must not mutate GraphMap. Empty candidates are a valid result and must
not be encoded as an artificial point. The existing implementation is wrapped as
`ETPDepthWaypointProvider`; a later `MID360WaypointProvider` must satisfy the same
contract.

### Executor

Accept a semantic command rather than a Habitat action number:

- `Stop(target_node, target_pose)`; or
- `Navigate(back_path, frontier_node, target_waypoint)`.

Report queued/running/succeeded/failed/cancelled, current pose, progress, collision
or obstacle reason, and whether replanning is required. The executor owns low-level
control and emergency cancellation. It must never directly alter GraphMap state;
the application commits graph transitions only after execution feedback.

## Implementations by deployment

| Interface | Original Habitat baseline | Future real robot |
| --- | --- | --- |
| ObservationProvider | `HabitatObservationProvider` | `ROSObservationProvider` |
| PoseProvider | `HabitatPoseProvider` | `FastLIOPoseProvider` |
| WaypointProvider | `ETPDepthWaypointProvider` | `MID360WaypointProvider` |
| Executor | `HabitatExecutor` | `MPPIExecutor` |

An Isaac Sim deployment should implement the same four interfaces. It must not add
Isaac conditionals inside the ETP model or graph modules.

## Ownership after extraction

**ETP-core retains:**

- instruction tokenization and language embeddings;
- CLIP/panorama representation where applicable;
- candidate feature consumption;
- visited/ghost topology, localization thresholds, merging, and shortest paths;
- seven-dimensional graph position features after canonical geometry conversion;
- global graph transformer, masks, stop score, and ghost selection.

**Adapters own:**

- 12-view virtual camera naming and ordering;
- raw RGB/depth/range units, clipping, calibration, and synchronization;
- simulator/SLAM pose retrieval and coordinate transforms;
- navigability/geodesic queries used only by training or evaluation;
- turning, path following, collision avoidance, stop, metrics, and episode lifecycle.

## Staged migration

Current progress while licensed MP3D is pending: the narrow WaypointProvider seam,
bit-identical Original adapter, metric-depth point-cloud adapter, and an
evaluation-only MID360-like geometric adapter have been implemented and smoke-tested
on five official Habitat test-scene poses. PoseProvider and Executor remain design
only. Formal episode equivalence still requires authorized MP3D.

### Phase 0: characterize the original baseline

Keep the current official checkpoint and code path as the oracle. Save fixed-input
outputs for config loading, RGB/depth feature shapes, waypoint candidates, panorama
embeddings, GraphMap updates, planner logits, and semantic actions. The current
smoke utility already covers config, exact checkpoint load, waypoint, panorama, and
planner execution; real MP3D episode traces are added after licensed scenes exist.

Exit criterion: one and five `val_unseen` episodes complete with the unrefactored
baseline, and representative traces are captured without committing scene/model
data.

### Phase 1: extract PoseProvider and geometry

Wrap `VLNCEDaggerEnv.get_pos_ori` as `HabitatPoseProvider`. Move quaternion/heading
and candidate projection helpers behind `graph/geometry.py` while preserving exact
numeric output. Feed the existing GraphMap through the new interface.

Exit criterion: fixed poses and candidates yield the same node/ghost positions,
seven-dimensional position features, graph edges, and shortest paths within a
declared tolerance.

### Phase 2: extract ObservationProvider and WaypointProvider

Wrap the current 12-view Habitat observation packing and learned depth waypoint
network. Preserve existing outputs as `ETPDepthWaypointProvider`. Make the trainer
consume the provider result instead of directly naming sensor dictionary keys.

Exit criterion: on identical Habitat observations, candidate count/order, angles,
distances, features, panorama embeddings, and planner logits match the oracle.

### Phase 3: extract Executor

Replace action dictionaries with semantic commands. Implement `HabitatExecutor` by
delegating to the existing back-path, turn/forward, tryout, and stop code. Keep
metrics and video in the Habitat application shell.

Exit criterion: original and adapted Habitat executions produce equivalent action
traces and final pose/metrics for the characterization episodes.

### Phase 4: MID360 frontend inside Habitat

Only after phases 1-3 pass, implement `MID360WaypointProvider` against simulated
range data while retaining Habitat observation, pose, executor, episode, and metric
adapters. Compare its candidate contract and downstream GraphMap behavior against
`ETPDepthWaypointProvider` without changing the graph or transformer.

### Phase 5: Isaac Sim and robot deployments

First validate all four Isaac adapters, including timestamps and frame transforms.
Then connect ROS RGB, FAST-LIO2 pose, MID360 waypoints, and MPPI execution. Promote a
deployment only after stale-data, localization-loss, empty-waypoint, cancellation,
and controller-failure paths are tested.

## Frame and timing rules

- Define canonical robot coordinates explicitly; recommended semantics are x
  forward, y left, z up with right-handed rotations.
- Store full SE(3) poses even if GraphMap currently projects to the horizontal plane.
- Convert Habitat `[x, y, z]` and `[qx, qy, qz, qw]` only at the adapter boundary.
- Use observation-time pose for waypoint projection, not the newest pose available
  after neural inference.
- Reject or flag data outside configured synchronization and localization-covariance
  thresholds.
- Keep metres and radians as canonical units; UI/logging may convert to degrees.

## Failure and safety semantics

The policy can request a target but cannot assert that motion succeeded. Executor
feedback must distinguish collision, no valid local path, timeout, localization
loss, cancellation, and emergency stop. GraphMap should consume only confirmed pose
updates; failed targets remain available or are invalidated by an explicit policy,
never as an incidental controller side effect.

The real executor must have a safety layer below MPPI. ETP stop is a navigation
decision, not an emergency-stop mechanism.

## Required regression tests

- provider contract shape/dtype/frame/timestamp validation;
- Habitat pose conversion against current quaternion helpers;
- deterministic candidate-to-global projection;
- GraphMap node localization, ghost merge/consume, and shortest paths;
- provider equivalence on fixed Habitat observations;
- planner-logit equivalence for saved tensors;
- semantic-command translation to the original Habitat controller;
- empty candidate, stale observation, degraded pose, collision, and cancellation;
- one and five complete original `val_unseen` episodes before and after extraction.

## Next recommended step

Do not expand the example-scene algorithm. Wait for officially authorized full
MP3D scenes, then immediately run Original versus Geometry R2R-CE evaluation for
1, 5, and 20 fixed `val_unseen` episodes. Only after that evidence should the next
choice be made between a LiDAR geometry feature adapter and Isaac Sim validation.
