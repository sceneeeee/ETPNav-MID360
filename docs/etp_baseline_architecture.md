# Original ETPNav baseline architecture

## Scope and boundary

This note describes the released R2R-CE evaluation path at official commit
`160e65fe5d23ad83d024d57d710e47be6892fc68`. It documents the current algorithm;
it does not propose changes to the waypoint network, ETP transformer, or GraphMap.

The two ownership labels used below are:

- **ETP-core**: language/visual representation, learned waypoint proposal,
  topological memory, cross-modal graph planning, and high-level target selection.
- **Habitat-specific**: sensor construction, simulator observations and pose,
  navmesh/geodesic services, discrete actuation, episode lifecycle, metrics, and
  video rendering.

```text
Habitat RGB/depth + instruction                 [Habitat-specific]
                |
                v
12-view observation packing                    [Habitat-specific adapter]
                |
                v
CLIP RGB + DD-PPO depth -> waypoint TRM/NMS    [ETP-core]
                |
                v
panorama encoder -> GraphMap update            [ETP-core]
                ^             |
 Habitat pose --+             v
 [Habitat-specific]   language-conditioned graph planner [ETP-core]
                              |
                              v
                    stop or target ghost         [ETP-core]
                              |
                              v
 turn/forward/collision controller + reset      [Habitat-specific]
```

## Entry points and orchestration

`run.py` loads `run_r2r/iter_train.yaml`, imports registrations from
`habitat_extensions` and `vlnce_baselines`, looks up trainer `SS-ETP`, and dispatches
evaluation. `run_r2r/r2r_vlnce.yaml` defines the Habitat task, sensors, action set,
dataset paths, and simulator parameters. `run_r2r/main.bash` is the original
multi-GPU reference; `run_r2r/main_local.bash` supplies a one-GPU, one-environment,
bounded-episode evaluation without changing it.

The online loop is implemented in `vlnce_baselines/ss_trainer_ETP.py::rollout`:

1. reset vector environments and tokenize instructions;
2. encode the instruction once;
3. predict waypoints from the current 12-view panorama;
4. build panorama features;
5. query simulator pose and update GraphMap;
6. build global-map tensors and run the graph planner;
7. select stop or an unvisited ghost;
8. execute the graph path and target through the Habitat environment;
9. repeat until stop, timeout, or episode completion.

## 1. Observation input

**Habitat-specific.** `SS-ETP._set_config` expands the base RGB and depth sensors to
12 cameras separated by 30 degrees. Each environment reset/step returns the camera
keys, instruction sensor, and task metadata. Habitat Baselines batches the resulting
dictionary and applies configured observation transforms.

Current model-facing assumptions are a batch of 12 headings, RGB at 224x224 and
depth at 256x256, with the base view plus suffixes `_30` through `_330`. This naming,
layout, dtype, and normalization are adapter concerns and should not leak into a
robot observation source.

## 2. RGB pipeline

**ETP-core with a Habitat-format adapter.** `CLIPEncoder` flattens the panorama to
`B*12` images, converts Habitat's channel-last uint8 tensors to CLIP input format,
and runs frozen OpenAI CLIP ViT-B/32. The encoder produces a 512-dimensional feature
for each view. The local compatibility patch ensures its cache is under ignored
`data/pretrained/CLIP`.

## 3. Depth pipeline

**ETP-core with a simulator-depth adapter.** `VlnResnetDepthEncoder` uses the frozen
DD-PPO ResNet-50 checkpoint `gibson-2plus-resnet50.pth`. It processes the 12
256x256 single-channel depth frames and emits spatial depth features, later pooled
to 128 dimensions per view. Habitat currently supplies normalized depth according
to its sensor configuration; a future adapter must define units, clipping, invalid
samples, and normalization explicitly.

## 4. Waypoint predictor

**ETP-core.** `BinaryDistPredictor_TRM` receives the reshaped RGB/depth embeddings
and predicts a heatmap with 120 heading bins and 12 distance bins. The policy wraps
the angular boundary, applies non-maximum suppression, and keeps at most five
candidates. Distance index `d` maps to `(d + 1) * 0.25` metres. Candidate headings,
distances, RGB/depth embeddings, angle features, and source panorama indices are
returned together.

The model contains 17,614,200 parameters and loads
`data/wp_pred/check_cwp_bestdist_hfov90`. Encoder outputs are restored from the
clockwise predictor ordering to the counter-clockwise panorama ordering before
pooling and candidate feature lookup; this original ordering is preserved.

The implementation is now owned by
`vlnce_baselines/waypoint_provider/original_etp.py`. `ETP.forward(mode='waypoint')`
delegates through the narrow `WaypointProvider.predict(observations, in_train)`
seam and returns the unchanged ten-field mapping. The alternate
`MID360GeometryWaypointProvider` uses the same panorama encoding and candidate
feature construction, replacing only heatmap proposal. No provider mutates
GraphMap or calls the graph planner.

## 5. Panorama encoder

**ETP-core.** Candidate views are placed first and tagged as navigable; remaining
panorama views are retained as context. RGB features, depth features, angular
features, navigation-type embeddings, and view masks enter the two-layer panorama
transformer in the ETP VLN-BERT model. Its output is a 768-dimensional embedding per
view. The masked mean represents the visited graph node; candidate embeddings seed
or update ghosts.

## 6. GraphMap

**ETP-core, with current Habitat frame conventions embedded in geometry helpers.**
`vlnce_baselines/models/graph_utils.py::GraphMap` stores:

- visited node position, panorama embedding, visit step, and stop score;
- ghost observations, mean/augmented position, averaged candidate embedding, and
  one or more frontier nodes;
- a weighted NetworkX graph of visited nodes;
- all-pairs Dijkstra paths and distances recomputed after updates.

Candidates within `loc_noise` of an existing node become graph edges. When ghost
merging is enabled, nearby candidate proposals share a ghost and their positions
and embeddings are averaged. Consumed ghosts are removed. Graph position features
contain four angular components plus normalized Euclidean distance, graph shortest
distance, and graph step count (seven values total).

`estimate_cand_pos` assumes Habitat coordinates: y is vertical, negative z is
forward at zero heading, and quaternion coefficients are `[x, y, z, w]`. The graph
data structures are reusable, but coordinate conversion belongs behind a pose and
geometry boundary.

## 7. Graph planner

**ETP-core.** `_nav_gmap_variable` creates a sequence consisting of stop, visited
nodes, and unvisited ghosts. It supplies image embeddings, visit-step embeddings,
seven-dimensional position features, masks, visited flags, and pairwise graph
distances. `forward_navigation` uses the language-conditioned global-map transformer
and a scalar action head. Visited entries are masked; index zero represents stop.

The released checkpoint has iteration 12000, 771 state tensors, and loads into the
policy with zero missing or unexpected keys. A synthetic graph forward pass produces
valid logits and selects an unvisited ghost.

## 8. Simulator pose acquisition

**Habitat-specific.** `SS-ETP.get_pos_ori` calls each vector environment's
`VLNCEDaggerEnv.get_pos_ori`, which calls `sim.get_agent_state()`. Position is the
Habitat xyz vector; orientation is serialized from the simulator quaternion to
`[imag_x, imag_y, imag_z, real]`. Training/video modes can also simulate each
candidate with `get_cand_real_pos` and query geodesic distance or navigability.

These direct simulator calls are the principal seam for a future `PoseProvider`.
The ETP core only needs a current pose in a declared map frame and deterministic
relative-to-global waypoint geometry.

## 9. High-level action

**ETP-core.** Evaluation uses argmax planner feedback. Planner index zero means
stop. Any other index identifies a ghost. GraphMap chooses that ghost's nearest
front node and, when `IL.back_algo=control`, provides the shortest visited-node path
from the current node to the frontier. The high-level command therefore has one of
two semantic forms:

- stop at a selected visited node; or
- follow a visited-node back-path, then approach the selected ghost target.

The current trainer immediately translates this semantic command into a Habitat
action dictionary. That translation should move behind `Executor` during extraction.

## 10. Low-level control

**Habitat-specific.** `VLNCEDaggerEnv.step` accepts only high-to-low action 4 or
stop action 0. For a target it follows the graph back-path, computes relative
heading/distance, rounds turns to Habitat's discrete turn unit, and issues repeated
MOVE_FORWARD commands at the configured 0.25 m step. With `tryout=True`, collision
causes a randomized left/right trial sequence over several headings. The alternative
`teleport` back algorithm directly sets simulator state and is not used by the local
baseline. Stop optionally follows a graph path and then calls Habitat STOP.

Observation refresh, collision status, episode completion, SPL/NDTW metrics, and
video generation all remain inside the Habitat environment layer.

## 11. Habitat-specific dependency inventory

| Dependency | Current call sites / responsibility | Replacement seam |
| --- | --- | --- |
| Habitat Config/registry | `run.py`, config defaults, trainer/env registration | application composition/config adapter |
| Dataset and vector env | episode enumeration, reset/step/pause | `ObservationProvider` plus episode runner |
| RGB/depth sensors | 12 virtual cameras, rendering, transforms | `ObservationProvider` / `RGBAdapter` |
| `habitat_sim` pose | xyz/quaternion and observations-at-pose | `PoseProvider` |
| navmesh/geodesics | teacher paths, distances, candidate real pose | training/evaluation-only map service |
| Habitat actions | turn, forward, stop, collision tryout | `Executor` |
| metrics/video | SPL/NDTW/success and render output | evaluation observer |

## 12. Real-robot replacement points

The smallest stable interfaces are:

- `ObservationProvider`: synchronized instruction/RGB/depth-or-range input with
  timestamps and calibration metadata;
- `PoseProvider`: current SE(3) pose, covariance/health, and explicit source/target
  frames;
- `WaypointProvider`: relative candidate poses plus ETP-ready features and validity;
- `Executor`: accept stop or a graph target/path, report progress, collision/failure,
  and final pose.

For the Habitat baseline these map to `HabitatObservationProvider`,
`HabitatPoseProvider`, `ETPDepthWaypointProvider`, and `HabitatExecutor`. Future robot
implementations map to `ROSObservationProvider`, `FastLIOPoseProvider`,
`MID360WaypointProvider`, and `MPPIExecutor`. The ETP language encoder, panorama
encoder, GraphMap lifecycle, graph planner, and action selection stay above those
interfaces.

Detailed contracts and a staged extraction plan are in
`docs/etp_robot_refactor_plan.md`.
