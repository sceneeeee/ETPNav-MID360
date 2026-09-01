# ETP to Isaac adapter contract

This document defines the initial integration seams. It does not import Isaac
into ETPNav and does not alter the frozen MID360 V0 algorithm.

## Boundary

```text
Isaac task/runtime
  -> ObservationProvider
  -> PoseProvider
  -> unchanged OriginalETPWaypointProvider or MID360GeometryWaypointProvider
  -> unchanged ETP GraphMap/planner
  -> Executor
  -> Isaac task/runtime
```

There are four independent interfaces:

| Interface | Simulator baseline | Initial Isaac implementation | Later robot implementation |
|---|---|---|---|
| ObservationProvider | HabitatObservationProvider | IsaacObservationProvider | ROSObservationProvider |
| PoseProvider | HabitatPoseProvider | IsaacGroundTruthPoseProvider | FastLIOPoseProvider |
| WaypointProvider | Original / MID360Geometry | same frozen providers | later LiDAR geometry provider |
| Executor | HabitatExecutor | IsaacExecutor | MPPIExecutor |

Simulator ground-truth pose is mandatory for the first Isaac experiment.
FAST-LIO is intentionally excluded so localization cannot confound the adapter
and planner validation.

## ObservationProvider

`ObservationProvider.get_observations()` returns the existing ordered ETP map.
It does not return Isaac sensor objects.

Required keys, in insertion order:

```text
rgb, depth,
rgb_30, depth_30,
...
rgb_330, depth_330
```

### Canonical values

| Field | Contract |
|---|---|
| headings | `0,30,...,330` degrees |
| heading sign | positive counter-clockwise about local +Z |
| heading zero | robot/base +X forward |
| RGB | batched BHWC `uint8`, 224x224x3 |
| depth | batched BHW1 `float32`, normalized to [0,1] |
| depth metric range | 0.0--10.0 m before normalization |
| RGB FoV | 90 degrees horizontal |
| depth FoV | 90 degrees horizontal |
| capture time | all 12 yaw stations from one simulator tick |

Isaac's raw RGB annotator can return RGBA; the adapter drops alpha. Isaac
`distance_to_image_plane` is metric. NaN/Inf becomes the configured far value,
then depth is clipped and normalized. The adapter refuses to resize because a
silent resize changes the experimental sensor definition.

The checked-in `IsaacPanoramaObservationAdapter` is deliberately dependency
isolated. It accepts two NumPy mappings keyed by integer heading:

```python
observations = adapter.adapt(rgb_by_heading, depth_by_heading)
```

It is an offline contract adapter, not proof that Isaac ran. A future runtime
provider owns Isaac imports, Camera data access, and any GPU/CPU or interprocess
transport, then calls this adapter at the ETP process boundary.

### Rig construction

The exact rig is 12 yaw stations, each containing two co-located cameras:

- RGB camera: 224x224, `rgb`;
- depth camera: 256x256, `distance_to_image_plane`;
- shared pose and 90-degree horizontal FoV.

This is 24 Isaac Camera prims because resolution is per `CameraCfg`, not per
annotator type. They are perception-only and must not alter robot mass,
collision shapes, articulation or control rate.

### Heading mapping

| Isaac base yaw | ETP keys | Provider-facing panorama index |
|---:|---|---:|
| 0 | `rgb`, `depth` | 0 |
| 30 | `rgb_30`, `depth_30` | 1 |
| 60 | `rgb_60`, `depth_60` | 2 |
| 90 | `rgb_90`, `depth_90` | 3 |
| 120 | `rgb_120`, `depth_120` | 4 |
| 150 | `rgb_150`, `depth_150` | 5 |
| 180 | `rgb_180`, `depth_180` | 6 |
| 210 | `rgb_210`, `depth_210` | 7 |
| 240 | `rgb_240`, `depth_240` | 8 |
| 270 | `rgb_270`, `depth_270` | 9 |
| 300 | `rgb_300`, `depth_300` | 10 |
| 330 | `rgb_330`, `depth_330` | 11 |

Do not negate or reverse yaw in this adapter. The frozen ETP panorama encoder
already converts this counter-clockwise observation map to the learned
predictor's historical clockwise order, then reconstructs counter-clockwise
pano/candidate outputs.

## PoseProvider

The initial `IsaacGroundTruthPoseProvider` reads:

- world translation: robot `root_pos_w`, `(x,y,z)` metres;
- world orientation: robot `root_quat_w`, scalar-first `(w,x,y,z)`;
- planar heading: yaw about world +Z;
- camera extrinsics: Camera `pos_w` and `quat_w_world` when required.

It returns a canonical right-handed navigation pose with +X forward, +Y left,
+Z up, metres and radians. A scene/episode start transform should provide an
episode-local frame to ETP while preserving the original world pose for
metrics. The provider, not GraphMap, owns quaternion ordering, camera optical
convention and world/local transforms.

The R2R-to-Isaac dataset commonly maps Habitat positions as `(x,y,z)` to
`(x,-z,y)`, but some starts are manually adjusted. Runtime pose must therefore
come from Isaac state; it must never be recreated from the source Habitat
episode using only that formula.

## WaypointProvider

No new waypoint interface is needed for ISAAC-V0. Both frozen implementations
consume the exact observation map above and return the established contract:

```text
cand_rgb, cand_depth, cand_angle_fts, cand_img_idxes,
cand_angles, cand_distances,
pano_rgb, pano_depth, pano_angle_fts, pano_img_idxes
```

ISAAC-V0 feeds normalized Isaac depth to the existing reconstruction and
MID360-like filter. Original and Geometry must see identical pano RGB/depth for
an ablation. The frozen provider and scoring code are not to be retuned during
adapter work.

ISAAC-V1 adds a point-cloud acquisition adapter before the existing
traversability stage. Initial preferred source is the pinned Lab
`RayCasterCfg.ray_hits_w`, transformed from world into base frame. RTX LiDAR is
a later alternative, not an implicit MID360 model.

## Executor

ETP candidate angles are counter-clockwise relative to the current navigation
heading. A local polar waypoint `(theta,d)` becomes:

```text
target_base = [d*cos(theta), d*sin(theta), 0]
target_world = root_position + Rz(root_yaw) * target_base
```

The Isaac executor owns target following and STOP:

```text
ETP target / STOP
  -> local target or short path
  -> benchmark velocity command [vx,vy,omega]
  -> existing Go2 low-level RSL policy
  -> joint-position action
```

The first physics adapter should reuse Go2 and its released low-level executor.
A kinematic executor may be used first for a clearly labelled integration
smoke test, but its results are not benchmark results. A wheeled robot is a
later controlled experiment.

STOP must set the benchmark wrapper's stop flag. Success remains the metric
implementation's conjunction of STOP and distance below the episode radius;
reaching a geometric target alone is not a successful VLN episode.

## Interface invariants and tests

The offline tests enforce:

- all twelve headings are present and no extra heading is accepted;
- stable interleaved key insertion order;
- exact RGB/depth sizes, channel layouts and dtypes;
- RGBA-to-RGB conversion;
- metric depth clipping, invalid handling and 0--10 m normalization;
- positive Isaac yaw remains the positive ETP suffix;
- no Isaac module import is required by the ETP test suite.

Runtime acceptance remains separate:

1. boot pinned AppLauncher on a supported native host;
2. render a primitive with valid RGB and metric depth;
3. validate camera poses/intrinsics and all 12 timestamps;
4. compare adapter output to these offline invariants;
5. run Original and frozen Geometry waypoint forwards on the same tick;
6. only then add Matterport USD and the executor.
