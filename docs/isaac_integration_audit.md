# VLN-CE-Isaac compatibility audit

Audit date: 2026-09-02

Status: **SOURCE AUDIT PASS; ISAAC RUNTIME NOT INSTALLED**

This audit is tied to exact source revisions. It does not claim that
VLN-CE-Isaac, Matterport USD, or Isaac Sim ran on this machine.

## Reproducible source pins

| Source | External path | Commit | State |
|---|---|---|---|
| VLN-CE-Isaac | `~/SURF/isaac-vlnce/VLN-CE-Isaac` | `bd301c042c77ef3d84c329056a9d5856ebdf53ae` | clean `main`, `origin=https://github.com/zikuicai/VLN-CE-Isaac.git` |
| Modified Isaac Lab | `~/SURF/isaac-vlnce/IsaacLab` | `4d558ec83878c4892a46591c85ba91ac9d3c1834` | clean `main`, `origin=https://github.com/yang-zj1026/IsaacLab.git` |

Neither repository is vendored into ETPNav. No update was performed after the
pins were recorded.

The benchmark README asks for Ubuntu 22.04, Python 3.10, and pip packages at
Isaac Sim `4.2.0.2`. Its modified Isaac Lab has `VERSION=1.1.0`, uses the old
`omni.isaac.lab` API generation, pins `torch==2.2.2` and
`gymnasium==0.29.0`, and declares Python `>=3.10`. The VLN extension declares
Python `>=3.10` and `torch>=2.2.0`; its bundled RSL-RL reports version 2.0.2.

This is a custom compatibility combination. Upstream Isaac Lab 1.1 targets
Isaac Sim 4.0/4.1; Isaac Lab 1.2 was the first upstream generation announced
for Isaac Sim 4.2. The modified fork and exact commit are therefore part of
the benchmark definition. It must not be replaced with current Isaac Lab or
Isaac Sim 5/6 as an installation convenience.

Primary references:

- [VLN-CE-Isaac pinned README](https://github.com/zikuicai/VLN-CE-Isaac/blob/bd301c042c77ef3d84c329056a9d5856ebdf53ae/README.md)
- [Modified Isaac Lab pinned setup](https://github.com/yang-zj1026/IsaacLab/blob/4d558ec83878c4892a46591c85ba91ac9d3c1834/source/extensions/omni.isaac.lab/setup.py)
- [Isaac Lab 1.2 / Isaac Sim 4.2 announcement](https://github.com/isaac-sim/IsaacLab/discussions/1008)
- [Isaac Sim 4.2 Python installation](https://docs.isaacsim.omniverse.nvidia.com/4.2.0/installation/install_python.html)

## Host audit and hard gate

| Item | Measured value |
|---|---|
| Linux | Ubuntu 22.04.5 LTS, kernel `6.18.33.2-microsoft-standard-WSL2` |
| Virtualization | WSL2 (`systemd-detect-virt` = `wsl`) |
| Windows host | Windows 10 Enterprise, build 19045 |
| WSL | 2.7.12.0; WSLg 1.0.73.2 |
| GPU | NVIDIA GeForce RTX 4060 Ti, compute capability 8.9 |
| Driver | Windows KMD 610.88; CUDA UMD 13.3 reported by WSL `nvidia-smi` |
| VRAM | 8188 MiB total; 1852 MiB used and 6098 MiB free at audit time |
| RAM | 15 GiB visible, about 14 GiB available; 4 GiB swap |
| WSL disk | 1007 GiB total, 865 GiB free |
| Windows C: | 130 GiB total, about 9.5 GiB free |
| Windows D: | 824 GiB total, about 34.6 GiB free |
| Graphics | WSLg D3D12 renderer on RTX 4060 Ti; direct rendering; OpenGL 4.2 |

NVIDIA's Isaac Sim 4.2 minimum is 32 GB RAM, 50 GB SSD, an RTX 3070-class
GPU, and 8 GB VRAM. The GPU only meets the VRAM floor; RAM is half the minimum.
Neither Windows volume currently has the stated free-space minimum. The
modified Isaac Lab itself recommends 32 GB RAM and 16 GB VRAM. See the
[Isaac Sim 4.2 requirements](https://docs.isaacsim.omniverse.nvidia.com/4.2.0/installation/requirements.html).

### WSL2 verdict

**NO-GO for a formal runtime.** CUDA visibility and accelerated WSLg OpenGL do
not establish support for the Vulkan/RTX Kit application. NVIDIA's supported
4.2 operating systems are native Ubuntu 20.04/22.04 and Windows 10/11; NVIDIA
also states that the `isaacsim` package is not supported in WSL2. NVIDIA's
documented Windows/WSL arrangement runs Isaac Sim natively on Windows and uses
WSL for ROS nodes, not for the simulator.

- [Isaac Sim 4.2 ROS launch note: package unsupported in WSL2](https://docs.isaacsim.omniverse.nvidia.com/4.2.0/ros2_tutorials/tutorial_ros2_launch.html)
- [NVIDIA staff: WSL2 and Xvfb not supported; Linux requires Vulkan](https://forums.developer.nvidia.com/t/isaac-sim-x86-64-headless-docker-wsl2-support/278252/3)
- [Windows-native Isaac Sim with WSL ROS](https://docs.isaacsim.omniverse.nvidia.com/4.2.0/installation/install_ros.html)

The pip installation being resolvable does not change this verdict. In
particular, headless cameras still require the renderer and camera-enabled
AppLauncher experience. Matterport USD RGB, depth, Replicator camera products,
and RTX LiDAR all depend on that runtime.

### Installation decision

Decision for this host: **E. OTHER — SOURCE_AUDIT_ONLY**.

Formal target route: **A. NATIVE_LINUX_4_2**, using the two pinned repositories
above and Isaac Sim 4.2.0.2. A Windows-native port is platform-plausible because
Isaac Sim 4.2 and the modified Isaac Lab support Windows, and the fork includes
`isaaclab.bat`. It is not benchmark-validated: VLN-CE-Isaac documents only bash,
Linux symlinks and apt/GLX troubleshooting. The present Windows disks and RAM
also fail the minimum resource gate.

No `vlnce-isaac` environment was created and no Isaac packages were downloaded.
This avoids a large known-unsupported installation and leaves `etpnav_legacy`
unchanged.

## Benchmark task registry

The extension registers exactly four Gymnasium tasks:

| ID | Environment | Robot/config |
|---|---|---|
| `go2_matterport_base` | `ManagerBasedRLEnv` | Go2 base config |
| `go2_matterport_vision` | `ManagerBasedRLEnv` | Go2 vision config |
| `h1_matterport_base` | `ManagerBasedRLEnv` | H1 base config |
| `h1_matterport_vision` | `ManagerBasedRLEnv` | H1 vision config |

There is no registered generic Matterport task and no task whose ID contains
`blind`. The `base` variants are locomotion/base-policy configurations, not a
separately specified VLN blind benchmark. The demo commands use the two
`*_vision` IDs. Go2 is configured for one environment; H1 vision defaults to
4096 in source, but `demo_planner.py --num_envs=1` overrides it through
`parse_env_cfg`.

The benchmark README still contains `TODO: Release the VLA example and
evaluation code`. The released runnable path is an expert-path PD demo plus a
VLN metric wrapper, not the released NaVILA inference/evaluation stack.

## Observation contract

### Released forward camera

Both vision configs provide one forward `rgbd_camera` attached to the Go2
`base` or H1 `pelvis`:

- offset position `(0.1, 0.0, 0.5)` metres;
- offset quaternion `(-0.5, 0.5, -0.5, 0.5)` in Isaac Lab's scalar-first
  convention and default camera-offset `ros` convention;
- 512 x 512 pixels;
- `rgb` and `distance_to_image_plane`;
- pinhole focal length 24.0 and horizontal aperture 54.0, which the pinned
  Isaac Lab formula converts to about 96.7 degrees horizontal FoV.

The model-facing observation groups are:

| Group | Shape/dtype for one env | Meaning |
|---|---|---|
| `camera_obs` | `(1,512,512,4)`, RGB annotator dtype (normally `uint8`) | Unmodified Isaac RGB/RGBA; demo slices `:3` |
| `depth_obs` | `(1,1,512,512)`, `float32` | Metric image-plane depth; NaN/Inf replaced by 5.0 m |
| `viz_camera_obs` | `(1,512,512,4)` | Rear/visualization RGB camera |
| `policy` / `proprio` | flattened float tensors | Robot state, velocity command, joints and last action |

The Go2 vision policy additionally consumes a `(1,459)` local height-map tensor
computed from a 32-channel, 360-degree `RayCasterCfg` LiDAR pattern. Raw ray
hits are not exposed as a public observation group.

Camera intrinsics and poses exist in the sensor object but are not registered
in the released observation groups:

- `intrinsic_matrices`: `(N,3,3)`;
- `pos_w`: `(N,3)` in metres;
- `quat_w_world`: `(N,4)`, scalar-first `(w,x,y,z)`.

The wrapper can retain low-level proprioceptive history (Go2 demo uses length
9). It does not retain RGB/depth history. The high-level demo receives only
`camera_obs`.

### Sensor API implication

The pinned Camera API supports multiple simultaneous camera objects, RGB and
metric depth. Replicator also supports multiple render products. The benchmark
does not currently configure a panorama, but a simultaneous panorama is an
API-level extension; no robot rotation or physics mutation is required.

## Episode contract and provenance

Only the small public `vln_ce_isaac_v1.json.gz` (501,265 bytes, SHA-256
`ceb2a2a9ac1f6d1a1ebbf9fe205867b101b1b7bb61d532a4d491124c7c0b0eec`)
was downloaded to `/tmp` for schema inspection. It was not added to Git.

Top level: `{ "episodes": [...] }`. There is no split field. Each episode has:

- `episode_id`, `episode_new_id`, `trajectory_id`;
- Habitat-style `scene_id` such as `mp3d/<scene>/<scene>.glb`;
- `start_position`, `start_rotation`;
- one or more `goals` with `position` and `radius`;
- `instruction.instruction_text` and 200 `instruction_tokens`;
- `reference_path`, `gt_locations`, `gt_actions`, `gt_forward_steps`;
- `info.geodesic_distance`.

Measured contents:

- 1077 episodes = 359 trajectories x 3 instructions;
- 11 Matterport scenes;
- goal radius always 3.0 m;
- reference paths contain 4--7 vertices;
- GT paths contain 18--69 locations;
- `episode_new_id` is contiguous 0--1076.

Comparison against the official ETP local R2R-CE val_unseen JSON shows that
this is a **modified subset**, not a direct copy:

- source val_unseen: 1839 episodes / 613 trajectories;
- Isaac subset: 1077 episodes / 359 trajectories (58.6%);
- scene ID, trajectory ID, instruction and `info` are retained for included
  episodes;
- the usual point transform is Habitat `(x,y,z)` to Isaac `(x,-z,y)`;
- 72 retained starts contain additional 0.05--0.5 m position adjustments;
- rotations are converted to scalar-first, Z-up yaw quaternions and quantized
  to 30-degree headings;
- `gt_locations`, `gt_actions`, `gt_forward_steps`, and `episode_new_id` are
  newly supplied and do not exactly copy the R2R-CE GT sidecar.

The absence of a split field plus exact scene/instruction correspondence is
evidence that it derives from R2R-CE val_unseen. Formal results must nevertheless
use this benchmark JSON as its own episode definition.

## Metrics and termination

The actual implementation is in `utils/measures.py`:

| Metric | Source behavior |
|---|---|
| Path length | Sum of Euclidean distances between successive robot root positions |
| Distance to goal / NE | Find nearest `gt_locations` waypoint, add Euclidean distance to it plus remaining GT polyline length |
| Success | `stop` was called and distance-to-goal is strictly below the episode goal radius |
| SPL | Success x initial distance / max(initial distance, accumulated Euclidean robot path) |
| Oracle navigation error | Minimum observed distance-to-goal |
| Oracle success / OS | Whether distance-to-goal was ever below goal radius |

This distance-to-goal is a GT-polyline approximation, not a simulator geodesic
query. The final `distance_to_goal` value is the implementation's navigation
error even though no class is literally named `NavigationError`.

Success does not itself terminate the base environment. The demo sets the stop
flag after ten consecutive iterations within 0.5 m of the final expert point.
An episode can also end through base/torso contact, bad orientation, underlying
timeout, 1000 stuck checks, the wrapper's 10,000 high-level-step maximum, or
the demo's goal break. No collision metric is registered; contact is a
termination signal only.

These details are material differences from assuming Habitat-Lab's evaluator.

## Pose and coordinate contract

The pinned stack uses a right-handed, metre-scale world with +Z up and +X
forward. Isaac Lab state quaternions are scalar-first `(w,x,y,z)`. Camera data
offers three explicit conventions:

- world/robotics camera: +X forward, +Z up;
- ROS optical camera: +Z forward, -Y up;
- USD/OpenGL camera: -Z forward, +Y up.

Robot root pose comes from `scene["robot"].data.root_pos_w` and
`root_quat_w`. Camera world pose and intrinsics come from the Camera data
object. The ETP bridge must consume those explicit fields and must not infer
heading from image order.

Reference: [Isaac Sim 4.2 conventions](https://docs.isaacsim.omniverse.nvidia.com/4.2.0/reference_conventions.html).

## Control contract

The released demo does not call a VLM. It follows `gt_locations` with a simple
controller:

```text
expert path point
  -> proportional forward speed and yaw-rate
  -> [vx, vy=0, omega]
  -> inject command into pretrained low-level-policy observation
  -> bundled RSL-RL policy
  -> Go2/H1 joint-position action
  -> PhysX step
```

The wrapper documents the high-level action as `[vx,vy,omega]`. It warm-starts
the low-level policy, injects the command into robot-specific observation
indices, obtains joint actions, and steps the `ManagerBasedRLEnv`.

The repository also contains `NavigationAction`, `VLMActions`, path follower,
mid-level and robot-velocity command classes. They are not registered by the
four released task configs. ViPlanner is acknowledged but is not imported by
the runnable demo. Therefore:

- generic released infrastructure: task registry, Matterport scene config,
  episode loader, observation groups, RSL wrapper, and metrics;
- Go2/H1-specific: assets, policy observation layouts, pretrained low-level
  checkpoints, warmup counts, contacts and camera mount;
- NaVILA-specific/dormant: VLM action and command-generator scaffolding;
- unavailable: released NaVILA model/evaluation loop.

### Executor staging recommendation

1. Use a kinematic/point-agent executor only for adapter and coordinate
   closed-loop validation; label it non-benchmark.
2. For the first physics experiment, use Go2 plus the released low-level policy
   and let ETP select only the high-level target. This controls the locomotion
   variable against the benchmark demo.
3. Add a wheeled robot only after ETP sensor, pose, STOP and metrics contracts
   are validated. It is more relevant to the final robot but introduces a new
   dynamics/controller confounder.

## Matterport USD audit

The public Hugging Face repository is owned by `Zhaojing`, is public and
ungated, and declares MIT metadata. Its visible files are:

| File | Visible size |
|---|---:|
| `matterport_usd.zip` | 5,020,545,930 bytes (about 4.68 GiB) |
| `vln_ce_isaac_v1.json` | 9,874,756 bytes |
| `vln_ce_isaac_v1.json.gz` | 501,265 bytes |

Source: [Hugging Face file listing](https://huggingface.co/datasets/Zhaojing/VLN-CE-Isaac/tree/main).

`matterport_usd.zip` was **not downloaded**. The loader resolves each episode
as:

```text
assets/matterport_usd/<scene_id>/<scene_id>.usd
```

The source therefore requires one root USD per scene directory. It adds the
USD as a referenced prim, applies collision properties and a physics material,
and does not load a navmesh or separate navigation file. Metrics use the JSON
GT polyline. The benchmark path does not configure semantic data. Whether the
archive embeds or references textures, contains extra meshes, or includes PLY
semantics cannot be truthfully established without inspecting the archive.

`omni.isaac.matterport` contains a real but limited conversion path:

- input: a Matterport textured `.obj` (with its material/texture companions);
- converter: `omni.kit.asset_converter` through the extension's asynchronous
  GUI load path;
- output: a same-basename `.usd`, metres and Z-up requested in converter
  options;
- synchronous benchmark loading accepts the resulting USD.

There is no released bulk conversion script that reproduces the Hugging Face
archive, and no code that proves identical material layout, transforms,
collision behavior or episode start corrections. Once officially authorized
Matterport OBJ data arrives, local conversion is technically possible but must
be parity-tested before it is accepted as benchmark-equivalent.

The Hugging Face `license: mit` card does not override Matterport scene rights.
The official Matterport repository states that code is MIT while data is under
the Matterport3D Terms of Use and requires a signed access agreement. See the
[official Matterport repository](https://github.com/niessner/Matterport#license).

## ETP sensor compatibility

The current benchmark has one forward 512-square RGB-D camera. ETP requires 12
simultaneous headings, RGB 224 square, depth 256 square, 90-degree FoV, and the
Habitat 0--10 m normalized depth convention. The minimum faithful rig is:

- 12 yaw stations at `0,30,...,330` degrees;
- positive yaw counter-clockwise about robot +Z from +X forward;
- at each station, co-located RGB 224x224 and depth 256x256 pinhole cameras;
- 90-degree horizontal FoV (with focal length 24, horizontal aperture 48 gives
  90 degrees in the pinned Camera formula);
- ETP sensor origin/height matched independently of robot collision geometry;
- all stations rendered on the same simulation tick.

Because one Isaac `CameraCfg` has one resolution for all its data types, exact
ETP resolutions require 24 camera prims grouped as 12 co-located RGB/depth
pairs. A 12-camera 256-square RGB-D rig followed by RGB resize is cheaper but
is not the exact sensor contract and must be treated as a separate ablation.

The dependency-isolated adapter maps Isaac station yaw `psi` directly to ETP
keys `rgb_psi` and `depth_psi`. It deliberately does not reverse the views;
`OriginalETPWaypointProvider.encode_panorama` remains the single owner of the
legacy clockwise predictor packing.

## MID360 path

### ISAAC-V0

```text
Isaac 12-view metric depth
  -> normalize to the existing 0--10 m ETP observation contract
  -> frozen depth_panorama_to_pointcloud
  -> frozen MID360LikeFilter
  -> frozen traversability and candidate contract
```

This changes only the observation source. No MID360 V0 scoring code should be
modified.

### ISAAC-V1

The pinned benchmark already uses Isaac Lab's Warp `RayCasterCfg` with a
`LidarPatternCfg`. It exposes world-frame `ray_hits_w` and is the lowest-risk
first point-cloud source on this API generation. It supports static targets and
one mesh target, so the imported Matterport USD must first be verified to
resolve as one usable static ray-cast mesh.

Isaac Sim 4.2 also offers PhysX generic range sensors and RTX LiDAR. RTX LiDAR
supports solid-state profiles and point-cloud annotators but consumes the RTX
render path and has no official Livox MID360 profile. On this constrained
hardware it is not the first choice. No option may be described as real MID360
physics without a validated Livox scan profile.

- [Isaac Sim 4.2 RTX LiDAR](https://docs.isaacsim.omniverse.nvidia.com/4.2.0/features/sensors_simulation/isaac_sim_sensors_rtx_based_lidar.html)
- [Isaac Sim 4.2 generic range sensor](https://docs.isaacsim.omniverse.nvidia.com/4.2.0/advanced_tutorials/tutorial_advanced_range_sensor_generic.html)

## Runtime status

| Check | Result |
|---|---|
| Isaac installed | NO — intentionally skipped by platform/resource gate |
| AppLauncher / SimulationApp | NOT RUN |
| Primitive scene | NOT RUN |
| PhysX step | NOT RUN |
| RGB/depth renderer | NOT RUN |
| Matterport USD | NOT DOWNLOADED / NOT RUN |

This is a source-level compatibility result, not an Isaac runtime pass.
