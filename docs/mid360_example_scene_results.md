# MID360-like waypoint frontend: public-scene results

> **ENGINEERING SMOKE TEST ONLY — NOT R2R-CE BENCHMARK**
>
> These five fixed poses have no R2R instruction/goal distribution. No SR, SPL,
> NE, or comparison with the ETPNav paper is valid here.

## Fixed test conditions

- Public official Habitat `van-gogh-room` test scene and navmesh.
- Five navigable poses sampled deterministically with seed 17 and identity heading.
- Released R2R checkpoint at iteration 12000 and official learned waypoint weights.
- Identical 12-view RGB/depth observations and CLIP/DD-PPO encoders.
- Identical panorama feature construction and downstream ETP ten-field contract.
- Only learned heatmap proposal versus geometric score/NMS changes.

The original pre-refactor implementation and `OriginalETPWaypointProvider` were run
back-to-back on the first fixed observation. Candidate count, angles, distances,
image indices, all candidate features, and all panorama features had maximum
absolute difference `0.0`. The checked-in golden regression retains shapes and
numeric fingerprints after the temporary compatibility path was removed.

## MID360-like V0

This is an engineering approximation for architecture validation, not a physical
simulation of Livox's non-repetitive scanning pattern.

The frontend back-projects the twelve metric depth images into Habitat base
coordinates (`+Y` up, `-Z` forward), rotates each view by its sensor yaw, merges the
cloud, applies deterministic pixel-stride, 1-degree azimuth/elevation cells, and
voxel downsampling, then filters range and vertical FoV. Height-filtered obstacles
are projected into a local 2-D grid,
inflated by footprint plus clearance, and ray-cast into a `120 x 12` score map:

- 120 angular bins at 3 degrees;
- 12 distance bins at 0.25 m, covering 0.25–3.0 m;
- at most five candidates through the original circular NMS contract.

`robot_radius_m=0.10` matches Habitat-Lab 0.1.7's default simulated agent. It is
marked `HABITAT_DEFAULT_AGENT_TEST_ONLY`; no verified real-robot footprint exists in
the current workspace. This value must not be carried into real-robot work.

The geometric provider does not replace CLIP, the DD-PPO depth encoder, panorama
features, GraphMap, the language model, graph planner, or controller. After choosing
angle/distance bins, it selects `cand_rgb` and `cand_depth` from the same original
counter-clockwise panorama features.

## Five-pose comparison

Rerun on 2026-09-03 after the final range-guard change: PASS. The ignored source
artifact is `outputs/example_scene_current/comparison/comparison_results.json`.
The current run retained five geometry candidates per pose, zero panorama-feature
difference, and 1,016.34 MiB peak allocated CUDA memory. The deterministic table
below remains the comparison record; current mean full-provider latencies were
32.69 ms Original and 75.48 ms Geometry.

Angles below are counter-clockwise ETP output bearings. Latencies include the full
provider call, including the shared RGB/depth encoders. Pose 0 contains cold-start
cost.

| Pose | Original candidates | Geometry candidates | Original ms | Geometry ms | merged -> filtered points |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | 357deg/1.00m; 99deg/0.50m; 84deg/1.25m; 66deg/1.00m | 345deg/2.25m; 318deg/3.00m; 300deg/3.00m; 282deg/3.00m; 264deg/3.00m | 67.24 | 71.00 | 38,363 -> 10,299 |
| 1 | 300deg/0.75m; 237deg/1.00m; 159deg/0.75m; 144deg/1.00m; 126deg/1.25m | 288deg/3.00m; 270deg/3.00m; 252deg/3.00m; 234deg/3.00m; 216deg/2.50m | 23.28 | 60.65 | 40,988 -> 11,184 |
| 2 | 282deg/0.75m; 267deg/0.75m; 243deg/1.00m; 213deg/0.75m; 183deg/0.75m | 282deg/3.00m; 264deg/3.00m; 246deg/3.00m; 228deg/3.00m; 210deg/2.50m | 28.80 | 65.73 | 43,699 -> 10,858 |
| 3 | 333deg/0.75m; 318deg/1.25m; 300deg/0.75m; 126deg/1.00m; 108deg/1.00m | 291deg/3.00m; 273deg/3.00m; 255deg/3.00m; 237deg/3.00m; 219deg/2.25m | 24.29 | 61.58 | 42,235 -> 13,373 |
| 4 | 300deg/0.75m; 183deg/1.00m; 168deg/1.25m; 138deg/1.25m; 123deg/1.75m | 288deg/3.00m; 270deg/3.00m; 252deg/3.00m; 234deg/3.00m; 126deg/3.00m | 30.86 | 70.52 | 34,110 -> 8,243 |

Summary:

- original candidates: mean 4.8 per pose;
- geometry candidates: exactly 5 per pose;
- original full-provider latency: mean 34.90 ms (about 26.81 ms excluding cold pose);
- geometry full-provider latency: mean 65.89 ms (about 64.62 ms excluding cold pose);
- peak allocated CUDA memory across the comparison: 1,016.34 MiB;
- geometry score shape: `[120, 12]` at every pose;
- Original-versus-Geometry pano RGB/depth maximum absolute difference: `0.0` at
  every pose.

## Visual sanity check

Ignored artifacts under `outputs/example_scene/comparison/` include:

- top-down filtered point cloud;
- obstacle occupancy;
- footprint-inflated occupancy;
- `120 x 12` traversability heatmap;
- side-by-side Original learned versus MID360-like candidates for all five poses.

Manual inspection shows wall/obstacle arcs aligned with the panorama, inflation
closing obstacle-side rays, and geometric candidates falling in visible free-space
sectors. An initially guessed 0.18 m footprint rejected all candidates at one valid
Habitat pose; replacing it with the documented Habitat default 0.10 m test-agent
radius fixed that inconsistency. This confirms why the real footprint must remain a
required deployment parameter rather than an algorithm constant.

## Formal benchmark gate

R2R-CE Original baseline and Original-versus-Geometry 1/5/20 episode evaluation
remain gated exclusively on officially authorized full Matterport3D scenes. The
example-scene smoke result is sufficient to stop frontend expansion now; it is not
evidence of navigation quality or algorithmic superiority.
