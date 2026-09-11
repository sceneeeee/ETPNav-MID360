# Official Habitat example-scene smoke test

> **ENGINEERING SMOKE TEST ONLY — NOT AN R2R-CE BENCHMARK**
>
> No SR, SPL, or NE value is computed or implied by this test.

## Scene provenance and access

| Item | Value |
| --- | --- |
| Habitat asset UID | `habitat_test_scenes` |
| Official archive | `http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes_v1.0.zip` |
| Scene | `van-gogh-room.glb` plus `van-gogh-room.navmesh` |
| Local path | `/home/evens/SURF/datasets/habitat/scene_datasets/habitat-test-scenes/van-gogh-room.glb` |
| Storage | symlink to `/home/evens/SURF/datasets/habitat/versioned_data/habitat_test_scenes_1.0` |
| Scene SHA-256 | `24570d71c3ca9802e8642aa7ecd2479eb7fdda5b925259e27d9aa59760f39e84` |
| Navmesh SHA-256 | `0d0b50ceb92882aca27e02a4f3db2a0d0d10ed5271de075f04bf6bcb56ab15ab` |
| Format | GLB scene plus Habitat navmesh |
| Access status | public Habitat test asset; no Matterport3D access used |
| Asset license note | Van Gogh Room by ruslans3d, CC BY 4.0 per the official Habitat-Sim README |

The installed Habitat-Sim 0.1.7 package predates the dataset downloader module:

```text
python -m habitat_sim.utils.datasets_download --help
No module named habitat_sim.utils.datasets_download
```

The already materialized asset was therefore verified against the official
Habitat-Sim repository's `habitat_test_scenes` definition and README. In newer
official source that UID resolves to the Facebook-hosted archive above. The same
source also defines `mp3d_example_scene`, but changing the legacy simulator solely
to obtain it was unnecessary. No third-party MP3D mirror was queried or used, and
no example-scene asset is tracked in Git.

## Legacy renderer compatibility

The old 0.1.7 headless build cannot enumerate a CUDA EGL device under current WSL2:

```text
Platform::WindowlessEglApplication::tryCreateContext(): unable to find EGL device for CUDA device 0
```

The official 0.1.7 display build loads under WSLg. Its Conda RPATH initially mixed
an old Conda GLVND dispatcher with Ubuntu's OpenGL library, yielding black RGB and
zero depth despite successful scene initialization. Preloading the matching Ubuntu
GL dispatch libraries fixes the context without changing Habitat or ETP code:

```bash
scripts/run_example_scene_smoke.bash --output-dir outputs/example_scene \
  --pose-count 5 --seed 17
```

The validated context reports:

```text
Renderer: D3D12 (NVIDIA GeForce RTX 4060 Ti) by Microsoft Corporation
OpenGL version: 4.2 (Core Profile) Mesa 23.2.1
```

This is a GPU-backed GLX/WSLg compatibility path, not a claim that legacy EGL now
works. PyTorch CUDA is independently validated on the same RTX 4060 Ti.

## Simulator and sensor result

Habitat-Sim 0.1.7 loaded the scene and navmesh with 9.177721 square metres of
navigable area. Seed 17 selected five deterministic navigable points. The first was
`[3.1923761, 0.17669876, 0.80009353]`; its identity rotation was set explicitly.

The simulator returned valid RGB and metric depth, then successfully executed
MOVE_FORWARD, TURN_LEFT, and TURN_RIGHT. Each action returned fresh RGB/depth
arrays and a collision flag; all three smoke actions reported `collided=false`.
Simulator initialization took 935.69 ms, and the actions took 26.16, 16.37, and
15.52 ms respectively.

One RGB and one depth debug frame, the panorama contact sheet, observations, and
metadata are under ignored `outputs/example_scene/`. They are intentionally not
committed.

## Twelve-view observation contract

Each pose captures headings `0, 30, ..., 330` degrees with:

- RGB: `224 x 224 x 3`, uint8;
- depth: `256 x 256 x 1`, float32;
- horizontal FoV: 90 degrees;
- camera height: 1.25 m, matching Habitat-Lab's sensor default;
- policy depth: normalized from `[0, 10]` metres to `[0, 1]`;
- geometry depth: retained separately in metres.

Positive Habitat sensor yaw rotates left/counter-clockwise, as confirmed by the
contact sheet. The observation mapping is stored in that order. The original ETP
packing logic reverses views 1 through 11 to the clockwise order expected by its
waypoint predictor, then restores counter-clockwise order before panorama pooling.

At the first pose, valid metric ranges across the twelve cameras span approximately
0.696 to 3.968 m. The generated files deliberately separate
`fixed_12view_policy_observation.npz` from `fixed_12view_depth_meters.npz` so learned
depth features never receive metres and point-cloud geometry never receives
normalized values.

## Gate to formal evaluation

Available now: simulator engineering smoke, real 12-view example-scene inputs,
provider equivalence, and geometric-frontend tests.

Waiting for official Matterport3D access: complete R2R-CE episodes, SR/SPL/NE, and
formal Original-versus-Geometry ablation. Once authorized scenes arrive, evaluation
returns directly to 1, 5, and 20 `val_unseen` episodes; this public test scene will
not be substituted for MP3D.
