# ETPNav original baseline setup log

## Status snapshot

- Date: 2026-08-24 (Asia/Shanghai)
- Result: **PASS WITH EXTERNAL DATA BLOCKER**
- External blocker: licensed Matterport3D Habitat scenes are not present.
- Scope deliberately excluded: MID360 implementation, ROS 2, Isaac Sim, FAST-LIO2,
  MPPI, training, and changes to GraphMap or the ETP transformer.

The complete public code/model path is ready. Habitat-Sim rendering, reset, and one
action pass on a public test scene. The released ETP policy, waypoint predictor, and
graph planner initialize and execute on the GPU. A real R2R-CE evaluation reaches
the first requested MP3D scene and then stops because its `.glb` is absent.

## Machine inventory

| Item | Observed value |
| --- | --- |
| Host | WSL2, Ubuntu 22.04.5 LTS |
| Kernel | Linux 6.18 WSL2 |
| CPU/RAM | 15 GiB RAM, 4 GiB swap |
| GPU | NVIDIA GeForce RTX 4060 Ti, compute capability 8.9 |
| GPU driver | NVIDIA 610.88 under WSL |
| Reported CUDA compatibility | 13.3 |
| VRAM | 8,188 MiB (`torch`: 7.996 GiB) |
| Disk at setup | 1,007 GiB total, 877 GiB free under `/home` |
| Proxy | HTTP/HTTPS/ALL proxy at `172.23.208.1:7897`; localhost in `NO_PROXY` |

The workspace inventory found existing VLN, NaVIDA, and Habitat repositories and
environments. None were modified. In particular, the existing dirty
`_external/habitat-sim-0.2.4-glx` worktree was left untouched. No existing official
ETPNav clone was found, so a separate clone was created.

Reasonable data locations searched were `~/SURF`, `~/data`, `~/datasets`, and
`/mnt/d`. Public Habitat test scenes were found under
`~/SURF/datasets/habitat/versioned_data/habitat_test_scenes_1.0`; no directory of
the form `mp3d/{scene}/{scene}.glb` was found.

## Repository

| Item | Value |
| --- | --- |
| Path | `/home/evens/SURF/etpnav-mid360` |
| Origin | `https://github.com/MarSaKi/ETPNav.git` |
| Official base | `160e65fe5d23ad83d024d57d710e47be6892fc68` |
| Development branch | `etp-mid360` |
| Official branch policy | `main` was not modified |

The clone initially hit an HTTP/2 GnuTLS termination through the local proxy. A
retry using Git HTTP/1.1 completed successfully. Local Habitat-Lab source is kept
under ignored `.external/` and is not vendored.

## Isolated legacy environment

| Package | Installed version/source |
| --- | --- |
| Conda environment | `etpnav_legacy` |
| Python | 3.6.15 |
| pip / setuptools / wheel | 21.3.1 / 58.0.4 / 0.37.1 |
| PyTorch | 1.9.1+cu111 |
| torchvision | 0.10.1+cu111 |
| Habitat-Sim | 0.1.7, build `py3.6_linux_856d4b08...` |
| Habitat-Sim mutex | `display_nobullet` |
| Habitat-Lab | tag v0.1.7, commit `d6ed1c0a0e786f16f261de2beafe347f4186d0d8`, editable |
| Gym | 0.21.0 |
| NumPy / SciPy / Numba | 1.19.5 / 1.5.3 / 0.53.1 |
| Transformers | 4.12.5 |
| PyTorch Transformers | 1.2.0 |
| OpenAI CLIP | commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` |

This environment was built from runtime imports instead of blindly solving the
old `environment.yaml`. That file pins `gym==0.10.9`, while the project README
explicitly requires `gym==0.21.0` for Habitat-Lab 0.1.7. The README value is used.
TensorFlow is not a runtime dependency of the released PyTorch evaluation path.

Principal supporting packages include OpenCV 4.5.5.62, NetworkX 2.5.1, LMDB 1.3,
WebDataset 0.1.40, `fastdtw` 0.3.4, `dtw` 1.4, `jsonlines` 3.0, `ifcfg` 0.22,
`moviepy` 1.0.3, and TensorBoard 2.7.

### Dependency and platform fixes

1. Creating the environment with `setuptools=59.6` failed because no Python 3.6
   Conda build exists. The solver-supported official-era 58.0.4 build was used.
2. Habitat-Lab's editable build initially invoked PEP 517 and attempted an online
   build isolation. Installing `pytest-runner==5.3.1` and using
   `pip --no-use-pep517 --no-deps -e .` installed the checked-out v0.1.7 tree.
3. The official headless Habitat-Sim 0.1.7 package imports, loads a scene and
   navmesh, but fails to create a WSL EGL context with:

   ```text
   Platform::WindowlessEglApplication::tryCreateContext(): unable to find EGL device for CUDA device 0
   WindowlessContext: Unable to create windowless context
   ```

   This is old EGL CUDA-device enumeration under WSL2, not a model, Python, or
   scene incompatibility. The same official Habitat-Sim 0.1.7 source commit's
   display/GLX build was installed. It uses WSLg and passes real simulator setup.
4. Three trainer modules imported unused TensorFlow unconditionally even though the
   official setup instructs users to omit it. Those unused imports were removed.
5. CLIP ViT-B/32 downloads and loads from ignored `data/pretrained/CLIP`, keeping
   model cache out of the user's home cache and inside the baseline asset boundary.

During early smoke debugging, a temporary local edit pooled panorama tensors before
the original clockwise-to-counter-clockwise restoration. Two-axis code review caught
the semantic mismatch before handoff; the final code retains the official ordering
and the post-restoration pooling exactly.

## Public model and dataset assets

Assets came from the official ModelScope repository `admagic/ETPNav`, repository
commit `a542e27a3603477abf10a06978e5455cda51631f`. They were fetched using a modern
base toolchain; ModelScope was not installed in `etpnav_legacy`. Only baseline
assets were materialized, and the temporary Git-LFS object cache was removed after
checksum verification to avoid a second copy of the 2.1 GB checkpoint.

| Required asset | Size | SHA-256 |
| --- | ---: | --- |
| `data/connectivity_graphs.pkl` | 2,282,592 B | `806653531d5a08bf6bbe8a12b7cb555ef82db04d19f4aa630f4afae494f82ba6` |
| `data/ddppo-models/gibson-2plus-resnet50.pth` | 49,853,716 B | `a6a600277efacf5fd98e293267221185d843eb3012aeff62fabfeee24c2bcdad` |
| `data/logs/checkpoints/release_r2r/ckpt.iter12000.pth` | 2,071,802,352 B | `4e70d2a1b4a6cfb32158901b7300cbc2be657641f16c9a17f35ca8570470d8b3` |
| `data/wp_pred/check_cwp_bestdist_hfov90` | 201,954,133 B | `09d0f42cbd801e05b0fa0212b901d409f033f4d0bce2fce1fa8a1331b502159f` |

The R2R-CE `val_unseen` BERT-indexed dataset and ground-truth files are also
materialized. Unneeded ModelScope paths may remain as small Git-LFS pointer files;
the smoke utility explicitly rejects pointers for every required baseline asset.
The complete local `data/` tree is ignored by Git and occupies about 2.5 GB.

Expected licensed scene layout remains:

```text
data/scene_datasets/mp3d/{scene}/{scene}.glb
```

No checkpoint, scene, dataset, log, cache, or video is tracked by this repository.

## Smoke tests and measured results

All GPU and simulator commands were run outside the restricted command sandbox so
they could access WSL's GPU and display devices.

| Stage | Result | Evidence |
| --- | --- | --- |
| Config load | PASS | one GPU, one environment, `val_unseen`, one episode |
| Torch CUDA | PASS | CUDA available; RTX 4060 Ti; real 1024x1024 matmul |
| Habitat imports | PASS | `habitat` and `habitat_sim` 0.1.7 |
| Simulator initialization | PASS | GLX/WSLg renderer, public van-gogh-room scene |
| Habitat reset | PASS | RGB and depth observations returned |
| One simulator step | PASS | MOVE_FORWARD returned observation and collision flag |
| ETP policy import/init | PASS | 299,141,092 parameters |
| Released checkpoint load | PASS | iteration 12000; 771 tensors; zero missing/unexpected keys |
| Waypoint inference | PASS | five candidates from a synthetic 12-view panorama |
| Panorama + graph planner | PASS | logits shape `[1, 3]`; selectable action |
| One real R2R-CE episode | BLOCKED BY MP3D | startup reaches first scene load |
| Five real R2R-CE episodes | NOT RUN | same licensed-scene prerequisite |

Measured synthetic inference timings on the RTX 4060 Ti:

- Policy initialization: 8,505.27 ms.
- Checkpoint load: 1,852.33 ms.
- Waypoint inference: 67.32 ms cold, then 24.64 and 26.84 ms; mean 39.60 ms.
- Graph planner: 6.74, 6.46, 6.43, 5.93, and 5.66 ms; mean 6.25 ms.
- Allocated GPU memory after model load: 975.15 MiB.
- Peak allocated GPU memory through waypoint/planner: 1,016.30 MiB.
- Process RSS after model load: 7,196 MiB; peak RSS: 8,531.89 MiB.
- Public test-scene simulator initialization: 803.94 ms; one step: 1.33 ms.

The synthetic planner timing isolates a single forward pass. A real episode's
average planner latency and total runtime cannot be measured until MP3D is mounted.

### Commands

Activate the environment first:

```bash
cd /home/evens/SURF/etpnav-mid360
conda activate etpnav_legacy
```

Run the staged, data-independent checks:

```bash
python scripts/etp_baseline_smoke.py
python scripts/etp_baseline_smoke.py \
  --scene /home/evens/SURF/datasets/habitat/versioned_data/habitat_test_scenes_1.0/van-gogh-room.glb
python scripts/etp_baseline_smoke.py \
  --policy --waypoint --planner --device cuda \
  --waypoint-runs 3 --planner-runs 5
```

After mounting MP3D in the expected directory, run one and then five episodes:

```bash
bash run_r2r/main_local.bash 1
bash run_r2r/main_local.bash 5
```

### Exact external-data failure

Command:

```bash
conda run -n etpnav_legacy bash run_r2r/main_local.bash 1
```

The dataset initializes and enumerates 11 `val_unseen` scenes, then Habitat-Sim
reports:

```text
Render asset ... data/scene_datasets/mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb ...
does not correspond to any existing file
```

The worker then exits and the parent observes `ConnectionResetError`. The command
took 5.28 s with 1,365,540 KiB maximum RSS before stopping. The missing GLB is the
root cause; downloading or fabricating MP3D data would violate the license boundary.

## Git/data discipline

`.gitignore` covers `/data`, `/logs`, Python caches, pretraining datasets/features,
and local `.external` dependencies. ModelScope's temporary LFS metadata/cache was
removed after asset validation; it is recoverable by re-fetching the public source.
Smoke JSON files remain under ignored `data/logs/smoke/` and are deliberately not
committed.
