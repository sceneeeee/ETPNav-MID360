# MID360-like sparse-depth zero-shot control

Aim: measure the released Original ETP learned waypoint proposer's sensitivity to sparse-depth representation shift with fixed weights and downstream navigation code.

Engineering approximation for zero-shot sensor-domain control. This does **not** model the Livox MID360 non-repetitive scan physics. No training, fine-tuning, completion, interpolation, dilation, random noise, or navigation-success tuning.

## Audited contract

Baseline source commit: `8619498a048c5394fd78410e429c3a81514100c2`.

- `original_etp.py` owns encoding, learned prediction, softmax, original NMS, extraction and candidate-feature construction. The new provider delegates directly to it.
- `TRM_net.py` has RGB projection/fusion commented out; `vis_x = depth_x`. RGB still supplies the batch size, and Original still computes RGB panorama features for downstream use.
- Twelve sensor views at positive Habitat yaw 0,30,...,330 degrees; Original's insertion-order packing and reverse-index mapping remain untouched. Heatmap remains 120 angular by 12 radial bins, 0.25 m spacing.
- Installed `.external/habitat-lab-0.1.7/habitat/config/default.py` defines depth min=0, max=10, normalization enabled, position=(0,1.25,0). Its HabitatSimDepthSensor clips z-depth and applies `(depth-min)/(max-min)`.
- Effective `run_r2r/iter_train.yaml` configuration: 256x256 depth, 90-degree HFOV. Only CenterCropperPerSensor is enabled, with a 256x256 depth crop: no depth resize/crop change. The new runtime guard validates all twelve calibrations, orientations and unchanged crops.
- Existing backprojection: -Z forward, +Y up, positive yaw rotates forward toward -X. Inverse projection removes height and yaw, uses `z=-camera_z`, `f=W/(2*tan(HFOV/2))`, center `((W-1)/2,(H-1)/2)`, and rounds to a single nearest pixel with nearest-z z-buffering. No neighboring pixels are filled.

## Transformation and initial parameters

Normalized dense panorama -> metric z-depth -> existing panoramic backprojection -> existing deterministic angular nearest-return filter -> inverse projection into the same cameras -> normalized sparse panorama -> Original provider.

The initial config uses stride=1, 0.5-degree azimuth/elevation cells, full camera elevation support (-90 to 90), voxel_size=0 and no-return fill=1.0. Voxel averaging is rejected because it creates synthetic returns. Finite source pixels strictly between normalized 0 and 1 supply geometry. Saturated 1.0 is censored max-depth data, not a known surface; invalid/zero/saturated pixels do not create artificial 3D surfaces.

`minimum_range_m=0` and `maximum_range_m=10` explicitly mean **policy camera z-depth support**. The shared filter uses Euclidean range, so its maximum is the enclosing frustum radius (17.320508 m for square 90-degree cameras). This prevents a 10 m Euclidean sphere from unintentionally clipping off-axis points whose original z-depth was usable. Every projected camera z remains <=10 m. This is deliberately a sparsity control, not an additional physical range/FOV experiment.

All downstream construction algorithms stay frozen. Depth feature **values** naturally change because the same encoder receives changed sensor values; candidate features are still constructed by Original ETP. GraphMap association threshold, language/topological planner, navigation selection, executor, RGB pixels, instruction, checkpoint and auxiliary predictor weight are untouched. Before/after hashes cover all loaded policy/predictor parameters and buffers; no optimizer step occurs.

## Validation

Executed from repository root with:

```bash
PY=/home/evens/miniforge3/envs/etpnav_legacy/bin/python
$PY -m unittest discover -s tests -p test_mid360_sparse_depth.py -v
$PY -m unittest discover -s tests -p test_original_etp_waypoint_provider.py -v
$PY -m unittest discover -s tests -p test_waypoint_provider_contract.py -v
$PY -m unittest discover -s tests -p test_sparse_depth_evidence.py -v
$PY -m unittest discover -s tests -v
$PY scripts/test_waypoint_provider_equivalence.py --output outputs/mid360_sparse_depth_original_equivalence.json
bash -n scripts/run_mid360_sparse_depth_zero_shot20.bash
source scripts/renderer_bootstrap.bash
etp_apply_known_good_renderer_environment
$PY scripts/smoke_mid360_sparse_depth53.py --output outputs/mid360_sparse_depth_smoke53_attempt3
```

PASS: 9 sparse tests (CPU and CUDA; geometry, exact fill, z-buffer, determinism, input immutability, batch/dtype/device, delegation and embedding routing); 1 unchanged Original provider test; 3 unchanged contract tests; 3 calibration/factory/evidence tests. Full repository suite: **127 PASS**. Existing CUDA golden regression: **PASS, exact tensor hashes and candidates**. Shell syntax and Python compilation pass.

Review: separate Standards and Spec reviews found one float64 configurable-fill precision edge, fixed with source-dtype projection allocation and a `fill=0.1` regression. No unresolved findings. Default float32/fill=1 outputs are unchanged by that fix.

Smoke attempts are preserved, never overwritten. Attempt1 could not access CUDA inside the sandbox. Attempt2 completed the GPU rollout and all input-path checks but its last bookkeeping assertion confused integer trainer keys with string JSON keys. Attempt3 reads the serialized episode evidence and **PASS**es. Its full rollout retains max_traj_len=15 and every baseline evaluation setting; no outcome-based tuning between attempts. The subsequent dtype-edge fix affects float64 configurable fills, not the audited float32 experiment.

## Episode 53 evidence

Artifact directory: `outputs/mid360_sparse_depth_smoke53_attempt3/`.

- `first_decision_comparison.json`: same-observation proposals, scalar sensor statistics, encoder/predictor input hashes and before/after model-state hashes.
- `run_manifest.json`: branch/HEAD/dirty status, verified baseline asset hashes, config, renderer and command.
- `controlled_code_audit.json`: 15 frozen source files byte-identical to baseline, including GraphMap, planner, Original, encoders and executor.
- `mid360_sparse_depth_smoke53/`: complete episode metrics, telemetry and raw-Habitat sensor-validity evidence.

First-decision candidates, in Original output order; angles are relative counter-clockwise degrees (rounded only for display), distances in meters:

| Original dense | Sparse zero-shot |
|---|---|
| 198°, 1.25 m | 345°, 0.75 m |
| 78°, 1.50 m | 315°, 0.75 m |
| 27°, 0.50 m | 285°, 0.75 m |
| 12°, 1.25 m | 255°, 0.75 m |
| — | 225°, 0.75 m |

First-frame source cloud: **785,849** points; filtered cloud: **128,976** returns. Valid projected pixels: **346,667 / 786,432 = 0.4408098856608073**. No-return fraction: **0.5591901143391926**, exact fill **1.0**.

Per-view projected pixels at 0,30,...,330 degrees:
`[28888, 28452, 28326, 28185, 28551, 29122, 29099, 29031, 29216, 29310, 29284, 29203]`.

| Normalized depth statistic | Dense | Sparse |
|---|---:|---:|
| Min | 0 | 0.02265177295 |
| Max | 1 | 1 |
| Mean | 0.13602562249 | 0.62148475647 |
| Finite fraction | 1 | 1 |

Smoke hooks assert: every raw Habitat depth view matches the dense proposer source; every packed encoder input equals the intended sparse view; predictor depth embeddings hash-identically match encoder outputs; dense and sparse encoder-input and embedding hashes differ; source RGB/instruction/depth are unmodified; panorama RGB features match. All model-state hashes match before and after the complete episode.

Sparse episode 53 navigation outcome: SR=0, SPL=0, NE=14.282913 m, OS=0, nDTW=0.157449, SDTW=0; 118 simulator steps, path length=9.557706 m. This is an unsuccessful navigation outcome, not a failed sensor-path smoke. It does not establish population performance or justify parameter changes. Smoke latency includes extra dense inference and hashing, so is not formal latency evidence.

## Prepared formal comparison — not executed

`scripts/run_mid360_sparse_depth_zero_shot20.bash` mirrors the frozen Original command, changing only the provider selector and adding the sparse config path. It uses one environment, the exact twenty requested IDs, unchanged val_unseen dataset/checkpoint/auxiliary predictor, renderer bootstrap, sliding=True and control executor. Its output directory is created atomically and cannot overwrite a prior attempt. Manifest creation verifies the three baseline asset SHA256 values, records the sparse config/hash, branch/HEAD/dirty state, timestamp and exact command.

`scripts/collect_mid360_sparse_depth_zero_shot20.py` requires all twenty ordered episodes, valid sensors and complete per-decision sparse telemetry, then creates `summary.json`, `episodes.json`, `sparse_depth_summary.json`, `telemetry.json`, and `sensor_validity.json`. It computes SR/SPL/NE/OS/nDTW/SDTW, mean trajectory length/steps, mean/p95 decision latency, peak allocated VRAM, signed and absolute deltas against frozen Original SR=.8, SPL=.6983661897804773, NE=2.9216403543949125, OS=.8, and per-view sparse-return statistics. Collector tests use synthetic evidence only.

Recommendation: scientifically clean enough for the frozen twenty-episode comparison, subject to explicit user approval. Interpret it as deterministic sparse-depth representation sensitivity with a constant far fill, not faithful MID360 physics. The formal runner has **not** been launched.
