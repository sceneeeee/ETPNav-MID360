#!/usr/bin/env bash
# GPU/manual only.  This is diagnostic capture; it does not alter Geometry V0.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
source scripts/renderer_bootstrap.bash
etp_apply_known_good_renderer_environment
ETP_GEOMETRY_SCORE_CAPTURE_EPISODE_IDS='1685,1024,1477,1636,1375,1478,53,821,164,1679' ETP_GEOMETRY_SCORE_CAPTURE_MAX_DECISIONS=5 \
/home/evens/miniforge3/bin/conda run -n etpnav_legacy python run.py --exp_name geometry_v0_score_diagnosis --run-type eval --exp-config run_r2r/iter_train.yaml SIMULATOR_GPU_IDS "[0]" TORCH_GPU_IDS "[0]" TORCH_GPU_ID 0 GPU_NUMBERS 1 NUM_ENVIRONMENTS 1 EVAL.SPLIT val_unseen EVAL.EPISODE_IDS "['1685','1024','1477','1636','1375','1478','53','821','164','1679']" EVAL.EPISODE_COUNT 10 EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r/ckpt.iter12000.pth RESULTS_DIR outputs/geometry_v0_score_diagnosis/ EVAL.DETAILED_TELEMETRY True IL.back_algo control MODEL.pretrained_path None MODEL.WAYPOINT_PROVIDER mid360_geometry MODEL.MID360_CONFIG_PATH configs/mid360_waypoint.yaml VIDEO_OPTION "[]" TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
/home/evens/miniforge3/bin/conda run -n etpnav_legacy python scripts/analyze_geometry_v0_score_capture.py --telemetry outputs/geometry_v0_score_diagnosis/geometry_v0_score_diagnosis/telemetry_ep_ckpt_59_val_unseen_r0_w1.json --output outputs/geometry_v0_score_diagnosis/score_diagnosis.json
