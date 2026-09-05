#!/usr/bin/env bash
# GPU/manual only. Runs precisely the Geometry V0 bounded-radial-NMS panel.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${repo_root}/outputs/geometry_v0_bounded_nms_10ep"
baseline_root="${repo_root}/outputs/geometry_v0_score_diagnosis/geometry_v0_score_diagnosis"
conda_executable="${CONDA_EXE:-/home/evens/miniforge3/bin/conda}"
python_executable="${conda_executable} run -n etpnav_legacy python"
episode_ids="['1685','1024','1477','1636','1375','1478','53','821','164','1679']"
checkpoint="data/logs/checkpoints/release_r2r/ckpt.iter12000.pth"
config="configs/mid360_waypoint.yaml"

cd "${repo_root}"
if [[ -e "${output_root}" ]]; then
  echo "refusing to overwrite evidence directory: ${output_root}" >&2
  exit 2
fi
if [[ ! -d "${baseline_root}" ]]; then
  echo "missing required same-panel Geometry V0 diagnostic baseline: ${baseline_root}" >&2
  exit 2
fi
mkdir -p "${output_root}"
source scripts/renderer_bootstrap.bash
etp_apply_known_good_renderer_environment

${python_executable} - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA unavailable; refusing bounded-NMS navigation run")
print("CUDA compute: PASS", torch.cuda.get_device_name(0))
PY

ETP_GEOMETRY_SCORE_CAPTURE_EPISODE_IDS='1685,1024,1477,1636,1375,1478,53,821,164,1679' ETP_GEOMETRY_SCORE_CAPTURE_MAX_DECISIONS=5 \
${python_executable} run.py --exp_name geometry_v0_bounded_nms_10ep --run-type eval --exp-config run_r2r/iter_train.yaml \
  SIMULATOR_GPU_IDS "[0]" TORCH_GPU_IDS "[0]" TORCH_GPU_ID 0 GPU_NUMBERS 1 NUM_ENVIRONMENTS 1 \
  EVAL.SPLIT val_unseen EVAL.EPISODE_IDS "${episode_ids}" EVAL.EPISODE_COUNT 10 \
  EVAL.CKPT_PATH_DIR "${checkpoint}" RESULTS_DIR "${output_root}/" EVAL.DETAILED_TELEMETRY True \
  IL.back_algo control MODEL.pretrained_path None MODEL.WAYPOINT_PROVIDER mid360_geometry \
  MODEL.MID360_CONFIG_PATH "${config}" VIDEO_OPTION "[]" TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True

${python_executable} scripts/analyze_geometry_v0_bounded_nms_10ep.py \
  --bounded-run "${output_root}/geometry_v0_bounded_nms_10ep" \
  --baseline-run "${baseline_root}" \
  --output "${output_root}/comparison.json"
