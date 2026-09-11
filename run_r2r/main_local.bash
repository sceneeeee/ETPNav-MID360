#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
source "${repo_root}/scripts/renderer_bootstrap.bash"
etp_apply_known_good_renderer_environment

episodes="${1:-1}"
provider="${2:-original_etp}"
if [[ ! "${episodes}" =~ ^[1-9][0-9]*$ ]]; then
  echo "usage: $0 [positive_episode_count] [original_etp|mid360_geometry]" >&2
  exit 2
fi
if [[ "${provider}" != "original_etp" && "${provider}" != "mid360_geometry" ]]; then
  echo "unknown waypoint provider: ${provider}" >&2
  exit 2
fi

if ! python -c 'import habitat, torch' >/dev/null 2>&1; then
  echo "activate the etpnav_legacy environment before running this script" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GLOG_minloglevel="${GLOG_minloglevel:-2}"
export MAGNUM_LOG="${MAGNUM_LOG:-quiet}"

python run.py \
  --exp_name "local_r2r_${provider}_${episodes}ep" \
  --run-type eval \
  --exp-config run_r2r/iter_train.yaml \
  SIMULATOR_GPU_IDS "[0]" \
  TORCH_GPU_IDS "[0]" \
  TORCH_GPU_ID 0 \
  GPU_NUMBERS 1 \
  NUM_ENVIRONMENTS 1 \
  EVAL.SPLIT val_unseen \
  EVAL.EPISODE_COUNT "${episodes}" \
  EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r/ckpt.iter12000.pth \
  IL.back_algo control \
  MODEL.pretrained_path None \
  MODEL.WAYPOINT_PROVIDER "${provider}" \
  VIDEO_OPTION "[]" \
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
