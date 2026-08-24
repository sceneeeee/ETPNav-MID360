#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

episodes="${1:-1}"
if [[ ! "${episodes}" =~ ^[1-9][0-9]*$ ]]; then
  echo "usage: $0 [positive_episode_count]" >&2
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
  --exp_name "local_r2r_${episodes}ep" \
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
  VIDEO_OPTION "[]" \
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
