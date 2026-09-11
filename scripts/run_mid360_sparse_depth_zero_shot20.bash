#!/usr/bin/env bash
# Prepared formal runner. Execute only after explicit approval of the episode-53 smoke.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${repo_root}/outputs/mid360_sparse_depth_zero_shot20"
conda_executable="${CONDA_EXE:-/home/evens/miniforge3/bin/conda}"
python_command=("${conda_executable}" run --no-capture-output -n etpnav_legacy python)
episode_ids="['53','1685','821','1024','1477','1636','675','164','1375','1679','250','2','475','1356','252','1439','1016','1285','1478','25']"
checkpoint="data/logs/checkpoints/release_r2r/ckpt.iter12000.pth"
cd "${repo_root}"
if [[ -e "${output_root}" ]]; then
  echo "refusing to overwrite evidence directory: ${output_root}" >&2
  exit 2
fi
# Atomic leaf creation also refuses a concurrent run.
mkdir -p "${repo_root}/outputs"
mkdir "${output_root}"
source "${repo_root}/scripts/renderer_bootstrap.bash"
etp_apply_known_good_renderer_environment
"${python_command[@]}" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable; refusing formal run"'
command=("${python_command[@]}" run.py --exp_name mid360_sparse_depth_zero_shot20 --run-type eval --exp-config run_r2r/iter_train.yaml
  SIMULATOR_GPU_IDS "[0]" TORCH_GPU_IDS "[0]" TORCH_GPU_ID 0 GPU_NUMBERS 1 NUM_ENVIRONMENTS 1
  EVAL.SPLIT val_unseen EVAL.EPISODE_IDS "${episode_ids}" EVAL.EPISODE_COUNT 20
  EVAL.CKPT_PATH_DIR "${checkpoint}" RESULTS_DIR "${output_root}/" EVAL.DETAILED_TELEMETRY True
  IL.back_algo control MODEL.pretrained_path None MODEL.WAYPOINT_PROVIDER mid360_sparse_depth_etp
  MODEL.MID360_SPARSE_DEPTH_CONFIG_PATH configs/mid360_sparse_depth_zero_shot.yaml VIDEO_OPTION "[]"
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True)
"${python_command[@]}" scripts/sparse_depth_run_manifest.py --output "${output_root}" --episode-ids "${episode_ids}" -- "${command[@]}"
"${command[@]}" 2>&1 | tee "${output_root}/evaluation.log"
"${python_command[@]}" scripts/collect_mid360_sparse_depth_zero_shot20.py --output "${output_root}"
printf 'COMPLETE\n' > "${output_root}/COMPLETE"
