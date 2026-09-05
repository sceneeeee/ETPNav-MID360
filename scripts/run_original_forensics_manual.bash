#!/usr/bin/env bash
# Run this from a normal WSL shell with GPU visibility.  Codex must not run it.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_executable="${CONDA_EXE:-/home/evens/miniforge3/bin/conda}"
phase="${1:-one}"
if [[ "${phase}" != "one" && "${phase}" != "repro" ]]; then
  echo "usage: $0 [one|repro]" >&2
  exit 2
fi
output_root="${repo_root}/outputs/original_forensics_${phase}_manual"
python_executable="${conda_executable} run -n etpnav_legacy python"

cd "${repo_root}"
source "${repo_root}/scripts/renderer_bootstrap.bash"
etp_apply_known_good_renderer_environment
if [[ -f "${output_root}/COMPLETE" ]]; then
  echo "refusing to overwrite existing forensic evidence: ${output_root}" >&2
  exit 2
fi
if [[ -e "${output_root}" && -n "$(find "${output_root}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  failed_root="${output_root}.incomplete.$(date +%Y%m%d-%H%M%S-%N)"
  mv "${output_root}" "${failed_root}"
  echo "preserved incomplete forensic attempt: ${failed_root}" >&2
fi
mkdir -p "${output_root}"

mark_failed_attempt() {
  local status="$?"
  if [[ "${status}" -ne 0 && -d "${output_root}" && ! -f "${output_root}/COMPLETE" ]]; then
    printf 'FAILED\n' > "${output_root}/FAILED"
  fi
}
trap mark_failed_attempt EXIT

${python_executable} - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA unavailable; refusing forensic run")
torch.cuda.reset_peak_memory_stats()
x = torch.randn(1024, 1024, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("CUDA compute: PASS", torch.cuda.get_device_name(0), float(y[0, 0]))
print("peak allocated:", torch.cuda.max_memory_allocated())
print("peak reserved:", torch.cuda.max_memory_reserved())
PY

# This resets the authorized R2R MP3D episode at its true start pose and exits
# before ETP if Habitat-Sim does not provide valid RGB and depth.
${python_executable} scripts/mp3d_episode_sensor_smoke.py \
  --episode-id 53 --output "${output_root}/mp3d_sensor_smoke.json"

run_one() {
  local label="$1"
  local detailed="$2"
  export ETP_ORIGINAL_FORENSICS_RAW_PATH="${output_root}/raw_${label}.json"
  export ETP_ORIGINAL_FORENSICS_TRACE_PATH="${output_root}/${label}/forensic_trace.json"
  mkdir -p "${output_root}/${label}/original_forensics_${label}"
  ${python_executable} run.py --exp_name "original_forensics_${label}" --run-type eval --exp-config run_r2r/iter_train.yaml \
    SIMULATOR_GPU_IDS "[0]" TORCH_GPU_IDS "[0]" TORCH_GPU_ID 0 GPU_NUMBERS 1 NUM_ENVIRONMENTS 1 \
    EVAL.SPLIT val_unseen EVAL.EPISODE_IDS "['53']" EVAL.EPISODE_COUNT 1 \
    EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r/ckpt.iter12000.pth \
    RESULTS_DIR "${output_root}/${label}/" EVAL.DETAILED_TELEMETRY "${detailed}" \
    IL.back_algo control MODEL.pretrained_path None MODEL.WAYPOINT_PROVIDER original_etp VIDEO_OPTION "[]" \
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
}

if [[ "${phase}" == "one" ]]; then
  # Do not begin the reproducibility phase until this output has been reviewed
  # and the restored waypoint distribution is scene-dependent.
  run_one fixed_one True
  echo "One-episode post-smoke result: ${output_root}"
  echo "If valid and sensible, run: bash scripts/run_original_forensics_manual.bash repro"
else
  for number in 1 2 3 4 5; do run_one "run${number}" True; done
  run_one telemetry_off False
  run_one telemetry_on True
  ${python_executable} scripts/collect_original_forensics.py --root "${output_root}"
  echo "Five-run reproducibility results: ${output_root}"
fi

printf 'COMPLETE\n' > "${output_root}/COMPLETE"
trap - EXIT

# Optional historical comparison (does not touch the active branch):
# git worktree add --detach ../etpnav-mid360-c794ba6 c794ba6
# cd ../etpnav-mid360-c794ba6 && bash scripts/run_original_forensics_manual.bash
