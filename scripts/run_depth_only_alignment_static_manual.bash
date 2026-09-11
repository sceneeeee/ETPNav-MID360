#!/usr/bin/env bash
# HOST ONLY: the user runs this on the GPU-enabled WSL host. Never run in Codex.
set -euo pipefail
# The accepted evaluation's exact forensic gates use Python assertions.
unset PYTHONOPTIMIZE
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
run_dir="$(mktemp -d "${repo_root}/outputs/alignment_depth_only_static5_$(date +%Y%m%dT%H%M%S)_XXXXXX")"
python_command=("${CONDA_EXE:-/home/evens/miniforge3/bin/conda}" run --no-capture-output -n etpnav_legacy python)
echo "OUTPUT_DIR=${run_dir}"

finish() {
  local status="$?"
  trap - EXIT
  if [[ -f "${run_dir}/preflight/calibration_validation.json" ]]; then
    if ! "${python_command[@]}" scripts/review_depth_only_alignment.py preserve --output "${run_dir}" >> "${run_dir}/console.log" 2>&1; then
      status=1
    fi
  fi
  if [[ "${status}" -eq 0 ]]; then
    printf 'PASS STATIC_ALIGNMENT_COMPLETE 20 ROWS; ZERO NAVIGATION/TRAINING\n' | tee "${run_dir}/COMPLETE"
  else
    printf 'FAIL exit=%s; STOP: do not interpret alignment; review console.log\n' "${status}" | tee "${run_dir}/FAILED"
  fi
  echo "OUTPUT_DIR=${run_dir}"
  exit "${status}"
}
trap finish EXIT

{
  # CPU provenance/selection/moment checks finish before any GPU import/job.
  "${python_command[@]}" scripts/review_depth_only_alignment.py validate --output "${run_dir}/preflight"
  source scripts/renderer_bootstrap.bash
  etp_apply_known_good_renderer_environment
  "${python_command[@]}" scripts/run_sparse_depth_feature_alignment.py evaluate \
    --calibration outputs/mid360_sparse_depth_alignment_calibration_depth_only \
    --output "${run_dir}/evaluation"
  "${python_command[@]}" scripts/review_depth_only_alignment.py report --output "${run_dir}/evaluation"
} 2>&1 | tee "${run_dir}/console.log"
