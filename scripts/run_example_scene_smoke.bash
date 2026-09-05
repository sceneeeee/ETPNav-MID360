#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_executable="${CONDA_EXE:-/home/evens/miniforge3/bin/conda}"
source "${repo_root}/scripts/renderer_bootstrap.bash"
etp_apply_known_good_renderer_environment

cd "${repo_root}"
exec "${conda_executable}" run -n etpnav_legacy \
  python scripts/example_scene_smoke.py "$@"
