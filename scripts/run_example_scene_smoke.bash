#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_executable="${CONDA_EXE:-/home/evens/miniforge3/bin/conda}"
system_gl_dir="/lib/x86_64-linux-gnu"
system_gl_preload="${system_gl_dir}/libGLdispatch.so.0:${system_gl_dir}/libGLX.so.0:${system_gl_dir}/libOpenGL.so.0"

for library in libGLdispatch.so.0 libGLX.so.0 libOpenGL.so.0; do
  if [[ ! -e "${system_gl_dir}/${library}" ]]; then
    echo "missing required WSL host OpenGL library: ${system_gl_dir}/${library}" >&2
    exit 2
  fi
done

cd "${repo_root}"
export LD_PRELOAD="${system_gl_preload}${LD_PRELOAD:+:${LD_PRELOAD}}"
exec "${conda_executable}" run -n etpnav_legacy \
  python scripts/example_scene_smoke.py "$@"
