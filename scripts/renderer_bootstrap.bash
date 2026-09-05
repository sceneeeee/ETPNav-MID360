#!/usr/bin/env bash
# Source this file before starting Habitat-Sim under WSLg.
# Kept identical to the bootstrap validated by run_example_scene_smoke.bash.

etp_apply_known_good_renderer_environment() {
  local system_gl_dir="/lib/x86_64-linux-gnu"
  local system_gl_preload="${system_gl_dir}/libGLdispatch.so.0:${system_gl_dir}/libGLX.so.0:${system_gl_dir}/libOpenGL.so.0"
  local library
  for library in libGLdispatch.so.0 libGLX.so.0 libOpenGL.so.0; do
    if [[ ! -e "${system_gl_dir}/${library}" ]]; then
      echo "missing required WSL host OpenGL library: ${system_gl_dir}/${library}" >&2
      return 2
    fi
  done
  export LD_PRELOAD="${system_gl_preload}${LD_PRELOAD:+:${LD_PRELOAD}}"
}
