# Habitat-Sim renderer bootstrap

The previously validated public-scene launcher set only one renderer-related
environment variable before entering `etpnav_legacy`:

`LD_PRELOAD=/lib/x86_64-linux-gnu/libGLdispatch.so.0:/lib/x86_64-linux-gnu/libGLX.so.0:/lib/x86_64-linux-gnu/libOpenGL.so.0${LD_PRELOAD:+:$LD_PRELOAD}`

Before this change, `run_r2r/main_local.bash` and
`scripts/run_original_forensics_manual.bash` set no `LD_PRELOAD`,
`LD_LIBRARY_PATH`, `LIBGL_*`, `EGL_*`, `DISPLAY`, or `WAYLAND_DISPLAY` values.
The public launcher likewise did not set those latter variables; it invoked
Conda with `conda run -n etpnav_legacy`, whereas the R2R launcher required an
already activated environment and the forensic launcher used `conda run`.

`scripts/renderer_bootstrap.bash` now supplies exactly that tested preload to
all three launchers.  It deliberately does not select a Habitat-Sim package,
set `LD_LIBRARY_PATH`, modify display/WSLg variables, or add Mesa/GLVND
libraries beyond the three public-launcher libraries.

## Invalidated renderer-failure experiment

The older evidence under `outputs/original_forensics_manual/` recorded
all-zero RGB/depth observations.  It remains preserved as renderer-failure
evidence, but is invalid for algorithmic waypoint analysis.  In particular,
do not use its all-0.75 m Original candidates, selected-distance P95 of
0.75 m, a Geometry V0.1 cap selected from that distribution, or comparisons
that treat that run as a valid Original baseline.  Algorithmic comparisons
must instead use a run with the shared renderer bootstrap and valid sensor
observations.
