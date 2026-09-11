#!/usr/bin/env bash
# FIRST frozen G-Min development panel. No performance retries or evidence overwrite.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
output_root="${repo_root}/outputs/geometry_gmin_frozen20_valid_renderer_attempt2"
python_command=("${CONDA_EXE:-/home/evens/miniforge3/bin/conda}" run --no-capture-output -n etpnav_legacy python)
episode_ids="['53','1685','821','1024','1477','1636','675','164','1375','1679','250','2','475','1356','252','1439','1016','1285','1478','25']"
checkpoint="data/logs/checkpoints/release_r2r/ckpt.iter12000.pth"
dataset="data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/val_unseen/val_unseen_bertidx.json.gz"
auxiliary="data/wp_pred/check_cwp_bestdist_hfov90"
config="configs/mid360_waypoint_gmin.yaml"
[[ ! -e "${output_root}" ]] || { echo "refusing to overwrite ${output_root}" >&2; exit 2; }
# Check correctness attestation, current source, assets, and physical controls
# before creating the dedicated run directory.
"${python_command[@]}" - "${output_root}" "${dataset}" "${checkpoint}" "${auxiliary}" "${config}" "${episode_ids}" <<'PY'
import ast, hashlib, json, os, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import torch
import yaml
root, dataset, checkpoint, auxiliary, config, ids = sys.argv[1:]
root = Path(root)
def sha(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()
gate = json.loads(Path('outputs/gmin_stage1_validation/correctness_gate.json').read_text())
if gate['status'] != 'PASS': raise SystemExit('correctness gate not passed')
for path, digest in gate['source_sha256'].items():
    if sha(path) != digest: raise SystemExit('source changed after correctness gate: ' + path)
expected = {'dataset': '7db0a39bef374cf5ce986473f8b35c9721f962c6c1a9af069257aa710df16365',
            'checkpoint': '4e70d2a1b4a6cfb32158901b7300cbc2be657641f16c9a17f35ca8570470d8b3',
            'auxiliary': '09d0f42cbd801e05b0fa0212b901d409f033f4d0bce2fce1fa8a1331b502159f'}
paths = dict(dataset=dataset, checkpoint=checkpoint, auxiliary=auxiliary)
hashes = {k: sha(p) for k, p in paths.items()}
if hashes != expected: raise SystemExit('asset provenance mismatch: {}'.format(hashes))
values = yaml.safe_load(Path(config).read_text())
baseline = yaml.safe_load(Path('configs/mid360_waypoint.yaml').read_text())
if values.get('candidate_extraction') != 'gmin_coverage_v0': raise SystemExit('G-Min not enabled')
if {k: v for k, v in values.items() if k != 'candidate_extraction'} != baseline:
    raise SystemExit('G-Min physical parameters differ from corrected baseline')
if values['maximum_candidates'] != 5: raise SystemExit('frozen budget must be five')
if not torch.cuda.is_available(): raise SystemExit('CUDA unavailable')
git = lambda *args: subprocess.check_output(['git', *args], universal_newlines=True)
files = git('ls-files', '--cached', '--others', '--exclude-standard').splitlines()
source = {p: sha(p) for p in files if Path(p).is_file()}
root.mkdir(exist_ok=False)
(root / 'source_snapshot').mkdir()
for p in gate['source_sha256']:
    destination = root / 'source_snapshot' / p
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, str(destination))
(root / 'source_sha256.json').write_text(json.dumps(source, indent=2))
(root / 'source_dirty.patch').write_text(git('diff', '--binary'))
shutil.copy2('outputs/gmin_stage1_validation/correctness_gate.json', str(root / 'correctness_gate.json'))
manifest = dict(label='G-Min coverage V0 FIRST frozen-20 development evaluation',
    branch=git('branch', '--show-current').strip(), HEAD=git('rev-parse', 'HEAD').strip(),
    git_status=git('status', '--short'), source_sha256=source, asset_paths=paths, asset_sha256=hashes,
    ordered_episode_ids=ast.literal_eval(ids), split='val_unseen', geometry_config=values,
    config_path=config, policy_config='run_r2r/iter_train.yaml',
    renderer_bootstrap='scripts/renderer_bootstrap.bash:etp_apply_known_good_renderer_environment',
    controls=dict(NUM_ENVIRONMENTS=1, GPU_NUMBERS=1, back_algo='control', ALLOW_SLIDING=True, VIDEO_OPTION=[]),
    command='bash scripts/run_gmin_frozen20.bash', timestamp=datetime.now(timezone.utc).isoformat(),
    waypoint_predictor_lifecycle='unchanged; initialized and CUDA resident', model_memory_released='NONE',
    intervention='boolean score > 0; radial schedule + circular angular lexicographic coverage; observation gate',
    gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
    timing_boundaries=dict(decision='CUDA synchronized before waypoint and after navigation network; excludes action execution and post-action trace serialization',
        geometry_frontend='CPU depth reconstruction + MID360-like filtering + occupancy/inflation/traversability; excludes availability gate and extraction',
        gmin_extraction='CPU score > 0, availability-mask choice, pure extraction; excludes construct_waypoint_output',
        shared_visual_encoding='encode_panorama wall time with CUDA synchronization at end, after synchronized availability gate',
        baseline_component_timings='historical unsynchronized host timings; not precise GPU kernel times'))
(root / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Provenance and correctness gates PASS; GPU:', manifest['gpu'])
PY
source scripts/renderer_bootstrap.bash
etp_apply_known_good_renderer_environment
"${python_command[@]}" run.py --exp_name geometry_gmin_frozen20_valid_renderer --run-type eval --exp-config run_r2r/iter_train.yaml \
  SIMULATOR_GPU_IDS "[0]" TORCH_GPU_IDS "[0]" TORCH_GPU_ID 0 GPU_NUMBERS 1 NUM_ENVIRONMENTS 1 \
  EVAL.SPLIT val_unseen EVAL.EPISODE_IDS "${episode_ids}" EVAL.EPISODE_COUNT 20 \
  EVAL.CKPT_PATH_DIR "${checkpoint}" RESULTS_DIR "${output_root}/" EVAL.DETAILED_TELEMETRY True \
  IL.back_algo control MODEL.pretrained_path None MODEL.WAYPOINT_PROVIDER mid360_geometry \
  MODEL.MID360_CONFIG_PATH "${config}" VIDEO_OPTION "[]" \
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True 2>&1 | tee "${output_root}/evaluation.log"
printf 'COMPLETE\n' > "${output_root}/EVALUATION_COMPLETE"
