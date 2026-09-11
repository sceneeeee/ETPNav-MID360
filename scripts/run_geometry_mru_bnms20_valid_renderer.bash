#!/usr/bin/env bash
# Formal frozen-20 Geometry MRU evaluation.  Refuses to overwrite evidence.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${repo_root}/outputs/geometry_mru_bnms20_valid_renderer"
conda_executable="${CONDA_EXE:-/home/evens/miniforge3/bin/conda}"
python_executable="${conda_executable} run -n etpnav_legacy python"
episode_ids="['53','1685','821','1024','1477','1636','675','164','1375','1679','250','2','475','1356','252','1439','1016','1285','1478','25']"
checkpoint="data/logs/checkpoints/release_r2r/ckpt.iter12000.pth"
auxiliary_weight="data/wp_pred/check_cwp_bestdist_hfov90"
dataset="data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/val_unseen/val_unseen_bertidx.json.gz"
config="configs/mid360_waypoint_mru.yaml"
bnms_run="outputs/geometry_v0_bounded_nms20_valid_renderer/geometry_v0_bounded_nms20_valid_renderer"
original_run="outputs/original_baseline20_valid_renderer/original_baseline20_valid_renderer"
saved_diagnostic="outputs/geometry_v0_score_diagnosis/geometry_v0_score_diagnosis/telemetry_ep_ckpt_59_val_unseen_r0_w1.json"

cd "${repo_root}"
[[ ! -e "${output_root}" ]] || { echo "refusing to overwrite ${output_root}" >&2; exit 2; }
[[ -d "${bnms_run}" && -d "${original_run}" && -f "${saved_diagnostic}" ]] || {
  echo "missing frozen baseline or saved pre-NMS diagnostic evidence" >&2; exit 2;
}
mkdir -p "${output_root}"

${python_executable} - "${output_root}" "${checkpoint}" "${auxiliary_weight}" "${dataset}" "${config}" "${episode_ids}" <<'PY'
import ast, hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import yaml
root, checkpoint, auxiliary, dataset, config, ids = sys.argv[1:]
expected = {"dataset": "7db0a39bef374cf5ce986473f8b35c9721f962c6c1a9af069257aa710df16365", "checkpoint": "4e70d2a1b4a6cfb32158901b7300cbc2be657641f16c9a17f35ca8570470d8b3", "auxiliary": "09d0f42cbd801e05b0fa0212b901d409f033f4d0bce2fce1fa8a1331b502159f"}
def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()
hashes = {"dataset": sha(dataset), "checkpoint": sha(checkpoint), "auxiliary": sha(auxiliary)}
if hashes != expected: raise SystemExit("provenance SHA256 mismatch: {}".format(hashes))
values = yaml.safe_load(Path(config).read_text())
baseline = yaml.safe_load(Path("configs/mid360_waypoint.yaml").read_text())
if "maximum_candidate_distance_m" in values: raise SystemExit("distance cap is forbidden")
if values.get("radial_utility") != "mru": raise SystemExit("MRU config is not enabled")
if {key: value for key, value in values.items() if key != "radial_utility"} != baseline:
    raise SystemExit("MRU config changes something other than radial_utility")
git = lambda *args: subprocess.check_output(["git", *args], universal_newlines=True).strip()
manifest = {"label": "Geometry Mid-Range Radial Utility + Bounded Radial NMS", "short_label": "Geometry MRU-BNMS", "split": "val_unseen", "ordered_episode_ids": ast.literal_eval(ids), "dataset_path": dataset, "dataset_sha256": hashes["dataset"], "checkpoint_path": checkpoint, "checkpoint_sha256": hashes["checkpoint"], "auxiliary_waypoint_weight_path": auxiliary, "auxiliary_waypoint_weight_sha256": hashes["auxiliary"], "provenance_matches_restored_formal_runs": True, "provider": "mid360_geometry", "controlled_change": "Pre-NMS radial utility only; Geometry V0-BNMS NMS remains angular-circular and radial-bounded/noncircular.", "radial_utility": {"name": "MRU", "formula": "x * exp(1 - x), x = 2*d/3.0", "D_max_m": 3.0, "analytic_peak_m": 1.5, "hard_cutoff": None, "learned_parameters": None, "fitted_parameters": None}, "geometry_parameters": {"angular_bins": 120, "radial_bins": 12, "radial_spacing_m": 0.25, "maximum_range_m": 3.0, "sigma": [7, 5], "maximum_candidates": 5, "config": values}, "renderer_bootstrap": "scripts/renderer_bootstrap.bash:etp_apply_known_good_renderer_environment", "sensor_guard": "require_valid_sensor_observations at every evaluation reset", "branch": git("branch", "--show-current"), "HEAD": git("rev-parse", "HEAD"), "git_status_at_manifest_creation": git("status", "--short"), "timestamp": datetime.now(timezone.utc).isoformat(), "command": "bash scripts/run_geometry_mru_bnms20_valid_renderer.bash"}
with open(str(Path(root) / "run_manifest.json"), "w") as handle: json.dump(manifest, handle, indent=2); handle.write("\n")
PY

${python_executable} scripts/write_geometry_mru_manifest.py --output "${output_root}" \
  --checkpoint "${checkpoint}" --auxiliary "${auxiliary_weight}" --dataset "${dataset}" \
  --config "${config}" --episode-ids "${episode_ids}"

# The replay operates only on already saved maps and cannot affect evaluation.
${python_executable} scripts/replay_geometry_mru_bnms.py --telemetry "${saved_diagnostic}" \
  --output "${output_root}/offline_sanity_replay.json"

source scripts/renderer_bootstrap.bash
etp_apply_known_good_renderer_environment
${python_executable} -c 'import torch; assert torch.cuda.is_available(), "PyTorch CUDA unavailable"; print("CUDA compute: PASS", torch.cuda.get_device_name(0))'

${python_executable} run.py --exp_name geometry_mru_bnms20_valid_renderer --run-type eval --exp-config run_r2r/iter_train.yaml \
  SIMULATOR_GPU_IDS "[0]" TORCH_GPU_IDS "[0]" TORCH_GPU_ID 0 GPU_NUMBERS 1 NUM_ENVIRONMENTS 1 \
  EVAL.SPLIT val_unseen EVAL.EPISODE_IDS "${episode_ids}" EVAL.EPISODE_COUNT 20 \
  EVAL.CKPT_PATH_DIR "${checkpoint}" RESULTS_DIR "${output_root}/" EVAL.DETAILED_TELEMETRY True \
  IL.back_algo control MODEL.pretrained_path None MODEL.WAYPOINT_PROVIDER mid360_geometry \
  MODEL.MID360_CONFIG_PATH "${config}" VIDEO_OPTION "[]" \
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True

${python_executable} scripts/analyze_geometry_mru_bnms20.py --output "${output_root}" \
  --mru-run "${output_root}/geometry_mru_bnms20_valid_renderer" --bnms-run "${bnms_run}" \
  --original-run "${original_run}"
printf 'COMPLETE\n' > "${output_root}/COMPLETE"
