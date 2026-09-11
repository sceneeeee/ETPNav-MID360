#!/usr/bin/env bash
# Run only from the normal GPU-enabled WSL shell; Codex must not execute it.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${repo_root}/outputs/geometry_v0_baseline20_valid_renderer"
conda_executable="${CONDA_EXE:-/home/evens/miniforge3/bin/conda}"
python_executable="${conda_executable} run -n etpnav_legacy python"
episode_ids="['53','1685','821','1024','1477','1636','675','164','1375','1679','250','2','475','1356','252','1439','1016','1285','1478','25']"
checkpoint="data/logs/checkpoints/release_r2r/ckpt.iter12000.pth"
auxiliary_weight="data/wp_pred/check_cwp_bestdist_hfov90"
dataset="data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/val_unseen/val_unseen_bertidx.json.gz"
config="configs/mid360_waypoint.yaml"
original_telemetry="outputs/original_baseline20_valid_renderer/original_baseline20_valid_renderer/telemetry_ep_ckpt_59_val_unseen_r0_w1.json"

cd "${repo_root}"
if [[ -e "${output_root}" ]]; then
  echo "refusing to overwrite evidence directory: ${output_root}" >&2
  exit 2
fi
mkdir -p "${output_root}"
source "${repo_root}/scripts/renderer_bootstrap.bash"
etp_apply_known_good_renderer_environment

${python_executable} - "${output_root}" "${checkpoint}" "${auxiliary_weight}" "${dataset}" "${config}" "${episode_ids}" <<'PY'
import hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import yaml
root, checkpoint, auxiliary, dataset, config, ids = sys.argv[1:]
expected = {
    "dataset": "7db0a39bef374cf5ce986473f8b35c9721f962c6c1a9af069257aa710df16365",
    "checkpoint": "4e70d2a1b4a6cfb32158901b7300cbc2be657641f16c9a17f35ca8570470d8b3",
    "auxiliary": "09d0f42cbd801e05b0fa0212b901d409f033f4d0bce2fce1fa8a1331b502159f",
}
def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
hashes = {"dataset": sha(dataset), "checkpoint": sha(checkpoint), "auxiliary": sha(auxiliary)}
if hashes != expected:
    raise SystemExit("provenance SHA256 mismatch: {}".format(hashes))
values = yaml.safe_load(Path(config).read_text())
if "maximum_candidate_distance_m" in values:
    raise SystemExit("refusing Geometry V0.1 distance-capped configuration")
git = lambda *args: subprocess.check_output(["git", *args], text=True).strip()
status = git("status", "--short")
json.dump({
    "branch": git("branch", "--show-current"), "HEAD": git("rev-parse", "HEAD"),
    "dirty_status": bool(status), "git_status_at_manifest_creation": status,
    "relevant_changed_files": [line[3:] for line in status.splitlines() if line],
    "ordered_episode_ids": json.loads(ids), "dataset_path": dataset,
    "dataset_sha256": hashes["dataset"], "checkpoint_path": checkpoint,
    "checkpoint_sha256": hashes["checkpoint"], "auxiliary_waypoint_weight_path": auxiliary,
    "auxiliary_waypoint_weight_sha256": hashes["auxiliary"], "provider": "mid360_geometry",
    "provider_version": "Geometry V0 (pre-V0.1 distance-cap configuration)",
    "config_path": config, "exact_geometry_v0_parameters": values,
    "no_v0_1_max_distance_cap": True,
    "renderer_bootstrap": "scripts/renderer_bootstrap.bash:etp_apply_known_good_renderer_environment",
    "sensor_guard": "require_valid_sensor_observations at every evaluation reset",
    "manual_command": "bash scripts/run_geometry_v0_baseline20_valid_renderer_manual.bash",
    "timestamp": datetime.now(timezone.utc).isoformat(),
}, open(Path(root) / "run_manifest.json", "w"), indent=2)
PY

${python_executable} - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA unavailable; refusing formal Geometry V0 run")
torch.cuda.reset_peak_memory_stats()
print("CUDA compute: PASS", torch.cuda.get_device_name(0))
PY

# Minimal, non-tuning startup/single-decision gate.  It uses the same renderer,
# checkpoint, V0 config, and sensor guard, then validates finite raw candidates.
${python_executable} run.py --exp_name geometry_v0_sanity53 --run-type eval --exp-config run_r2r/iter_train.yaml \
  SIMULATOR_GPU_IDS "[0]" TORCH_GPU_IDS "[0]" TORCH_GPU_ID 0 GPU_NUMBERS 1 NUM_ENVIRONMENTS 1 \
  EVAL.SPLIT val_unseen EVAL.EPISODE_IDS "['53']" EVAL.EPISODE_COUNT 1 \
  EVAL.CKPT_PATH_DIR "${checkpoint}" RESULTS_DIR "${output_root}/sanity/" EVAL.DETAILED_TELEMETRY True \
  IL.back_algo control IL.max_traj_len 1 MODEL.pretrained_path None MODEL.WAYPOINT_PROVIDER mid360_geometry \
  MODEL.MID360_CONFIG_PATH "${config}" VIDEO_OPTION "[]" TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True

${python_executable} - "${output_root}/sanity/geometry_v0_sanity53" <<'PY'
import json, math, sys
from pathlib import Path
root = Path(sys.argv[1])
sensor = json.load(open(next(root.glob("sensor_validity_ep_ckpt_*_val_unseen_r0_w1.json"))))
if set(sensor) != {"53"} or not sensor["53"]["rgb_valid"] or not sensor["53"]["depth_valid"]:
    raise SystemExit("Geometry V0 sanity gate failed sensor validation")
telemetry = json.load(open(next(root.glob("telemetry_ep_ckpt_*_val_unseen_r0_w1.json"))))
trace = telemetry["episodes"].get("53", {}).get("waypoint_trace", [])
if not trace:
    raise SystemExit("Geometry V0 sanity gate emitted no decisions")
candidates = trace[0]["candidates"]
if not isinstance(candidates, list):
    raise SystemExit("Geometry V0 sanity gate candidate outcome is malformed")
for candidate in candidates:
    if not math.isfinite(candidate["angle"]) or not math.isfinite(candidate["distance"]):
        raise SystemExit("Geometry V0 sanity gate found non-finite candidate")
print("Geometry V0 sanity gate: PASS; first-decision candidates={}".format(len(candidates)))
PY

${python_executable} run.py --exp_name geometry_v0_baseline20_valid_renderer --run-type eval --exp-config run_r2r/iter_train.yaml \
  SIMULATOR_GPU_IDS "[0]" TORCH_GPU_IDS "[0]" TORCH_GPU_ID 0 GPU_NUMBERS 1 NUM_ENVIRONMENTS 1 \
  EVAL.SPLIT val_unseen EVAL.EPISODE_IDS "${episode_ids}" EVAL.EPISODE_COUNT 20 \
  EVAL.CKPT_PATH_DIR "${checkpoint}" RESULTS_DIR "${output_root}/" EVAL.DETAILED_TELEMETRY True \
  IL.back_algo control MODEL.pretrained_path None MODEL.WAYPOINT_PROVIDER mid360_geometry \
  MODEL.MID360_CONFIG_PATH "${config}" VIDEO_OPTION "[]" TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True

${python_executable} scripts/collect_geometry_v0_baseline20_evidence.py --output "${output_root}" \
  --original-telemetry "${original_telemetry}" --config "${config}"
printf 'COMPLETE\n' > "${output_root}/COMPLETE"
