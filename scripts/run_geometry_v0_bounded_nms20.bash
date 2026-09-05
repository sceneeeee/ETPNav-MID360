#!/usr/bin/env bash
# Formal frozen-20 Geometry V0-BNMS evaluation. Refuses to overwrite evidence.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${repo_root}/outputs/geometry_v0_bounded_nms20_valid_renderer"
conda_executable="${CONDA_EXE:-/home/evens/miniforge3/bin/conda}"
python_executable="${conda_executable} run -n etpnav_legacy python"
episode_ids="['53','1685','821','1024','1477','1636','675','164','1375','1679','250','2','475','1356','252','1439','1016','1285','1478','25']"
checkpoint="data/logs/checkpoints/release_r2r/ckpt.iter12000.pth"
auxiliary_weight="data/wp_pred/check_cwp_bestdist_hfov90"
dataset="data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/val_unseen/val_unseen_bertidx.json.gz"
config="configs/mid360_waypoint.yaml"
legacy_run="outputs/geometry_v0_baseline20_valid_renderer/geometry_v0_baseline20_valid_renderer"
original_run="outputs/original_baseline20_valid_renderer/original_baseline20_valid_renderer"

cd "${repo_root}"
[[ ! -e "${output_root}" ]] || { echo "refusing to overwrite ${output_root}" >&2; exit 2; }
[[ -d "${legacy_run}" && -d "${original_run}" ]] || { echo "missing formal baseline artifacts" >&2; exit 2; }
mkdir -p "${output_root}"
source scripts/renderer_bootstrap.bash
etp_apply_known_good_renderer_environment

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
if "maximum_candidate_distance_m" in values: raise SystemExit("Geometry V0.1 distance cap is forbidden")
git = lambda *args: subprocess.check_output(["git", *args], universal_newlines=True).strip()
status = git("status", "--short")
json.dump({"label": "Geometry V0 + Bounded Radial NMS", "short_label": "Geometry V0-BNMS", "split": "val_unseen", "ordered_episode_ids": ast.literal_eval(ids), "dataset_path": dataset, "dataset_sha256": hashes["dataset"], "checkpoint_path": checkpoint, "checkpoint_sha256": hashes["checkpoint"], "auxiliary_waypoint_weight_path": auxiliary, "auxiliary_waypoint_weight_sha256": hashes["auxiliary"], "provenance_matches_restored_formal_runs": hashes == expected, "provider": "mid360_geometry", "controlled_change": "Geometry-specific NMS boundary semantics only: angular circular; radial bounded/noncircular.", "geometry_parameters": {"score": "S(a,k)=d/3", "angular_bins": 120, "radial_bins": 12, "sigma": [7, 5], "maximum_candidates": 5, "radial_distance_cap": None, "config": values}, "renderer_bootstrap": "scripts/renderer_bootstrap.bash:etp_apply_known_good_renderer_environment", "sensor_guard": "require_valid_sensor_observations at every evaluation reset", "branch": git("branch", "--show-current"), "HEAD": git("rev-parse", "HEAD"), "git_status_at_manifest_creation": status, "timestamp": datetime.now(timezone.utc).isoformat(), "command": "bash scripts/run_geometry_v0_bounded_nms20.bash"}, open(Path(root) / "run_manifest.json", "w"), indent=2)
PY

${python_executable} - <<'PY'
import torch
if not torch.cuda.is_available(): raise SystemExit("PyTorch CUDA unavailable")
print("CUDA compute: PASS", torch.cuda.get_device_name(0))
PY

${python_executable} run.py --exp_name geometry_v0_bounded_nms20_valid_renderer --run-type eval --exp-config run_r2r/iter_train.yaml \
  SIMULATOR_GPU_IDS "[0]" TORCH_GPU_IDS "[0]" TORCH_GPU_ID 0 GPU_NUMBERS 1 NUM_ENVIRONMENTS 1 \
  EVAL.SPLIT val_unseen EVAL.EPISODE_IDS "${episode_ids}" EVAL.EPISODE_COUNT 20 \
  EVAL.CKPT_PATH_DIR "${checkpoint}" RESULTS_DIR "${output_root}/" EVAL.DETAILED_TELEMETRY True \
  IL.back_algo control MODEL.pretrained_path None MODEL.WAYPOINT_PROVIDER mid360_geometry \
  MODEL.MID360_CONFIG_PATH "${config}" VIDEO_OPTION "[]" TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True

${python_executable} scripts/analyze_geometry_v0_bounded_nms20.py --output "${output_root}" \
  --bounded-run "${output_root}/geometry_v0_bounded_nms20_valid_renderer" --legacy-run "${legacy_run}" --original-run "${original_run}"
${python_executable} scripts/write_geometry_v0_bnms_manifest.py --output "${output_root}" \
  --checkpoint "${checkpoint}" --auxiliary "${auxiliary_weight}" --dataset "${dataset}" \
  --config "${config}" --episode-ids "${episode_ids}"
printf 'COMPLETE\n' > "${output_root}/COMPLETE"
