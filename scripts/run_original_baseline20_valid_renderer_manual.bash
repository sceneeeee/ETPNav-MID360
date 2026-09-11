#!/usr/bin/env bash
# Run only from the normal GPU-enabled WSL shell; Codex must not execute it.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${repo_root}/outputs/original_baseline20_valid_renderer"
conda_executable="${CONDA_EXE:-/home/evens/miniforge3/bin/conda}"
python_executable="${conda_executable} run -n etpnav_legacy python"
episode_ids="['53','1685','821','1024','1477','1636','675','164','1375','1679','250','2','475','1356','252','1439','1016','1285','1478','25']"
checkpoint="data/logs/checkpoints/release_r2r/ckpt.iter12000.pth"
auxiliary_weight="data/wp_pred/check_cwp_bestdist_hfov90"
dataset="data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/val_unseen/val_unseen_bertidx.json.gz"

cd "${repo_root}"
if [[ -e "${output_root}" ]]; then
  echo "refusing to overwrite evidence directory: ${output_root}" >&2
  exit 2
fi
mkdir -p "${output_root}"
source "${repo_root}/scripts/renderer_bootstrap.bash"
etp_apply_known_good_renderer_environment

${python_executable} - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA unavailable; refusing formal baseline run")
torch.cuda.reset_peak_memory_stats()
print("CUDA compute: PASS", torch.cuda.get_device_name(0))
PY

${python_executable} - "${output_root}/run_manifest.json" "${checkpoint}" "${auxiliary_weight}" "${dataset}" "${episode_ids}" <<'PY'
import hashlib, json, subprocess, sys
from datetime import datetime, timezone
manifest, checkpoint, auxiliary, dataset, ids = sys.argv[1:]
def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
git = lambda *args: subprocess.check_output(["git", *args], text=True).strip()
json.dump({"branch": git("branch", "--show-current"), "HEAD": git("rev-parse", "HEAD"),
           "dirty_status": git("status", "--short"), "dataset_split": "val_unseen",
           "dataset_path": dataset, "dataset_hash": sha(dataset),
           "episode_ids": json.loads(ids), "checkpoint_path": checkpoint,
           "checkpoint_sha256": sha(checkpoint), "auxiliary_weight_path": auxiliary,
           "auxiliary_weight_sha256": sha(auxiliary), "provider": "original_etp",
           "renderer_bootstrap": "scripts/renderer_bootstrap.bash:etp_apply_known_good_renderer_environment",
           "timestamp": datetime.now(timezone.utc).isoformat(),
           "command": "bash scripts/run_original_baseline20_valid_renderer_manual.bash"},
          open(manifest, "w"), indent=2)
PY

${python_executable} run.py --exp_name original_baseline20_valid_renderer --run-type eval --exp-config run_r2r/iter_train.yaml \
  SIMULATOR_GPU_IDS "[0]" TORCH_GPU_IDS "[0]" TORCH_GPU_ID 0 GPU_NUMBERS 1 NUM_ENVIRONMENTS 1 \
  EVAL.SPLIT val_unseen EVAL.EPISODE_IDS "${episode_ids}" EVAL.EPISODE_COUNT 20 \
  EVAL.CKPT_PATH_DIR "${checkpoint}" RESULTS_DIR "${output_root}/" EVAL.DETAILED_TELEMETRY True \
  IL.back_algo control MODEL.pretrained_path None MODEL.WAYPOINT_PROVIDER original_etp VIDEO_OPTION "[]" \
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True

${python_executable} scripts/collect_original_baseline20_evidence.py --output "${output_root}"
printf 'COMPLETE\n' > "${output_root}/COMPLETE"
