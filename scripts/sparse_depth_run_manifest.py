#!/usr/bin/env python3
"""Record immutable inputs for the sparse-depth control and its bounded smoke."""
import argparse
import ast
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

DATASET = 'data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/val_unseen/val_unseen_bertidx.json.gz'
CHECKPOINT = 'data/logs/checkpoints/release_r2r/ckpt.iter12000.pth'
AUXILIARY = 'data/wp_pred/check_cwp_bestdist_hfov90'
CONFIG = 'configs/mid360_sparse_depth_zero_shot.yaml'
EXPECTED = {
    DATASET: '7db0a39bef374cf5ce986473f8b35c9721f962c6c1a9af069257aa710df16365',
    CHECKPOINT: '4e70d2a1b4a6cfb32158901b7300cbc2be657641f16c9a17f35ca8570470d8b3',
    AUXILIARY: '09d0f42cbd801e05b0fa0212b901d409f033f4d0bce2fce1fa8a1331b502159f',
}


def sha(path):
    digest = hashlib.sha256()
    with open(str(path), 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(root, episode_ids, command):
    git = lambda *args: subprocess.check_output(['git'] + list(args), universal_newlines=True).strip()
    branch = git('branch', '--show-current')
    if branch != 'mid360-sparse-depth-zero-shot':
        raise RuntimeError('wrong experiment branch: ' + branch)
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise RuntimeError('frozen baseline input SHA256 mismatch: ' + path)
    manifest = dict(branch=branch, HEAD=git('rev-parse', 'HEAD'), dirty_status=git('status', '--short'),
                    dataset_path=DATASET, dataset_sha256=sha(DATASET),
                    checkpoint_path=CHECKPOINT, checkpoint_sha256=sha(CHECKPOINT),
                    auxiliary_weight_path=AUXILIARY, auxiliary_weight_sha256=sha(AUXILIARY),
                    sparse_depth_config_path=CONFIG, sparse_depth_config_sha256=sha(CONFIG),
                    sparse_depth_config=Path(CONFIG).read_text(), provider='mid360_sparse_depth_etp',
                    episode_ids=episode_ids, timestamp=datetime.now(timezone.utc).isoformat(),
                    command=command, shell_command=' '.join(shlex.quote(x) for x in command),
                    renderer_bootstrap='scripts/renderer_bootstrap.bash:etp_apply_known_good_renderer_environment',
                    renderer_bootstrap_sha256=sha('scripts/renderer_bootstrap.bash'),
                    LD_PRELOAD=os.environ.get('LD_PRELOAD', ''),
                    experiment_config_path='run_r2r/iter_train.yaml',
                    experiment_config_sha256=sha('run_r2r/iter_train.yaml'))
    with open(str(Path(root) / 'run_manifest.json'), 'x') as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--episode-ids', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    write_manifest(args.output, ast.literal_eval(args.episode_ids), command)
