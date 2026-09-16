"""Stage 5 M1 A1 ResidualFeatureAdapter Training Pipeline.

Trains the lightweight adapter on offline paired feature cache using the frozen
M0 loss: SmoothL1 + (1 - cosine) with equal 1.0 weights.
Features are fixed; Depth Encoder and Waypoint Predictor are never touched.
Outputs standardized checkpoints and run manifests conforming to Stage 5 contract.
Compatible with Python 3.6 / PyTorch 1.9.1.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.optim import Adam

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from sparse_depth_learned_adapter import (
    ResidualFeatureAdapter,
    feature_distillation_loss,
    save_adapter_checkpoint,
    load_adapter_checkpoint,
    ARCHITECTURE,
)
from sparse_depth_feature_cache import FeatureCache, compute_file_sha256
from sparse_depth_feature_dataset import (
    create_scene_level_splits,
    create_dataloaders,
    PairedFeatureDataset,
)


def get_git_info():
    # type: () -> Dict[str, str]
    """Retrieve git provenance."""
    def run_git(*args):
        try:
            return subprocess.check_output(['git'] + list(args), universal_newlines=True).strip()
        except Exception:
            return 'unknown'
    return {
        'branch': run_git('branch', '--show-current'),
        'commit': run_git('rev-parse', 'HEAD'),
        'dirty': bool(run_git('status', '--porcelain')),
    }


def evaluate_adapter(adapter, val_loader, device='cpu'):
    # type: (ResidualFeatureAdapter, Any, str) -> Dict[str, float]
    """Evaluate adapter on validation dataloader."""
    adapter.eval()
    total_loss_sum = 0.0
    smooth_l1_sum = 0.0
    cosine_loss_sum = 0.0
    mean_cosine_sim_sum = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            sparse = batch['sparse'].to(device=device)  # [B, 12, 128, 4, 4]
            dense = batch['dense'].to(device=device)    # [B, 12, 128, 4, 4]
            b_size = sparse.size(0)

            # Flatten batch and view dimensions: [B * 12, 128, 4, 4]
            sparse_flat = sparse.view(-1, 128, 4, 4)
            dense_flat = dense.view(-1, 128, 4, 4)

            adapted_flat = adapter(sparse_flat)
            metrics = feature_distillation_loss(adapted_flat, dense_flat)

            total_loss_sum += metrics['total'].item() * b_size
            smooth_l1_sum += metrics['smooth_l1'].item() * b_size
            cosine_loss_sum += metrics['cosine_loss'].item() * b_size
            mean_cosine_sim_sum += metrics['mean_cosine_similarity'].item() * b_size
            num_samples += b_size

    if num_samples == 0:
        return {'total': 0.0, 'smooth_l1': 0.0, 'cosine_loss': 0.0, 'mean_cosine_similarity': 0.0}

    return {
        'total': total_loss_sum / num_samples,
        'smooth_l1': smooth_l1_sum / num_samples,
        'cosine_loss': cosine_loss_sum / num_samples,
        'mean_cosine_similarity': mean_cosine_sim_sum / num_samples,
    }


def train_adapter_pipeline(
    train_loader,
    val_loader,
    epochs=50,
    lr=1e-3,
    seed=42,
    device='cpu',
    output_dir=None,
    split_info=None,
):
    # type: (Any, Any, int, float, int, str, Optional[Path], Optional[Dict[str, Any]]) -> Tuple[ResidualFeatureAdapter, Dict[str, Any]]
    """Execute A1 adapter training loop and validation tracking."""
    torch.manual_seed(seed)
    adapter = ResidualFeatureAdapter(channels=128).to(device=device)

    # Sanity check initial zero-init identity
    dummy_input = torch.randn(12, 128, 4, 4, device=device)
    with torch.no_grad():
        identity_diff = (adapter(dummy_input) - dummy_input).abs().max().item()
    if identity_diff != 0.0:
        raise RuntimeError('Fresh adapter must be exact identity, max diff: {}'.format(identity_diff))

    optimizer = Adam(adapter.parameters(), lr=lr)

    # Initial pre-training metrics
    initial_val_metrics = evaluate_adapter(adapter, val_loader, device=device)

    history = []
    best_val_loss = float('inf')
    best_epoch = -1
    best_metrics = initial_val_metrics

    best_checkpoint_path = None
    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        best_checkpoint_path = output_dir / 'best_adapter.pt'

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        adapter.train()
        train_total_sum = 0.0
        train_l1_sum = 0.0
        train_cos_sum = 0.0
        train_sim_sum = 0.0
        train_samples = 0

        for batch in train_loader:
            sparse = batch['sparse'].to(device=device)
            dense = batch['dense'].to(device=device)
            b_size = sparse.size(0)

            sparse_flat = sparse.view(-1, 128, 4, 4)
            dense_flat = dense.view(-1, 128, 4, 4)

            optimizer.zero_grad()
            adapted_flat = adapter(sparse_flat)
            loss_dict = feature_distillation_loss(adapted_flat, dense_flat)

            loss = loss_dict['total']
            if not torch.isfinite(loss):
                raise RuntimeError('Encountered non-finite loss in epoch {}'.format(epoch))

            loss.backward()

            # Check gradients
            for p in adapter.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise RuntimeError('Non-finite gradient encountered in epoch {}'.format(epoch))

            optimizer.step()

            train_total_sum += loss.item() * b_size
            train_l1_sum += loss_dict['smooth_l1'].item() * b_size
            train_cos_sum += loss_dict['cosine_loss'].item() * b_size
            train_sim_sum += loss_dict['mean_cosine_similarity'].item() * b_size
            train_samples += b_size

        train_metrics = {
            'total': train_total_sum / max(1, train_samples),
            'smooth_l1': train_l1_sum / max(1, train_samples),
            'cosine_loss': train_cos_sum / max(1, train_samples),
            'mean_cosine_similarity': train_sim_sum / max(1, train_samples),
        }

        # Validation phase
        val_metrics = evaluate_adapter(adapter, val_loader, device=device)

        epoch_record = {
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
        }
        history.append(epoch_record)

        # Track best model
        if val_metrics['total'] < best_val_loss:
            best_val_loss = val_metrics['total']
            best_epoch = epoch
            best_metrics = val_metrics

            if best_checkpoint_path is not None:
                if best_checkpoint_path.exists():
                    best_checkpoint_path.unlink()
                save_adapter_checkpoint(
                    best_checkpoint_path,
                    adapter,
                    smooth_l1_weight=1.0,
                    cosine_weight=1.0,
                    seed=seed,
                    step=epoch,
                )

    total_duration = time.time() - start_time

    # Load best adapter for final verification if saved
    if best_checkpoint_path is not None and best_checkpoint_path.exists():
        restored_adapter, _ = load_adapter_checkpoint(best_checkpoint_path, expected_channels=128)
        restored_adapter.to(device=device)
        checkpoint_sha256 = compute_file_sha256(best_checkpoint_path)
    else:
        restored_adapter = adapter
        checkpoint_sha256 = None

    run_manifest = {
        'stage': 'M1_TRAINING',
        'status': 'COMPLETED',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'duration_seconds': total_duration,
        'git': get_git_info(),
        'hyperparameters': {
            'epochs': epochs,
            'learning_rate': lr,
            'seed': seed,
            'device': device,
            'architecture': ARCHITECTURE,
            'channels': 128,
            'loss_weights': {'smooth_l1': 1.0, 'cosine': 1.0},
        },
        'split_info': split_info or {},
        'initial_val_metrics': initial_val_metrics,
        'best_epoch': best_epoch,
        'best_val_metrics': best_metrics,
        'final_val_metrics': history[-1]['val'] if history else initial_val_metrics,
        'best_checkpoint': str(best_checkpoint_path) if best_checkpoint_path else None,
        'best_checkpoint_sha256': checkpoint_sha256,
        'history': history,
    }

    if output_dir is not None:
        manifest_path = output_dir / 'run_manifest.json'
        with open(str(manifest_path), 'w') as handle:
            json.dump(run_manifest, handle, indent=2, sort_keys=True)

    return restored_adapter, run_manifest


def main():
    parser = argparse.ArgumentParser(description='Stage 5 M1 Adapter Training Pipeline')
    parser.add_argument('--cache-dir', required=True, type=Path, help='Path to offline feature cache directory')
    parser.add_argument('--output-dir', required=True, type=Path, help='Target directory for checkpoint and manifest')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size (default: 4)')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation scene ratio (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cpu', help='Device: cpu or cuda (default: cpu)')

    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('Output directory already exists and is non-empty: {}'.format(output_dir))

    print('Loading feature cache from {}...'.format(cache_dir))
    cache = FeatureCache.load(cache_dir)
    print('Loaded {} paired entries across {} scenes.'.format(len(cache), len(cache.scene_ids)))

    train_ds, val_ds, split_info = create_scene_level_splits(cache, val_ratio=args.val_ratio, seed=args.seed)
    print('Scene-level split: {} train episodes ({} scenes), {} val episodes ({} scenes)'.format(
        len(train_ds), len(split_info['train_scenes']), len(val_ds), len(split_info['val_scenes'])))

    train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=args.batch_size)

    print('Starting adapter training for {} epochs on {}...'.format(args.epochs, args.device))
    _, manifest = train_adapter_pipeline(
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        output_dir=output_dir,
        split_info=split_info,
    )

    print('Training complete! Best epoch: {}, Best Val Loss: {:.6f}, Cosine Sim: {:.6f}'.format(
        manifest['best_epoch'], manifest['best_val_metrics']['total'],
        manifest['best_val_metrics']['mean_cosine_similarity']))
    print('Saved best checkpoint to {}'.format(manifest['best_checkpoint']))


if __name__ == '__main__':
    main()
