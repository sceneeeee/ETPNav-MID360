#!/usr/bin/env python3
"""Stage 5 M1 Paired Feature Data Collector.

Extracts paired dense and sparse depth feature representations [12, 128, 4, 4]
across multiple scenes/episodes using the Stage 4 Habitat runtime seam.
Saves the collected pairs into an offline FeatureCache for reproducible training.

Strict Safety Rules:
1. Zero Data Leakage: strictly rejects HELD_OUT_IDS ('53', '821', '1024', '250', '25')
   and FROZEN_IDS (the 20 val_unseen benchmark IDs).
2. Pure Detached Storage: features are detached and cloned to CPU; Original models
   are kept in eval mode with requires_grad=False and their parameters unchanged.
3. Compatible with Python 3.6 / PyTorch 1.9.1.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Set, Tuple, Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from sparse_depth_feature_cache import FeatureCache, validate_feature_tensor
from sparse_depth_feature_dataset import HELD_OUT_IDS, FROZEN_IDS

CALIBRATION_DATA = 'data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/val_seen/val_seen_bertidx.json.gz'
CALIBRATION_SHA = '83b231674ae3a18cb02b3d65847377e978f45cc35682f807c03becc38b09410c'


def canonicalize_episode_id(episode_id):
    # type: (Any) -> str
    """Convert episode ID to normalized numeric string."""
    try:
        return str(int(str(episode_id).strip()))
    except (ValueError, TypeError):
        raise ValueError('Episode ID must be numeric, got: {!r}'.format(episode_id))


def validate_episode_id(episode_id):
    # type: (Any) -> str
    """Validate single episode ID against held-out and frozen exclusions."""
    eid = canonicalize_episode_id(episode_id)
    if eid in HELD_OUT_IDS:
        raise ValueError('Episode {} is in HELD_OUT_IDS; collection strictly forbidden!'.format(eid))
    if eid in FROZEN_IDS:
        raise ValueError('Episode {} is in FROZEN_IDS; collection strictly forbidden!'.format(eid))
    return eid


def validate_episode_ids(episode_ids):
    # type: (List[Any]) -> List[str]
    """Validate and deduplicate a sequence of episode IDs."""
    if not episode_ids:
        raise ValueError('Episode list cannot be empty')
    validated = []
    seen = set()  # type: Set[str]
    for raw_id in episode_ids:
        eid = validate_episode_id(raw_id)
        if eid in seen:
            raise ValueError('Duplicate episode ID detected: {}'.format(eid))
        seen.add(eid)
        validated.append(eid)
    return validated


def load_dataset_episodes(dataset_path=None):
    # type: (Optional[Path]) -> List[Dict[str, Any]]
    """Load episodes from val_seen dataset json.gz."""
    path = Path(dataset_path) if dataset_path else ROOT / CALIBRATION_DATA
    if not path.exists():
        raise FileNotFoundError('Dataset file not found: {}'.format(path))
    with gzip.open(str(path), 'rt') as handle:
        data = json.load(handle)
    return data.get('episodes', [])


def filter_eligible_episodes(episodes, one_per_scene=True, max_episodes=None):
    # type: (List[Dict[str, Any]], bool, Optional[int]) -> List[Dict[str, Any]]
    """Select eligible episodes excluding held-out and frozen IDs."""
    scene_map = {}  # type: Dict[str, List[Dict[str, Any]]]
    for ep in episodes:
        raw_id = ep.get('episode_id')
        try:
            eid = validate_episode_id(raw_id)
        except ValueError:
            # Skip held-out, frozen, or non-numeric IDs
            continue
        scene_id = ep.get('scene_id')
        if not scene_id:
            continue
        if scene_id not in scene_map:
            scene_map[scene_id] = []
        scene_map[scene_id].append({'episode_id': eid, 'scene_id': scene_id})

    selected = []  # type: List[Dict[str, Any]]
    for scene_id in sorted(scene_map.keys()):
        eps = sorted(scene_map[scene_id], key=lambda x: int(x['episode_id']))
        if one_per_scene:
            selected.append(eps[0])
        else:
            selected.extend(eps)

    if max_episodes is not None and max_episodes > 0:
        selected = selected[:max_episodes]

    return selected


def collect_features_for_episode(stage4, episode_info, tmp_output_dir):
    # type: (Any, Dict[str, str], Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]
    """Collect paired dense and sparse features for a single episode reset."""
    eid = episode_info['episode_id']
    scene_id = episode_info['scene_id']
    validate_episode_id(eid)

    envs = None
    try:
        tm, config, trainer, envs, provider = stage4.runtime(
            'val_seen', [eid], tmp_output_dir, calibration_only=True
        )

        for module in (trainer.policy, provider.original.rgb_encoder,
                       provider.original.depth_encoder, provider.original.waypoint_predictor):
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)
                p.grad = None

        modules = {
            'depth_encoder': provider.original.depth_encoder,
            'predictor': provider.original.waypoint_predictor,
            'policy': trainer.policy,
        }
        before_hashes = {name: stage4.state_fingerprint(m) for name, m in modules.items()}

        sensor_diag_path = tmp_output_dir / 'sensors_{}.json'.format(eid)
        batch, record = stage4.reset_batch(
            tm, config, trainer, envs,
            diagnostic_path=sensor_diag_path,
            calibration_only=True
        )

        # Close env before running encoder forward
        envs.close()
        envs = None

        features = stage4.calibration_features(provider, batch)
        dense = features['dense'].detach().cpu().clone().float()
        sparse = features['sparse'].detach().cpu().clone().float()

        validate_feature_tensor(dense, 'dense')
        validate_feature_tensor(sparse, 'sparse')

        after_hashes = {name: stage4.state_fingerprint(m) for name, m in modules.items()}
        if before_hashes != after_hashes:
            raise RuntimeError('Original models modified during feature collection for episode {}'.format(eid))

        provenance = {
            'episode_id': eid,
            'scene_id': scene_id,
            'record': record,
            'original_models_unchanged': True,
        }
        return {'dense': dense, 'sparse': sparse}, provenance

    finally:
        if envs is not None:
            envs.close()


def collect_feature_pairs_pipeline(
    episodes_to_collect,
    output_cache_dir,
    stage4_module=None,
    tmp_working_dir=None,
):
    # type: (List[Dict[str, str]], Path, Optional[Any], Optional[Path]) -> FeatureCache
    """Orchestrate multi-observation paired feature collection into a FeatureCache."""
    if not episodes_to_collect:
        raise ValueError('No episodes provided for collection')

    output_cache_dir = Path(output_cache_dir).resolve()
    if tmp_working_dir is None:
        tmp_working_dir = output_cache_dir / '_tmp_runtime'
    tmp_working_dir.mkdir(parents=True, exist_ok=True)

    validated_ids = validate_episode_ids([e['episode_id'] for e in episodes_to_collect])

    cache = FeatureCache()

    for idx, ep_info in enumerate(episodes_to_collect):
        eid = ep_info['episode_id']
        scene_id = ep_info['scene_id']
        print('[{}/{}] Collecting paired features for episode {} (scene: {})...'.format(
            idx + 1, len(episodes_to_collect), eid, scene_id
        ))

        features, prov = collect_features_for_episode(
            stage4=stage4_module,
            episode_info=ep_info,
            tmp_output_dir=tmp_working_dir
        )

        cache.add_entry({
            'episode_id': eid,
            'scene_id': scene_id,
            'dense': features['dense'],
            'sparse': features['sparse'],
            'provenance': prov,
        })

    manifest_extra = {
        'collection_tool': 'collect_sparse_depth_feature_pairs.py',
        'requested_count': len(episodes_to_collect),
        'collected_count': len(cache),
    }
    cache.save(output_cache_dir, manifest_extra=manifest_extra)
    print('Successfully saved FeatureCache with {} entries to {}'.format(len(cache), output_cache_dir))
    return cache


def parse_args():
    # type: () -> argparse.Namespace
    parser = argparse.ArgumentParser(description='Stage 5 M1 Paired Feature Collector')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/stage5_m1_feature_cache',
                        help='Destination directory for the FeatureCache')
    parser.add_argument('--dataset-path', type=Path, default=ROOT / CALIBRATION_DATA,
                        help='Path to val_seen dataset json.gz')
    parser.add_argument('--max-episodes', type=int, default=None,
                        help='Maximum number of episodes to collect')
    parser.add_argument('--one-per-scene', action='store_true', default=True,
                        help='Collect at most one observation per scene')
    parser.add_argument('--episodes', type=str, default=None,
                        help='Comma-separated episode IDs to collect')
    return parser.parse_args()


def main():
    # type: () -> None
    args = parse_args()

    if args.episodes:
        raw_ids = [s.strip() for s in args.episodes.split(',') if s.strip()]
        validated_ids = validate_episode_ids(raw_ids)
        all_eps = load_dataset_episodes(args.dataset_path)
        ep_map = {str(e['episode_id']): e['scene_id'] for e in all_eps}
        selected = []
        for eid in validated_ids:
            if eid not in ep_map:
                raise ValueError('Episode ID {} not found in dataset'.format(eid))
            selected.append({'episode_id': eid, 'scene_id': ep_map[eid]})
    else:
        all_eps = load_dataset_episodes(args.dataset_path)
        selected = filter_eligible_episodes(
            all_eps,
            one_per_scene=args.one_per_scene,
            max_episodes=args.max_episodes
        )

    if not selected:
        raise RuntimeError('No eligible episodes found for collection')

    print('Selected {} eligible episodes for collection.'.format(len(selected)))

    # Import Habitat runtime seam when executed directly on host
    import run_sparse_depth_feature_alignment as stage4
    collect_feature_pairs_pipeline(
        episodes_to_collect=selected,
        output_cache_dir=args.output_dir,
        stage4_module=stage4
    )


if __name__ == '__main__':
    main()
