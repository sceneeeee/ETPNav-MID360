"""Stage 5 M1 Paired Feature Dataset and Scene-Level Splitting.

Implements PyTorch Dataset/DataLoader interfaces with strict Scene-level
isolation to guarantee zero spatial data leakage between train and validation.
Pure PyTorch; compatible with Python 3.6 / PyTorch 1.9.1.
"""
import random
from typing import Dict, List, Optional, Tuple, Set, Any

import torch
from torch.utils.data import Dataset, DataLoader

from sparse_depth_feature_cache import FeatureCache, validate_cache_entry

# Strict exclusions from Stage 3/4/5 specifications
HELD_OUT_IDS = {'53', '821', '1024', '250', '25'}
FROZEN_IDS = {
    '53', '1685', '821', '1024', '1477', '1636', '675', '164', '1375', '1679',
    '250', '2', '475', '1356', '252', '1439', '1016', '1285', '1478', '25'
}


def assert_no_leakage(train_scenes, val_scenes, all_episodes):
    # type: (Set[str], Set[str], List[Dict[str, Any]]) -> None
    """Strictly assert no scene-level leakage and no held-out contamination."""
    overlap = train_scenes.intersection(val_scenes)
    if overlap:
        raise ValueError('Scene-level data leakage detected! Overlapping scenes: {}'.format(sorted(list(overlap))))

    for entry in all_episodes:
        eid = str(entry['episode_id'])
        # Canonicalize leading zeros before check
        try:
            eid_norm = str(int(eid))
        except (ValueError, TypeError):
            eid_norm = eid

        if eid in HELD_OUT_IDS or eid_norm in HELD_OUT_IDS:
            raise ValueError('Held-out ID {} detected in dataset! Contamination forbidden.'.format(eid))
        if eid in FROZEN_IDS or eid_norm in FROZEN_IDS:
            raise ValueError('Frozen ID {} detected in dataset! Contamination forbidden.'.format(eid))


class PairedFeatureDataset(Dataset):
    """PyTorch Dataset returning paired sparse and dense features."""

    def __init__(self, entries, transform=None):
        # type: (List[Dict[str, Any]], Optional[Any]) -> None
        self.entries = entries
        self.transform = transform
        for entry in self.entries:
            validate_cache_entry(entry)

    def __len__(self):
        # type: () -> int
        return len(self.entries)

    def __getitem__(self, idx):
        # type: (int) -> Dict[str, Any]
        item = self.entries[idx]
        sample = {
            'sparse': item['sparse'].clone(),
            'dense': item['dense'].clone(),
            'episode_id': item['episode_id'],
            'scene_id': item['scene_id'],
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

    @property
    def scene_ids(self):
        # type: () -> List[str]
        return sorted(list(set(e['scene_id'] for e in self.entries)))

    @property
    def episode_ids(self):
        # type: () -> List[str]
        return [e['episode_id'] for e in self.entries]


def create_scene_level_splits(cache, val_ratio=0.2, seed=42):
    # type: (FeatureCache, float, int) -> Tuple[PairedFeatureDataset, PairedFeatureDataset, Dict[str, Any]]
    """Partition feature cache into train and val datasets strictly by scene_id."""
    if not (0.0 < val_ratio < 1.0):
        raise ValueError('val_ratio must be strictly between 0 and 1, got {}'.format(val_ratio))
    if len(cache) == 0:
        raise ValueError('cannot split empty feature cache')

    all_scenes = cache.scene_ids
    if len(all_scenes) < 2:
        raise ValueError('at least 2 distinct scenes required for scene-level train/val split, found: {}'.format(all_scenes))

    # Deterministic scene shuffle
    rng = random.Random(seed)
    shuffled_scenes = list(all_scenes)
    shuffled_scenes.sort()  # Canonical start order
    rng.shuffle(shuffled_scenes)

    val_count = max(1, int(round(len(shuffled_scenes) * val_ratio)))
    val_count = min(val_count, len(shuffled_scenes) - 1)  # Ensure train has at least 1 scene

    val_scenes = set(shuffled_scenes[:val_count])
    train_scenes = set(shuffled_scenes[val_count:])

    # Perform strict leakage and contamination audit
    assert_no_leakage(train_scenes, val_scenes, cache.entries)

    train_entries = [e for e in cache.entries if e['scene_id'] in train_scenes]
    val_entries = [e for e in cache.entries if e['scene_id'] in val_scenes]

    train_dataset = PairedFeatureDataset(train_entries)
    val_dataset = PairedFeatureDataset(val_entries)

    split_info = {
        'seed': seed,
        'val_ratio': val_ratio,
        'total_entries': len(cache),
        'train_entries': len(train_entries),
        'val_entries': len(val_entries),
        'total_scenes': len(all_scenes),
        'train_scenes': sorted(list(train_scenes)),
        'val_scenes': sorted(list(val_scenes)),
    }

    return train_dataset, val_dataset, split_info


def paired_collate_fn(batch):
    # type: (List[Dict[str, Any]]) -> Dict[str, Any]
    """Collate batch into stacked tensors."""
    sparse_batch = torch.stack([item['sparse'] for item in batch], dim=0)  # [B, 12, 128, 4, 4]
    dense_batch = torch.stack([item['dense'] for item in batch], dim=0)    # [B, 12, 128, 4, 4]
    return {
        'sparse': sparse_batch,
        'dense': dense_batch,
        'episode_id': [item['episode_id'] for item in batch],
        'scene_id': [item['scene_id'] for item in batch],
    }


def create_dataloaders(train_dataset, val_dataset, batch_size=4, num_workers=0):
    # type: (PairedFeatureDataset, PairedFeatureDataset, int, int) -> Tuple[DataLoader, DataLoader]
    """Create PyTorch DataLoaders for train and validation datasets."""
    if batch_size <= 0:
        raise ValueError('batch_size must be a positive integer, got {}'.format(batch_size))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=paired_collate_fn,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=paired_collate_fn,
        drop_last=False,
    )

    return train_loader, val_loader
