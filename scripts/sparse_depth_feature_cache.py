"""Stage 5 M1 reproducible offline feature cache.

Stores paired dense and sparse features [12, 128, 4, 4] with observation
metadata, enabling fast adapter training completely independent of Habitat.
Pure PyTorch and standard library; fully compatible with Python 3.6 / PyTorch 1.9.1.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch

CACHE_FORMAT_VERSION = 1
EXPECTED_SHAPE = (12, 128, 4, 4)
EXPECTED_DTYPE = torch.float32


def validate_feature_tensor(tensor, name='feature', expected_channels=128):
    # type: (torch.Tensor, str, int) -> None
    """Validate feature tensor conforms strictly to contract."""
    if not torch.is_tensor(tensor):
        raise TypeError('expected {} to be a torch.Tensor, got {}'.format(name, type(tensor)))
    if tensor.ndim != 4:
        raise ValueError('expected 4D tensor for {}, got shape {}'.format(name, tuple(tensor.shape)))
    expected = (12, expected_channels, 4, 4)
    if tuple(tensor.shape) != expected:
        raise ValueError('expected {} shape {}, got {}'.format(name, expected, tuple(tensor.shape)))
    if tensor.dtype != EXPECTED_DTYPE:
        raise ValueError('expected {} dtype {}, got {}'.format(name, EXPECTED_DTYPE, tensor.dtype))
    if not torch.isfinite(tensor).all():
        raise ValueError('expected {} to contain only finite values'.format(name))


def validate_cache_entry(entry, expected_channels=128):
    # type: (Dict[str, Any], int) -> None
    """Validate a single paired feature entry."""
    if not isinstance(entry, dict):
        raise TypeError('expected entry to be a dict, got {}'.format(type(entry)))
    required_keys = {'episode_id', 'scene_id', 'dense', 'sparse'}
    missing = required_keys - set(entry.keys())
    if missing:
        raise ValueError('entry missing required keys: {}'.format(sorted(list(missing))))
    if not isinstance(entry['episode_id'], str) or not entry['episode_id'].strip():
        raise ValueError('entry episode_id must be a non-empty string')
    if not isinstance(entry['scene_id'], str) or not entry['scene_id'].strip():
        raise ValueError('entry scene_id must be a non-empty string')
    validate_feature_tensor(entry['dense'], 'dense', expected_channels)
    validate_feature_tensor(entry['sparse'], 'sparse', expected_channels)


def compute_file_sha256(file_path):
    # type: (Path) -> str
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(str(file_path), 'rb') as handle:
        for chunk in iter(lambda: handle.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


class FeatureCache(object):
    """Container managing paired dense/sparse feature cache."""

    def __init__(self, entries=None, metadata=None):
        # type: (Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]) -> None
        self.entries = []  # type: List[Dict[str, Any]]
        self.metadata = metadata or {}  # type: Dict[str, Any]
        if entries:
            for entry in entries:
                self.add_entry(entry)

    def __len__(self):
        # type: () -> int
        return len(self.entries)

    def __getitem__(self, idx):
        # type: (int) -> Dict[str, Any]
        return self.entries[idx]

    def add_entry(self, entry):
        # type: (Dict[str, Any]) -> None
        validate_cache_entry(entry)
        eid = str(entry['episode_id'])
        if any(str(e['episode_id']) == eid for e in self.entries):
            raise ValueError('duplicate episode_id in cache: {}'.format(eid))
        # Store detached CPU tensors
        clean_entry = {
            'episode_id': eid,
            'scene_id': str(entry['scene_id']),
            'dense': entry['dense'].detach().cpu().clone(),
            'sparse': entry['sparse'].detach().cpu().clone(),
        }
        if 'provenance' in entry:
            clean_entry['provenance'] = entry['provenance']
        self.entries.append(clean_entry)

    @property
    def episode_ids(self):
        # type: () -> List[str]
        return [e['episode_id'] for e in self.entries]

    @property
    def scene_ids(self):
        # type: () -> List[str]
        return sorted(list(set(e['scene_id'] for e in self.entries)))

    def save(self, target_dir, manifest_extra=None):
        # type: (Path, Optional[Dict[str, Any]]) -> Path
        """Save cache directory with features.pt and manifest.json."""
        target_dir = Path(target_dir).resolve()
        if target_dir.exists() and any(target_dir.iterdir()):
            raise FileExistsError('refusing to overwrite non-empty cache directory: {}'.format(target_dir))
        target_dir.mkdir(parents=True, exist_ok=True)

        features_path = target_dir / 'features.pt'
        manifest_path = target_dir / 'manifest.json'

        # Serialize tensor payload
        payload = {
            'format_version': CACHE_FORMAT_VERSION,
            'feature_shape': list(EXPECTED_SHAPE),
            'dtype': 'float32',
            'entries': self.entries,
            'metadata': self.metadata,
        }
        with open(str(features_path), 'wb') as handle:
            torch.save(payload, handle)

        features_sha256 = compute_file_sha256(features_path)

        # Build manifest summary (without raw tensors)
        manifest = {
            'format_version': CACHE_FORMAT_VERSION,
            'entry_count': len(self.entries),
            'scene_count': len(self.scene_ids),
            'scenes': self.scene_ids,
            'episode_ids': self.episode_ids,
            'features_file': 'features.pt',
            'features_sha256': features_sha256,
            'feature_shape': list(EXPECTED_SHAPE),
            'dtype': 'float32',
            'metadata': self.metadata,
        }
        if manifest_extra:
            manifest.update(manifest_extra)

        with open(str(manifest_path), 'w') as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

        return manifest_path

    @classmethod
    def load(cls, target_dir, verify_sha=True):
        # type: (Path, bool) -> FeatureCache
        """Load cache from target directory."""
        target_dir = Path(target_dir).resolve()
        manifest_path = target_dir / 'manifest.json'
        features_path = target_dir / 'features.pt'

        if not manifest_path.exists():
            raise FileNotFoundError('manifest.json not found in cache: {}'.format(target_dir))
        if not features_path.exists():
            raise FileNotFoundError('features.pt not found in cache: {}'.format(target_dir))

        with open(str(manifest_path), 'r') as handle:
            manifest = json.load(handle)

        if manifest.get('format_version') != CACHE_FORMAT_VERSION:
            raise ValueError('unsupported cache format version: {}'.format(manifest.get('format_version')))

        if verify_sha:
            computed_sha = compute_file_sha256(features_path)
            if computed_sha != manifest.get('features_sha256'):
                raise ValueError('features.pt SHA256 mismatch (possible file corruption)')

        payload = torch.load(str(features_path), map_location='cpu')
        entries = payload.get('entries', [])
        metadata = payload.get('metadata', {})

        cache = cls(metadata=metadata)
        for e in entries:
            cache.add_entry(e)

        return cache
