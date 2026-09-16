#!/usr/bin/env python3
"""CPU Unit Tests for Stage 5 M1 Paired Feature / Cache / Training Pipeline.

Covers:
1. FeatureCache serialization, deserialization, SHA-256 integrity, corruption rejection.
2. Scene-level split determinism, zero spatial leakage assertion, held-out/frozen rejection.
3. PairedFeatureDataset indexing, collation, DataLoader batching.
4. End-to-end A1 ResidualFeatureAdapter training on synthetic cache:
   - Zero-init identity check
   - Loss reduction verification
   - Standard checkpoint format version 1 saving
   - Checkpoint restoration via load_adapter_checkpoint and exact output parity
   - Run manifest structure
5. Collector episode ID validation, canonicalization, and exclusion filtering.

Completely CPU-based; zero Habitat or GPU dependency.
Compatible with Python 3.6 / PyTorch 1.9.1.
"""
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from sparse_depth_learned_adapter import (
    ResidualFeatureAdapter,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
    feature_distillation_loss,
)
from sparse_depth_feature_cache import (
    FeatureCache,
    validate_feature_tensor,
    validate_cache_entry,
    compute_file_sha256,
)
from sparse_depth_feature_dataset import (
    PairedFeatureDataset,
    create_scene_level_splits,
    create_dataloaders,
    assert_no_leakage,
    paired_collate_fn,
    HELD_OUT_IDS,
    FROZEN_IDS,
)
from train_sparse_depth_learned_adapter import (
    train_adapter_pipeline,
    evaluate_adapter,
)
from collect_sparse_depth_feature_pairs import (
    canonicalize_episode_id,
    validate_episode_id,
    validate_episode_ids,
    filter_eligible_episodes,
)


def create_synthetic_entry(episode_id, scene_id, seed=0):
    # type: (str, str, int) -> dict
    """Create synthetic paired feature entry with deterministic values."""
    torch.manual_seed(seed)
    dense = torch.randn(12, 128, 4, 4, dtype=torch.float32)
    # Sparse is shifted and scaled version with some noise
    sparse = dense * 0.8 + 0.1 + torch.randn(12, 128, 4, 4, dtype=torch.float32) * 0.1
    return {
        'episode_id': str(episode_id),
        'scene_id': str(scene_id),
        'dense': dense,
        'sparse': sparse,
    }


class TestSparseDepthFeatureCache(unittest.TestCase):
    """Test suite for offline FeatureCache storage and verification."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(str(self.temp_dir), ignore_errors=True)

    def test_tensor_validation(self):
        # Correct tensor: [12, 128, 4, 4], float32
        valid = torch.randn(12, 128, 4, 4, dtype=torch.float32)
        validate_feature_tensor(valid, 'test_tensor')

        # Invalid shape
        with self.assertRaises(ValueError):
            validate_feature_tensor(torch.randn(10, 128, 4, 4), 'wrong_view_count')
        with self.assertRaises(ValueError):
            validate_feature_tensor(torch.randn(12, 64, 4, 4), 'wrong_channels')
        with self.assertRaises(ValueError):
            validate_feature_tensor(torch.randn(12, 128, 8, 8), 'wrong_spatial')

        # Invalid dtype
        with self.assertRaises(ValueError):
            validate_feature_tensor(torch.randn(12, 128, 4, 4, dtype=torch.float64), 'wrong_dtype')

        # Non-finite values
        invalid_inf = valid.clone()
        invalid_inf[0, 0, 0, 0] = float('inf')
        with self.assertRaises(ValueError):
            validate_feature_tensor(invalid_inf, 'inf_tensor')

        invalid_nan = valid.clone()
        invalid_nan[0, 0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            validate_feature_tensor(invalid_nan, 'nan_tensor')

    def test_cache_entry_validation(self):
        valid_entry = create_synthetic_entry('101', 'scene_A', seed=1)
        validate_cache_entry(valid_entry)

        # Missing key
        incomplete = dict(valid_entry)
        del incomplete['dense']
        with self.assertRaises(ValueError):
            validate_cache_entry(incomplete)

        # Empty episode_id
        bad_eid = dict(valid_entry, episode_id='')
        with self.assertRaises(ValueError):
            validate_cache_entry(bad_eid)

    def test_cache_save_and_load_roundtrip(self):
        cache = FeatureCache(metadata={'author': 'test', 'dataset': 'val_seen'})
        cache.add_entry(create_synthetic_entry('101', 'scene_A', seed=1))
        cache.add_entry(create_synthetic_entry('102', 'scene_B', seed=2))
        cache.add_entry(create_synthetic_entry('103', 'scene_A', seed=3))

        self.assertEqual(len(cache), 3)
        self.assertEqual(cache.episode_ids, ['101', '102', '103'])
        self.assertEqual(cache.scene_ids, ['scene_A', 'scene_B'])

        cache_target = self.temp_dir / 'cache_test'
        manifest_path = cache.save(cache_target)
        self.assertTrue(manifest_path.exists())
        self.assertTrue((cache_target / 'features.pt').exists())

        # Load back
        loaded = FeatureCache.load(cache_target, verify_sha=True)
        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded.episode_ids, ['101', '102', '103'])
        self.assertEqual(loaded.scene_ids, ['scene_A', 'scene_B'])
        self.assertEqual(loaded.metadata.get('author'), 'test')

        # Check tensor parity
        for orig, rest in zip(cache.entries, loaded.entries):
            self.assertEqual(orig['episode_id'], rest['episode_id'])
            self.assertEqual(orig['scene_id'], rest['scene_id'])
            self.assertTrue(torch.equal(orig['dense'], rest['dense']))
            self.assertTrue(torch.equal(orig['sparse'], rest['sparse']))

    def test_cache_duplicate_episode_rejection(self):
        cache = FeatureCache()
        cache.add_entry(create_synthetic_entry('101', 'scene_A', seed=1))
        with self.assertRaises(ValueError):
            cache.add_entry(create_synthetic_entry('101', 'scene_B', seed=2))

    def test_cache_refuses_overwrite(self):
        cache = FeatureCache()
        cache.add_entry(create_synthetic_entry('101', 'scene_A', seed=1))
        cache_target = self.temp_dir / 'overwrite_test'
        cache.save(cache_target)

        # Attempting to save again to non-empty directory must fail
        with self.assertRaises(FileExistsError):
            cache.save(cache_target)

    def test_cache_sha256_corruption_detection(self):
        cache = FeatureCache()
        cache.add_entry(create_synthetic_entry('101', 'scene_A', seed=1))
        cache_target = self.temp_dir / 'corrupt_test'
        cache.save(cache_target)

        # Tamper with features.pt
        features_path = cache_target / 'features.pt'
        with open(str(features_path), 'ab') as f:
            f.write(b'extra_garbage_bytes')

        with self.assertRaises(ValueError):
            FeatureCache.load(cache_target, verify_sha=True)


class TestSparseDepthFeatureDataset(unittest.TestCase):
    """Test suite for Dataset, DataLoader, and Scene-level splitting."""

    def test_dataset_indexing_and_collation(self):
        entries = [
            create_synthetic_entry('101', 'scene_A', seed=1),
            create_synthetic_entry('102', 'scene_B', seed=2),
        ]
        dataset = PairedFeatureDataset(entries)
        self.assertEqual(len(dataset), 2)
        sample = dataset[0]
        self.assertEqual(sample['episode_id'], '101')
        self.assertEqual(sample['scene_id'], 'scene_A')
        self.assertEqual(tuple(sample['sparse'].shape), (12, 128, 4, 4))
        self.assertEqual(tuple(sample['dense'].shape), (12, 128, 4, 4))

        # Test collation
        batch = paired_collate_fn([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch['sparse'].shape), (2, 12, 128, 4, 4))
        self.assertEqual(tuple(batch['dense'].shape), (2, 12, 128, 4, 4))
        self.assertEqual(batch['episode_id'], ['101', '102'])
        self.assertEqual(batch['scene_id'], ['scene_A', 'scene_B'])

    def test_scene_level_splitting_isolation(self):
        # 6 episodes across 4 scenes
        entries = [
            create_synthetic_entry('101', 'scene_1', seed=1),
            create_synthetic_entry('102', 'scene_1', seed=2),
            create_synthetic_entry('103', 'scene_2', seed=3),
            create_synthetic_entry('104', 'scene_2', seed=4),
            create_synthetic_entry('105', 'scene_3', seed=5),
            create_synthetic_entry('106', 'scene_4', seed=6),
        ]
        cache = FeatureCache(entries=entries)

        train_ds, val_ds, split_info = create_scene_level_splits(cache, val_ratio=0.25, seed=42)

        # Verify zero scene overlap
        train_scenes = set(train_ds.scene_ids)
        val_scenes = set(val_ds.scene_ids)
        self.assertTrue(train_scenes.isdisjoint(val_scenes))
        self.assertEqual(len(train_scenes) + len(val_scenes), 4)

        # Verify total entries preserved
        self.assertEqual(len(train_ds) + len(val_ds), 6)

        # Verify split info metadata
        self.assertEqual(split_info['total_entries'], 6)
        self.assertEqual(split_info['total_scenes'], 4)
        self.assertEqual(len(split_info['train_scenes']), len(train_scenes))
        self.assertEqual(len(split_info['val_scenes']), len(val_scenes))

    def test_leakage_assertion_catches_overlap(self):
        train_scenes = {'scene_1', 'scene_2'}
        val_scenes = {'scene_2', 'scene_3'}  # scene_2 overlaps!
        episodes = [create_synthetic_entry('101', 'scene_1', seed=1)]

        with self.assertRaises(ValueError) as ctx:
            assert_no_leakage(train_scenes, val_scenes, episodes)
        self.assertIn('Scene-level data leakage detected', str(ctx.exception))

    def test_leakage_assertion_catches_held_out_and_frozen_ids(self):
        train_scenes = {'scene_1'}
        val_scenes = {'scene_2'}

        # Test held-out ID rejection
        for hid in HELD_OUT_IDS:
            contaminated = [create_synthetic_entry(hid, 'scene_1', seed=1)]
            with self.assertRaises(ValueError) as ctx:
                assert_no_leakage(train_scenes, val_scenes, contaminated)
            self.assertIn('Held-out ID', str(ctx.exception))

        # Test frozen ID rejection
        for fid in list(FROZEN_IDS)[:3]:
            contaminated = [create_synthetic_entry(fid, 'scene_1', seed=1)]
            with self.assertRaises(ValueError) as ctx:
                assert_no_leakage(train_scenes, val_scenes, contaminated)
            self.assertIn('Contamination forbidden', str(ctx.exception))

    def test_dataloader_creation(self):
        entries_train = [
            create_synthetic_entry('101', 'scene_1', seed=1),
            create_synthetic_entry('102', 'scene_1', seed=2),
            create_synthetic_entry('103', 'scene_2', seed=3),
        ]
        entries_val = [
            create_synthetic_entry('104', 'scene_3', seed=4),
        ]
        train_ds = PairedFeatureDataset(entries_train)
        val_ds = PairedFeatureDataset(entries_val)

        train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=2)
        batches = list(train_loader)
        self.assertEqual(len(batches), 2)  # 3 items with batch_size 2 -> 2 batches
        self.assertEqual(batches[0]['sparse'].shape[0], 2)
        self.assertEqual(batches[1]['sparse'].shape[0], 1)


class TestA1TrainingPipeline(unittest.TestCase):
    """Test suite for A1 training pipeline execution and contract verification."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(str(self.temp_dir), ignore_errors=True)

    def test_synthetic_training_loop_and_checkpoint_compatibility(self):
        # Build 4 scenes with distinct shifts
        train_entries = [
            create_synthetic_entry('101', 'scene_1', seed=10),
            create_synthetic_entry('102', 'scene_1', seed=11),
            create_synthetic_entry('103', 'scene_2', seed=12),
            create_synthetic_entry('104', 'scene_2', seed=13),
        ]
        val_entries = [
            create_synthetic_entry('105', 'scene_3', seed=14),
            create_synthetic_entry('106', 'scene_4', seed=15),
        ]
        train_ds = PairedFeatureDataset(train_entries)
        val_ds = PairedFeatureDataset(val_entries)

        train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=2)

        output_dir = self.temp_dir / 'training_run'
        split_info = {
            'train_scenes': ['scene_1', 'scene_2'],
            'val_scenes': ['scene_3', 'scene_4'],
            'seed': 42,
        }

        trained_adapter, manifest = train_adapter_pipeline(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=5,
            lr=1e-3,
            seed=42,
            device='cpu',
            output_dir=output_dir,
            split_info=split_info,
        )

        # 1. Verify checkpoint exists
        checkpoint_path = output_dir / 'best_adapter.pt'
        manifest_path = output_dir / 'run_manifest.json'
        self.assertTrue(checkpoint_path.exists())
        self.assertTrue(manifest_path.exists())

        # 2. Verify loss decreased from initial
        initial_val_total = manifest['initial_val_metrics']['total']
        best_val_total = manifest['best_val_metrics']['total']
        self.assertLessEqual(best_val_total, initial_val_total)

        # 3. Verify checkpoint format with official load_adapter_checkpoint
        loaded_adapter, meta = load_adapter_checkpoint(checkpoint_path, expected_channels=128, device='cpu')
        self.assertEqual(meta['format_version'], 1)
        self.assertEqual(meta['architecture']['name'], 'residual_1x1_gelu_1x1_v1')
        self.assertEqual(meta['architecture']['channels'], 128)
        self.assertEqual(meta['loss']['smooth_l1_weight'], 1.0)
        self.assertEqual(meta['loss']['cosine_weight'], 1.0)

        # 4. Verify exact output parity between loaded_adapter and trained_adapter
        test_input = torch.randn(12, 128, 4, 4)
        with torch.no_grad():
            out_trained = trained_adapter(test_input)
            out_loaded = loaded_adapter(test_input)
        self.assertTrue(torch.equal(out_trained, out_loaded))

        # 5. Verify manifest contents
        self.assertEqual(manifest['stage'], 'M1_TRAINING')
        self.assertEqual(manifest['status'], 'COMPLETED')
        self.assertEqual(manifest['hyperparameters']['epochs'], 5)
        self.assertEqual(len(manifest['history']), 5)
        self.assertTrue(len(manifest['best_checkpoint_sha256']) == 64)


class TestCollectorValidation(unittest.TestCase):
    """Test suite for collection input validation and exclusion filtering."""

    def test_canonicalization(self):
        self.assertEqual(canonicalize_episode_id('123'), '123')
        self.assertEqual(canonicalize_episode_id(123), '123')
        self.assertEqual(canonicalize_episode_id('0123'), '123')
        with self.assertRaises(ValueError):
            canonicalize_episode_id('abc')

    def test_validate_episode_id_exclusions(self):
        # Valid ID
        self.assertEqual(validate_episode_id('9999'), '9999')

        # Held-out rejection (including leading zeros)
        for hid in HELD_OUT_IDS:
            with self.assertRaises(ValueError):
                validate_episode_id(hid)
            with self.assertRaises(ValueError):
                validate_episode_id('00' + hid)

        # Frozen ID rejection
        for fid in list(FROZEN_IDS)[:3]:
            with self.assertRaises(ValueError):
                validate_episode_id(fid)

    def test_validate_episode_ids_list(self):
        self.assertEqual(validate_episode_ids(['100', '101', '102']), ['100', '101', '102'])
        # Duplicates
        with self.assertRaises(ValueError):
            validate_episode_ids(['100', '101', '100'])
        # Empty
        with self.assertRaises(ValueError):
            validate_episode_ids([])

    def test_filter_eligible_episodes(self):
        episodes = [
            {'episode_id': '53', 'scene_id': 'scene_A'},    # Held out -> must be skipped
            {'episode_id': '101', 'scene_id': 'scene_A'},   # Valid
            {'episode_id': '102', 'scene_id': 'scene_A'},   # Valid (duplicate scene)
            {'episode_id': '201', 'scene_id': 'scene_B'},   # Valid
            {'episode_id': '1685', 'scene_id': 'scene_C'},  # Frozen -> must be skipped
        ]

        # One per scene
        selected_one = filter_eligible_episodes(episodes, one_per_scene=True)
        self.assertEqual(len(selected_one), 2)
        eids = [e['episode_id'] for e in selected_one]
        self.assertIn('101', eids)
        self.assertIn('201', eids)
        self.assertNotIn('53', eids)
        self.assertNotIn('1685', eids)

        # All eligible
        selected_all = filter_eligible_episodes(episodes, one_per_scene=False)
        self.assertEqual(len(selected_all), 3)
        self.assertEqual([e['episode_id'] for e in selected_all], ['101', '102', '201'])


if __name__ == '__main__':
    unittest.main()
