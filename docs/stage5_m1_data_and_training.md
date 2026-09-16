# Stage 5 M1: Paired Feature Data, Offline Cache, and A1 Training Pipeline

## 1. Overview & Architectural Design

Stage 5 M1 transitions the Stage 5 learned depth adaptation effort from the single-observation sanity check established in M0 (`71745e0`) into a production-grade, reproducible **offline feature cache, dataset, and training pipeline**.

The pipeline cleanly decouples expensive 3D simulator rendering from neural adapter optimization:
1. **Collector Seam**: Multi-observation paired feature extractor wrapping the Stage 4 Depth-only runtime.
2. **Offline Feature Cache**: Self-contained PyTorch archives (`features.pt` + `manifest.json`) storing pairs of dense and sparse depth representations `[12, 128, 4, 4]` with SHA-256 integrity verification.
3. **Scene-Level Splitting & Dataset**: PyTorch `Dataset` and `DataLoader` abstractions that enforce strict **Scene-level isolation** (`scene_id`), guaranteeing zero spatial data leakage between training and validation.
4. **A1 Training Pipeline**: Standalone CPU/GPU training script optimizing `ResidualFeatureAdapter` with the frozen M0 composite loss (`SmoothL1 + (1 - cosine)`), outputting contract-compliant format version 1 checkpoints and run manifests.

```
       +------------------------------------+
       |  Habitat Simulation & Sensor Seam  |
       |  (collect_sparse_depth_feature_...) |
       +-----------------+------------------+
                         |
                         v
       +------------------------------------+
       |   Offline Feature Cache Directory  |
       |   - features.pt  (torch.float32)   |
       |   - manifest.json (SHA256 audited) |
       +-----------------+------------------+
                         |
                         v
       +------------------------------------+
       |   Scene-Level Splitting & Dataset  |
       |   - Train Scenes   |  Val Scenes   |
       |   (Zero spatial / scene leakage)   |
       +-----------------+------------------+
                         |
                         v
       +------------------------------------+
       |      A1 Training & Validation      |
       |  (train_sparse_depth_learned_...)  |
       |   - Frozen SmoothL1 + Cosine Loss  |
       |   - Best Validation Checkpointing  |
       +-----------------+------------------+
                         |
                         v
       +------------------------------------+
       |      Standard Format 1 Output      |
       |   - best_adapter.pt                |
       |   - run_manifest.json              |
       +-----------------+------------------+
                         |
                         v
       [ Consumed by stage5-m1-static-eval ]
```

---

## 2. Zero-Leakage & Safety Invariants

To maintain strict scientific validity and avoid data contamination, the M1 pipeline enforces three inviolable constraints:

### 2.1 Strict Exclusion of Benchmark & Held-Out Sets
- **Held-Out Test Set**: Episode IDs `53`, `821`, `1024`, `250`, `25` are reserved exclusively for Stage 5 mechanism evaluation and must NEVER enter the training set, validation set, or offline cache.
- **Val-Unseen 20**: The 20 frozen numeric benchmark IDs (`53 1685 821 1024 1477 1636 675 164 1375 1679 250 2 475 1356 252 1439 1016 1285 1478 25`) are strictly excluded.
- **Canonicalization**: All episode IDs are normalized via `canonicalize_episode_id()` (converting `int(str(x))` -> `str`) so leading zeros (e.g., `'053'`) cannot bypass exclusion rules.

### 2.2 Scene-Level Isolation (No Room Leakage)
In embodied AI, adjacent camera viewpoints within the same building share structural layout, textures, and geometry. Splitting observations uniformly at random causes massive spatial data leakage.
- Splitting in `sparse_depth_feature_dataset.py::create_scene_level_splits` is performed strictly at the **`scene_id` level**.
- `assert_no_leakage(train_scenes, val_scenes, episodes)` mathematically verifies:
  $$\text{train\_scenes} \cap \text{val\_scenes} = \emptyset$$
  and confirms that no held-out or frozen ID exists anywhere in the split datasets.

---

## 3. Storage Specification: `FeatureCache`

The `FeatureCache` format provides tamper-evident, zero-dependency storage for feature pairs.

### Directory Layout
```
<cache_dir>/
├── features.pt      # PyTorch serialized payload containing feature tensors
└── manifest.json    # Human-readable metadata and SHA-256 checksums
```

### Specifications
- **Tensor Shape**: Exactly `[12, 128, 4, 4]` ($N=12$ panoramic views, $C=128$ channels, $H=4, W=4$ spatial grid).
- **Data Type**: `torch.float32`.
- **Integrity**: `manifest.json` stores `features_sha256`. Upon `FeatureCache.load(verify_sha=True)`, the SHA-256 hash of `features.pt` is recomputed and compared.
- **Safety**: `save()` strictly refuses to overwrite existing non-empty directories (`FileExistsError`).

---

## 4. ResidualFeatureAdapter (A1) & Training Recipe

### Architecture
As established in M0:
$$\mathbf{F}_{\text{adapted}} = \mathbf{F}_{\text{sparse}} + \text{Conv1x1}_2\left(\text{GELU}\left(\text{Conv1x1}_1\left(\mathbf{F}_{\text{sparse}}\right)\right)\right)$$
- $\text{Conv1x1}_2$ weights and biases are initialized to **exact zero**.
- Zero-initialization ensures identity transformation at step 0 ($\mathbf{F}_{\text{adapted}} \equiv \mathbf{F}_{\text{sparse}}$).
- Total parameter count: $128 \times 128 \times 1 + 128 + 128 \times 128 \times 1 + 128 = 33,024$.

### Loss Formulation
$$\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{SmoothL1}}(\mathbf{F}_{\text{adapted}}, \mathbf{F}_{\text{dense}}) + 1.0 \cdot \mathcal{L}_{\text{cosine}}(\mathbf{F}_{\text{adapted}}, \mathbf{F}_{\text{dense}})$$
where:
$$\mathcal{L}_{\text{cosine}} = 1 - \frac{1}{B} \sum_{i=1}^B \frac{\mathbf{f}_{\text{adapted}}^{(i)} \cdot \mathbf{f}_{\text{dense}}^{(i)}}{\|\mathbf{f}_{\text{adapted}}^{(i)}\|_2 \|\mathbf{f}_{\text{dense}}^{(i)}\|_2 + \epsilon}$$

### Checkpoint Format Version 1
Checkpoints are saved using `save_adapter_checkpoint()`:
- Format version: `1`
- Architecture metadata: `residual_1x1_gelu_1x1_v1`
- State dict: CPU tensors of `conv1` and `conv2` parameters
- Loss weights: `smooth_l1_weight=1.0, cosine_weight=1.0`

This format is guaranteed to be fully compatible with `load_adapter_checkpoint()` on the parallel `stage5-m1-static-eval` branch.

---

## 5. CLI Usage & Pipeline Workflows

### 5.1 Collecting Feature Pairs (Habitat Host Only)
```bash
python scripts/collect_sparse_depth_feature_pairs.py \
    --output-dir outputs/stage5_m1_feature_cache \
    --one-per-scene \
    --max-episodes 50
```

### 5.2 Training A1 Adapter (CPU or GPU)
```bash
python scripts/train_sparse_depth_learned_adapter.py \
    --cache-dir outputs/stage5_m1_feature_cache \
    --output-dir outputs/stage5_m1_training_run \
    --epochs 50 \
    --lr 0.001 \
    --batch-size 4 \
    --val-ratio 0.2 \
    --seed 42 \
    --device cpu
```

### 5.3 Running Unit Verification (CPU)
```bash
python -m unittest discover -s tests -p "test_stage5_m1_pipeline.py" -v
```
