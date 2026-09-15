# Stage 5 M1 static evaluation infrastructure

**Infrastructure only. Formal held-out evaluation is NOT YET AUTHORIZED.**
Do not run it until the independent data/training branch freezes the
multi-observation A1 training recipe and checkpoint. CPU fixtures establish
plumbing correctness; they provide no held-out or navigation evidence.

## Scope and conditions

The runner accepts one explicitly supplied adapter checkpoint, loaded with the
existing strict M0 loader. Architecture, zero initialization, feature loss,
checkpoint format, frozen encoders/predictor, and Original NMS are unchanged.
The loader verifies reconstruction metadata and finite state, and M1 additionally
requires float32, 128 channels, and the contractual 1.0/1.0 loss weights. Metadata
alone cannot establish that training used the independently frozen recipe;
that remains a prerequisite for the later formal host run. An M0 checkpoint may
be used for NONFORMAL plumbing only.

| Condition | Same reset observation and model path |
| --- | --- |
| DENSE | Dense depth and RGB → frozen Original encoders → frozen Original predictor |
| SPARSE | Same observation → deterministic accepted MID360 depth transform → same frozen encoders → same predictor |
| A1_ADAPTED | Exact same SPARSE encoding object → frozen A1 on `depth_embedding` → same predictor |

There are two encodings and three predictor/NMS calls per observation. All
modules are in eval mode with gradients disabled and parameters frozen.
`encoded.depth_features`, `pano_depth`, `cand_depth`, and RGB remain raw.
Assertions/fingerprints detect source, encoding, model, or adapter mutation.
The policy is constructed only to reuse the accepted model/environment setup;
GraphMap and planner are not run. All observation acquisition uses reset.

Stage 3 provides condition construction, input statistics, tensor/state hashes,
logit statistics, similarities, and feature diversity. Stage 4 provides unchanged
runtime/reset helpers, activity statistics, and `predict_encoded`, which calls
Original candidate-map/NMS and output construction. Stage 5's local
`capture_encoded` duplicates only the bounded capture-hook orchestration needed
to reuse one sparse encoding and retain its candidate map. It preserves Stage 4's
distinct ReLU sites and immediate cloning before the in-place upstream ReLU.
CPU tests compare every historical capture tensor and metric exactly against
Stage 4 for DENSE, SPARSE, and a nonidentity adapted condition.

The unchanged legacy runtime constructs an AdamW optimizer while loading models.
This is allowed construction only: M1 never calls optimizer methods or performs
backward/training. Reports record `legacy_optimizer_constructed`,
`training_steps=0`, and `navigation_steps=0`. Stage 5 checks the prepared runtime
config and rejects training requeue before loading, then freezes all evaluated
modules. CPU lifecycle tests construct a real CPU optimizer and fail on any
subsequent optimizer API or backward/gradient call, covering success and failure.

Stage 4 source, the core trainer, their default behavior, and historical provenance
locks remain exactly at HEAD. No persistent global monkeypatch is used by M1.
ALL_FAR is constructed by the shared helper but is not encoded or evaluated by M1.

## Metrics and evidence

For every observation, JSON records:

- Feature-domain SPARSE vs DENSE and A1_ADAPTED vs DENSE: mean SmoothL1
  (`beta=1`) and mean per-view cosine over flattened C,H,W (`eps=1e-8`).
  Deltas are **A1 minus SPARSE**: lower SmoothL1 and higher cosine are improvements.
- `visual_fc_depth[1]` Linear / `[2]` ReLU: pre/post positive count/fraction,
  min, max, mean, population std, and existing Stage 4 statistics. This is
  upstream diagnostic evidence.
- **Primary mechanism:** `vis_classifier[0]` Linear pre-ReLU statistics and
  `vis_classifier[1]` post-ReLU positive count/fraction. Sparse collapse remains
  defined at this classifier ReLU. Both sites retain full pre/post diagnostics.
- Logits: entropy in nats, min/max/mean/population std, radial/angular marginals,
  peaks, ties, periodicity, and existing Stage 3 post-NMS cells/radial counts.
  Pairwise comparisons report cosine, normalized L2, maximum absolute difference,
  and exact equality for logits and intermediate features; classifier sign
  crossings record direction explicitly in the pair name.
- Candidates: count, Original counterclockwise radians, normalized counterclockwise
  degrees, distances in meters, radial indexes, distinct bins/distances, mean and
  median distance (null if empty). Candidate ordering remains Original ordering.

`report.md` contains a per-observation table separating feature, upstream,
classifier, logit, and candidate evidence. `report.json` contains the full
metrics and provenance; `capture_<id>.npz` contains feature/activity/logit tensors
and Original candidate maps for audit. No full depth images are serialized.

Formal reports contain `stage="M1_STATIC"`, `formal_evaluation=true`,
`training_steps=0`, `navigation_steps=0`, and exactly the IDs
**53, 821, 1024, 250, 25**. CLI subsets, supersets, duplicates, replacements, or
reordering are rejected; leading zeros are canonicalized. Each ID must occur
exactly once in the accepted hash-checked val_unseen dataset. Reset IDs/scenes are
checked and poses/raw depth hashes recorded. The shared reset helper also saves
sensor diagnostics.

The output directory must be fresh, with no overwrite or resume. Provenance
includes the A1 checkpoint SHA256/metadata, adapter state before/after, git
commit/branch/dirty status, accepted ETPNav checkpoint hashes, DDPPO asset hash,
dataset/config/source hashes, rendered scene hashes, runtime config, versions,
seed/determinism settings, and full Original/policy fingerprints before/after.
Failures after manifest creation retain a FAILED report and partial artifacts.
COMPLETE means collection completed; it is not a scientific pass decision.

NONFORMAL smoke requires one explicitly selected, uniquely present val_seen ID,
excluding every frozen-20 numeric ID. Episode 137 is the existing eligible smoke
observation; membership and dataset hash are rechecked at execution. Its output
basename must start `nonformal_`, its reports are `NONFORMAL_report.json/.md`,
its stage is `M1_STATIC_NONFORMAL_SMOKE`, and its statuses start `NONFORMAL_`.
It never claims held-out evidence.

## Leakage protections and interpretation

Optimizer construction occurs only in the unchanged legacy loader. There is no
optimizer use afterward, backward call, training step, navigation step, checkpoint
search, threshold search, or automatic next stage in the evaluation path. The
runner accepts one checkpoint and never tunes it. Formal IDs cannot be used in
the smoke mode. Full source and state provenance supports later independent
review; this runner cannot verify the training branch's data provenance by itself.

Feature rescue != classifier rescue != candidate-diversity rescue.

- **CASE A:** features improve, classifier activity recovers, and meaningful
  candidate diversity recovers → candidate for later frozen-20 navigation.
- **CASE B:** features improve while classifier remains collapsed → consider
  A2 predictor-aware distillation.
- **CASE C:** features fail to generalize → investigate data, capacity, and
  placement before navigation.

Classifier activity without meaningful diversity is only partial rescue. No
numerical pass threshold or automatic case assignment is introduced.

## Host command A: optional NONFORMAL plumbing smoke

Use a trusted compatible A1 checkpoint. This command is provided for later host
use and has not been executed by Codex. Choose a fresh output directory.

```bash
cd ~/SURF/etpnav-mid360
source scripts/renderer_bootstrap.bash
etp_apply_known_good_renderer_environment
/home/evens/miniforge3/envs/etpnav_legacy/bin/python \
  scripts/run_sparse_depth_learned_adapter_static.py \
  --mode nonformal-smoke --episode-ids 137 \
  --adapter-checkpoint <A1_SMOKE_CHECKPOINT> \
  --output outputs/nonformal_stage5_m1_ep137
```

## Host command B: formal template — NOT YET AUTHORIZED

**DO NOT RUN UNTIL TRAINING RECIPE AND CHECKPOINT ARE FROZEN.**
Supply the single frozen multi-observation A1 artifact from the training branch.
The fixed five IDs are selected by the runner; no episode-ID override is needed.

```bash
cd ~/SURF/etpnav-mid360
source scripts/renderer_bootstrap.bash
etp_apply_known_good_renderer_environment
/home/evens/miniforge3/envs/etpnav_legacy/bin/python \
  scripts/run_sparse_depth_learned_adapter_static.py \
  --mode formal \
  --adapter-checkpoint <TRAINED_A1_CHECKPOINT> \
  --output outputs/stage5_m1_static_held_out_five
```

Replace angle-bracket placeholders before host use. The current interpreter path
resolves to Python 3.6.15 / PyTorch 1.9.1+cu111, which the code supports.

## CPU/static validation

```bash
CUDA_VISIBLE_DEVICES='' /home/evens/miniforge3/envs/etpnav_legacy/bin/python \
  -m unittest discover -s tests -p 'test_sparse_depth_learned_adapter_static.py' -v
CUDA_VISIBLE_DEVICES='' /home/evens/miniforge3/envs/etpnav_legacy/bin/python \
  -m unittest discover -s tests -p 'test_sparse_depth_learned_adapter.py' -v
CUDA_VISIBLE_DEVICES='' /home/evens/miniforge3/envs/etpnav_legacy/bin/python \
  -m unittest discover -s tests -p 'test_sparse_depth_feature_alignment.py' -v
```

Tests use synthetic tensors, the Stage 3 mock encoders, a predictor with distinct
upstream/classifier ReLUs, and unchanged Original NMS. A coarse angular-cell
configuration makes the tiny 8×8 fixture sparse; the host configuration is never
changed. Synthetic tests cover identity and nonidentity routing, feature-only
improvement with a still-collapsed classifier, boundary activity recovery,
correct feature/candidate metrics, fingerprints, metadata/ID/dataset guards,
NONFORMAL success/failure reporting, and absence of training/navigation calls in
the static entry point and local capture/shared reset/predictor seams. These are CPU
plumbing checks, not evidence of real learned recovery.

Refactor validation on the requested legacy interpreter: M1 static 15/15
passed; learned adapter M0 8/8 passed; Stage 4 alignment 7 passed with 1 unavailable
static-evidence test skipped; `git diff --check` passed. No real observation, CUDA/Habitat runtime, navigation,
or formal held-out evaluation was executed.
