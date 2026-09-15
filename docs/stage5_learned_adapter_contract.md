# Stage 5 learned sparse-to-dense feature adapter: M0 contract

## Question and scope

Can a small learned nonlinear adapter map sparse depth-encoder features toward
the dense feature domain while keeping the Original ETP waypoint predictor frozen?

The Original dense development-20 baseline achieved SR=0.80, SPL=0.698.
MID360-like sparse-depth zero-shot achieved SR=0.00 and collapsed to a roughly
0.75 m waypoint pattern. Forensics located complete collapse at the predictor's
`vis_classifier` ReLU (0/9216 active), while `visual_fc_depth` remained about
23–24% active. Bias-only logits then produced the fixed pattern. Stage 4 fixed
channel mean/std alignment failed to recover classifier activity or waypoint
diversity on all five held-out observations.

M0 implements plumbing and an optimization sanity check on one feature pair.
It does not establish classifier recovery, waypoint diversity, generalization,
or navigation improvement. No formal Stage 5 experiment is authorized here.

## Frozen variants

| Variant | Definition |
| --- | --- |
| A0 | Identity: no learned change (`alignment=None`). |
| A1 | `x + Conv2d(128,128,1)(GELU(Conv2d(128,128,1)(x)))`. |

Both convolutions have biases. The second convolution's weight and bias are
initialized to exactly zero. There are **33,024 trainable parameters**. Input
and output have shape `[N,128,4,4]`; the real pair has `N=12` ordered panorama
views of one observation. The twelve views are not twelve independent episodes.

Fresh A1 is exactly identity for representative finite float32 inputs, subject
to representable intermediate arithmetic. Nonfinite inputs/outputs are rejected.
Shape, dtype, device, and the input values are preserved. As with ordinary
PyTorch modules, callers use `.to(device=..., dtype=...)` to match the input;
the adapter refuses mismatches. Normal `train()`/`eval()` semantics apply.

There is no normalization, dropout, spatial resizing, attention, predictor-aware
loss, classifier/logit loss, GraphMap loss, or navigation loss.

## Feature loss

For adapted features A and fixed dense features D, use equal weights:

```text
smooth_l1 = mean over N,C,H,W of SmoothL1(A - D, beta=1)
cosine_n = torch.nn.functional.cosine_similarity(
    A.flatten(1), D.flatten(1), dim=1, eps=1e-8)
cosine_loss = mean over N of (1 - cosine_n)
L = 1.0 * smooth_l1 + 1.0 * cosine_loss
```

Each cosine compares one view flattened across C,H,W. PyTorch's epsilon handles
zero norms; a zero vector gives cosine zero. Report mean cosine similarity as
well as both loss terms and total. The loss helper preserves autograd. The M0
optimizer owns detached feature copies, so no gradient reaches ETPNav or the
collected input/target tensors.

## Isolation and placement

The only prospective insertion point is:

```text
sparse depth_embedding -> A1 -> frozen Original waypoint predictor
```

Reuse `predict_encoded(original, encoded, alignment=adapter)` from Stage 4 for
the identity regression. It retains the Original candidate map/NMS and candidate
construction. `encoded.depth_features` stays raw for panorama/GraphMap consumers,
including after the adapter becomes nonidentity. M0 registers no provider in the
evaluator or trainer and makes no change to `ss_trainer_ETP.py`.

Frozen-ID definitions are imported from Stage 4 and never changed. All frozen
val_unseen-20 numeric IDs are excluded even if they occur in val_seen. Held-out
IDs **53, 821, 1024, 250, 25** are never training/calibration inputs. Numeric IDs
are canonicalized before checking, including leading-zero spellings. The runner
requires an explicit ID present exactly once in the accepted, hash-checked
val_seen dataset; it offers no alternate split or automatic sample selection.

## Manual host runner

Run only on the normal WSL host in Python 3.10 / PyTorch 1.9.1 `etpnav_legacy`.
The renderer uses the accepted shared bootstrap. Episode 137 is an eligible
val_seen observation used by the existing Stage 4 depth-only smoke; the runner
verifies membership and exclusion rules again before collection.

```bash
cd ~/SURF/etpnav-mid360
source scripts/renderer_bootstrap.bash
etp_apply_known_good_renderer_environment
/home/evens/miniforge3/envs/etpnav_legacy/bin/python \
  scripts/run_sparse_depth_learned_adapter_m0.py \
  --episode-id 137 \
  --output outputs/stage5_learned_adapter_m0_ep137 \
  --steps 100 --learning-rate 0.001 --seed 0 --save-features
```

The output directory must be fresh. Reruns require a new path. The runner:

1. Checks the val_seen dataset and accepted ETPNav checkpoint hashes, records
   source/config hashes, command, branch/HEAD, dirty status, and renderer preload.
2. Reuses Stage 4 `runtime(..., calibration_only=True)` and `reset_batch` for
   exactly one reset with twelve Depth sensors and no RGB rendering.
3. Freezes the policy, encoders, and predictor, and fingerprints their states.
4. Closes Habitat after reset, before encoding, matching the accepted Stage 4
   one-observation smoke. Calls `calibration_features(provider, batch)` exactly
   once for the same-observation dense/sparse pair. Checks finite float32
   `[12,128,4,4]` tensors and unchanged encoder/predictor/policy fingerprints.
5. Keeps detached CPU feature copies and releases the collection scope. Creates
   a fresh A1, verifies max absolute identity error is exactly zero, and runs
   Adam on **adapter parameters only**, on CPU. Default: 100 steps, lr=0.001,
   seed=0. The allowed range is 1–1000 steps; weights are fixed to 1.0/1.0.
6. Checks every adapter gradient for finiteness and verifies the fixed pair did
   not change. Reports initial/final total, SmoothL1, cosine loss, mean cosine,
   identity error, shapes, parameter count, and optimization settings.
7. Saves adapter-only `adapter.pt`, reloads into a fresh adapter, and requires
   exactly equal CPU outputs. With `--save-features`, saves the small pair in
   `features.pt`. No ETPNav model is serialized.

`report.json` is both report and manifest. It records `stage="M0"`,
`navigation_steps=0`, `formal_evaluation=false`, sensor/observation provenance,
model and feature hashes, and checkpoint reconstruction metadata. The runtime
also saves `calibration_runtime_config.yaml` and `sensors.json`. Failures after
manifest creation leave a `FAILED` report and preserve artifacts for inspection.
Failure to reduce loss is reported as a failed smoke, without automatic retries
or changes of episode. This is an overfit check, not statistical evidence.

No `env.step`, navigation, GraphMap, planner rollout, frozen-20 run, or formal
held-out-five evaluation occurs. The runtime constructs the existing trainer
solely to reuse its accepted model/environment setup.

## Checkpoint contract

Format version 1 stores the architecture identifier
`residual_1x1_gelu_1x1_v1`, channel count, kernel/activation/bias/residual/spatial
metadata, dtype, loss weights, beta, cosine epsilon/reduction, adapter state,
and optional seed/step. Loading validates architecture and loss metadata and
strictly loads finite adapter tensors. Optimizer state and ETPNav models are
absent; this is an inference/reconstruction checkpoint, not an exact optimizer
resume contract. Legacy `torch.load` uses pickle: load trusted local artifacts.

## CPU validation

```bash
CUDA_VISIBLE_DEVICES='' /home/evens/miniforge3/envs/etpnav_legacy/bin/python \
  -m unittest discover -s tests -p 'test_sparse_depth_learned_adapter.py' -v
CUDA_VISIBLE_DEVICES='' /home/evens/miniforge3/envs/etpnav_legacy/bin/python \
  -m unittest discover -s tests -p 'test_sparse_depth_feature_alignment.py' -v
```

New tests cover exact initialization identity, input immutability, shape/dtype/
device, loss definition, isolated finite gradients, synthetic overfit, checkpoint
equality/metadata rejection, frozen-ID guards, and mocked collection lifecycle
and state-mutation detection. The existing Stage 4/Original CPU fixture verifies
exact logits, candidate maps, angles and distances, unchanged model fingerprints,
and raw panorama/candidate depth preservation even for a nonidentity adapter.
These fixture tests exercise the Original seam/NMS, not a real checkpoint or
renderer. Host CUDA/Habitat and real-pair validation remain manual.

Implementation validation used the requested interpreter path, which currently
resolves to Python 3.6.15 with PyTorch 1.9.1+cu111 in this workspace. The new
code also passes Python 3.10 syntax parsing. All eight new CPU tests passed;
the Stage 4 suite passed seven tests and skipped one unavailable static-evidence
test. Synthetic 100-step loss fell from 0.231242 to 0.035898, with final mean
cosine 0.991794. Initial identity error was exactly zero and checkpoint output
equality was exact. These results do not substitute for the manual real-pair run.
