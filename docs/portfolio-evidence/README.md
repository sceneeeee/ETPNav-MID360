# ETPNav-MID360 portfolio evidence pack

**Generated:** 2026-11-16. **Purpose:** publication-safe, traceable figures for the ETPNav project case study.

These figures are post-processing of previously recorded experimental values and reports. This pack did **not** launch Habitat, load a checkpoint, run navigation, train a model, or create new experimental measurements.

## Figures

| Asset | What it shows | Evidence boundary |
|---|---|---|
| [first-decision-proposals.svg](assets/first-decision-proposals.svg) | Original dense-depth and MID360-like sparse-depth waypoint proposals from the same Habitat episode-53 observation | This is a proposal-output comparison, **not** a raw dense/sparse depth-image pair. All five sparse candidates are at 0.75 m. Use this as the Portfolio replacement for the unavailable matched sensor screenshot only if the caption preserves this distinction. |
| [graphmap-association-audit.svg](assets/graphmap-association-audit.svg) | G-Min proposal outcomes after GraphMap association, including radial-bin breakdown | Aggregate counts: 595 raw = 411 new ghosts + 25 ghost merges + 159 visited-node absorptions. Not a per-candidate screenshot. |
| [predictor-activation-collapse.svg](assets/predictor-activation-collapse.svg) | Five-observation static comparison of encoder and classifier post-ReLU activity | Dense classifier activity 5/5; sparse and sparse-aligned classifier activity 0/5. No navigation or GraphMap rollout. |
| [frozen20-development-results.svg](assets/frozen20-development-results.svg) | Frozen-20 method comparison | Controlled 20-episode val_unseen development panel; not official R2R-CE benchmark or paper reproduction. |

## Provenance

1. **Episode-53 first decision:** `outputs/mid360_sparse_depth_smoke53_attempt3/first_decision_comparison.json`; the same candidate list and sensor statistics are preserved in [the zero-shot report](../mid360_sparse_depth_zero_shot.md). The dense and sparse conditions use the same observation. Sparse depth is a deterministic MID360-like approximation, not native Livox scan simulation.
2. **GraphMap audit:** `outputs/gmin_graph_survival_probe/report.md` and `outputs/geometry_gmin_frozen20_valid_renderer_attempt2/analysis/report.md`; frozen source snapshot branch `isaac-integration`, HEAD `e7d346217b020cec09a6460ea32ff219543f800b`. `0.25 m` candidates were absorbed 119/119; across the run, 595 proposals became 411 new ghosts, 25 ghost merges and 159 visited absorptions.
3. **Activation audit:** `outputs/alignment_depth_only_static5_20260911T183237_i6ZU54/evaluation/static_report.md`; observations 53, 821, 1024, 250 and 25. The five-observation gate is descriptive, not a statistical significance test. It involved no navigation, training, GraphMap rollout or parameter updates.
4. **Frozen-20 metrics:** [README §4](../../README.md) and [project closeout](../project-closeout.md). Values in `frozen20_results.csv` are rounded to three decimals to match the documented comparison table.

The original `outputs/` run directories were local/ignored and are not present in this public repository. The checked-in CSVs preserve the values used by these figures; they do not replace the original raw NPZ/JSON/telemetry artifacts.

## Missing raw observation image

The Portfolio asset request originally called for a matched dense-depth / sparse-depth image pair from the same viewpoint. The currently available source presentation contains a Habitat dense-depth screenshot and a separate Livox scan visualization, but they are **not** a matched pair. They are intentionally not combined or relabelled as such. The episode-53 proposal plot is supplied as a truthful alternative. A raw matched image can only be produced if the original observation NPZ/PNG artifacts are recovered or a new authorized capture is run.

## Regeneration

The SVGs are vector assets and are accompanied by the exact CSV data used for each plot. `evidence-summary.json` contains the key counts and validation checks. Do not infer new benchmark results from these figures.
