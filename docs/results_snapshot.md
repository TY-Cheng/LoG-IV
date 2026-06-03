# Results And Discussion Snapshot

This document is the live **Results and Discussion** companion to
`docs/paper_plan.md`. The paper plan states the intended research design; this
snapshot records what the current repository actually supports, what each
experiment has produced, and which conclusions are still premature.

Current evidence state: **useful preliminary and candidate-selection evidence,
not final manuscript evidence**.

## 1. Results Overview

### 1.1 Main Takeaway

The strongest recorded result is the A1 stratified benchmark from 2026-05-04.
It reports that graph variants substantially outperform train-only and
within-surface interpolation baselines on masked IV reconstruction. However, the
local artifact directory for that benchmark is currently missing, so the result
must be restored or rerun before it can support final manuscript claims.

The harder-mask evidence from 2026-05-07 is useful for candidate selection but
not yet final evidence. It suggests:

- `lagos_liq_feature_only` is the strongest short-run masked-IV candidate under
  `liquidity_correlated` and `block_wing`;
- `gnn_decoded_calendar_convexity` is the strongest diagnostics candidate,
  especially on decoded-price error and sampled no-arbitrage violations;
- `lagos_loss_only` should not be promoted despite winning a superseded
  stratified two-epoch screen.

### 1.2 Evidence Classes

| Evidence class | Artifact status | How to use it |
| --- | --- | --- |
| A1 stratified, 20 epochs, 3 seeds | Recorded in this document; local `reports/runs/a1-str/` missing | Treat as preliminary evidence that must be restored or rerun. |
| Broad two-epoch model screen | Archived under `reports/archive/20260511-legacy/` | Use only as provenance for candidate selection. |
| Top4 harder-mask screen | Archived under `reports/archive/20260511-legacy/` | Use as candidate-selection evidence. |
| Promoted 20-epoch hard-mask runs | `reports/runs/a1-e20-lc/` incomplete; `block_wing` pending | Do not cite as accepted evidence. |
| Full SVI accounting | Pending | Required for manuscript-level classical baseline comparison. |
| OOD/ticker-holdout analysis | Pending | Required for robustness claims. |
| Figures and final tables | Pending | Required before manuscript drafting. |

## 2. Data Results

### 2.1 Dataset Acceptance Gates

| Dataset | Role | Live status | Gate interpretation |
| --- | --- | --- | --- |
| U.S. expanded options silver | Main in-domain masked-reconstruction benchmark | `data_v1`; 2,979,716 rows; 40 underlyings; 62 observation dates; 2,867,637 IV-usable rows; 2,480 usable surfaces | Passes A1/data_v1. Not enough for broad market-cycle claims. |
| Japan expanded options silver | Out-of-distribution evaluation material | `data_v0`; 557,982 rows; 249 underlyings; 32 observation dates; 557,982 IV-usable rows; 7,488 usable surfaces | Passes the current OOD date gate, but needs matched baselines and normalization checks before claims. |
| Synthetic-LoG-IV | Reproducibility and controlled diagnostics | Generator implemented | Useful for sanity checks, not a substitute for real U.S. evidence. |

### 2.2 Data Discussion

The U.S. table is sufficient for Protocol A development and first multi-seed
benchmarking because it passes the 2,400-surface and 60-date gates. It is not a
final paper universe because it covers only 2026-02-02 through 2026-04-30 and
only 40 liquid U.S. underlyings.

The Japan table is now larger than the older docs stated: the live manifest
shows 249 underlyings and 7,488 usable surfaces. It should still be framed as
out-of-distribution evidence rather than the main empirical claim, because
timestamp semantics, market-specific normalization, and matched baselines still
need explicit treatment.

## 3. Experiment 1: A1 Stratified Benchmark

### 3.1 Protocol

Recorded run family:

```text
reports/runs/a1-str/
```

Current artifact caveat: this directory is **not present locally** as of the
latest repository check. Treat the table below as a recorded result that must be
restored or rerun before final claims.

Protocol:

- task: `masked_reconstruction`;
- split: temporal;
- mask regime: `stratified`;
- seeds: 1, 2, 3;
- epochs: 20;
- max nodes per surface: 250;
- baseline preset: `fast`;
- training surfaces: 1,680;
- validation surfaces: 400;
- test surfaces: 400;
- train / validation / test rows: 419,625 / 99,824 / 99,711;
- validation / test masked rows: 19,964 / 19,942;
- train ends on 2026-04-01;
- validation covers 2026-04-02 through 2026-04-16;
- test starts on 2026-04-17.

### 3.2 Main Result Table

Headline table recorded from `reports/runs/a1-str/benchmark_summary.csv`:

| Variant | Masked IV MAE | p90 abs error | Price MAE | vs train kNN | vs within kNN | Calendar viol. | Convexity viol. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `encoder_mlp` | 0.0925 +/- 0.0011 | 0.1910 | 0.0801 | +7.9% better | worse | 2,553 | 8,143 |
| `gnn_no_liq` | 0.0343 +/- 0.0007 | 0.0650 | 0.0807 | +65.9% better | +49.5% better | 2,952 | 8,909 |
| `gnn_liq` | 0.0345 +/- 0.0007 | 0.0624 | 0.0782 | +65.6% better | +49.2% better | 3,144 | 8,954 |
| `gnn_decoded_calendar_convexity` | **0.0338 +/- 0.0010** | **0.0619** | **0.0759** | **+66.3% better** | **+50.2% better** | **2,344** | **7,751** |

Reference baselines:

- train-only moneyness-tenor kNN masked IV MAE: 0.1004;
- within-surface kNN masked IV MAE: 0.0679;
- within-surface RBF masked IV MAE: 0.0721;
- within-surface local linear masked IV MAE: 0.1332.

### 3.3 Interpretation

This is the first recorded multi-seed result in which graph variants outperform
both train-only kNN and within-surface interpolation baselines. The decoded
calendar/convexity variant is best on masked IV MAE, p90 masked error,
normalized price error, and sampled no-arbitrage diagnostic counts.

The result supports the core graph-representation claim under stratified
masking, subject to restoring or rerunning the missing local artifact. It does
not yet support a broad liquidity-aware claim: `gnn_liq` improves normalized
price error relative to `gnn_no_liq`, but does not improve masked IV MAE.
Heteroscedastic and reliability-gated ablations are still needed before
attributing performance to liquidity modeling.

### 3.4 Discussion And Limitations

Convexity violations remain frequent even for the best A1 model. The current
result supports masked reconstruction quality, not strict no-arbitrage surface
construction. Raw SVI and constrained SVI/SSVI accounting are also pending, so
classical surface-fitting comparisons are incomplete.

## 4. Experiment 2: Broad Two-Epoch Model Screen

### 4.1 Protocol And Artifact Status

This engineering screen is retained only as candidate-selection provenance. It
does **not** supersede the A1 stratified benchmark and it is superseded for
promotion decisions by the harder-mask screen in Section 5.

Archived run families:

- `reports/archive/20260511-legacy/p-all-e2-str/`;
- `reports/archive/20260511-legacy/p-llo-e2-str/`;
- `reports/archive/20260511-legacy/p-la-e2-str/`;
- `reports/archive/20260511-legacy/p-lhf-e2-str/`;
- `reports/archive/20260511-legacy/p-lre-e2-str/`;
- `reports/archive/20260511-legacy/p-lse-e2-str/`.

### 4.2 Outcome Summary

Key outcomes:

- `lagos_loss_only` won the stratified two-epoch screen, but the harder-mask
  screen later showed that it should not be promoted.
- `lagos_liq_feature_only` and `gnn_decoded_calendar_convexity` were close
  enough under stratified masking to justify harder-mask follow-up.
- `lagos_random_edges` and `lagos_shuffled_edges` were weak negative controls,
  which supports keeping graph-topology ablations in the manuscript plan.
- Related-work proxy variants still need separate screening if they are to
  appear in the final comparison: `anchor_deep_smoothing_proxy`,
  `anchor_operator_deep_smoothing_proxy`, `anchor_hexagon_proxy`,
  `anchor_hyperiv_proxy`, `anchor_volnp_proxy`, and `cnp_baseline`.

### 4.3 Discussion

The broad screen is valuable as a search log but should not be cited as
benchmark evidence. It uses one seed and two epochs, so it is too short to rank
final models. Its main use is to identify candidates and failure modes for
longer experiments.

## 5. Experiment 3: Harder Missingness Top4 Screen

### 5.1 Protocol And Artifact Status

Archived run families:

- `reports/archive/20260511-legacy/p-top4-e2-lc/`;
- `reports/archive/20260511-legacy/p-top4-e2-bw/`.

Protocol:

- task: `masked_reconstruction`;
- split: temporal;
- mask regimes: `liquidity_correlated`, `block_wing`;
- seed: 1;
- epochs: 2;
- variants: `lagos_loss_only`, `lagos_liq_feature_only`, `gnn_liq`,
  `gnn_decoded_calendar_convexity`;
- baseline preset: `fast`;
- no-arbitrage diagnostics: sampled surface, 10 validation and 10 test
  surfaces;
- Japan OOD prediction: skipped with `skip_ood=true`.

### 5.2 Main Result Table

| Mask | Variant | Masked IV MAE | p90 abs error | Price MAE | vs train kNN | vs within kNN | Calendar viol. | Convexity viol. | Vertical viol. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `liquidity_correlated` | `lagos_liq_feature_only` | 0.0815 | 0.1463 | 0.0209 | +25.0% | +26.2% | 508 | 1,194 | 416 |
| `liquidity_correlated` | `gnn_decoded_calendar_convexity` | 0.1047 | 0.2092 | 0.0127 | +3.6% | +5.1% | 251 | 654 | 199 |
| `liquidity_correlated` | `gnn_liq` | 0.1146 | 0.2309 | 0.0226 | -5.5% | -3.8% | 564 | 1,152 | 517 |
| `liquidity_correlated` | `lagos_loss_only` | 0.1215 | 0.2326 | 0.0210 | -11.9% | -10.1% | 668 | 1,122 | 435 |
| `block_wing` | `lagos_liq_feature_only` | 0.0993 | 0.1650 | 0.0246 | +18.0% | +49.0% | 480 | 1,148 | 337 |
| `block_wing` | `gnn_decoded_calendar_convexity` | 0.1058 | 0.1798 | 0.0188 | +12.7% | +45.7% | 307 | 718 | 115 |
| `block_wing` | `gnn_liq` | 0.1077 | 0.1700 | 0.0248 | +11.1% | +44.7% | 566 | 1,147 | 452 |
| `block_wing` | `lagos_loss_only` | 0.1401 | 0.2270 | 0.0423 | -15.7% | +28.1% | 580 | 1,361 | 631 |

### 5.3 Interpretation

`lagos_liq_feature_only` is the strongest short-run masked-IV candidate across
the two harder masks. It is best on masked IV MAE for both
`liquidity_correlated` and `block_wing`.

`gnn_decoded_calendar_convexity` is the strongest diagnostics candidate. It has
lower price MAE and fewer sampled calendar, convexity, and vertical-spread
violations than `lagos_liq_feature_only` under both harder masks.

`lagos_loss_only` should not be promoted from the current evidence. It wins the
superseded stratified two-epoch screen, but it fails both train kNN and
within-surface kNN under `liquidity_correlated`, and fails train kNN under
`block_wing`.

### 5.4 Discussion

This experiment sharpens the next modeling decision. It suggests a trade-off
between masked IV error and surface-quality diagnostics:

- choose `lagos_liq_feature_only` as the primary error candidate;
- keep `gnn_decoded_calendar_convexity` as the diagnostics candidate;
- do not promote `lagos_loss_only` without additional evidence.

The screen is still two epochs and one seed. It is evidence for what to run
next, not evidence for a final model ranking.

## 6. Experiment 4: Promoted Hard-Mask Confirmation

### 6.1 Current Artifact Status

Current non-archived run family:

```text
reports/runs/a1-e20-lc/
```

Current visible artifact:

```text
reports/runs/a1-e20-lc/b-us-gdcc-mr-t-e20-s1/best_model.pt
```

No top-level `benchmark_summary.csv` exists. This is an incomplete run and must
not be cited as accepted evidence.

### 6.2 Required Confirmation Runs

The intended promoted top2 confirmation commands are:

```bash
just benchmark-a1 mask=liquidity_correlated seeds=1 epochs=20 variant_suite=anchor_proxy variants=lagos_liq_feature_only,gnn_decoded_calendar_convexity out=reports/runs/a1-e20 baseline_preset=fast no_arb_surfaces=50 skip_ood=true
just benchmark-a1 mask=block_wing seeds=1 epochs=20 variant_suite=anchor_proxy variants=lagos_liq_feature_only,gnn_decoded_calendar_convexity out=reports/runs/a1-e20 baseline_preset=fast no_arb_surfaces=50 skip_ood=true
```

### 6.3 Discussion

The current incomplete `liquidity_correlated` run indicates that promoted
confirmation was started but not completed. The cleanest path is to delete or
archive the partial directory, rerun `liquidity_correlated`, then run
`block_wing`. The promotion gate is that `lagos_liq_feature_only` should remain
ahead of train-only and within-surface baselines on harder masks, while
`gnn_decoded_calendar_convexity` should remain competitive when judged jointly
on error, price diagnostics, and sampled no-arbitrage diagnostics.

## 7. Experiment 5: Full Baseline And SVI Accounting

### 7.1 Current Status

Raw SVI per-expiry and constrained/projected SVI/SSVI accounting are pending.
Current A1 and harder-mask screens use `baseline_preset=fast`, which skips the
raw SVI family.

### 7.2 Required Run

```bash
just benchmark-a1 mask=stratified baseline_preset=full out=reports/runs/a1-full
```

### 7.3 Discussion

SVI/SSVI accounting is required because the paper compares learned
reconstruction against classical surface-fitting references. The final results
should report not only successful SVI errors, but also failure, timeout,
underidentified-slice, and projection/constraint rates. Silent fallback would
make the baseline comparison misleading.

## 8. Experiment 6: OOD And Robustness

### 8.1 Current Status

Ticker-holdout, temporal-ticker-holdout, and Japan OOD evaluations are pending.
The implementation supports `ticker_holdout` and `temporal_ticker_holdout`
split modes, and the Japan silver table passes the current date gate. However,
no accepted aggregate OOD result is present in `reports/runs/`.

### 8.2 Required Outputs

The final paper should include:

- ticker-holdout or temporal-ticker-holdout masked reconstruction;
- Japan OOD evaluation only after matched baselines and normalization checks;
- OOD degradation ratios normalized against train-only and within-surface
  baselines;
- split manifests and artifact paths for each accepted run.

### 8.3 Discussion

OOD should be framed as robustness analysis. Weak OOD results would not
invalidate the main U.S. masked-reconstruction benchmark, but they would limit
claims about cross-underlying or cross-market generalization.

## 9. Figures And Tables Status

### 9.1 Tables Available In This Snapshot

Current tables in this document:

1. Evidence class summary.
2. Dataset acceptance summary.
3. A1 stratified headline table.
4. Harder-mask top4 candidate-selection table.

### 9.2 Tables Still Needed

Final manuscript tables still needed:

1. Completed `stratified`, `liquidity_correlated`, and `block_wing` benchmark
   table with accepted artifacts.
2. Full SVI/SSVI baseline accounting table.
3. Graph-necessity and liquidity-ablation table.
4. Reliability and bucketed-error table.
5. OOD/ticker-holdout degradation table.
6. Failure-mode and diagnostic summary table.

### 9.3 Figures Still Needed

No manuscript-ready figure artifacts are currently present in `reports/`.

Required figures:

1. Data/preprocessing and masking pipeline.
2. Example irregular option-surface graph.
3. Mask-regime examples.
4. Error versus liquidity/moneyness/tenor buckets.
5. Error versus no-arbitrage violation trade-off.
6. Per-seed dispersion or confidence visualization.

## 10. Overall Discussion

### 10.1 What The Current Evidence Supports

The current recorded evidence supports an internal claim that graph models can
be strong for leakage-controlled masked IV reconstruction on real U.S. option
chains. The A1 stratified result is especially promising because graph variants
beat both train-only and within-surface interpolation baselines.

The harder-mask screen supports a specific next-step decision: promote
`lagos_liq_feature_only` and `gnn_decoded_calendar_convexity`, while avoiding
premature promotion of `lagos_loss_only`.

### 10.2 What The Current Evidence Does Not Yet Support

The current evidence does not yet support:

- a final LoG paper claim;
- a strict no-arbitrage surface construction claim;
- a claim that liquidity-aware GNNs beat all baselines;
- a cross-market U.S.-to-Japan causal or risk-transfer claim;
- a trading or alpha claim;
- a broad market-cycle generalization claim.

### 10.3 Main Failure Modes To Watch

The final results could be weakened if:

- A1 stratified cannot be restored or reproduced;
- harder-mask confirmation fails to beat within-surface baselines;
- SVI/SSVI baselines are competitive after full accounting;
- no-arbitrage diagnostics worsen as masked IV error improves;
- liquidity gains disappear after ablations;
- OOD degradation is larger than simple baseline degradation.

## 11. Immediate Next Results To Produce

1. Restore or rerun A1 stratified:

   ```bash
   just benchmark-a1 mask=stratified out=reports/runs/a1
   ```

2. Delete or archive the incomplete promoted `liquidity_correlated` directory
   and rerun the promoted hard-mask family:

   ```bash
   rm -rf reports/runs/a1-e20-lc
   just benchmark-a1 mask=liquidity_correlated seeds=1 epochs=20 variant_suite=anchor_proxy variants=lagos_liq_feature_only,gnn_decoded_calendar_convexity out=reports/runs/a1-e20 baseline_preset=fast no_arb_surfaces=50 skip_ood=true
   just benchmark-a1 mask=block_wing seeds=1 epochs=20 variant_suite=anchor_proxy variants=lagos_liq_feature_only,gnn_decoded_calendar_convexity out=reports/runs/a1-e20 baseline_preset=fast no_arb_surfaces=50 skip_ood=true
   ```

3. Run full SVI accounting:

   ```bash
   just benchmark-a1 mask=stratified baseline_preset=full out=reports/runs/a1-full
   ```

4. Add accepted aggregate summaries, figures, and final interpretations to this
   document after each completed run.
