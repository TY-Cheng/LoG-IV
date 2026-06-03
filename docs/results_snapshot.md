# Results And Discussion

This document is the live Results and Discussion chapter for the LoG-IV
manuscript plan. `docs/paper_plan.md` defines the research question, methods,
metrics, and expected experiments; this document records the evidence currently
available in the repository and interprets it using that plan.

The current evidence state is **preliminary and candidate-selection evidence,
not final manuscript evidence**. Results are reported only as strongly as the
local artifacts support them. Missing, incomplete, and pending experiments are
kept in the chapter so that the paper narrative and the execution backlog stay
aligned.

## 1. Results Overview

### 1.1 Main Finding So Far

The strongest recorded result is the A1 stratified benchmark from 2026-05-04.
It reports that graph variants substantially outperform train-only and
within-surface interpolation baselines on leakage-controlled masked IV
reconstruction. The best recorded A1 model is
`gnn_decoded_calendar_convexity`, with masked IV MAE of `0.0338 +/- 0.0010`,
about `66.3%` better than train-only moneyness-tenor kNN and about `50.2%`
better than within-surface kNN.

That finding is promising, but it is not yet a final claim because the local
artifact directory `reports/runs/a1-str/` is absent. The result must be restored
or rerun before it can serve as manuscript evidence.

The harder-mask evidence from 2026-05-07 is useful for model selection rather
than final ranking. It suggests:

- `lagos_liq_feature_only` is the strongest short-run masked-IV candidate under
  `liquidity_correlated` and `block_wing`;
- `gnn_decoded_calendar_convexity` is the strongest diagnostics candidate,
  especially on decoded-price error and sampled no-arbitrage violations;
- `lagos_loss_only` should not be promoted despite winning a superseded
  stratified two-epoch screen.

### 1.2 Result Status By Planned Experiment

Table 1 aligns the paper plan with the current evidence state. "Accepted" means
the local artifact family contains the expected aggregate outputs. No current
non-archived run family has an accepted `benchmark_summary.csv`.

| Planned result | Current artifact | Status | Current outcome | Interpretation |
| --- | --- | --- | --- | --- |
| Data acceptance gates | `data/silver/option_quotes/*.manifest.json` | Complete for A1 development | U.S. and Japan silver gates pass | Data are sufficient for Protocol A development, not broad market-cycle claims. |
| A1 stratified, 20 epochs, 3 seeds | Recorded from `reports/runs/a1-str/`; local directory missing | Recorded but not locally backed | Graph variants beat train-only and within-surface baselines | Restore or rerun before final claims. |
| Broad two-epoch model screen | `reports/archive/20260511-legacy/p-*-e2-str/` | Archived candidate-selection evidence | Identified LAGOS and graph candidates | Useful search provenance, not benchmark evidence. |
| Harder-mask top4 screen | `reports/archive/20260511-legacy/p-top4-e2-lc/` and `p-top4-e2-bw/` | Archived candidate-selection evidence | `lagos_liq_feature_only` best on error; `gnn_decoded_calendar_convexity` best on diagnostics | Determines next promoted runs. |
| Promoted hard-mask confirmation | `reports/runs/a1-e20-lc/` | Incomplete | Only `b-us-gdcc-mr-t-e20-s1/best_model.pt` visible | Do not cite; rerun or archive partial directory. |
| Full SVI/SSVI accounting | None | Pending | No accepted full-baseline table | Required for classical surface-fitting comparison. |
| OOD/ticker-holdout robustness | None | Pending | No accepted aggregate OOD result | Required for robustness claims. |
| Reliability and bucket analysis | Not yet aggregated into manuscript tables | Pending | Diagnostics implementation exists, accepted tables absent | Required for liquidity-reliability claims. |
| Final figures | None in `reports/` | Pending | No manuscript-ready figures | Required before manuscript drafting. |

### 1.3 Claim Status

Current evidence supports an internal, preliminary statement: graph-based
irregular option-surface representations appear strong for leakage-controlled
masked IV reconstruction on the current U.S. A1 data window.

Current evidence does **not** yet support:

- a final LoG paper claim;
- a strict no-arbitrage surface construction claim;
- a claim that liquidity-aware GNNs beat all baselines;
- a cross-market U.S.-to-Japan causal or risk-transfer claim;
- a trading, alpha, hedging, execution, or portfolio claim;
- a broad market-cycle generalization claim.

## 2. Data Results And Discussion

### 2.1 Dataset Acceptance Results

Table 2 reports the live data gates used by Protocol A.

| Dataset | Role | Live status | Gate interpretation |
| --- | --- | --- | --- |
| U.S. expanded options silver | Main in-domain masked-reconstruction benchmark | `data_v1`; 2,979,716 rows; 40 underlyings; 62 observation dates; 2,867,637 IV-usable rows; 2,480 usable surfaces | Passes A1/data_v1. Not enough for broad market-cycle claims. |
| Japan expanded options silver | Out-of-distribution evaluation material | `data_v0`; 557,982 rows; 249 underlyings; 32 observation dates; 557,982 IV-usable rows; 7,488 usable surfaces | Passes the current OOD date gate, but needs matched baselines and normalization checks before claims. |
| Synthetic-LoG-IV | Reproducibility and controlled diagnostics | Generator implemented | Useful for sanity checks, not a substitute for real U.S. evidence. |

### 2.2 Data Interpretation

The U.S. table is sufficient for Protocol A development and first multi-seed
benchmarking because it passes the 2,400-surface and 60-date gates. The
empirical window is 2026-02-02 through 2026-04-30 and the universe contains 40
liquid U.S. underlyings. That scope is appropriate for a first benchmark, but
it is not sufficient for claims about market-cycle robustness, low-liquidity
universes, or long-regime generalization.

The Japan table is now larger than older planning notes stated: the live
manifest shows 249 underlyings and 7,488 usable surfaces. Japan should still be
framed as out-of-distribution evidence rather than the main empirical claim.
Timestamp semantics, market-specific normalization, and matched baselines need
explicit treatment before Japan results can be interpreted cleanly.

The current IV inversion uses option mid prices with same-date underlying daily
close, Black-forward bisection, zero rate, and zero dividend. This is
acceptable for masked-IV reconstruction benchmarking, but it weakens strict
pricing and no-arbitrage claims. Results should therefore emphasize IV
reconstruction and treat decoded-price and no-arbitrage checks as diagnostics
until rate, dividend, and forward assumptions are upgraded.

## 3. Main Masked-Reconstruction Result

### 3.1 A1 Stratified Protocol

Recorded run family:

```text
reports/runs/a1-str/
```

Current artifact caveat: this directory is **not present locally** as of the
latest repository check. Treat Table 3 as a recorded result that must be
restored or rerun before final claims.

Protocol details:

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

### 3.2 A1 Stratified Outcome

Table 3 is the headline result recorded from
`reports/runs/a1-str/benchmark_summary.csv`.

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

### 3.3 A1 Interpretation

The recorded A1 result supports the core graph-representation direction under
stratified masking. Every graph variant beats the no-graph `encoder_mlp`, the
train-only kNN baseline, and the within-surface kNN baseline on masked IV MAE.
This is exactly the first evidence gate defined in the paper plan: the model
must beat train-only baselines and should beat the relevant within-surface
baseline on masked-node metrics.

`gnn_decoded_calendar_convexity` is the strongest recorded A1 model. It is best
on masked IV MAE, p90 masked absolute error, normalized price MAE, and sampled
calendar/convexity diagnostic counts. This supports the idea that decoded
regularization can improve the reconstruction/diagnostics trade-off.

The result does not yet prove that liquidity-aware message passing is the
driver. `gnn_liq` improves normalized price MAE relative to `gnn_no_liq`, but
it does not improve masked IV MAE. Strong liquidity claims require the LAGOS
ablation ladder, reliability diagnostics, and harder missingness confirmation.

### 3.4 A1 Limitations

The missing local artifact is the most important limitation. A manuscript
cannot cite the A1 table as final evidence until `reports/runs/a1-str/` is
restored or the benchmark is rerun and reproduces the aggregate table.

Convexity violations remain frequent even for the best recorded A1 model. The
result therefore supports masked reconstruction quality, not strict
no-arbitrage surface construction. Raw SVI and constrained SVI/SSVI accounting
are also pending, so classical surface-fitting comparisons are incomplete.

## 4. Model Screens And Harder Missingness

### 4.1 Broad Two-Epoch Model Screen

This engineering screen is retained only as candidate-selection provenance. It
does **not** supersede the A1 stratified benchmark and it is superseded for
promotion decisions by the harder-mask screen in Section 4.2.

Archived run families:

- `reports/archive/20260511-legacy/p-all-e2-str/`;
- `reports/archive/20260511-legacy/p-llo-e2-str/`;
- `reports/archive/20260511-legacy/p-la-e2-str/`;
- `reports/archive/20260511-legacy/p-lhf-e2-str/`;
- `reports/archive/20260511-legacy/p-lre-e2-str/`;
- `reports/archive/20260511-legacy/p-lse-e2-str/`.

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

Interpretation: this screen is valuable as a search log, not as benchmark
evidence. It uses one seed and two epochs, so it is too short to rank final
models.

### 4.2 Harder Missingness Top4 Screen

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

Table 4 reports the harder-mask candidate-selection results.

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

Interpretation:

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

Discussion:

This experiment sharpens the next modeling decision. It suggests a trade-off
between scalar IV reconstruction and surface-quality diagnostics:

- choose `lagos_liq_feature_only` as the primary error candidate;
- keep `gnn_decoded_calendar_convexity` as the diagnostics candidate;
- do not promote `lagos_loss_only` without additional evidence.

The screen is still two epochs and one seed. It is evidence for what to run
next, not evidence for a final model ranking.

### 4.3 Promoted Hard-Mask Confirmation

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

Required confirmation commands:

```bash
just benchmark-a1 mask=liquidity_correlated seeds=1 epochs=20 variant_suite=anchor_proxy variants=lagos_liq_feature_only,gnn_decoded_calendar_convexity out=reports/runs/a1-e20 baseline_preset=fast no_arb_surfaces=50 skip_ood=true
just benchmark-a1 mask=block_wing seeds=1 epochs=20 variant_suite=anchor_proxy variants=lagos_liq_feature_only,gnn_decoded_calendar_convexity out=reports/runs/a1-e20 baseline_preset=fast no_arb_surfaces=50 skip_ood=true
```

The promotion gate is that `lagos_liq_feature_only` should remain ahead of
train-only and within-surface baselines on harder masks, while
`gnn_decoded_calendar_convexity` should remain competitive when judged jointly
on error, price diagnostics, and sampled no-arbitrage diagnostics.

## 5. Baseline, SVI, And Robustness Results

### 5.1 Full Baseline And SVI Accounting

Raw SVI per-expiry and constrained/projected SVI/SSVI accounting are pending.
Current A1 and harder-mask screens use `baseline_preset=fast`, which skips the
raw SVI family.

Required run:

```bash
just benchmark-a1 mask=stratified baseline_preset=full out=reports/runs/a1-full
```

Discussion:

SVI/SSVI accounting is required because the paper compares learned
reconstruction against classical surface-fitting references. The final results
should report successful SVI errors together with failure, timeout,
underidentified-slice, and projection/constraint rates. Silent fallback would
make the baseline comparison misleading.

### 5.2 OOD And Ticker-Holdout Robustness

Ticker-holdout, temporal-ticker-holdout, and Japan OOD evaluations are pending.
The implementation supports `ticker_holdout` and `temporal_ticker_holdout`
split modes, and the Japan silver table passes the current date gate. However,
no accepted aggregate OOD result is present in `reports/runs/`.

Required outputs:

- ticker-holdout or temporal-ticker-holdout masked reconstruction;
- Japan OOD evaluation only after matched baselines and normalization checks;
- OOD degradation ratios normalized against train-only and within-surface
  baselines;
- split manifests and artifact paths for each accepted run.

Discussion:

OOD should be framed as robustness analysis. Weak OOD results would not
invalidate the main U.S. masked-reconstruction benchmark, but they would limit
claims about cross-underlying or cross-market generalization.

### 5.3 Reliability, Bucket, And Failure-Mode Analysis

The implementation writes reliability diagnostics and bucketed metrics, but no
accepted manuscript-level reliability table is currently present. The final
chapter should include:

- predicted precision versus realized absolute error;
- reliability-bucket NLL and interval coverage;
- masked IV MAE by liquidity, moneyness, tenor, underlying, option type, and
  split;
- worst-bucket behavior under `liquidity_correlated` and `block_wing`;
- error versus no-arbitrage violation trade-offs.

Discussion:

This analysis is necessary before claiming that liquidity variables improve
reliability rather than merely acting as extra node features. It is also the
main place where negative or uneven behavior should be made visible.

## 6. Tables And Figures

### 6.1 Tables Currently Available In This Chapter

| Table | Content | Status |
| --- | --- | --- |
| Table 1 | Planned experiment versus artifact/outcome inventory | Current live summary |
| Table 2 | Dataset acceptance results | Artifact-backed by manifests |
| Table 3 | A1 stratified headline result | Recorded, but local artifact missing |
| Table 4 | Harder-mask top4 screen | Archived candidate-selection evidence |

### 6.2 Tables Still Needed For The Final Manuscript

| Needed table | Required artifact |
| --- | --- |
| Completed `stratified`, `liquidity_correlated`, and `block_wing` benchmark table | Accepted `benchmark_summary.csv` files |
| Full SVI/SSVI baseline accounting table | `baseline_preset=full` run with failure and timeout fields |
| Graph-necessity and liquidity-ablation table | Candidate-budget runs for no-edge, random-edge, shuffled-edge, liquidity-feature, and reliability-gated variants |
| Reliability and bucketed-error table | Accepted diagnostics and bucket summaries |
| OOD/ticker-holdout degradation table | Accepted OOD or ticker-holdout aggregate summaries |
| Failure-mode and diagnostic summary table | Consolidated price, reliability, no-arbitrage, and SVI diagnostics |

### 6.3 Figure Register

No manuscript-ready figure artifacts are currently present in `reports/`.

| Figure | Intended content | Current status |
| --- | --- | --- |
| Figure 1 | Data/preprocessing and masking pipeline | Mermaid chart exists in `docs/paper_plan.md`; needs final rendering |
| Figure 2 | Example irregular option-surface graph | Pending |
| Figure 3 | Mask-regime examples for `stratified`, `liquidity_correlated`, and `block_wing` | Pending |
| Figure 4 | Error versus liquidity, moneyness, and tenor buckets | Pending |
| Figure 5 | Error versus no-arbitrage violation trade-off | Pending |
| Figure 6 | Per-seed dispersion or confidence visualization | Pending |

## 7. Overall Discussion

### 7.1 What The Evidence Supports

The current recorded evidence supports an internal claim that graph models can
be strong for leakage-controlled masked IV reconstruction on real U.S. option
chains. The A1 stratified result is especially promising because graph variants
beat both train-only and within-surface interpolation baselines.

The harder-mask screen supports a specific next-step decision: promote
`lagos_liq_feature_only` and `gnn_decoded_calendar_convexity`, while avoiding
premature promotion of `lagos_loss_only`.

### 7.2 What The Evidence Does Not Yet Support

The current evidence does not yet support:

- a final LoG paper claim;
- a strict no-arbitrage surface construction claim;
- a claim that liquidity-aware GNNs beat all baselines;
- a cross-market U.S.-to-Japan causal or risk-transfer claim;
- a trading or alpha claim;
- a broad market-cycle generalization claim.

### 7.3 Main Failure Modes To Watch

The final results could be weakened if:

- A1 stratified cannot be restored or reproduced;
- harder-mask confirmation fails to beat within-surface baselines;
- SVI/SSVI baselines are competitive after full accounting;
- no-arbitrage diagnostics worsen as masked IV error improves;
- liquidity gains disappear after ablations;
- OOD degradation is larger than simple baseline degradation.

### 7.4 Manuscript Positioning

The paper should be sold as a graph/geometric ML benchmark and modeling paper:
irregular option-token graphs, leakage-controlled masking, market-relevant
missingness, and reliability-aware evaluation. The current results are
promising enough to justify the benchmark direction, but the paper should not
overstate final performance until accepted multi-mask, multi-seed, SVI, OOD,
reliability, and figure artifacts exist.

## 8. Immediate Next Results To Produce

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
