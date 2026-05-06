# Results Snapshot

Current evidence state: **A1 stratified remains the core multi-seed preliminary
benchmark; the 2026-05-07 top4 harder-mask screen is candidate-selection
evidence, not final manuscript evidence**.

This page records the current empirical evidence. It intentionally excludes
small pipeline checks and older single-seed tables. The current evidence ledger
separates multi-seed benchmark evidence from single-seed engineering screens.

## Project Introduction

LoG-IV studies option chains as **irregular graphs of option contracts** rather
than fixed-grid volatility images. Each contract is represented by strike,
tenor, option type, quote information, and liquidity variables. Edges encode
local strike-maturity structure and, where used, liquidity similarity.

The core research question is:

> Can graph models reconstruct sparse and irregular option-implied volatility
> surfaces under leakage-controlled masking and liquidity-dependent observation
> noise more accurately than interpolation and train-only baselines?

The project should be presented as a graph-learning benchmark and modeling
study, not as a trading strategy or a cross-market causality study. Japan is an
out-of-distribution evaluation setting rather than the primary empirical claim.

## Design Choices

- **Leakage-controlled masked reconstruction.** The main task is masked
  reconstruction with fixed
  temporal splits. Masked query nodes cannot carry same-day bid, ask, mid, IV,
  spread, volume, open interest, decoded price, quote-derived liquidity score,
  or IV-derived Greeks.
- **Irregular option-surface graph representation.** The benchmark evaluates
  option chains in their native sparse strike-tenor layout instead of forcing a
  regular image grid.
- **Liquidity-dependent observation noise.** Liquidity variables are treated as
  possible signals of quote reliability, not only as ordinary node features.
- **Baseline ladder.** Current A1 already compares against train-only
  means, train-only moneyness-tenor kNN, within-surface kNN, local linear, and
  RBF interpolation. Raw SVI accounting is still pending.
- **Surface-quality diagnostics.** Runs write masked IV metrics, normalized
  decoded-price diagnostics, and Europeanized calendar / convexity /
  vertical-spread diagnostics.
- **Synthetic data for reproducibility.** The synthetic generator provides
  AR(1) SSVI-style surfaces, diagnostic-only fields, stable logical hashing,
  and no-arbitrage diagnostics for controlled tests.

## Current Data

| Dataset | Current role | Current status |
| --- | --- | --- |
| U.S. expanded options silver | Main masked-reconstruction benchmark | 2,979,716 rows, 40 underlyings, 62 observation dates, 2,480 usable surfaces. |
| Japan expanded options silver | Out-of-distribution evaluation | 531,280 rows, 31 observation dates, 6,179 usable surfaces. |
| Synthetic-LoG-IV | Reproducibility and controlled diagnostics | Generator implemented; canonical release artifact still pending. |

The U.S. data window is enough for masked-reconstruction benchmarking. It is not
yet enough for broad market-cycle or regime-generalization claims.

## A1 Stratified Result On 2026-05-04

Run family: `reports/runs/benchmark_a1_stratified/`.

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

Headline table from
`reports/runs/benchmark_a1_stratified/benchmark_summary.csv`:

| Variant | Masked IV MAE | p90 abs error | Price MAE | vs train kNN | vs within kNN | Calendar viol. | Convexity viol. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `encoder_mlp` | 0.0925 ± 0.0011 | 0.1910 | 0.0801 | +7.9% better | worse | 2,553 | 8,143 |
| `gnn_no_liq` | 0.0343 ± 0.0007 | 0.0650 | 0.0807 | +65.9% better | +49.5% better | 2,952 | 8,909 |
| `gnn_liq` | 0.0345 ± 0.0007 | 0.0624 | 0.0782 | +65.6% better | +49.2% better | 3,144 | 8,954 |
| `gnn_decoded_calendar_convexity` | **0.0338 ± 0.0010** | **0.0619** | **0.0759** | **+66.3% better** | **+50.2% better** | **2,344** | **7,751** |

Reference baselines:

- train-only moneyness-tenor kNN masked IV MAE: 0.1004;
- within-surface kNN masked IV MAE: 0.0679;
- within-surface RBF masked IV MAE: 0.0721;
- within-surface local linear masked IV MAE: 0.1332.

Interpretation:

- This is the first complete multi-seed result in which graph variants
  outperform both train-only kNN and within-surface interpolation baselines.
- The decoded calendar/convexity variant is currently best on masked IV MAE,
  p90 masked error, normalized price error, and sampled no-arbitrage diagnostic
  counts.
- Liquidity-aware `gnn_liq` improves normalized price error relative to
  `gnn_no_liq`, but it is not yet better on masked IV MAE. The heteroscedastic
  model ablation is needed before attributing the result to liquidity modeling.
- Convexity violations remain frequent. The current result supports
  reconstruction performance, not strict no-arbitrage quality.

## Superseded Broad 2-Epoch Screen On 2026-05-06

This engineering screen is retained only as candidate-selection provenance. It
does **not** supersede the A1 stratified benchmark, and it is superseded for
current promotion decisions by the 2026-05-07 top4 mask-regime screen below.

Run families:

- `reports/runs/prelim_all_models_e2_stratified/`;
- `reports/runs/prelim_retry_lagos_loss_only_e2_stratified/`;
- `reports/runs/prelim_retry_lagos_attn_only_e2_stratified/`;
- `reports/runs/prelim_retry_lagos_hetero_full_e2_stratified/`;
- `reports/runs/prelim_retry_lagos_random_edges_e2_stratified/`;
- `reports/runs/prelim_retry_lagos_shuffled_edges_e2_stratified/`.

Key takeaways:

- `lagos_loss_only` won the stratified two-epoch screen, but the harder-mask
  screen later showed that it should not be promoted.
- `lagos_liq_feature_only` and `gnn_decoded_calendar_convexity` were close
  enough under stratified masking to justify harder-mask follow-up.
- `lagos_random_edges` and `lagos_shuffled_edges` were weak negative controls,
  which supports keeping graph-topology ablations in the manuscript plan.
- Remaining related-work proxy variants still need separate screening if they
  are to appear in the final comparison: `anchor_deep_smoothing_proxy`,
  `anchor_operator_deep_smoothing_proxy`, `anchor_hexagon_proxy`,
  `anchor_hyperiv_proxy`, `anchor_volnp_proxy`, and `cnp_baseline`.

## Preliminary Top4 Mask-Regime Screen On 2026-05-07

Run families:

- `reports/runs/prelim_top4_liquidity_correlated_e2_liquidity_correlated/`;
- `reports/runs/prelim_top4_block_wing_e2_block_wing/`.

This screen reruns the top stratified candidates under two harder masking
regimes. It is still a two-epoch, seed-1 engineering screen, not final
benchmark evidence.

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

Preliminary interpretation:

- `lagos_liq_feature_only` is the most robust short-run masked-IV candidate
  across `stratified`, `liquidity_correlated`, and `block_wing`. It is best on
  masked IV MAE in both harder mask regimes.
- `gnn_decoded_calendar_convexity` is the best diagnostics candidate. It has
  lower price MAE and fewer sampled calendar, convexity, and vertical-spread
  violations than `lagos_liq_feature_only` under both harder masks.
- `lagos_loss_only` should not be promoted from the current evidence. It wins
  the stratified two-epoch screen, but fails both train kNN and within-surface
  kNN under `liquidity_correlated`, and fails train kNN under `block_wing`.
- `gnn_liq` is competitive under `block_wing`, but not under
  `liquidity_correlated` after two epochs.

## What Is Still Missing

The current result is strong enough for an internal presentation, but not enough
for a final LoG paper claim.

Required next evidence:

- longer or multi-seed confirmation for `liquidity_correlated` and
  `block_wing`;
- raw SVI per-expiry accounting with timeout and failure rates;
- promoted `lagos_liq_feature_only` and `gnn_decoded_calendar_convexity`
  candidate runs;
- ticker-holdout or other out-of-distribution evaluation;
- liquidity-bucket and worst-bucket error tables;
- error-versus-violation trade-off curves;
- final figures and manuscript-level tables.

## Suggested Positioning

The central positioning is:

> LoG-IV introduces a leakage-controlled benchmark for learning from irregular
> option-surface graphs with liquidity-dependent observation noise. On real
> U.S. option chains, graph
> models substantially outperform train-only and within-surface interpolation
> baselines on masked IV reconstruction. The project then studies when
> graph models conditioned on quote reliability help under liquidity-correlated
> missingness, structured sparsity, and out-of-distribution evaluation.

Do not present it as:

- a trading alpha system;
- a causal U.S.-to-Japan risk-transfer paper;
- a strict no-arbitrage surface construction method;
- final evidence that liquidity-aware GNNs outperform all baselines.

## Immediate Next Runs

Promote the two candidates that passed the top4 mask-regime screen:

```bash
just benchmark-a1 mask=liquidity_correlated seeds=1 epochs=20 variant_suite=anchor_proxy variants=lagos_liq_feature_only,gnn_decoded_calendar_convexity out=reports/runs/candidate_top2_liquidity_correlated_e20 baseline_preset=fast no_arb_surfaces=50 skip_ood=true
just benchmark-a1 mask=block_wing seeds=1 epochs=20 variant_suite=anchor_proxy variants=lagos_liq_feature_only,gnn_decoded_calendar_convexity out=reports/runs/candidate_top2_block_wing_e20 baseline_preset=fast no_arb_surfaces=50 skip_ood=true
```

Then run raw SVI accounting on the promoted setting:

```bash
just benchmark-a1 mask=stratified baseline_preset=full out=reports/runs/benchmark_a1_full
```

Promotion gate: `lagos_liq_feature_only` should remain ahead of train-only and
within-surface baselines on both harder masks after longer training.
`gnn_decoded_calendar_convexity` should be judged jointly on masked IV error,
price error, and sampled no-arbitrage diagnostics.
