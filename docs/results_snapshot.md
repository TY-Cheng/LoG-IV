# Results Snapshot

Current evidence state: **A1 stratified is a complete preliminary benchmark, not
final manuscript evidence**.

This page records the current empirical evidence. It intentionally excludes
small pipeline checks and older single-seed tables. The latest complete result
is the 3-seed A1 stratified masked-reconstruction benchmark under
`reports/runs/benchmark_a1_stratified/`.

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

## What Is Still Missing

The current result is strong enough for an internal presentation, but not enough
for a final LoG paper claim.

Required next evidence:

- A1 `liquidity_correlated` mask;
- A1 `block_wing` mask;
- raw SVI per-expiry accounting with timeout and failure rates;
- heteroscedastic graph-model ablation results;
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

Run the remaining A1 mask regimes:

```bash
just benchmark-a1 mask=liquidity_correlated
just benchmark-a1 mask=block_wing
```

Then run raw SVI accounting:

```bash
just benchmark-a1 mask=stratified baseline_preset=full out=reports/runs/benchmark_a1_full
```

Promotion gate: graph variants should remain ahead of the strongest
within-surface baseline on `liquidity_correlated` and `block_wing` masks, not
only on `stratified`.
