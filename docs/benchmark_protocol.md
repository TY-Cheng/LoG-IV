# Benchmark Protocol

This page defines the evaluation protocol: task construction, data splits,
baseline scope, leakage controls, regularizer semantics, and evidence gates.

## Entrypoint

The main benchmark entrypoint is:

```bash
just benchmark-a1 stratified
```

The default protocol uses:

- `task=masked_reconstruction`;
- `split_mode=temporal`;
- `mask_regime=stratified`;
- `target_space=iv`;
- `seeds=1,2,3`;
- `min_us_surfaces=2400`;
- `min_us_dates=60`;
- `min_jp_dates=20`.

Use smaller `--max-us-surfaces`, `--max-jp-surfaces`, and
`--max-nodes-per-surface` caps only for pipeline checks. Capped or single-seed
runs are diagnostic.

## Data Acceptance

The benchmark may run only when the data gate passes:

| Market | Gate | Current status |
| --- | --- | --- |
| U.S. | At least 2,400 usable `(underlying, observation_date)` surfaces after `min_nodes_per_surface=20` | Pass: 2,480 usable surfaces in the current expanded silver table. |
| Japan | At least 20 usable option observation dates for out-of-distribution evaluation | Pass: 31 usable dates in the current expanded silver table. |

If the gate fails, the CLI writes a data-expansion report and exits before
training. This is intended behavior, not a training crash.

## Task

The benchmark task is masked reconstruction.

For each surface:

1. Deterministically choose masked target nodes from option token identity and
   seed under the configured mask regime.
2. Remove IV, bid, ask, volume, and open interest from the masked input nodes.
3. Preserve the full target quotes only for loss, metrics, and diagnostics.
4. Report main metrics on masked nodes.

Supported Protocol A mask regimes:

- `stratified`: deterministic liquidity, moneyness, tenor, and option-type
  bucketed mask; default for `data_v0`.
- `random`: deterministic random-node mask.
- `liquidity_correlated`: preferentially masks low-volume and low-open-interest
  nodes.
- `block_wing`: masks the largest absolute log-moneyness wing nodes.

Visible-context features must be computed after masking and strictly from
visible nodes. Masked query nodes must not carry bid, ask, mid, IV, spread,
decoded price, quote-derived liquidity score, or IV-derived Greeks.

Observed reconstruction and next-day forecasting remain useful diagnostics, but
they are not the first manuscript-level task.

## Splits

Supported split modes:

- `temporal`: train, validation, and test are separated by observation date.
- `ticker_holdout`: selected underlyings are withheld from training.
- `temporal_ticker_holdout`: temporal split plus held-out ticker evaluation.
- `random`: engineering diagnostic only.

The default split is `temporal`. Ticker-holdout and temporal-ticker-holdout
runs are robustness or transfer checks. Random splits should not be used for
manuscript-level claims.

Every run writes `splits.json` so the train, validation, test, masked count, and
held-out ticker metadata are auditable.

## Baselines

Manuscript-level baselines must be fit only on training surfaces:

- `train_mean_iv_global`;
- `train_mean_iv_by_underlying`;
- `train_mean_iv_by_moneyness_tenor_bucket`;
- `train_knn_moneyness_tenor`;
- `random_uniform_iv`.

Protocol A P0 within-surface baselines may use evaluation-surface visible nodes
after masking, but never masked quote fields:

- `within_surface_knn_moneyness_tenor`;
- `within_surface_local_linear`;
- `within_surface_rbf`;
- `within_surface_svi_raw_per_expiry`;
- `within_surface_svi_constrained_per_expiry`;
- `within_surface_svi_calendar_projection`;
- `within_surface_ssvi_calendar_constrained`;
- `within_surface_rbf_calendar_projection`.

Raw SVI per-expiry remains the primary classical calibration baseline. It uses
bounded optimization with a per-slice wall-clock limit and max-iteration limit.
Underidentified slices, optimizer failures, constraint failures, and timeouts
are recorded directly; there is no silent fallback. The constrained and
projected variants are explicit rows, not replacements for raw SVI. The
calendar-projection rows enforce total-variance monotonicity only and should not
be described as full static-arbitrage repair.

The CLI also exposes an `anchor_proxy` variant suite for related-work
orientation. These variants are faithful-spirit baselines for the relevant
method family, not external-paper reproductions:

- `anchor_deep_smoothing_proxy`: fixed-grid CNN smoother with decoded
  no-arbitrage regularization;
- `anchor_operator_deep_smoothing_proxy`: continuous-coordinate kernel operator
  over visible option tokens, without liquidity reliability modeling;
- `anchor_hexagon_proxy`: heterogeneous edge-family attention over same-expiry,
  same-strike, near-moneyness, near-tenor, put-call, and liquidity-bucket
  relations, with a lightweight cross-view alignment loss;
- `anchor_hyperiv_proxy`: fixed-grid neural smoother with decoded
  no-arbitrage regularization, without hypernetwork or hard guarantees;
- `anchor_volnp_proxy`: attentive neural-process-style visible-context baseline
  for sparse-quote completion;
- `cnp_baseline`: permutation-invariant conditional neural process baseline.

Runs using these variants record `anchor_reference`, `implementation_status`,
and `implementation_notes` in `manifest.json`. HyperIV is additionally supported
through `hyperiv-compare`, which records or runs an external adapter manifest;
the native proxy is not a hard-arbitrage hypernetwork.

The moneyness-tenor kNN baseline is the current lower bound. A GNN result that
beats only the global mean is not enough for a LoG claim. A manuscript-level
Protocol A claim should beat relevant within-surface baselines, not just
train-only means.

Leave-one-out baselines are written separately to
`diagnostic_leakage_prone_baselines.csv`. They are useful for debugging but must
not be used as benchmark comparisons because they can see evaluation-surface
targets.

## Regularization

Benchmark runs must not use embedding-distance or embedding-norm proxies as
manuscript-level no-arbitrage penalties.

Allowed training regularizers:

- decoded Black-forward calendar total-variance penalty;
- decoded Black-forward strike-convexity penalty;
- graph smoothness regularizer as an engineering model prior.

Put-call parity remains diagnostic only until rates, dividends, forwards, and
style assumptions are explicit. No-arbitrage diagnostics are computed after
training on decoded prices and should compare true-IV and predicted-IV decoded
prices.

## Artifacts

Each benchmark run should write:

- `manifest.json`;
- `splits.json`;
- `metrics_epoch.jsonl`;
- `metrics_summary.json`;
- `predictions.parquet`;
- `baselines_summary.csv`;
- `diagnostic_leakage_prone_baselines.csv`;
- `diagnostics_price.json`;
- `diagnostics_reliability.json`;
- `diagnostics_no_arbitrage.json`.

The aggregate command writes `benchmark_summary.csv` with seed-level means and
standard deviations.

`baselines_summary.csv` records `fit_scope`, `visible_context_policy`,
`uses_masked_quote_features`, predicted row counts, failure rates, and raw SVI
failure accounting.

## Benchmark Matrix

Avoid full Cartesian-product expansion.

| Stage | Data | Seeds | Mask regimes | Baselines |
| --- | --- | --- | --- | --- |
| A0 | `data_v0` | 1 | `stratified` | P0 sanity and within-surface baselines |
| A1 | `data_v1` | 3 | `stratified`, `liquidity_correlated`, `block_wing` | P0 baselines |
| A2 | `data_v2` | 3-5 | Final selected masks | screened manuscript-level baselines |

Price and no-arbitrage outputs are diagnostics first, not separate target
spaces. Out-of-distribution degradation ratios should use normalized errors
rather than raw MAE alone.

## Training Controls

Training uses deterministic split and mask construction under the configured
seed. The model code also seeds NumPy and PyTorch before model construction, so
random initialization, surface order shuffling, Fourier-feature initialization,
dropout, and PyTorch module defaults are seed-controlled within a run. The
current implementation does not override PyTorch's default linear-layer
initialization with a custom Xavier or Kaiming policy. That is intentional for
now: initialization changes should be treated as an ablation only after the
main protocol is stable.

Validation early stopping is available but disabled by default. Use
`--early-stopping-patience N` and optionally `--early-stopping-min-delta X` for
screening or long paper-candidate runs. Runs that use early stopping must report
the selected epoch and should keep the same patience rule across compared
variants. Fixed-epoch runs remain the cleanest comparison when the training
budget is the object being controlled.

The default neural architecture uses residual MLP token encoding, Fourier
geometry features, LayerNorm, dropout, and GELU activations, followed by a
heterogeneous GAT/SAGE-style graph operator over strike-neighbor,
maturity-neighbor, and liquidity-similarity edges. GELU is a standard modern
choice for Transformer-style and residual MLP models. It is not claimed as a
novel contribution. ReLU, SiLU, or ELU can be screened later, but activation
search is lower priority than leakage control, baseline strength, and graph
necessity ablations.

Epoch budgets should be empirical rather than copied from unrelated IV-surface
papers. Current working tiers are:

| Purpose | Epochs | Notes |
| --- | ---: | --- |
| Pipeline smoke | 1-2 | Confirms artifacts and device path only. |
| Screening | 10 | Useful for rough ranking, not a claim. |
| A1 preliminary | 20 | Current default for local multi-seed development. |
| Paper-candidate | 50 | Use if validation curves still improve at 20 epochs. |
| Final selected variants | 50-100 | Use fixed budget or registered early stopping consistently. |

## Manuscript-Level Scorecard

Headline tables should remain compact:

- masked IV MAE and RMSE;
- masked p90 absolute error;
- liquidity-bucket masked MAE;
- difference relative to the strongest relevant within-surface baseline.

Supporting scorecards should report:

- normalized decoded-price error and, where bid/ask are available, bid-ask hit
  rate;
- error-versus-violation trade-off summaries: held-out quote error versus
  decoded calendar, butterfly, and vertical-spread violation severity;
- risk-neutral-density roughness diagnostics only for synthetic or
  European-style index-option subsets where the Europeanized assumptions are
  defensible;
- reliability calibration on held-out masked nodes: predicted precision versus
  realized absolute error rank correlation; masked IV MAE, normalized price MAE,
  and NLL by predicted-reliability bucket; liquidity-bucket NLL and MAE by
  spread, volume, and open-interest bucket; and prediction-interval coverage if
  intervals are emitted;
- graph-necessity ablations: no-edge set/context model, liquidity-as-feature-only
  GNN, scalar-gate GNN, random-edge or shuffled-edge GNN, and ODS-style
  continuous-coordinate operator without liquidity reliability;
- performance degradation under distribution shift, normalized against a naive
  train-only baseline rather than a raw out-of-distribution/in-distribution MAE
  ratio.

Reliability calibration and graph-necessity ablations are required for
manuscript-level heteroscedastic-graph claims. Synthetic prior pretraining is
optional and should not be promoted to the main real-data claim unless it
improves real-data generalization under the same fixed split and masking
protocol.

For U.S. single-name and ETF options, decoded no-arbitrage outputs are
Europeanized surface-geometry diagnostics. Strict static-arbitrage claims are
reserved for synthetic data or European-style index-option subsets with explicit
rate, dividend, and forward assumptions.

## Promotion Rule

A result can be discussed as manuscript-level evidence only when:

- the U.S. and Japan data gates pass;
- fixed non-random splits are recorded;
- main metrics are masked-node metrics;
- train-only baselines are present and stronger leakage-prone baselines are
  separated;
- at least three seeds are run;
- training length is long enough for model comparison, not only a pipeline check;
- the graph model beats the relevant within-surface kNN/interpolation baseline;
- raw SVI per-expiry failure accounting is present;
- reliability calibration shows that predicted precision ranks realized
  held-out errors or improves low-reliability buckets;
- graph-necessity ablations show that the result is not explained by no-edge
  set aggregation, liquidity features alone, or shuffled/random graph structure;
- decoded-price and no-arbitrage diagnostics do not undermine the scalar metric.

Until then, results should be labeled diagnostic.
