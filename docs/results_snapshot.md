# Results Snapshot

Current evidence state: **expanded data gate plus a single-seed 10-epoch
benchmark-stage run, not paper evidence**.

No result below is a `paper_candidate`. The current runs are useful because they
exercise real data, model controls, baseline summaries, and post-run diagnostics.
They are still too short, single-seed, and missing the full strong-baseline
ladder required for LoG claims. Short pipeline-check runs are intentionally
excluded from this page so it can serve as a reporting-oriented result ledger.

## Snapshot On 2026-05-01

Implemented:

- `engineering_gate`: `just check` passes with all extras, including `torch`
  and `torch-geometric`.
- `real_us_mvp`: U.S. silver table has 980 rows over 5 tickers. With
  `min_nodes_per_surface=3`, 31 of 70 `(underlying, date)` surfaces survive.
- `japan_ood_probe`: Japan silver table has 17,781 rows over one observation
  date. The first matrix caps this to 80 of 249 surfaces.
- `real_us_mvp` / `japan_ood_probe`: first 20-epoch matrix writes full
  validation/test predictions, baseline summaries, normalized price diagnostics,
  and no-arbitrage diagnostics under `reports/runs/`.
Blocked or incomplete:

- IV inversion and rate/dividend handling;
- multi-seed real-data matrix;
- full-window U.S. and Japan data coverage;
- masked quote reconstruction and next-day forecasting;
- stronger non-neural baselines trained on fixed train splits;
- paper-facing discussion beyond diagnostic interpretation.

## Protocol Update On 2026-05-02

Implemented, but not yet paper evidence:

- `benchmark-protocol`: fixed split manifests with `random`, `temporal`,
  `ticker_holdout`, and `temporal_ticker_holdout` modes. The default credible
  split is temporal.
- `masked_reconstruction`: deterministic per-surface masks keyed by seed and
  option token. Masked inputs remove IV, bid, and ask while preserving target
  values only for loss/metrics.
- Train-only baselines: global train mean IV, train mean by underlying, train
  mean by moneyness-tenor bucket, and train-fitted moneyness-tenor kNN.
- Leakage-prone leave-one-out baselines are still written, but only to
  `diagnostic_leakage_prone_baselines.csv`.
- Decoded-price regularization replaces embedding proxy PCP/convexity for
  benchmark runs. PCP remains diagnostic only.
- Data expansion probes now cover U.S. options flat-file templates and J-Quants
  V2 options date loops.

This protocol update was implemented before the expanded local data gate passed.
The current expanded-data status is recorded below and supersedes this older
blocked state.

## Expansion Update On 2026-05-03

Implemented:

- `data-expansion` now writes deduplicated expanded silver tables when rows are
  available:
  `data/silver/option_quotes/us_option_quotes_expanded.parquet` and
  `data/silver/option_quotes/jp_option_quotes_expanded.parquet`.
- The expansion report records source rows, deduplicated rows, distinct dates,
  IV-usable rows, usable surface counts, and the gate result.
- J-Quants bronze-cache rebuild is available through `data-expansion --market jp`.
  The current expanded Japan table has 531,280 deduplicated rows, 31 usable
  observation dates, and 6,179 usable `(underlying, observation_date)` surfaces
  under `min_nodes_per_surface=20`.
- `MASSIVE_OPTIONS_FLAT_FILE_TEMPLATE` is configured locally for the
  `us_options_opra/day_aggs_v1` S3 flat-file path.
- U.S. expanded silver now covers 40 underlyings from 2026-03-18 to
  2026-04-30. It has 1,516,612 deduplicated rows, 1,456,358 IV-usable rows, 31
  usable observation dates, and 1,240 usable `(underlying, observation_date)`
  surfaces under `min_nodes_per_surface=20`.
- Massive option `day_aggs_v1` provides price rows, not vendor IV. The U.S.
  adapter uses same-date `us_stocks_sip/day_aggs_v1` closes as spot/forward
  proxies and applies a zero-rate, zero-dividend Black-forward bisection
  inversion. These inferred IVs are engineering benchmark targets, not
  vendor-IV or no-arbitrage claims.

Current gate status:

- U.S. expanded gate: pass, 1,240 usable surfaces against the 1,000-surface
  default.
- Japan expanded gate: pass, 31 usable observation dates against the 20-date
  default.

## Benchmark Stage1 On 2026-05-03

Run family: `reports/runs/benchmark_stage1/`.

This is the latest local benchmark result currently recorded in the repo. It
uses the credible protocol path with `task=masked_reconstruction`,
`split_mode=temporal`, `seed=1`, `epochs=10`, `d_model=64`, two encoder layers,
two GNN layers where applicable, and `max_nodes_per_surface=250`.

Temporal split manifest:

- train ends on 2026-04-16;
- validation covers 2026-04-17 through 2026-04-23;
- test starts on 2026-04-24;
- train/validation/test surfaces: 840 / 200 / 200;
- train/validation/test rows: 209,768 / 49,906 / 49,805;
- validation/test masked rows: 9,981 / 9,961.

Headline table from `benchmark_summary.csv`:

| Variant | Masked IV MAE | IV MAE | Train kNN MAE | Delta vs kNN | Price MAE | Cal Viol | Conv Viol | Label |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `encoder_mlp` | 0.1028 | 0.1270 | 0.1043 | -0.0016 | 0.0148 | 1,894 | 9,689 | `real_us_mvp` |
| `gnn_no_liq` | 0.0546 | 0.0878 | 0.1043 | -0.0497 | 0.0099 | 3,967 | 15,869 | `real_us_mvp` |
| `gnn_liq` | 0.0581 | 0.0799 | 0.1043 | -0.0463 | 0.0090 | 3,592 | 13,718 | `real_us_mvp` |
| `gnn_decoded_calendar_convexity` | 0.0511 | 0.0865 | 0.1043 | -0.0533 | 0.0090 | 2,579 | 14,200 | `real_us_mvp` |

Interpretation:

- This is the first expanded-data run where graph variants beat the current
  train-only moneyness-tenor kNN baseline on masked IV MAE.
- The decoded calendar/convexity variant has the best masked IV MAE in this
  run, but it does not solve the no-arbitrage story because convexity
  violations remain high.
- Liquidity edges improve normalized price error versus `gnn_no_liq`, but
  `gnn_liq` is not the best masked-IV variant in this single-seed run.
- The result remains `real_us_mvp`, not `paper_candidate`, because it is only
  10 epochs and one seed, and it does not yet include SVI/SSVI, interpolation,
  tabular ML, set-model, and standard-GNN baseline ladders.

## First Matrix

All runs use 20 epochs, seed 42, `d_model=64`, two encoder layers, and two GNN
layers where applicable. `Price MAE` is normalized Black-forward price error
against observed bid/ask mid. No-arbitrage counts are diagnostics on decoded
prices, not training losses.

| Run | Market | Variant | Label | IV MAE | IV RMSE | Global Mean MAE | kNN MAE | Price MAE | Cal Viol | Conv Viol |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `real_us_encoder_mlp_20e_seed42` | US | `encoder_mlp` | `real_us_mvp` | 0.1182 | 0.1785 | 0.2141 | 0.0797 | 0.0058 | 1 | 146 |
| `real_us_gnn_no_liq_20e_seed42` | US | `gnn_no_liq` | `real_us_mvp` | 0.1644 | 0.2175 | 0.2141 | 0.0797 | 0.0081 | 12 | 144 |
| `real_us_gnn_liq_20e_seed42` | US | `gnn_liq` | `real_us_mvp` | 0.1631 | 0.2192 | 0.2141 | 0.0797 | 0.0078 | 0 | 152 |
| `real_us_gnn_calendar_reg_20e_seed42` | US | `gnn_calendar_reg` | `real_us_mvp` | 0.1575 | 0.2071 | 0.2141 | 0.0797 | 0.0077 | 0 | 146 |
| `real_us_gnn_all_reg_diag_20e_seed42` | US | `gnn_all_reg_diag` | `diagnostic_only` | 0.2634 | 0.3683 | 0.2141 | 0.0797 | 0.0117 | 0 | 149 |
| `real_jp_encoder_mlp_20e_seed42` | JP | `encoder_mlp` | `japan_ood_probe` | 0.0090 | 0.0155 | 0.1730 | 0.1126 | 0.1017 | 0 | 13 |
| `real_jp_gnn_no_liq_20e_seed42` | JP | `gnn_no_liq` | `japan_ood_probe` | 0.0218 | 0.0454 | 0.1730 | 0.1126 | 0.1008 | 0 | 32 |
| `real_jp_gnn_liq_20e_seed42` | JP | `gnn_liq` | `japan_ood_probe` | 0.0225 | 0.0588 | 0.1730 | 0.1126 | 0.1018 | 0 | 25 |
| `real_jp_gnn_calendar_reg_20e_seed42` | JP | `gnn_calendar_reg` | `japan_ood_probe` | 0.0254 | 0.0594 | 0.1730 | 0.1126 | 0.1033 | 0 | 32 |
| `real_jp_gnn_all_reg_diag_20e_seed42` | JP | `gnn_all_reg_diag` | `diagnostic_only` | 0.0253 | 0.0604 | 0.1730 | 0.1126 | 0.1008 | 0 | 31 |

Interpretation:

- U.S. graph variants now beat the global mean-IV baseline, but not the
  moneyness-tenor kNN baseline.
- U.S. liquidity edges slightly improve over no-liquidity edges, and the
  calendar-only regularizer is the strongest U.S. GNN variant in this pass.
- The all-regularizer diagnostic is worse on U.S.; it should not be promoted.
- Japan single-date reconstruction is easy for the encoder MLP and still only a
  coverage/OOD probe. It is not transfer evidence.
- Convexity diagnostics remain frequent, so no no-arbitrage claim is supported.

## Evidence Gate

Do not promote a result to `paper_candidate` unless the run records:

- larger and timestamp-clean source manifests;
- fixed train/validation/test/OOD splits;
- masked reconstruction headline metrics;
- credible non-neural baselines trained without target leakage;
- at least three seeds;
- at least 1,000 usable U.S. surfaces and 20 Japan OOD observation dates;
- liquidity-stratified metrics;
- normalized price metrics with rate/dividend assumptions;
- no-arbitrage diagnostics that do not worsen materially out of sample.
