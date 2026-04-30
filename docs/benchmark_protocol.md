# Benchmark Protocol

This page is the canonical contract for LoG-IV evaluation. It defines the task,
splits, baselines, leakage controls, regularizer semantics, and promotion gates.

## Entrypoint

The credible benchmark entrypoint is:

```bash
PYTHONPATH=src uv run python -m log_iv.cli benchmark-protocol \
  --us-data data/silver/option_quotes/us_option_quotes_expanded.parquet \
  --jp-data data/silver/option_quotes/jp_option_quotes_expanded.parquet
```

The default protocol uses:

- `task=masked_reconstruction`;
- `split_mode=temporal`;
- `seeds=1,2,3`;
- `min_us_surfaces=1000`;
- `min_jp_dates=20`.

Use smaller `--max-us-surfaces`, `--max-jp-surfaces`, and
`--max-nodes-per-surface` caps only for pipeline smoke checks. Capped or
single-seed runs are diagnostic, not paper evidence.

## Data Acceptance

The benchmark may run only when the data gate passes:

| Market | Gate | Current status |
| --- | --- | --- |
| U.S. | At least 1,000 usable `(underlying, observation_date)` surfaces after `min_nodes_per_surface=20` | Pass: 1,240 usable surfaces in the current expanded silver table. |
| Japan | At least 20 usable option observation dates for OOD probing | Pass: 31 usable dates in the current expanded silver table. |

If the gate fails, the CLI writes a data-expansion report and exits before
training. This is intended behavior, not a training crash.

## Task

The benchmark task is masked reconstruction.

For each surface:

1. Deterministically choose masked target nodes from option token identity and
   seed.
2. Remove IV, bid, and ask from the masked input nodes.
3. Preserve the full target quotes only for loss, metrics, and diagnostics.
4. Report headline metrics on masked nodes.

Observed reconstruction and next-day forecasting remain useful diagnostics, but
they are not the first paper-facing task.

## Splits

Supported split modes:

- `temporal`: train, validation, and test are separated by observation date.
- `ticker_holdout`: selected underlyings are withheld from training.
- `temporal_ticker_holdout`: temporal split plus held-out ticker evaluation.
- `random`: engineering diagnostic only.

The default credible split is `temporal`. Ticker-holdout and
temporal-ticker-holdout runs are robustness or transfer checks. Random splits
should not be used for paper-facing claims.

Every run writes `splits.json` so the train, validation, test, masked count, and
held-out ticker metadata are auditable.

## Baselines

Paper-facing baselines must be fit only on training surfaces:

- `train_mean_iv_global`;
- `train_mean_iv_by_underlying`;
- `train_mean_iv_by_moneyness_tenor_bucket`;
- `train_knn_moneyness_tenor`.

The moneyness-tenor kNN baseline is the current credibility floor. A GNN result
that beats only the global mean is not enough for a LoG claim.

Leave-one-out baselines are written separately to
`diagnostic_leakage_prone_baselines.csv`. They are useful for debugging but must
not be used as benchmark comparisons because they can see evaluation-surface
targets.

## Regularization

Benchmark runs must not use embedding-distance or embedding-norm proxies as
paper-facing no-arbitrage penalties.

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
- `diagnostics_no_arbitrage.json`.

The aggregate command writes `benchmark_summary.csv` with seed-level means and
standard deviations.

## Promotion Rule

A result can be discussed as paper-candidate evidence only when:

- the U.S. and Japan data gates pass;
- fixed non-random splits are recorded;
- headline metrics are masked-node metrics;
- train-only baselines are present and stronger leakage-prone baselines are
  separated;
- at least three seeds are run;
- training length is long enough for model comparison, not only a smoke check;
- the graph model beats the credible kNN/interpolation floor;
- decoded-price and no-arbitrage diagnostics do not undermine the scalar metric.

Until then, results should be labeled diagnostic.
