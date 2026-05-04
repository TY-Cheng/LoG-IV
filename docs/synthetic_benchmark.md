# Synthetic Benchmark

Vendor raw option data cannot be redistributed. The synthetic benchmark provides
a controlled option-surface graph generator for leakage tests, robustness tests,
and model ablations.

## Generator Contract

The generator lives in `src/log_iv/synthetic.py` and extends the existing SSVI
path rather than creating a separate synthetic-only codebase. The v0 contract is:

- SSVI clean surfaces with underlying-specific AR(1) temporal drift;
- liquidity AR(1) drift for spread, volume, and open-interest regimes;
- optional rare local total-variance bumps for event-style stress tests;
- heteroscedastic observation noise with quote-level diagnostic noise metadata;
- stable logical dataset hashing over sorted, rounded rows rather than Parquet
  bytes;
  fields prefixed with `oracle_`.

Fields prefixed with `oracle_` are diagnostics only. Graph builders and model
inputs must not use them.

## Diagnostics

Synthetic clean surfaces expose diagnostics for:

- calendar total-variance monotonicity;
- call butterfly / strike convexity;
- call vertical-spread monotonicity.

Observed noisy quotes may violate these diagnostics. The clean latent surface is
the reference surface, while noisy observations test whether a model can balance
quote fit against surface geometry.

## Hashing Policy

The reproducibility hash is a logical hash:

1. flatten surfaces into rows;
2. sort by surface id, option type, expiry, and strike;
3. keep canonical columns only;
4. round floats to stable precision;
5. hash the CSV-like logical payload.

This avoids false hash changes from Parquet metadata, compression settings, or
row-group layout.

## Claim Boundary

The synthetic benchmark supports graph-learning stress tests and reproducible
diagnostics. It does not claim to simulate market dynamics. Real-data claims
come from fixed-split U.S. and Japan benchmark artifacts.
