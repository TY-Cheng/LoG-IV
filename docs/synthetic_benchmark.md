# Synthetic-LoG-IV

Synthetic-LoG-IV is the public reproducibility scaffold for LoG-IV. Vendor raw
option data cannot be redistributed, so the synthetic benchmark provides a
controlled, auditable option-surface graph generator for leakage tests, graph
shift stress tests, and method ablations.

## Generator Contract

The current generator lives in `src/log_iv/synthetic.py` and is an extension of
the existing SSVI path, not a separate synthetic-only codebase. The v0 contract
is:

- SSVI clean surfaces with underlying-specific AR(1) temporal drift;
- liquidity AR(1) drift for spread, volume, and open-interest regimes;
- optional rare local total-variance bumps for event-style stress tests;
- heteroscedastic observation noise with quote-level oracle noise metadata;
- stable logical dataset hashing over sorted, rounded rows rather than Parquet
  bytes;
- oracle fields prefixed with `oracle_`.

Oracle fields are diagnostics only. Graph builders and model inputs must not use
columns matching `oracle_*`.

## Diagnostics

Synthetic clean surfaces expose oracle diagnostics for:

- calendar total-variance monotonicity;
- call butterfly / strike convexity;
- call vertical-spread monotonicity.

Observed noisy quotes may violate these diagnostics; that is intentional. The
clean oracle surface should remain the no-arbitrage reference, while noisy
observations test whether a model can balance quote fit against surface
geometry.

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

Synthetic-LoG-IV supports graph-learning stress tests and public reproducibility.
It does not claim to fully simulate market dynamics. Real-data claims still come
from fixed-split U.S. and Japan benchmark artifacts.
