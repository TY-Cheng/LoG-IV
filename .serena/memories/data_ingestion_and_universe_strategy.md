# Data Ingestion And Universe Strategy

Updated 2026-05-11 after live manifest verification.

## Current ingestion status

- U.S. expanded silver is present at
  `data/silver/option_quotes/us_option_quotes_expanded.parquet`.
- Japan expanded silver is present at
  `data/silver/option_quotes/jp_option_quotes_expanded.parquet`.
- Current U.S. state: 2,979,716 rows, 40 underlyings, 62 observation dates,
  62 usable observation dates, 2,867,637 IV-usable rows, and 2,480 usable
  surfaces with at least 20 nodes.
- Current Japan state: 557,982 rows, 249 underlyings, 32 observation dates,
  32 usable observation dates, 557,982 IV-usable rows, and 7,488 usable
  surfaces with at least 20 nodes.
- Rebuild U.S. with `just data-v1-us`; rebuild Japan with `just data-jp`.
- Monitor long ingestion or benchmark jobs with:
  - `du -sh data reports`
  - `find data/bronze/massive/flat_options -name 'payload_*.parquet' | wc -l`
  - `ps -ef | rg 'log_iv.cli data-expansion|log_iv.cli benchmark-protocol|just data|just benchmark'`

## Ticker universe strategy

The current 40-ticker U.S. universe in `justfile` is adequate for `data_v1` /
A1 masked-reconstruction development because 40 tickers times roughly 62
trading dates gives the intended 2,480 surface gate. It is intentionally liquid
and recognizable, with mega-cap single names plus ETFs and sector coverage.

It is not the final paper universe. It likely overweights liquid mega-caps/ETFs
and a short 2026 window. For `data_v2` or manuscript-level evidence, use a
pre-specified universe selection rule based only on training-period information,
such as:

- top option-liquidity names by train-period option volume/OI/dollar notional;
- a sector-balanced liquid universe;
- separate reporting for ETFs vs single names;
- liquidity-stratified panels or deciles, including low-liquidity/wide-spread
  stress panels.

Do not select tickers after seeing model results. Do not use only top-liquidity
names for the main claim; that would make the benchmark easier and weaken the
liquidity-dependent noise story. A top-liquidity subset is useful as a
stable/core panel, but the paper should also report liquidity buckets and
worst-bucket behavior.
