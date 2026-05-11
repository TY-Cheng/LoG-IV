# Data Design and Contract

This page is the canonical contract for data sources, option identifiers,
timestamp semantics, cache layers, local environment fields, IV-inversion
assumptions, and out-of-distribution data gates.

## Source Roles

| Source | Role | Initial status | Notes |
| --- | --- | --- | --- |
| Massive U.S. equities | Underlying prices, returns, realized-volatility controls | Flat-file stock daily closes work locally | Current IV inversion uses same-date `us_stocks_sip/day_aggs_v1` closes as spot/forward proxies. |
| Massive U.S. options | U.S. option chains, quotes, trades, or aggregates depending entitlement | OPRA daily flat-file expansion works locally | Current expanded U.S. silver has 2,480 usable surfaces after filtering. |
| J-Quants equities | Japanese underlying prices and corporate-action context | V2 API-key probe works locally | Needed for Japan out-of-distribution features and realized-volatility targets. |
| J-Quants derivatives/options | Japanese listed option or derivatives chains where available | V2 date-loop expansion works locally | Current expanded Japan silver has 31 usable observation dates. |
| Public calendars | U.S. and Japan trading sessions and holidays | Core public dependency | Needed for date alignment and cutoff validity. |
| Public rates/dividends | Risk-free curves, dividend proxies, and robustness controls | Candidate public dependency | Needed before inferred-IV and no-arbitrage diagnostics become paper claims. |

## Option Token Identity

The canonical token key is:

```text
(market, underlying, observation_date, expiry, strike, option_type)
```

Required fields:

- `market`: `US` or `JP` at initialization.
- `underlying`: normalized root or ticker.
- `observation_date`: local market date of the option observation.
- `expiry`: contract expiry date.
- `strike`: split-adjusted strike where the source supports it.
- `option_type`: call or put, stored as `C` or `P`.

Recommended raw identifiers:

- vendor contract symbol;
- OCC-style symbol for U.S. contracts when available;
- exchange or J-Quants contract code for Japanese contracts;
- raw root, expiry code, strike code, and call-put flag.

## Quote and Liquidity Fields

Core quote fields:

- bid, ask, mid, last, settlement, or vendor aggregate price;
- implied volatility if supplied by the vendor;
- volume and open interest;
- quote timestamp or aggregate bar timestamp;
- vendor availability timestamp or conservative availability label.

Canonical constructor fields are limited to:

```text
market, underlying, observation_date, expiry, strike, option_type,
bid, ask, implied_vol, volume, open_interest, forward,
underlying_price, vendor_symbol
```

Adapters must not pass derived fields such as `mid`, `spread`, `tenor_days`, or
`log_moneyness` into `OptionQuote`; those are schema properties or downstream
features.

Derived fields:

- tenor in calendar days and years;
- log moneyness using forward where available, otherwise spot proxy;
- bid-ask spread and relative spread;
- liquidity score;
- missingness and stale-quote flags.

## Timestamp Policy

Every modeling row should distinguish:

```text
observation_ts <= vendor_available_ts <= model_cutoff_ts < target_ts
```

If a vendor does not provide a precise availability timestamp, use a conservative
source-specific availability rule and record the rule in the run manifest. Do not
mix intraday and end-of-day semantics without explicit cutoff fields.

## Cache Layout

```text
/Volumes/ExternalSSD/data/LoG-IV/bronze/  raw or lightly typed vendor payloads
/Volumes/ExternalSSD/data/LoG-IV/silver/  normalized option tokens and calendars
/Volumes/ExternalSSD/data/LoG-IV/gold/    modeling-ready graph batches and tasks
reports/                                  local reports and run manifests
```

Benchmark graph caches live under
`/Volumes/ExternalSSD/data/LoG-IV/gold/graph_cache/`. They are keyed by the
source silver table fingerprint plus graph-loading controls such as
`min_nodes_per_surface`, `max_nodes_per_surface`, and `max_surfaces`; use
`--refresh-graph-cache` when intentionally rebuilding them.

The OneDrive checkout should contain source and documentation only. Do not keep
a repo-local `data` directory or `data` symlink in the checkout.

## Environment Contract

Shareable defaults live in `.env.example`; local secrets live in `.env`.

```bash
UV_PROJECT_ENVIRONMENT="${HOME}/.venvs/log-iv"
PROJECT_NAME="log-iv"
LOG_LEVEL="INFO"

DATA_DIR="/Volumes/ExternalSSD/data/LoG-IV"
BRONZE_DATA_DIR="/Volumes/ExternalSSD/data/LoG-IV/bronze"
SILVER_DATA_DIR="/Volumes/ExternalSSD/data/LoG-IV/silver"
GOLD_DATA_DIR="/Volumes/ExternalSSD/data/LoG-IV/gold"
REPORTS_DIR="reports"

MASSIVE_API_KEY_FILE=""
MASSIVE_FLAT_FILE_KEY_FILE=""
MASSIVE_BASE_URL="https://api.massive.com"
MASSIVE_FLAT_FILE_ENDPOINT_URL="https://files.massive.com"
MASSIVE_FLAT_FILE_BUCKET="flatfiles"
MASSIVE_OPTIONS_FLAT_FILE_TEMPLATE="s3://flatfiles/us_options_opra/day_aggs_v1/{year}/{month}/{date}.csv.gz"
MASSIVE_OPTIONS_FLAT_FILE_DATASET="day_aggs_v1"
MASSIVE_UNDERLYING_FLAT_FILE_TEMPLATE="s3://flatfiles/us_stocks_sip/day_aggs_v1/{year}/{month}/{date}.csv.gz"

JQUANTS_API_KEY_FILE=""
JQUANTS_MAILADDRESS_FILE=""
JQUANTS_PASSWORD_FILE=""
JQUANTS_REFRESH_TOKEN_FILE=""
JQUANTS_API_BASE_URL="https://api.jquants.com/v2"
JQUANTS_MAX_OPTION_DATES="35"

FRED_API_KEY_FILE=""
FRED_BASE_URL="https://api.stlouisfed.org/fred"
```

The first ingestion milestone now writes raw payloads and manifests under
`data/bronze/...` and canonical tables under `data/silver/...`; these paths are
ignored by git. Live source probes are part of `just check`. Credentials are
file-only: direct secret values such as `MASSIVE_API_KEY` or `JQUANTS_API_KEY`
are ignored by the code path. Under J-Quants V2 the probe reads
`JQUANTS_API_KEY_FILE` and sends `x-api-key`.

## Expansion Contract

U.S. historical option panels should come from the Massive/Polygon options
flat-file path rather than the REST snapshot sample path. The code expects
`MASSIVE_OPTIONS_FLAT_FILE_TEMPLATE` to be a local, `file://`, HTTPS, or `s3://`
template with `{date}`, `{yyyymmdd}`, `{year}`, `{month}`, `{day}`, or
`{dataset}` placeholders and writes per-date manifests under
`data/bronze/massive/flat_options/`. See the official
[options flat files overview](https://polygon.io/docs/flat-files/options/overview).
Expanded canonical rows are deduplicated and written to
`data/silver/option_quotes/us_option_quotes_expanded.parquet` when rows are
available. The IV benchmark gate counts only rows with bid/ask and
`implied_vol`; Massive day-aggregate rows without vendor IV are promoted to IV
targets only when the adapter can solve the inversion, and unsolved rows remain
price rows. The first IV-inversion path uses same-date
`us_stocks_sip/day_aggs_v1` closes as spot/forward proxies and a zero-rate,
zero-dividend Black-forward bisection inversion. These inferred IVs are usable
engineering targets, not vendor-IV claims.

J-Quants options ingestion loops the V2 derivatives endpoints over bounded
business dates and records successful, empty, and failed dates in bronze
manifests. See the official
[J-Quants options API reference](https://jpx.gitbook.io/j-quants-en/api-reference/options).
Expanded canonical rows are deduplicated and written to
`data/silver/option_quotes/jp_option_quotes_expanded.parquet` when rows are
available. `data-expansion --market jp` can rebuild this table from existing
bronze payloads without re-calling the vendor.

## Current Expanded Silver State

The current expanded U.S. table is:

```text
data/silver/option_quotes/us_option_quotes_expanded.parquet
```

It covers 40 U.S. underlyings from 2026-02-02 to 2026-04-30 and records:

- 2,979,716 deduplicated rows;
- 62 usable observation dates;
- 2,480 usable `(underlying, observation_date)` surfaces under
  `min_nodes_per_surface=20`;
- `iv_source=option_mid_price_with_underlying_daily_close`;
- `iv_method=black_forward_bisection_zero_rate_zero_dividend`.

This table is frozen as `data_v1`. It is suitable for Protocol A development
and first multi-seed benchmark runs, but not yet for market-cycle temporal
claims.

## Data Version Ladder

The data-first benchmark ladder is:

| Version | Purpose | U.S. gate | Date policy | Ticker policy |
| --- | --- | --- | --- | --- |
| `data_v0` | Pipeline/protocol/baseline development | 1,000+ usable surfaces and 31+ usable dates | Current short window | Current 40-ticker universe |
| `data_v1` | First multi-seed Protocol A run | 2,400+ usable surfaces and 60+ usable dates | Extend to 2026-02-02 through 2026-04-30 | Same 40 tickers |
| `data_v2` | Paper-candidate table | 8,000+ usable surfaces and 126+ usable dates | Six to twelve months before the cutoff | Stable ticker universe before model selection |

The implementation records `data_stage`, `date_range`, `tickers`,
`ticker_count`, row counts, deduplication counts, IV-usable rows, usable
surfaces, and surface-size min/p50/max in each expanded silver manifest.

Use the same 40-ticker universe before expanding to a larger cross-section:

```bash
just data-v1-us
```

Run through `just` so the repo `.env` is loaded and `UV_PROJECT_ENVIRONMENT`
points to `${HOME}/.venvs/log-iv`; do not run bare `uv run ...` for benchmark
data work.

The current expanded Japan table is:

```text
data/silver/option_quotes/jp_option_quotes_expanded.parquet
```

It records:

- 531,280 deduplicated rows;
- 31 usable observation dates;
- 6,179 usable `(underlying, observation_date)` surfaces under
  `min_nodes_per_surface=20`.

## U.S. To Japan Evaluation Gate

Japan is an out-of-distribution evaluation setting for LoG-IV, not the single
point of failure for the U.S. benchmark.

Promote Japan option transfer only if:

- option-chain coverage supports enough dates, expiries, and strikes;
- liquidity fields are available or measurable;
- timestamp semantics are clear;
- a matched U.S. baseline and at least one non-graph baseline exist;
- evaluation splits are fixed before model selection.

Current gate status:

- U.S. expanded gate passes the default 2,400-surface requirement.
- Japan expanded gate passes the default 20-observation-date requirement.
- Japan should be labeled as out-of-distribution evaluation until
  market-specific normalization, baseline matching, and transfer/fine-tuning
  protocols are registered.
