# LoG-IV

Benchmarking graph models for irregular option-implied volatility surfaces with
liquidity-dependent observation noise.

This repository supports a research project on option-surface learning. The
central modeling choice is to represent an option chain as an irregular set or
graph of contracts, rather than as a fixed strike-maturity image. The current
phase is to establish a leakage-controlled masked-reconstruction benchmark
before making substantive claims about graph models.

## Research Question

Can graph or token models that condition on quote reliability reconstruct
implied-volatility surfaces more accurately than pointwise and interpolation
baselines when the observed option chain is sparse, irregular, and affected by
liquidity-dependent noise?

The intended empirical ladder is:

- U.S. option-chain representation learning from Massive options and equities.
- Surface reconstruction, forecasting, and missing-quote robustness tasks.
- Liquidity-aware message passing, heteroscedastic reliability modeling, and
  decoded-price surface diagnostics.
- Japan option or equity evaluations using J-Quants and public data where field
  coverage is sufficient.

Japan is an out-of-distribution evaluation setting, not the main empirical
claim. If Japanese listed-option data are too sparse for option-graph transfer,
the fallback is to use Japanese equity realized-volatility or sector tail-risk
targets as downstream representation tests.

## Repository Status

Current status: expanded U.S. and Japan silver data gates pass, the fixed-split
masked-reconstruction benchmark protocol runs, and the latest A1 stratified
benchmark is a preliminary multi-seed result rather than final paper evidence.

Implemented now:

- LoG-IV project metadata and docs front door.
- Source, environment, graph, data, and benchmark protocol contracts.
- Minimal Python package under `src/log_iv`.
- A typed option-quote schema and graph builder.
- Credential-safe Massive and J-Quants source probes.
- Canonical `OptionQuote` adapters for Massive snapshots and J-Quants option rows.
- Liquidity-aware graph construction with input-order node IDs and bidirectional
  strike/maturity edges.
- Default ML dependency gate for `torch` and `torch-geometric`.
- Synthetic training artifacts under `reports/runs/` for engineering checks.
- First real-data matrix artifacts under `reports/runs/`, including full
  validation/test predictions, baseline summaries, normalized price diagnostics,
  and post-run no-arbitrage diagnostics.
- A silver-table U.S. to Japan out-of-distribution evaluation path with no
  synthetic fallback.
- Fixed-split `benchmark-protocol` for masked reconstruction with temporal,
  ticker-holdout, and combined split modes.
- Train-only baselines for benchmark artifacts; the old leave-one-out baselines
  are now written only as leakage-prone diagnostics.
- Decoded-price calendar and strike-convexity training penalties; embedding
  proxy put-call/convexity regularizers are not used for benchmark claims.
- Historical data expansion for U.S. options flat files and J-Quants options
  date loops.
- First inferred-IV path for Massive day aggregates using option mid prices and
  same-date U.S. stock daily closes.
- Expanded U.S. silver with 2,480 usable surfaces and expanded Japan silver with
  31 usable observation dates.

Not yet implemented:

- Rate, dividend, forward-curve, or corporate-action upgrades for inferred IVs.
- Accepted multi-mask benchmark results or manuscript-level empirical claims.
- Manuscript-level Japan transfer evidence. Japan is currently an
  out-of-distribution evaluation setting.

## Quick Start

This repo uses `uv` and `just`. The local virtual environment is controlled by
`.env`, which is intentionally ignored by git:

```bash
UV_PROJECT_ENVIRONMENT="${HOME}/.venvs/log-iv"
```

Typical local checks:

```bash
just check
```

`just check` is the single local gate: it syncs all extras, runs formatting and
lint checks, mypy, tests with coverage, strict docs build, status, source probes,
and a graph-construction check.

To serve the documentation site:

```bash
just docs
```

To fetch the first real-data MVP sample:

```bash
just check
just fetch-sample all 2026-02-02 2026-04-30
```

J-Quants V2 uses API-key authentication through the `x-api-key` header; the live
credential probe is now part of `just check`.

To reproduce recorded benchmark families, use the commands listed in
`docs/results_snapshot.md`. These are intentionally not all separate `just`
recipes; `just check` remains the main local verification gate.

The main benchmark entrypoint is:

```bash
just data-v1-us
just data-jp
just benchmark-a1 stratified
```

By default it requires masked reconstruction, temporal splits, three seeds, and
at least 2,400 usable U.S. surfaces across 60 U.S. observation dates. If the
local data do not meet that gate, it writes
`reports/runs/data_expansion_report.json` instead of promoting a run.
The data-expansion report records expanded silver paths, deduplication counts,
distinct dates, ticker universe, data-stage label, and IV-usable surface gates.
The current short-window expanded U.S. table is `data_v0`; the next target is
`data_v1`, using the same 40 tickers over 2026-02-02 through 2026-04-30.
Current Massive day aggregates provide price rows but no vendor IV, so the U.S.
adapter infers IV from same-date U.S. stock closes with a zero-rate,
zero-dividend Black-forward inversion. Treat these as inferred engineering
targets until rate/dividend assumptions are upgraded.

## Data Sources

Planned source roles:

- Massive: U.S. equities, underlying bars, and U.S. option chains.
- J-Quants: Japanese equities and derivatives/options where entitlement and
  endpoint coverage support research use.
- Public data: calendars, rates, dividends, corporate actions, market metadata,
  and robustness controls where licensing permits.

Credentials belong in `.env`; shareable defaults belong in `.env.example`.
Never commit vendor secrets or licensed raw payloads.

## Claim Boundaries

- This is a graph-learning benchmark and option-surface representation-learning
  project, not a trading strategy or execution system.
- A lower reconstruction or forecast loss is not sufficient for a finance claim
  unless liquidity, missingness, no-arbitrage diagnostics, and
  out-of-distribution behavior are reported together.
- Japan transfer results should be interpreted as evaluation under distribution
  shift, not as evidence of causal market leadership.
- Vendor entitlement probes and small fixtures are engineering checks, not
  empirical validation.

## Documentation

- `docs/results_snapshot.md`: current evidence ledger, data gates, benchmark
  results, and manuscript-readiness boundaries.
- `docs/paper_plan.md`: research plan, tasks, baselines, metrics, and
  claim boundaries.
- `docs/data.md`: source roles, option identifiers, timestamp policy, expanded
  silver artifacts, IV-inversion assumptions, and out-of-distribution data
  gates.
- `docs/benchmark_protocol.md`: fixed splits, masked reconstruction, train-only
  baselines, leakage controls, and promotion rules.
- `docs/graph.md`: node features, edge families, liquidity weighting, masking
  semantics, and no-arbitrage hooks.
- `docs/future_work.md`: deferred extensions that should stay outside the first
  paper.
