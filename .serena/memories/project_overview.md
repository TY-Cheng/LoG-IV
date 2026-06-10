# LoG-IV Project Overview

LoG-IV is a Python research repository for leakage-controlled graph learning on
irregular option-implied-volatility surfaces.

The central research question is whether graph or token models can reconstruct
sparse, irregular option-implied-volatility surfaces under leakage-controlled
masking and liquidity-dependent observation noise more accurately than
train-only and within-surface interpolation or surface-fitting baselines.

## Current Paper-Facing Direction

- First credible task: Protocol A masked reconstruction.
- Main claim type: graph/geometric ML benchmark and modeling evidence.
- Not in scope for first paper: trading alpha, hedging, PnL, execution,
  portfolio construction, or live risk management.
- Japan is an out-of-distribution evaluation setting, not the core empirical
  claim.
- No-arbitrage outputs are diagnostics unless rates, dividends, forwards, and
  option-style assumptions are made explicit.

## Repository Root And Portability

Current WSL root:

```text
/home/tycheng/projects/LoG-IV
```

The tracked repo should stay portable across devices. Machine-specific paths
belong in the ignored `.env`. Tracked examples/defaults should use relative
paths like `data`, `reports`, and `uv` unless a command explicitly documents a
local override.

From Codex/PowerShell, use:

```bash
wsl -d Ubuntu --cd /home/tycheng/projects/LoG-IV -- bash -lc "..."
```

## Tech Stack

- Python requirement: `>=3.11,<3.14`.
- uv project with `[tool.uv] package = false`; no build-system package install
  is expected.
- Main libraries: numpy, pandas, polars, pyarrow, scipy, pydantic, httpx,
  python-dotenv, torch, torch-geometric.
- Dev/docs tools: ruff, mypy, pytest/coverage, mkdocs-material.

## Workflow Rule

Use `just` or `PYTHONPATH=src uv run --env-file .env ...` so `.env` supplies
`UV_PROJECT_ENVIRONMENT`, `UV_BIN`, local data paths, and key-file paths.
Repo-local `.venv/` is workflow drift and should not be recreated. If it
appears, delete it after confirming it is exactly
`/home/tycheng/projects/LoG-IV/.venv`.

`just check` passed on 2026-05-11 after forcing pytest temp paths to `/tmp` to
avoid WSL/Windows temp-directory capture issues.

## Current Data State

U.S. expanded option silver:

- `data/silver/option_quotes/us_option_quotes_expanded.parquet`
- `data_stage=data_v1`
- 2,979,716 rows
- 40 underlyings
- 62 observation dates
- 2,480 usable surfaces with at least 20 nodes

Japan expanded option silver:

- `data/silver/option_quotes/jp_option_quotes_expanded.parquet`
- `data_stage=data_v0`
- 557,982 rows
- 249 underlyings
- 32 observation dates and 32 usable observation dates
- 557,982 IV-usable rows
- 7,488 usable surfaces with at least 20 nodes

Synthetic-LoG-IV is for reproducibility and controlled diagnostics, not a
replacement for real U.S. benchmark evidence.

## Current Evidence State

- `docs/results_snapshot.md` is now the live Results And Discussion chapter; it
  records current data gates, planned-experiment status, all current experiment
  outcomes, tables/figure register, interpretations, missing evidence, and the
  fact that local artifact `reports/runs/a1-str/` is missing.
- `docs/paper_plan.md` is the paper-style manuscript skeleton with Introduction
  (motivation, literature positioning/existing results, research gap, research
  question, contributions, claim boundary), Detailed Literature Review,
  Materials and Methods, Expected Experiments, Results Reporting Plan,
  Discussion Plan, Reviewer Risk Register, Immediate Execution Plan, and Venue
  Positioning.
- Legacy/preliminary run directories were archived under
  `reports/archive/20260511-legacy/`.
- The only current non-archived run family is the incomplete
  `reports/runs/a1-e20-lc/`.
- Current evidence is not final manuscript evidence: full SVI accounting,
  longer/multi-seed harder-mask runs, OOD/ticker-holdout, reliability analysis,
  and final figures/tables are still pending.

## Docs Front Door

- `README.md`: short overview.
- `docs/index.md`: docs home.
- `docs/results_snapshot.md`: Results and Discussion snapshot with current
  outcomes, interpretations, missing evidence, and next runs.
- `docs/paper_plan.md`: paper plan with Introduction, Literature Review,
  Materials and Methods, Expected Experiments, and execution steps.
- `docs/benchmark_protocol.md`: benchmark contract.
- `docs/data.md`: source/cache/data contract.
- `docs/graph.md`: graph construction and feature contract.

Always re-check `reports/runs`, `reports/archive`, `docs/results_snapshot.md`,
and current benchmark commands before quoting readiness or results.
