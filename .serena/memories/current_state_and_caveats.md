# Current State and Caveats

Updated from `/home/tycheng/projects/LoG-IV` on 2026-05-11 after live repo,
data, docs, result-artifact, and `just check` verification.

## Local Execution Status

- Project is activated in Serena as `LoG-IV`.
- Current WSL workspace root is `/home/tycheng/projects/LoG-IV`.
- The tracked repo should stay portable. Concrete machine paths belong in the
  ignored local `.env`; tracked defaults should use relative paths such as
  `data` and `reports`.
- This machine's `.env` currently maps data and reports to:
  - `/home/tycheng/projects/LoG-IV/data`
  - `/home/tycheng/projects/LoG-IV/reports`
- Use `just ...` as the preferred front door. `justfile` supports `UV_BIN`; on
  this machine `.env` points it to `/home/tycheng/.local/bin/uv`.
- `UV_PROJECT_ENVIRONMENT` should point outside the repo, currently
  `/home/tycheng/.venvs/log-iv`.
- Current `pyproject.toml` requires Python `>=3.11,<3.14`; recent verification
  used CPython 3.13.13.
- `just check` passed on 2026-05-11:
  - ruff format/check passed;
  - mypy passed;
  - pytest passed with 79 tests and 97.01% coverage;
  - mkdocs build passed;
  - status/source-probe/toy-graph passed.
- WSL may inherit Windows `TMP`/`TEMP` paths under `/mnt/c/...`; `just check`
  now forces pytest temp paths to `/tmp` to avoid pytest capture temp-file
  failures.

## Current Git/Docs State

- `main` and `origin/main` were aligned when checked: `HEAD...origin/main = 0 0`.
- Current uncommitted tracked edits are docs-oriented: paper-plan restructuring,
  Results and Discussion snapshot restructuring, and live data-count alignment.
- `.serena/` is currently visible as an untracked directory because `.gitignore`
  no longer ignores it.
- `docs/paper_plan.md` is a paper-style plan with Introduction, Literature
  Review, Materials and Methods including a preprocessing Mermaid chart,
  Expected Experiments, Results Reporting Plan, Discussion Plan, Reviewer Risks,
  Immediate Execution, and Venue Positioning.
- `docs/results_snapshot.md` is the Results and Discussion snapshot, but it
  still records an A1 stratified result under `reports/runs/a1-str/`; that
  artifact directory is currently missing locally.

## Research Question and Scope

LoG-IV asks whether graph or token models can reconstruct sparse, irregular
option-implied-volatility surfaces under leakage-controlled masking and
liquidity-dependent observation noise more accurately than train-only and
within-surface interpolation or surface-fitting baselines.

The first paper-facing task is Protocol A masked reconstruction. It is not a
trading, hedging, portfolio, alpha, execution, or live risk-management paper.
Japan is an out-of-distribution evaluation setting, not the core empirical
claim.

## Data State

Main U.S. expanded silver table:

- `data/silver/option_quotes/us_option_quotes_expanded.parquet`
- `data_stage=data_v1`
- 2,979,716 rows
- 40 underlyings
- 62 observation dates and 62 usable observation dates
- 2,867,637 IV-usable rows
- 2,480 usable `(underlying, observation_date)` surfaces with at least 20 nodes
- window: 2026-02-02 to 2026-04-30
- IV source/method: option mid price with same-date underlying daily close,
  Black-forward bisection, zero rate, zero dividend

Japan OOD expanded silver table:

- `data/silver/option_quotes/jp_option_quotes_expanded.parquet`
- `data_stage=data_v0`
- 557,982 rows
- 249 underlyings
- 32 observation dates and 32 usable observation dates
- 557,982 IV-usable rows
- 7,488 usable surfaces with at least 20 nodes
- window: 2026-03-16 to 2026-04-30

This is enough for Protocol A development and A1/data_v1 preliminary evidence,
but not enough for broad market-cycle, long-regime, or final paper-universe
claims.

## Evidence State

Current status: useful preliminary/candidate-selection evidence, not final
manuscript evidence.

Current non-archived result family in `reports/runs/` after cleanup:

- `a1-e20-lc`

This candidate directory is incomplete: it currently contains only
`b-us-gdcc-mr-t-e20-s1/best_model.pt`
and no top-level `benchmark_summary.csv`. Treat it as a partial/incomplete run,
not an accepted result.

Archived legacy/preliminary result families were moved to:

- `reports/archive/20260511-legacy/`

Archived directories:

- `p-all-e2-str`
- `p-la-e2-str`
- `p-lhf-e2-str`
- `p-llo-e2-str`
- `p-lre-e2-str`
- `p-lse-e2-str`
- `p-top4-e2-bw`
- `p-top4-e2-lc`
- `timing-str`

Important caveat: `docs/results_snapshot.md` records an A1 stratified
3-seed/20-epoch result under `reports/runs/a1-str/`, but that
directory is currently absent. Before final claims, restore that artifact or
rerun the A1 stratified benchmark so the recorded result is backed by local
artifacts.

## Current Candidate Decision

The archived 2-epoch harder-mask screen selected candidates, but it is
candidate-selection evidence only.

Promote for longer confirmation:

- `lagos_liq_feature_only`: strongest short-run masked-IV candidate across
  `stratified`, `liquidity_correlated`, and `block_wing`.
- `gnn_decoded_calendar_convexity`: best diagnostics candidate, with lower
  normalized price MAE and fewer sampled calendar/convexity/vertical violations
  than `lagos_liq_feature_only` under both harder masks.

Do not promote yet:

- `lagos_loss_only`: best on the stratified 2-epoch screen, but fails both
  train kNN and within-surface kNN under `liquidity_correlated`, and fails train
  kNN under `block_wing`.
- `gnn_liq`: competitive under `block_wing`, not under `liquidity_correlated`
  after two epochs.

## Models and Baselines

Candidate/own model families include:

- `encoder_mlp`
- `set_context_mlp`
- `gnn_no_liq`
- `gnn_liq`
- `gnn_decoded_calendar_convexity`
- `lagos_liq_feature_only`
- `lagos_scalar_gate`
- `lagos_loss_only`
- `lagos_attn_only`
- `lagos_hetero_full`
- `lagos_random_edges`
- `lagos_shuffled_edges`

Baselines include:

- train global/underlying/moneyness-tenor-bucket mean IV
- train kNN moneyness-tenor
- random uniform IV
- within-surface kNN
- within-surface local linear
- within-surface RBF
- raw/constrained/projected SVI/SSVI when `baseline_preset=full`

Related-work proxies exist but are not all complete paper evidence:

- Deep Smoothing / fixed-grid CNN proxy
- ODS-style continuous-coordinate operator proxy
- Hexagon-style heterogeneous attention proxy
- HyperIV proxy/adapter path
- ANP/CNP sparse-quote baselines

## Metrics and Diagnostics

Headline metrics:

- masked IV MAE
- masked IV RMSE
- masked p90 absolute error
- delta versus train-only kNN
- delta versus strongest relevant within-surface baseline

Supporting diagnostics:

- normalized decoded-price MAE/RMSE
- calendar, butterfly/convexity, vertical-spread, and put-call diagnostics
- liquidity, moneyness, tenor, underlying, option-type, and split buckets
- reliability calibration and predicted-precision diagnostics
- OOD degradation ratios for Japan or ticker holdout
- SVI failure, timeout, underidentified, and constraint/projected rates

## Readiness Answer

Not complete for a final paper. Current state is suitable for internal
presentation and planning, but final manuscript evidence still needs:

- restored or rerun A1 stratified artifacts;
- completed promoted top2 `liquidity_correlated` and `block_wing` runs;
- longer or multi-seed confirmation for harder masks;
- raw SVI/full baseline accounting;
- graph-necessity and reliability ablation analysis at candidate budget;
- ticker-holdout or temporal-ticker-holdout OOD evaluation;
- reliability/bucket/worst-bucket tables;
- error-versus-violation curves;
- final figures and manuscript-level tables.

## Next Execution Steps

1. Restore or rerun `reports/runs/a1-str/`.
2. Re-run the incomplete promoted top2 `liquidity_correlated` run to completion,
   or delete the partial directory and rerun cleanly:

   ```bash
   just benchmark-a1 mask=liquidity_correlated seeds=1 epochs=20 variant_suite=anchor_proxy variants=lagos_liq_feature_only,gnn_decoded_calendar_convexity out=reports/runs/a1-e20 baseline_preset=fast no_arb_surfaces=50 skip_ood=true
   ```
3. Run promoted top2 `block_wing`:

   ```bash
   just benchmark-a1 mask=block_wing seeds=1 epochs=20 variant_suite=anchor_proxy variants=lagos_liq_feature_only,gnn_decoded_calendar_convexity out=reports/runs/a1-e20 baseline_preset=fast no_arb_surfaces=50 skip_ood=true
   ```

4. Run raw SVI/full baseline accounting:

   ```bash
   just benchmark-a1 mask=stratified baseline_preset=full out=reports/runs/a1-full
   ```

5. If top2 remain stable, rerun selected masks with at least three seeds.
6. Update `docs/results_snapshot.md` after each completed benchmark family.
