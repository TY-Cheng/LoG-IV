# Code Style and Conventions

Language/style:
- Python `>=3.11,<3.14` per `pyproject.toml`; recent WSL verification used
  CPython 3.13.13 via `/home/tycheng/.venvs/log-iv`.
- Use type hints. `mypy` is strict globally, with explicit module overrides for
  heavy ML/data modules.
- Ruff line length is 100; lint selects E, F, I, UP, B, SIM.
- Prefer dataclasses for internal config/metrics containers and Pydantic for
  external schema validation (`OptionQuote`).
- Use clear, explicit names for benchmark/pipeline policy fields, e.g.
  `visible_context_policy`, `uses_masked_quote_features`, `baseline_preset`,
  `no_arb_diagnostics_mode`.
- Keep comments sparse and useful; avoid narrating obvious assignments.

Repository workflow conventions:
- This is a non-package uv project (`[tool.uv] package = false`); use
  `PYTHONPATH=src` for CLI/module commands.
- `.env` is local and ignored. It should contain `UV_PROJECT_ENVIRONMENT`
  pointing outside the repo. On this machine it also sets `UV_BIN`.
- Tracked repo defaults should stay portable. Concrete device paths belong in
  ignored `.env`; tracked defaults should use relative paths such as `data` and
  `reports`.
- Secrets are file-only. Code/docs should not depend on direct raw key env vars
  such as `MASSIVE_API_KEY` or `JQUANTS_API_KEY`; use key-file vars.
- `data/`, `reports/`, `site/`, caches, and local envs are ignored by git.
- Avoid committing generated data, reports, site output, or vendor payloads.
- `just check` forces pytest temp directories to `/tmp` because inherited
  Windows `TMP`/`TEMP` paths under WSL can break pytest capture.

Data conventions:
- Bronze = raw/lightly typed vendor payloads and manifests.
- Silver = normalized option quote tables and canonical rows.
- Gold = graph/task caches.
- Reports = local run outputs, figures, summaries, manifests.
- Canonical option token identity:
  `(market, underlying, observation_date, expiry, strike, option_type)`.
- `OptionQuote` constructor fields are limited; derived fields such as
  mid/spread/tenor/log_moneyness are properties/downstream features.

Benchmark/leakage conventions:
- Paper-facing Protocol A uses masked reconstruction with fixed temporal/ticker
  splits.
- Masked query nodes must not carry bid, ask, mid, IV, spread, quote-derived
  liquidity score, or IV-derived Greeks.
- Visible context aggregates must be computed after masking and only from
  visible nodes.
- Train-only baselines and eval-visible-only baselines must remain clearly
  separated.
- Leakage-prone diagnostics go to `diagnostic_leakage_prone_baselines.csv`, not
  paper tables.
- Raw SVI is P0 only in `full` baseline preset; constrained SVI/SSVI and larger
  baselines are P1 until the data/protocol are stable.
- no-arb diagnostics are default sampled at complete surface/date level for
  local A1 runs; do not row-sample no-arb diagnostics.

Model/regularization conventions:
- Benchmark-facing no-arb training penalties should be decoded Black-forward
  calendar/convexity terms, not embedding norm/proxy losses.
- Put-call parity remains diagnostic until rates/dividends/forwards/style
  assumptions are explicit.
- On the current WSL machine, CUDA is available for the RTX 3060. The GNN path
  is still limited by many small per-surface passes and Python/CPU overhead, so
  GPU speedups are not linear.

Docs/results conventions:
- Docs site is the canonical textual front door.
- `docs/results_snapshot.md` is the Results and Discussion snapshot; exclude
  smoke-only checks and label each result as accepted, incomplete,
  candidate-selection, or pending.
- `docs/paper_plan.md` is the paper-style plan with Introduction, Literature
  Review, Materials and Methods, preprocessing Mermaid chart, Expected
  Experiments, Results Reporting Plan, Discussion Plan, Reviewer Risks,
  Immediate Execution, and Venue Positioning.
- When updating docs, reconcile live run paths/counts with current artifacts
  rather than copying stale numbers.
- Legacy/preliminary result directories should live under
  `reports/archive/<yyyymmdd>-legacy/` once they are no longer current
  execution targets.
- Keep benchmark artifact paths short and Windows-friendly. Use compact family
  names such as `a1-e20-lc` and per-run names such as
  `b-us-gdcc-mr-t-e20-s1`; store full variant/task/split/mask metadata in
  manifests, CSV summaries, and docs instead of directory names.
