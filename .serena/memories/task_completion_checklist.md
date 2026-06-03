# Task Completion Checklist

Before editing:
- Check `git status --short`; do not revert unrelated user changes.
- Confirm the command should be run from the repo root.
- Use `just` or `PYTHONPATH=src ${UV_BIN:-uv} run --env-file .env ...` so the
  external `UV_PROJECT_ENVIRONMENT` contract is honored.
- Avoid exposing secrets; `.env` can be inspected only for variable names, not
  values.

After code changes:
- Run focused checks first for touched files/tests.
- Typical focused verification for CLI/train work:
```bash
PYTHONPATH=src ${UV_BIN:-uv} run --env-file .env ruff check src/log_iv/train.py src/log_iv/cli.py tests/test_train_artifacts.py tests/test_cli.py
PYTHONPATH=src ${UV_BIN:-uv} run --env-file .env mypy src/log_iv/train.py src/log_iv/cli.py
PYTHONPATH=src ${UV_BIN:-uv} run --env-file .env pytest --no-cov tests/test_train_artifacts.py tests/test_cli.py
```
- For docs changes, run:
```bash
${UV_BIN:-uv} run --env-file .env mkdocs build --strict
```
- For broad repo confidence, run:
```bash
just check
```

After benchmark/data workflow changes:
- Dry-run `just` recipes when possible:
```bash
just --dry-run benchmark-a1
```
- Run a small capped smoke before long benchmark runs.
- Inspect generated artifacts under `reports/runs/...` and verify
  `metrics_summary.json`, `baselines_summary.csv`,
  `diagnostics_no_arbitrage.json`, and `postprocess_summary.json` when
  applicable.
- If `reports/` was deleted or a run family was archived, expect no aggregate
  summary in `reports/runs/...` until the benchmark completes again.
- Archive legacy/preliminary runs under `reports/archive/<yyyymmdd>-legacy/`
  rather than mixing them with current run targets.

When reporting results:
- State whether evidence is diagnostic, candidate-selection, paper-candidate,
  or incomplete.
- Use concrete artifact paths and counts.
- Do not claim GNN beats baseline from single-seed partial runs unless labeled
  as preliminary or candidate-selection evidence.
- Treat the archived 2026-05-07 top4 harder-mask results as
  candidate-selection evidence, not final manuscript evidence.
- Before citing A1 stratified as a current artifact, verify or restore
  `reports/runs/a1-str/`.
- For LoG/paper readiness, require data gates, fixed splits,
  masked-node metrics, train-only and within-surface baselines, multi-seed
  results, and documented diagnostics.

Before final response:
- Mention tests/commands actually run.
- Mention anything not run or still blocked.
- Keep final concise and grounded in current artifacts.
