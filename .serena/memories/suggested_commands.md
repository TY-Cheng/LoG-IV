# Suggested Commands

Always run from the WSL repo root:

```bash
cd /home/tycheng/projects/LoG-IV
```

From Codex/PowerShell, wrap commands like this:

```powershell
wsl -d Ubuntu --cd /home/tycheng/projects/LoG-IV -- bash -lc "git status --short"
```

Use the repo `.env` / external uv environment contract:

- Preferred front door: `just ...`
- Direct CLI pattern:
  `PYTHONPATH=src ${UV_BIN:-uv} run --env-file .env python -m log_iv.cli ...`
- Do not use bare `uv run` without `.env`; it can create a repo-local `.venv`.

Discover commands:

```bash
just --list
```

Full local gate:

```bash
just check
```

`just check` sets pytest temp directories to `/tmp`; this avoids pytest capture
failures when WSL inherits Windows `TMP`/`TEMP` under `/mnt/c/...`.

Targeted docs check:

```bash
${UV_BIN:-uv} run --env-file .env mkdocs build --strict
```

Format/fix:

```bash
just fix
```

Status and probes:

```bash
PYTHONPATH=src ${UV_BIN:-uv} run --env-file .env python -m log_iv.cli status
PYTHONPATH=src ${UV_BIN:-uv} run --env-file .env python -m log_iv.cli source-probe all auto
PYTHONPATH=src ${UV_BIN:-uv} run --env-file .env python -m log_iv.cli toy-graph
```

Docs server:

```bash
just docs
```

Data expansion:

```bash
just data-v1-us
just data-jp
```

Current main benchmark front door:

```bash
just benchmark-a1
```

Current defaults: 20 epochs, 3 seeds, `stratified` mask, fast baselines,
sampled-surface no-arbitrage diagnostics.

Useful benchmark overrides:

```bash
just benchmark-a1 epochs=50
just benchmark-a1 mask=liquidity_correlated
just benchmark-a1 mask=block_wing
just benchmark-a1 no_arb_surfaces=200
```

Restore/rerun the missing A1 stratified artifact before final claims:

```bash
just benchmark-a1 mask=stratified out=reports/runs/a1
```

Promoted top2 runs:

```bash
just benchmark-a1 mask=liquidity_correlated seeds=1 epochs=20 variant_suite=anchor_proxy variants=lagos_liq_feature_only,gnn_decoded_calendar_convexity out=reports/runs/a1-e20 baseline_preset=fast no_arb_surfaces=50 skip_ood=true
just benchmark-a1 mask=block_wing seeds=1 epochs=20 variant_suite=anchor_proxy variants=lagos_liq_feature_only,gnn_decoded_calendar_convexity out=reports/runs/a1-e20 baseline_preset=fast no_arb_surfaces=50 skip_ood=true
```

Raw SVI/full baseline accounting:

```bash
just benchmark-a1 mask=stratified baseline_preset=full out=reports/runs/a1-full
```

Artifact checks:

```bash
find reports/runs -maxdepth 2 -name benchmark_summary.csv -print | sort
find reports/runs -maxdepth 3 -name metrics_summary.json -print | sort
find reports/archive -maxdepth 3 -name benchmark_summary.csv -print | sort
git status --short
rg "lagos_liq_feature_only|gnn_decoded_calendar_convexity|liquidity_correlated|block_wing" docs .serena/memories
```

Current caveats:

- `docs/results_snapshot.md` records an A1 stratified multi-seed result under
  `reports/runs/a1-str/`; this directory is currently missing
  locally.
- Legacy/preliminary run directories were archived under
  `reports/archive/20260511-legacy/`.
- `reports/runs/a1-e20-lc/`
  is incomplete and should be rerun or completed before being cited.
