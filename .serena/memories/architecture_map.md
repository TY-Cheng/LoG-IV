# LoG-IV Architecture Map

Top-level layout:
- `src/log_iv/`: Python source.
- `tests/`: pytest test suite.
- `docs/`: MkDocs Material documentation source.
- `config/default.toml`: default project/vendor/graph config.
- `data/bronze`, `data/silver`, `data/gold`: local ignored data layers.
- `reports/`: local ignored benchmark/report outputs.
- `justfile`: canonical workflow entrypoints.

Core modules and responsibilities:
- `schema.py`: Pydantic `OptionQuote` and graph data structures. `OptionQuote` normalizes option types and exposes derived properties like `mid`, `spread`, `tenor_days`, `tenor_years`, and `log_moneyness`.
- `graph.py`: builds option-surface graphs with strike-neighbor, maturity-neighbor, and liquidity-similarity edges; includes liquidity scoring and graph summaries.
- `encoder.py`: token feature extraction, Fourier features, residual MLP encoder, and `OptionTokenEncoder`.
- `gnn.py`: `LiquidityGraphOperator`, PyG hetero-data construction, and incoming-edge-weight gate.
- `regularizer.py`: older regularizer helpers plus smoothness; benchmark claims should use decoded-price regularizer path in `train.py`, not embedding proxy regularizers.
- `synthetic.py`: SVI/SSVI-inspired synthetic surface generation and synthetic quote conversion.
- `data_fetch.py`: source probes, Massive/J-Quants ingestion, U.S. flat-file bronze cache, IV inversion from price rows, silver writes, dedupe, gates, and reports.
- `train.py`: `TrainingConfig`, `SurfaceData`, split preparation, masking, `OptionTokenGNN`, train/validate loops, benchmark artifacts, train-only and within-surface baselines, price diagnostics, sampled no-arb diagnostics.
- `cli.py`: CLI commands and benchmark orchestration; graph cache lives under `data/gold/graph_cache`.
- `config.py`: environment-backed project settings and key-file status.

Benchmark flow:
1. `data-expansion` builds bronze/silver data.
2. `benchmark-protocol` loads silver rows into OptionQuote surfaces, using gold graph cache when available.
3. `prepare_registered_splits` creates deterministic temporal/ticker/random split manifests.
4. Masked reconstruction surfaces remove masked quote-derived inputs while retaining targets for loss/metrics.
5. Models train on train surfaces and validate/test on fixed splits.
6. Run artifacts include metrics, predictions, baselines, diagnostics, manifest, split manifest, and README.
7. `benchmark_summary.csv` aggregates seed/variant rows when the full matrix finishes.

Current benchmark default for local development:
- `just benchmark-a1` runs stratified masked reconstruction with 3 seeds, 4 variants, 20 epochs, max 250 nodes per surface, fast baselines, and sampled-surface no-arb diagnostics.
- Full paper evidence should override epochs upward and eventually add stronger baseline ladders.

Recent implementation detail to preserve:
- no-arb diagnostics default to `sampled_surface`, `val,test`, max 100 surfaces per split, preserving complete surface/date graphs instead of row-level sampling.
- baseline preset `fast` skips raw SVI; `full` includes raw SVI with timeout/maxiter accounting.