set dotenv-load
set shell := ["zsh", "-cu"]

cli := "PYTHONPATH=src uv run --env-file .env python -m log_iv.cli"
us_tickers := "AAPL,AMD,AMZN,AVGO,BA,BAC,COST,CRM,CSCO,CVX,DIA,DIS,F,GE,GOOG,GOOGL,HD,IBM,INTC,IWM,JNJ,JPM,KO,LLY,MA,META,MSFT,NFLX,NVDA,ORCL,PEP,PG,QQQ,SPY,T,TSLA,UNH,V,WMT,XOM"

default:
    @just --list

_require-external-uv-env:
    @python3 -c 'import os, sys; from pathlib import Path; raw = os.environ.get("UV_PROJECT_ENVIRONMENT", ""); repo = Path("{{ justfile_directory() }}").resolve(); expanded = Path(os.path.expanduser(os.path.expandvars(raw))); missing = not raw; relative = bool(raw) and not expanded.is_absolute(); inside = False if missing or relative else expanded.resolve().is_relative_to(repo); reason = "is required" if missing else "must be an absolute path" if relative else "must be outside the repo" if inside else "ok"; print("UV_PROJECT_ENVIRONMENT=" + (raw or "<unset>")); sys.exit(0 if reason == "ok" else (print("error: UV_PROJECT_ENVIRONMENT " + reason, file=sys.stderr) or 1))'

check: _require-external-uv-env
    uv sync --all-extras --dev
    uv run ruff format --check src tests
    uv run ruff check src tests
    uv run mypy src tests
    uv run pytest
    uv run mkdocs build --strict
    {{cli}} status
    {{cli}} source-probe all auto
    {{cli}} toy-graph

fix: _require-external-uv-env
    uv sync --all-extras --dev
    uv run ruff format src tests
    uv run ruff check --fix src tests

docs port="8000": _require-external-uv-env
    uv sync --all-extras --dev
    uv run mkdocs build --strict
    @port=$(python3 -c 'import socket, sys; host = "127.0.0.1"; start = int(sys.argv[1]); print(next(p for p in range(start, start + 100) if socket.socket().connect_ex((host, p))))' "{{ port }}"); echo "Serving docs at http://127.0.0.1:${port}"; uv run mkdocs serve -a 127.0.0.1:${port}

fetch-sample market="all" start="2026-02-02" end="2026-04-30": _require-external-uv-env
    {{cli}} fetch-sample --market "{{ market }}" --start "{{ start }}" --end "{{ end }}"

data-v1-us start="2026-02-02" end="2026-04-30" max_workers="4": _require-external-uv-env
    {{cli}} data-expansion --market us --start "{{ start }}" --end "{{ end }}" --tickers "{{ us_tickers }}" --max-workers "{{ max_workers }}"

data-jp start="2026-03-16" end="2026-04-30" max_dates="35": _require-external-uv-env
    {{cli}} data-expansion --market jp --start "{{ start }}" --end "{{ end }}" --max-jp-option-dates "{{ max_dates }}"

benchmark-a1 *args: _require-external-uv-env
    @set -- {{args}}; \
    mask="stratified"; out="reports/runs/benchmark_a1"; seeds="1,2,3"; epochs="20"; batch_size="8"; max_nodes="250"; torch_threads="6"; device="auto"; baseline_preset="fast"; no_arb_surfaces="100"; variant_suite="core"; variants=""; \
    for arg in "$@"; do \
      case "$arg" in \
        mask=*) mask="${arg#mask=}" ;; \
        out=*) out="${arg#out=}" ;; \
        seeds=*) seeds="${arg#seeds=}" ;; \
        epochs=*) epochs="${arg#epochs=}" ;; \
        batch_size=*) batch_size="${arg#batch_size=}" ;; \
        max_nodes=*) max_nodes="${arg#max_nodes=}" ;; \
        torch_threads=*) torch_threads="${arg#torch_threads=}" ;; \
        device=*) device="${arg#device=}" ;; \
        baseline_preset=*) baseline_preset="${arg#baseline_preset=}" ;; \
        no_arb_surfaces=*) no_arb_surfaces="${arg#no_arb_surfaces=}" ;; \
        variant_suite=*) variant_suite="${arg#variant_suite=}" ;; \
        variants=*) variants="${arg#variants=}" ;; \
        *) echo "error: benchmark-a1 expects key=value args, got '$arg'" >&2; exit 2 ;; \
      esac; \
    done; \
    {{cli}} benchmark-protocol --us-data data/silver/option_quotes/us_option_quotes_expanded.parquet --jp-data data/silver/option_quotes/jp_option_quotes_expanded.parquet --output-dir "${out}_${mask}" --seeds "$seeds" --epochs "$epochs" --batch-size "$batch_size" --mask-regime "$mask" --min-us-surfaces 2400 --min-us-dates 60 --min-jp-dates 20 --max-nodes-per-surface "$max_nodes" --torch-threads "$torch_threads" --device "$device" --baseline-preset "$baseline_preset" --baseline-eval-splits val,test --no-arb-diagnostics-mode sampled_surface --no-arb-eval-splits val,test --no-arb-max-surfaces-per-split "$no_arb_surfaces" --variant-suite "$variant_suite" --variants "$variants"

_smoke-models out="/tmp/log_iv_model_smoke": _require-external-uv-env
    @rm -rf "{{ out }}"
    @models=( \
      "gnn_no_liq|--model-kind gnn --gnn-layers 1 --no-use-liquidity-features --no-use-liquidity-gate" \
      "gnn_liq_feature_only|--model-kind gnn --gnn-layers 1 --use-liquidity-features --no-use-liquidity-gate" \
      "gnn_scalar_gate|--model-kind gnn --gnn-layers 1 --use-liquidity-features --use-liquidity-gate" \
      "lagos_loss_only|--model-kind gnn --gnn-layers 1 --use-liquidity-features --no-use-liquidity-gate --heteroscedastic-weight 1.0 --reliability-gate-weight 0.0" \
      "lagos_attn_only|--model-kind gnn --gnn-layers 1 --use-liquidity-features --use-liquidity-gate --heteroscedastic-weight 0.0 --reliability-gate-weight 1.0" \
      "lagos_hetero_full|--model-kind gnn --gnn-layers 1 --use-liquidity-features --use-liquidity-gate --heteroscedastic-weight 1.0 --reliability-gate-weight 1.0" \
      "lagos_random_edges|--model-kind gnn --graph-style random_edges --gnn-layers 1 --use-liquidity-features --use-liquidity-gate" \
      "lagos_shuffled_edges|--model-kind gnn --graph-style shuffled_edges --gnn-layers 1 --use-liquidity-features --use-liquidity-gate" \
      "ods_operator|--model-kind ods_operator --gnn-layers 1" \
      "hexagon_attention|--model-kind hexagon_attention --graph-style hexagon --gnn-layers 1 --cross-view-alignment-weight 0.1" \
      "cnp|--model-kind cnp --gnn-layers 1" \
      "anp|--model-kind anp --gnn-layers 1" \
      "grid_cnn|--model-kind grid_cnn --gnn-layers 1 --no-arb-weight 0.1" \
      "set_context_mlp|--model-kind set_context_mlp --gnn-layers 0" \
    ); \
    for spec in $models; do \
      name=${spec%%|*}; args=${spec#*|}; echo "===== $name ====="; \
      {{cli}} train ${=args} --epochs 1 --batch-size 2 --d-model 16 --encoder-layers 1 --synthetic-surfaces 6 --synthetic-underlyings 2 --synthetic-maturities 3 --synthetic-strikes 5 --task masked_reconstruction --mask-fraction 0.3 --output-dir "{{ out }}" --experiment-name "$name" --baseline-preset fast --no-arb-diagnostics-mode none --quiet-postprocess --torch-threads 2; \
      test -f "{{ out }}/$name/metrics_summary.json"; \
      test -f "{{ out }}/$name/diagnostics_reliability.json"; \
    done; \
    {{cli}} hyperiv-compare --output-dir "{{ out }}/hyperiv_external"; \
    echo "SMOKE_MODELS_OK"

_hyperiv-compare repo="" out="reports/runs/hyperiv_external" adapter="" run_adapter="": _require-external-uv-env
    {{cli}} hyperiv-compare --hyperiv-repo "{{ repo }}" --output-dir "{{ out }}" --adapter-command "{{ adapter }}" {{ run_adapter }}

_kernel: _require-external-uv-env
    uv run python -m ipykernel install --user --name log-iv --display-name "Python (LoG-IV)"
