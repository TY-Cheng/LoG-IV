set dotenv-load
set shell := ["zsh", "-cu"]

cli := "PYTHONPATH=src uv run python -m log_iv.cli"

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

_kernel: _require-external-uv-env
    uv run python -m ipykernel install --user --name log-iv --display-name "Python (LoG-IV)"
