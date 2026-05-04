"""Credential-safe probes and small real-data fetches for LoG-IV.

The module keeps vendor payloads in bronze and only exposes canonical
``OptionQuote`` rows to the graph/model layer.  It intentionally does not hide
real vendor failures behind synthetic fallback during source probes.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import numpy as np
import pandas as pd

from log_iv.schema import OptionQuote, OptionType

DEFAULT_US_TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
DEFAULT_JP_TICKERS = ["7203", "8306", "6758", "9984", "9432"]
DEFAULT_START = date(2026, 2, 2)
DEFAULT_END = date(2026, 4, 30)
DEFAULT_OPTION_FLAT_FILE_DATASET = "day_aggs_v1"
DEFAULT_MIN_NODES_PER_SURFACE = 20
DEFAULT_MIN_US_SURFACES = 1_000
DEFAULT_MIN_JP_DATES = 20
DATA_VERSION_TARGETS: dict[str, dict[str, int]] = {
    "data_v0": {"usable_surfaces": 1_000, "usable_dates": 31},
    "data_v1": {"usable_surfaces": 2_400, "usable_dates": 60},
    "data_v2": {"usable_surfaces": 8_000, "usable_dates": 126},
}
US_INFERRED_IV_SOURCE = "option_mid_price_with_underlying_daily_close"
US_INFERRED_IV_METHOD = "black_forward_bisection_zero_rate_zero_dividend"
US_FLAT_BRONZE_CACHE_VERSION = 1
_DOTENV_LOADED = False


@dataclass(frozen=True)
class DataFetchConfig:
    """Small-sample ingestion configuration."""

    bronze_dir: Path = Path("data/bronze")
    silver_dir: Path = Path("data/silver")
    gold_dir: Path = Path("data/gold")
    reports_dir: Path = Path("reports")
    us_tickers: list[str] = field(default_factory=lambda: DEFAULT_US_TICKERS.copy())
    jp_tickers: list[str] = field(default_factory=lambda: DEFAULT_JP_TICKERS.copy())
    start: date = DEFAULT_START
    end: date = DEFAULT_END
    request_timeout_seconds: float = 30.0
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    use_synthetic_fallback: bool = False
    max_jp_option_dates: int = 20
    max_workers: int = 4

    @classmethod
    def from_env(cls) -> DataFetchConfig:
        _load_local_dotenv()
        return cls(
            bronze_dir=Path(os.environ.get("BRONZE_DATA_DIR", "data/bronze")),
            silver_dir=Path(os.environ.get("SILVER_DATA_DIR", "data/silver")),
            gold_dir=Path(os.environ.get("GOLD_DATA_DIR", "data/gold")),
            reports_dir=Path(os.environ.get("REPORTS_DIR", "reports")),
            request_timeout_seconds=_env_float("REQUEST_TIMEOUT_SECONDS", 30.0),
            max_retries=_env_int("MAX_RETRIES", 2),
            retry_backoff_seconds=_env_float("RETRY_BACKOFF_SECONDS", 1.0),
            max_jp_option_dates=_env_int("JQUANTS_MAX_OPTION_DATES", 20),
            max_workers=_env_int("DATA_EXPANSION_MAX_WORKERS", 4),
        )


@dataclass(frozen=True)
class ProbeResult:
    """A redacted source-probe result suitable for logs and manifests."""

    source: str
    mode: str
    ok: bool
    message: str
    endpoint: str | None = None
    row_count: int | None = None
    schema_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FetchSummary:
    """Summary of a small fetch-sample run."""

    market: str
    start: date
    end: date
    bronze_manifests: list[str]
    silver_tables: list[str]
    row_counts: dict[str, int]
    schema_fingerprints: dict[str, str]
    stopped_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["start"] = self.start.isoformat()
        payload["end"] = self.end.isoformat()
        return payload


def read_secret_file(file_env_var: str) -> str | None:
    """Read a secret from a configured key-file path without printing it."""

    _load_local_dotenv()
    file_value = os.environ.get(file_env_var)
    if file_value:
        path = Path(os.path.expanduser(os.path.expandvars(file_value)))
        if path.is_file():
            value = path.read_text().strip()
            return value or None
    return None


def _load_local_dotenv() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        _DOTENV_LOADED = True
        return
    load_dotenv()
    _DOTENV_LOADED = True


def run_source_probe(source: str = "all", mode: str = "auto") -> list[ProbeResult]:
    """Run import and vendor probes.

    ``source`` is one of ``all``, ``imports``, ``massive``, or ``jquants``.
    ``mode`` refines the vendor probe: ``rest``/``flat`` for Massive and
    ``auth``/``options``/``equities`` for J-Quants.
    """

    source = source.lower()
    mode = mode.lower()
    results: list[ProbeResult] = []

    if source == "all" and mode == "auto":
        results.extend(probe_imports())
        results.extend(probe_massive("rest"))
        results.extend(probe_jquants("auth"))
        return results

    if source in {"all", "imports"}:
        results.extend(probe_imports())

    if source in {"all", "massive"}:
        results.extend(probe_massive(mode))

    if source in {"all", "jquants"}:
        results.extend(probe_jquants(mode))

    if source not in {"all", "imports", "massive", "jquants"}:
        return [
            ProbeResult(
                source=source,
                mode=mode,
                ok=False,
                message="unknown source; expected all/imports/massive/jquants",
            )
        ]
    return results


def probe_imports() -> list[ProbeResult]:
    """Import all default-gate modules, including ML dependencies."""

    import importlib

    modules = [
        "log_iv",
        "log_iv.config",
        "log_iv.schema",
        "log_iv.graph",
        "log_iv.synthetic",
        "log_iv.encoder",
        "log_iv.gnn",
        "log_iv.regularizer",
        "log_iv.train",
        "log_iv.data_fetch",
        "torch",
        "torch_geometric",
    ]
    results: list[ProbeResult] = []
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - exact failure is environment-specific.
            results.append(
                ProbeResult("imports", "import", False, f"{module}: {type(exc).__name__}: {exc}")
            )
        else:
            results.append(ProbeResult("imports", "import", True, module))
    return results


def probe_massive(mode: str = "auto") -> list[ProbeResult]:
    """Probe Massive credentials and lightweight REST endpoints."""

    selected = {"rest", "flat"} if mode in {"auto", "all"} else {mode}
    results: list[ProbeResult] = []

    if "rest" in selected:
        key = read_secret_file("MASSIVE_API_KEY_FILE")
        if key is None:
            results.append(ProbeResult("massive", "rest", False, "missing MASSIVE_API_KEY_FILE"))
        else:
            cfg = _vendor_config("MASSIVE")
            client = _client(cfg)
            try:
                payload = _massive_get_json(
                    client,
                    "/v2/aggs/ticker/SPY/range/1/day/2026-04-29/2026-04-30",
                    {"adjusted": "true", "sort": "asc", "limit": "2"},
                    key,
                )
                rows = _extract_list(payload, ["results"])
                results.append(
                    ProbeResult(
                        "massive",
                        "rest",
                        bool(rows),
                        "REST aggregate probe returned rows" if rows else "REST returned no rows",
                        endpoint="/v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}",
                        row_count=len(rows),
                        schema_fingerprint=schema_fingerprint(rows),
                    )
                )
            except Exception as exc:
                results.append(ProbeResult("massive", "rest", False, _redact_error(exc)))

    if "flat" in selected:
        flat_key = read_secret_file("MASSIVE_FLAT_FILE_KEY_FILE")
        key_file = os.environ.get("MASSIVE_FLAT_FILE_KEY_FILE", "")
        configured_file = bool(key_file) and Path(os.path.expanduser(key_file)).is_file()
        results.append(
            ProbeResult(
                "massive",
                "flat",
                flat_key is not None or configured_file,
                "flat-file credential configured"
                if flat_key is not None or configured_file
                else "missing MASSIVE_FLAT_FILE_KEY_FILE",
                endpoint=os.environ.get(
                    "MASSIVE_FLAT_FILE_ENDPOINT_URL", "https://files.massive.com"
                ),
            )
        )
    return results


def probe_jquants(mode: str = "auto") -> list[ProbeResult]:
    """Probe J-Quants official auth and small data endpoints."""

    selected = {"auth", "equities", "options"} if mode in {"auto", "all"} else {mode}
    results: list[ProbeResult] = []
    cfg = _vendor_config("JQUANTS")
    client = _client(cfg)
    if _is_jquants_v2(cfg):
        api_key = read_secret_file("JQUANTS_API_KEY_FILE")
        if selected & {"auth", "equities", "options"}:
            if api_key is None:
                results.append(
                    ProbeResult("jquants", "auth", False, "missing JQUANTS_API_KEY_FILE")
                )
            else:
                results.append(_jquants_v2_endpoint_probe(client, api_key, "/equities/master"))
        if "equities" in selected and api_key is not None:
            results.append(_jquants_v2_endpoint_probe(client, api_key, "/equities/bars/daily"))
        if "options" in selected and api_key is not None:
            for endpoint in (
                "/derivatives/bars/daily/options/225",
                "/derivatives/bars/daily/options",
            ):
                results.append(_jquants_v2_endpoint_probe(client, api_key, endpoint))
        return results

    token: str | None = None
    if selected & {"auth", "equities", "options"}:
        try:
            token = jquants_id_token(client)
            results.append(ProbeResult("jquants", "auth", True, "ID token acquired"))
        except Exception as exc:
            results.append(ProbeResult("jquants", "auth", False, _redact_error(exc)))

    if "equities" in selected:
        if token is None:
            results.append(ProbeResult("jquants", "equities", False, "auth failed; skipped"))
        else:
            results.append(_jquants_v1_endpoint_probe(client, token, "/prices/daily_quotes"))

    if "options" in selected:
        if token is None:
            results.append(ProbeResult("jquants", "options", False, "auth failed; skipped"))
        else:
            for endpoint in ("/derivatives/options", "/option/index_option"):
                results.append(_jquants_v1_endpoint_probe(client, token, endpoint))

    return results


def fetch_sample_dataset(
    *,
    market: str = "all",
    start: date = DEFAULT_START,
    end: date = DEFAULT_END,
    config: DataFetchConfig | None = None,
) -> FetchSummary:
    """Fetch the first LoG-IV real-data MVP sample and persist bronze/silver artifacts."""

    cfg = config or DataFetchConfig.from_env()
    market = market.lower()
    if market not in {"us", "jp", "all"}:
        msg = "market must be one of us/jp/all"
        raise ValueError(msg)

    bronze_manifests: list[str] = []
    silver_tables: list[str] = []
    row_counts: dict[str, int] = {}
    fingerprints: dict[str, str] = {}
    stopped: str | None = None

    if market in {"us", "all"}:
        us_dataset, us_manifests = fetch_us_option_dataset(cfg, start=start, end=end)
        bronze_manifests.extend(str(path) for path in us_manifests)
        us_rows = [quote for surface in us_dataset.values() for quote in surface]
        if us_rows:
            silver_path = write_silver_option_quotes(us_rows, "us_option_quotes", cfg.silver_dir)
            silver_tables.append(str(silver_path))
            row_counts["us_option_quotes"] = len(us_rows)
            fingerprints["us_option_quotes"] = table_schema_fingerprint(silver_path)
        else:
            stopped = "no usable U.S. option rows after bid/ask and schema validation"

    if stopped is None and market in {"jp", "all"}:
        jp_dataset, jp_manifests = fetch_jp_option_dataset(cfg, start=start, end=end)
        bronze_manifests.extend(str(path) for path in jp_manifests)
        jp_rows = [quote for surface in jp_dataset.values() for quote in surface]
        if jp_rows:
            silver_path = write_silver_option_quotes(jp_rows, "jp_option_quotes", cfg.silver_dir)
            silver_tables.append(str(silver_path))
            row_counts["jp_option_quotes"] = len(jp_rows)
            fingerprints["jp_option_quotes"] = table_schema_fingerprint(silver_path)
        elif market == "jp":
            stopped = "no usable Japan option rows; equity/OOD probe may still be possible"

    summary = FetchSummary(
        market=market,
        start=start,
        end=end,
        bronze_manifests=bronze_manifests,
        silver_tables=silver_tables,
        row_counts=row_counts,
        schema_fingerprints=fingerprints,
        stopped_reason=stopped,
    )
    _write_json(cfg.bronze_dir / "fetch_sample_manifest.json", summary.to_dict())
    return summary


def write_data_expansion_report(
    *,
    start: date = DEFAULT_START,
    end: date = DEFAULT_END,
    config: DataFetchConfig | None = None,
    market: str = "all",
    use_bronze_cache: bool = True,
    max_workers: int | None = None,
    refresh_failed: bool = False,
    refresh_all: bool = False,
) -> Path:
    """Probe/write expansion manifests for U.S. flat files and JP date-loop coverage."""

    cfg = config or DataFetchConfig.from_env()
    market = market.lower()
    if market not in {"all", "us", "jp"}:
        msg = "market must be one of all/us/jp"
        raise ValueError(msg)

    us_dataset: dict[str, list[OptionQuote]] = {}
    us_manifests: list[Path] = []
    jp_dataset: dict[str, list[OptionQuote]] = {}
    jp_manifests: list[Path] = []
    if market in {"all", "us"}:
        us_dataset, us_manifests = fetch_us_flat_file_option_dataset(
            cfg,
            start=start,
            end=end,
            use_bronze_cache=use_bronze_cache,
            max_workers=max_workers,
            refresh_failed=refresh_failed,
            refresh_all=refresh_all,
        )
    if market in {"all", "jp"}:
        if use_bronze_cache:
            jp_dataset, jp_manifests = load_jp_option_dataset_from_bronze(
                cfg,
                start=start,
                end=end,
            )
        if not jp_dataset:
            jp_dataset, jp_manifests = fetch_jp_option_dataset(cfg, start=start, end=end)
    us_source_rows = [quote for surface in us_dataset.values() for quote in surface]
    jp_source_rows = [quote for surface in jp_dataset.values() for quote in surface]
    us_rows = dedupe_option_quotes(us_source_rows)
    jp_rows = dedupe_option_quotes(jp_source_rows)
    us_gate = option_surface_gate_summary(
        us_rows,
        min_nodes_per_surface=DEFAULT_MIN_NODES_PER_SURFACE,
        min_surfaces=DEFAULT_MIN_US_SURFACES,
        min_observation_dates=0,
    )
    jp_gate = option_surface_gate_summary(
        jp_rows,
        min_nodes_per_surface=DEFAULT_MIN_NODES_PER_SURFACE,
        min_surfaces=0,
        min_observation_dates=DEFAULT_MIN_JP_DATES,
    )
    us_silver = (
        write_silver_option_quotes(
            us_rows,
            "us_option_quotes_expanded",
            cfg.silver_dir,
            metadata={
                "data_stage": data_stage_from_gate(us_gate),
                "data_stage_targets": DATA_VERSION_TARGETS,
                "date_range": {"start": start.isoformat(), "end": end.isoformat()},
                "ticker_count": len({ticker.upper() for ticker in cfg.us_tickers}),
                "tickers": sorted({ticker.upper() for ticker in cfg.us_tickers}),
                "source": "massive_options_flat_files",
                "source_row_count": len(us_source_rows),
                "deduplicated_row_count": len(us_rows),
                "iv_source": US_INFERRED_IV_SOURCE,
                "iv_method": US_INFERRED_IV_METHOD,
                "surface_gate": us_gate,
            },
        )
        if us_rows
        else None
    )
    jp_silver = (
        write_silver_option_quotes(
            jp_rows,
            "jp_option_quotes_expanded",
            cfg.silver_dir,
            metadata={
                "data_stage": data_stage_from_gate(jp_gate),
                "data_stage_targets": DATA_VERSION_TARGETS,
                "date_range": {"start": start.isoformat(), "end": end.isoformat()},
                "ticker_count": len({ticker for ticker in cfg.jp_tickers}),
                "tickers": sorted({ticker for ticker in cfg.jp_tickers}),
                "source": "jquants_options_date_loop",
                "source_row_count": len(jp_source_rows),
                "deduplicated_row_count": len(jp_rows),
                "surface_gate": jp_gate,
            },
        )
        if jp_rows
        else None
    )
    report = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "market": market,
        "used_bronze_cache": use_bronze_cache,
        "refresh_failed": refresh_failed,
        "refresh_all": refresh_all,
        "max_workers": max_workers if max_workers is not None else cfg.max_workers,
        "ok": bool((market == "jp" or us_gate["ok"]) and (market == "us" or jp_gate["ok"])),
        "action": _data_expansion_action(market, bool(us_gate["ok"]), bool(jp_gate["ok"])),
        "data_stage_targets": DATA_VERSION_TARGETS,
        "expanded_silver_tables": {
            "us": str(us_silver) if us_silver is not None else None,
            "jp": str(jp_silver) if jp_silver is not None else None,
        },
        "us_flat": {
            "data_stage": data_stage_from_gate(us_gate),
            "date_range": {"start": start.isoformat(), "end": end.isoformat()},
            "ticker_count": len({ticker.upper() for ticker in cfg.us_tickers}),
            "tickers": sorted({ticker.upper() for ticker in cfg.us_tickers}),
            "surface_count": len(us_dataset),
            "source_row_count": len(us_source_rows),
            "deduplicated_row_count": len(us_rows),
            "manifest_count": len(us_manifests),
            "manifests": [str(path) for path in us_manifests],
            "max_workers": max_workers if max_workers is not None else cfg.max_workers,
            "used_bronze_cache": use_bronze_cache,
            "refresh_failed": refresh_failed,
            "refresh_all": refresh_all,
            "iv_source": US_INFERRED_IV_SOURCE,
            "iv_method": US_INFERRED_IV_METHOD,
            "surface_gate": us_gate,
            "acceptance_target": (
                ">=1000 usable (underlying, observation_date) surfaces after filtering"
            ),
        },
        "jp_date_loop": {
            "data_stage": data_stage_from_gate(jp_gate),
            "date_range": {"start": start.isoformat(), "end": end.isoformat()},
            "ticker_count": len({ticker for ticker in cfg.jp_tickers}),
            "tickers": sorted({ticker for ticker in cfg.jp_tickers}),
            "surface_count": len(jp_dataset),
            "source_row_count": len(jp_source_rows),
            "deduplicated_row_count": len(jp_rows),
            "observation_date_count": jp_gate["distinct_observation_date_count"],
            "observation_dates": jp_gate["observation_dates"],
            "manifest_count": len(jp_manifests),
            "manifests": [str(path) for path in jp_manifests],
            "surface_gate": jp_gate,
            "acceptance_target": ">=20 observation dates with option rows",
        },
    }
    path = cfg.reports_dir / "data_expansion_report.json"
    _write_json(path, report)
    return path


def _data_expansion_action(market: str, us_ok: bool, jp_ok: bool) -> str:
    if market == "all":
        return "run_benchmark_protocol" if us_ok and jp_ok else "expand_data_first"
    if market == "us":
        return "us_gate_ok_check_other_market" if us_ok else "expand_data_first"
    if market == "jp":
        return "jp_gate_ok_check_other_market" if jp_ok else "expand_data_first"
    return "expand_data_first"


def fetch_us_option_dataset(
    config: DataFetchConfig | None = None,
    api_key_file: str | None = None,
    *,
    start: date | None = None,
    end: date | None = None,
) -> tuple[dict[str, list[OptionQuote]], list[Path]]:
    """Fetch Massive U.S. option snapshot rows for the MVP ticker set."""

    cfg = config or DataFetchConfig.from_env()
    api_key = _secret_from_optional_file("MASSIVE_API_KEY_FILE", api_key_file)
    if api_key is None:
        return {}, [
            write_bronze_manifest(
                cfg.bronze_dir,
                "massive",
                "rest",
                {"ok": False, "message": "missing MASSIVE_API_KEY_FILE"},
            )
        ]

    client = _client(_vendor_config("MASSIVE"))
    dataset: dict[str, list[OptionQuote]] = {}
    manifests: list[Path] = []
    for ticker in cfg.us_tickers:
        try:
            payload = massive_fetch_option_chain(client, ticker, api_key)
            rows = _extract_list(payload, ["results"])
            manifests.append(
                write_bronze_payload(cfg.bronze_dir, "massive", "options", ticker, payload)
            )
            quotes = massive_snapshot_to_option_quotes(rows, ticker)
            valid = [quote for quote in quotes if _has_valid_bid_ask(quote)]
            if valid:
                surface_date = valid[0].observation_date.isoformat()
                dataset[f"US_{ticker}_{surface_date}"] = valid
        except Exception as exc:
            manifests.append(
                write_bronze_manifest(
                    cfg.bronze_dir,
                    "massive",
                    "options",
                    {"ok": False, "ticker": ticker, "message": _redact_error(exc)},
                )
            )

    return dataset, manifests


@dataclass(frozen=True)
class _USFlatDateResult:
    observation_date: date
    dataset: dict[str, list[OptionQuote]]
    manifest: Path


def fetch_us_flat_file_option_dataset(
    config: DataFetchConfig | None = None,
    *,
    start: date | None = None,
    end: date | None = None,
    path_template: str | None = None,
    underlying_path_template: str | None = None,
    use_bronze_cache: bool = True,
    max_workers: int | None = None,
    refresh_failed: bool = False,
    refresh_all: bool = False,
) -> tuple[dict[str, list[OptionQuote]], list[Path]]:
    """Ingest Massive/Polygon daily historical option flat files via a path template.

    The template is credential-safe and may point at a local file, ``file://`` URL,
    HTTPS URL, or S3 URL. It can contain ``{date}`` and ``{yyyymmdd}`` placeholders.
    """

    cfg = config or DataFetchConfig.from_env()
    start = start or cfg.start
    end = end or cfg.end
    template = _resolve_massive_options_flat_file_template(path_template)
    underlying_template = _resolve_massive_underlying_flat_file_template(underlying_path_template)
    manifests: list[Path] = []
    dataset: dict[str, list[OptionQuote]] = {}
    if not template:
        manifests.append(
            write_bronze_manifest(
                cfg.bronze_dir,
                "massive",
                "flat_options",
                {
                    "ok": False,
                    "message": "missing MASSIVE_OPTIONS_FLAT_FILE_TEMPLATE",
                    "source": "daily historical options flat files",
                },
            )
        )
        return dataset, manifests

    tickers = {ticker.upper() for ticker in cfg.us_tickers}
    dates = _bounded_business_dates(start, end, 10_000)
    worker_count = max(1, int(max_workers if max_workers is not None else cfg.max_workers))
    results: list[_USFlatDateResult] = []
    if worker_count == 1 or len(dates) <= 1:
        for obs_date in dates:
            results.append(
                _fetch_us_flat_file_date(
                    cfg,
                    obs_date,
                    template,
                    underlying_template,
                    tickers,
                    use_bronze_cache=use_bronze_cache,
                    refresh_failed=refresh_failed,
                    refresh_all=refresh_all,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _fetch_us_flat_file_date,
                    cfg,
                    obs_date,
                    template,
                    underlying_template,
                    tickers,
                    use_bronze_cache=use_bronze_cache,
                    refresh_failed=refresh_failed,
                    refresh_all=refresh_all,
                ): obs_date
                for obs_date in dates
            }
            for future in as_completed(futures):
                obs_date = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    source_path = _format_flat_file_template(template, obs_date)
                    manifests.append(
                        write_bronze_manifest(
                            cfg.bronze_dir,
                            "massive",
                            "flat_options",
                            {
                                "ok": False,
                                "complete": False,
                                "date": obs_date.isoformat(),
                                "path": source_path,
                                "source_path": source_path,
                                "tickers": sorted(tickers),
                                "tickers_hash": _tickers_hash(tickers),
                                "message": _redact_error(exc),
                            },
                        )
                    )

    for result in sorted(results, key=lambda item: item.observation_date):
        manifests.append(result.manifest)
        dataset.update(result.dataset)
    return dataset, manifests


def _fetch_us_flat_file_date(
    cfg: DataFetchConfig,
    obs_date: date,
    template: str,
    underlying_template: str | None,
    tickers: set[str],
    *,
    use_bronze_cache: bool,
    refresh_failed: bool,
    refresh_all: bool,
) -> _USFlatDateResult:
    source_path = _format_flat_file_template(template, obs_date)
    tickers_hash = _tickers_hash(tickers)
    if use_bronze_cache and not refresh_all:
        cached = _load_us_flat_cache_payload(
            cfg,
            obs_date=obs_date,
            source_path=source_path,
            tickers_hash=tickers_hash,
            refresh_failed=refresh_failed,
        )
        if cached is not None:
            quotes, manifest_path = cached
            return _USFlatDateResult(
                observation_date=obs_date,
                dataset=_us_flat_dataset_for_date(quotes, obs_date),
                manifest=manifest_path,
            )

    try:
        rows = _read_flat_file_rows(source_path)
        quotes = massive_flat_rows_to_option_quotes(rows, obs_date, tickers)
        valid = [quote for quote in quotes if _has_valid_bid_ask(quote)]
        underlying_error = None
        underlying_prices: dict[str, float] = {}
        underlying_path = None
        if underlying_template:
            underlying_path = _format_flat_file_template(underlying_template, obs_date)
            try:
                underlying_prices = fetch_us_underlying_flat_file_prices(
                    obs_date,
                    tickers=tickers,
                    path_template=underlying_template,
                )
            except Exception as exc:
                underlying_error = _redact_error(exc)
        valid = infer_implied_vols_from_option_prices(valid, underlying_prices)
        iv_inferred_count = sum(1 for quote in valid if quote.implied_vol is not None)
        payload_path = _us_flat_payload_path(cfg, obs_date, tickers_hash)
        _write_us_flat_payload(payload_path, valid)
        manifest_payload = {
            "ok": True,
            "complete": True,
            "cache_version": US_FLAT_BRONZE_CACHE_VERSION,
            "cache_hit": False,
            "date": obs_date.isoformat(),
            "path": source_path,
            "source_path": source_path,
            "underlying_path": underlying_path,
            "payload_path": str(payload_path),
            "payload_row_count": len(valid),
            "payload_schema_fingerprint": table_schema_fingerprint(payload_path),
            "row_count": len(rows),
            "usable_row_count": len(valid),
            "underlying_price_count": len(underlying_prices),
            "iv_inferred_row_count": iv_inferred_count,
            "iv_missing_row_count": len(valid) - iv_inferred_count,
            "schema_fingerprint": schema_fingerprint(rows),
            "tickers": sorted(tickers),
            "ticker_count": len(tickers),
            "tickers_hash": tickers_hash,
            "iv_source": US_INFERRED_IV_SOURCE,
            "iv_method": US_INFERRED_IV_METHOD,
        }
        if underlying_error:
            manifest_payload["underlying_price_message"] = underlying_error
        manifest = write_bronze_manifest(
            cfg.bronze_dir,
            "massive",
            "flat_options",
            manifest_payload,
        )
        return _USFlatDateResult(
            observation_date=obs_date,
            dataset=_us_flat_dataset_for_date(valid, obs_date),
            manifest=manifest,
        )
    except Exception as exc:
        manifest = write_bronze_manifest(
            cfg.bronze_dir,
            "massive",
            "flat_options",
            {
                "ok": False,
                "complete": False,
                "cache_version": US_FLAT_BRONZE_CACHE_VERSION,
                "date": obs_date.isoformat(),
                "path": source_path,
                "source_path": source_path,
                "tickers": sorted(tickers),
                "ticker_count": len(tickers),
                "tickers_hash": tickers_hash,
                "message": _redact_error(exc),
            },
        )
        return _USFlatDateResult(observation_date=obs_date, dataset={}, manifest=manifest)


def _us_flat_dataset_for_date(
    quotes: list[OptionQuote],
    obs_date: date,
) -> dict[str, list[OptionQuote]]:
    dataset: dict[str, list[OptionQuote]] = {}
    for ticker, group in _quotes_by_underlying(quotes).items():
        dataset[f"US_FLAT_{ticker}_{obs_date.isoformat()}"] = group
    return dataset


def _load_us_flat_cache_payload(
    cfg: DataFetchConfig,
    *,
    obs_date: date,
    source_path: str,
    tickers_hash: str,
    refresh_failed: bool,
) -> tuple[list[OptionQuote], Path] | None:
    latest = _latest_us_flat_cache_manifest(
        cfg,
        obs_date=obs_date,
        source_path=source_path,
        tickers_hash=tickers_hash,
    )
    if latest is None:
        return None
    manifest_path, manifest_payload = latest
    if manifest_payload.get("ok") is not True:
        return None
    if refresh_failed and manifest_payload.get("complete") is not True:
        return None
    return _validate_and_load_us_flat_cache_manifest(manifest_path, manifest_payload)


def _latest_us_flat_cache_manifest(
    cfg: DataFetchConfig,
    *,
    obs_date: date,
    source_path: str,
    tickers_hash: str,
) -> tuple[Path, dict[str, Any]] | None:
    manifest_dir = cfg.bronze_dir / "massive" / "flat_options"
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in manifest_dir.glob("manifest_*.json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("date") != obs_date.isoformat():
            continue
        if payload.get("source_path", payload.get("path")) != source_path:
            continue
        if payload.get("tickers_hash") != tickers_hash:
            continue
        if payload.get("cache_version") != US_FLAT_BRONZE_CACHE_VERSION:
            continue
        candidates.append((path, payload))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0].name)[-1]


def _validate_and_load_us_flat_cache_manifest(
    manifest_path: Path,
    manifest_payload: dict[str, Any],
) -> tuple[list[OptionQuote], Path] | None:
    if manifest_payload.get("ok") is not True or manifest_payload.get("complete") is not True:
        return None
    if not isinstance(manifest_payload.get("row_count"), int):
        return None
    if not isinstance(manifest_payload.get("usable_row_count"), int):
        return None
    if not isinstance(manifest_payload.get("payload_row_count"), int):
        return None
    if not isinstance(manifest_payload.get("schema_fingerprint"), str):
        return None
    expected_payload_fingerprint = manifest_payload.get("payload_schema_fingerprint")
    if not isinstance(expected_payload_fingerprint, str):
        return None
    payload_path_value = manifest_payload.get("payload_path")
    if not isinstance(payload_path_value, str) or not payload_path_value:
        return None
    payload_path = Path(payload_path_value)
    if not payload_path.is_file():
        return None
    try:
        frame = pd.read_parquet(payload_path)
    except Exception:
        return None
    if len(frame) != manifest_payload["payload_row_count"]:
        return None
    if len(frame) != manifest_payload["usable_row_count"]:
        return None
    if _frame_schema_fingerprint(frame) != expected_payload_fingerprint:
        return None
    quotes = _option_quotes_from_frame(frame)
    if len(quotes) != len(frame):
        return None
    return quotes, manifest_path


def _write_us_flat_payload(path: Path, quotes: list[OptionQuote]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [quote.model_dump(mode="json") for quote in quotes]
    frame = pd.DataFrame(rows, columns=list(OptionQuote.model_fields))
    tmp_path = path.with_name(f"{path.stem}.{os.getpid()}.tmp{path.suffix}")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


def _option_quotes_from_frame(frame: pd.DataFrame) -> list[OptionQuote]:
    quotes: list[OptionQuote] = []
    normalized = frame.replace({np.nan: None})
    for row in normalized.to_dict(orient="records"):
        try:
            quotes.append(OptionQuote.model_validate(_clean_parquet_record(row)))
        except ValueError:
            continue
    return quotes


def _clean_parquet_record(row: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in row.items():
        if value is pd.NaT:
            cleaned[key] = None
            continue
        if isinstance(value, float) and math.isnan(value):
            cleaned[key] = None
            continue
        cleaned[key] = value
    return cleaned


def _us_flat_payload_path(cfg: DataFetchConfig, obs_date: date, tickers_hash: str) -> Path:
    return (
        cfg.bronze_dir
        / "massive"
        / "flat_options"
        / f"payload_{obs_date.isoformat()}_{tickers_hash}.parquet"
    )


def _tickers_hash(tickers: set[str]) -> str:
    payload = json.dumps(sorted(tickers), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def massive_flat_rows_to_option_quotes(
    rows: list[dict[str, Any]],
    observation_date: date,
    tickers: set[str] | None = None,
) -> list[OptionQuote]:
    """Normalize daily option flat-file rows to canonical U.S. quotes."""

    selected = {ticker.upper() for ticker in tickers} if tickers else None
    quotes: list[OptionQuote] = []
    for row in rows:
        ticker = _pick(
            row,
            "underlying",
            "underlying_ticker",
            "root",
            "root_symbol",
            "ticker",
            "Underlying",
        )
        if ticker is None and (symbol := _pick(row, "contract_symbol", "symbol", "ticker")):
            ticker = _underlying_from_option_symbol(str(symbol))
        symbol = _pick(row, "contract_symbol", "symbol", "ticker")
        parsed_contract = _parse_option_symbol(str(symbol)) if symbol else None
        parsed_underlying = parsed_contract["underlying"] if parsed_contract else None
        ticker_text = str(ticker).upper() if ticker else ""
        if parsed_underlying and (
            ticker_text.startswith("O:") or not ticker_text or ticker_text not in (selected or {})
        ):
            ticker_text = str(parsed_underlying)
        if selected is not None and ticker_text not in selected:
            continue
        expiry = _parse_date(
            _pick(row, "expiration_date", "expiry", "expiration", "expiry_date")
        ) or (parsed_contract["expiry"] if parsed_contract else None)
        strike = _float(_pick(row, "strike_price", "strike", "StrikePrice")) or (
            float(parsed_contract["strike"]) if parsed_contract else None
        )
        option_type = _option_type(_pick(row, "contract_type", "option_type", "type", "cp")) or (
            parsed_contract["option_type"] if parsed_contract else None
        )
        bid = _float(_pick(row, "bid", "bid_price", "best_bid", "bp"))
        ask = _float(_pick(row, "ask", "ask_price", "best_ask", "ap"))
        if bid is None and ask is None:
            close = _float(_pick(row, "close", "c", "price", "mid"))
            bid = close
            ask = close
        if expiry is None or strike is None or option_type is None or bid is None or ask is None:
            continue
        if ask < bid:
            continue
        iv_raw = _float(_pick(row, "implied_volatility", "iv", "implied_vol"))
        iv = iv_raw / 100.0 if iv_raw is not None and iv_raw > 5.0 else iv_raw
        spot = _float(_pick(row, "underlying_price", "underlying_close", "spot", "forward"))
        try:
            quotes.append(
                OptionQuote(
                    market="US",
                    underlying=ticker_text,
                    observation_date=observation_date,
                    expiry=expiry,
                    strike=strike,
                    option_type=option_type,
                    bid=bid,
                    ask=ask,
                    implied_vol=iv if iv is not None and iv > 0 else None,
                    volume=_int(_pick(row, "volume", "v")),
                    open_interest=_int(_pick(row, "open_interest", "oi")),
                    forward=spot if spot is not None and spot > 0 else None,
                    underlying_price=spot if spot is not None and spot > 0 else None,
                    vendor_symbol=str(symbol) if symbol else None,
                )
            )
        except ValueError:
            continue
    return quotes


def fetch_us_underlying_flat_file_prices(
    observation_date: date,
    *,
    tickers: set[str],
    path_template: str | None = None,
) -> dict[str, float]:
    """Read U.S. underlying closes from Massive/Polygon stock daily flat files."""

    template = _resolve_massive_underlying_flat_file_template(path_template)
    if not template:
        return {}
    source_path = _format_flat_file_template(template, observation_date)
    rows = _read_flat_file_rows(source_path)
    selected = {ticker.upper() for ticker in tickers}
    prices: dict[str, float] = {}
    for row in rows:
        ticker = str(_pick(row, "ticker", "symbol", "T") or "").upper()
        if ticker not in selected:
            continue
        close = _float(_pick(row, "close", "c", "Close"))
        if close is not None and close > 0:
            prices[ticker] = close
    return prices


def infer_implied_vols_from_option_prices(
    quotes: list[OptionQuote],
    underlying_prices: dict[str, float],
) -> list[OptionQuote]:
    """Attach spot/forward proxies and Black-forward implied vols where solvable."""

    enriched: list[OptionQuote] = []
    for quote in quotes:
        underlying_price = underlying_prices.get(quote.underlying)
        if underlying_price is None or underlying_price <= 0:
            enriched.append(quote)
            continue
        mid = quote.mid
        implied_vol = quote.implied_vol
        if implied_vol is None and mid is not None:
            implied_vol = implied_vol_from_option_price(
                option_type=quote.option_type,
                forward=underlying_price,
                strike=quote.strike,
                tenor_years=quote.tenor_years,
                price=mid,
            )
        enriched.append(
            quote.model_copy(
                update={
                    "forward": quote.forward or underlying_price,
                    "underlying_price": quote.underlying_price or underlying_price,
                    "implied_vol": implied_vol,
                }
            )
        )
    return enriched


def implied_vol_from_option_price(
    *,
    option_type: OptionType,
    forward: float,
    strike: float,
    tenor_years: float,
    price: float,
    min_sigma: float = 1e-6,
    max_sigma: float = 8.0,
) -> float | None:
    """Invert an undiscounted Black-forward option price by bisection."""

    if forward <= 0 or strike <= 0 or tenor_years <= 0 or price <= 0:
        return None
    intrinsic = max(forward - strike, 0.0) if option_type == "C" else max(strike - forward, 0.0)
    upper_bound = forward if option_type == "C" else strike
    tolerance = max(1e-5, 1e-5 * max(forward, strike))
    if price < intrinsic - tolerance or price > upper_bound + tolerance:
        return None
    if price <= intrinsic + tolerance:
        return min_sigma

    low = min_sigma
    high = max_sigma
    high_price = black_forward_price(
        option_type=option_type,
        forward=forward,
        strike=strike,
        tenor_years=tenor_years,
        sigma=high,
    )
    if high_price < price - tolerance:
        return None

    for _ in range(80):
        mid = 0.5 * (low + high)
        mid_price = black_forward_price(
            option_type=option_type,
            forward=forward,
            strike=strike,
            tenor_years=tenor_years,
            sigma=mid,
        )
        if mid_price < price:
            low = mid
        else:
            high = mid
    return max(0.5 * (low + high), min_sigma)


def black_forward_price(
    *,
    option_type: OptionType,
    forward: float,
    strike: float,
    tenor_years: float,
    sigma: float,
) -> float:
    """Undiscounted Black-forward option price with zero rate/dividend discounting."""

    if forward <= 0 or strike <= 0 or tenor_years <= 0 or sigma <= 0:
        return 0.0
    vol_sqrt_t = sigma * math.sqrt(tenor_years)
    if vol_sqrt_t <= 0:
        return 0.0
    d1 = (math.log(forward / strike) + 0.5 * sigma * sigma * tenor_years) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    if option_type == "C":
        return max(forward * _normal_cdf(d1) - strike * _normal_cdf(d2), 0.0)
    return max(strike * _normal_cdf(-d2) - forward * _normal_cdf(-d1), 0.0)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def fetch_jp_option_dataset(
    config: DataFetchConfig | None = None,
    api_key_file: str | None = None,
    *,
    start: date | None = None,
    end: date | None = None,
) -> tuple[dict[str, list[OptionQuote]], list[Path]]:
    """Fetch J-Quants option endpoints where entitlement and schema allow it."""

    cfg = config or DataFetchConfig.from_env()
    del api_key_file
    start = start or cfg.start
    end = end or cfg.end
    manifests: list[Path] = []
    dataset: dict[str, list[OptionQuote]] = {}
    vendor_cfg = _vendor_config("JQUANTS")
    client = _client(vendor_cfg)
    if _is_jquants_v2(vendor_cfg):
        api_key = read_secret_file("JQUANTS_API_KEY_FILE")
        if api_key is None:
            manifests.append(
                write_bronze_manifest(
                    cfg.bronze_dir,
                    "jquants",
                    "auth",
                    {"ok": False, "message": "missing JQUANTS_API_KEY_FILE"},
                )
            )
            return dataset, manifests
        for obs_date in _bounded_business_dates(start, end, cfg.max_jp_option_dates):
            for endpoint in (
                "/derivatives/bars/daily/options/225",
                "/derivatives/bars/daily/options",
            ):
                slug = endpoint.strip("/").replace("/", "_")
                try:
                    payload = jquants_v2_get_json(
                        client, endpoint, api_key, {"date": obs_date.isoformat()}
                    )
                    manifests.append(
                        write_bronze_payload(
                            cfg.bronze_dir,
                            "jquants",
                            slug,
                            obs_date.isoformat(),
                            payload,
                        )
                    )
                    rows = _first_list(payload)
                    quotes = jquants_rows_to_option_quotes(rows, endpoint)
                    if quotes:
                        dataset[f"JP_{slug}_{obs_date.isoformat()}"] = quotes
                except Exception as exc:
                    manifests.append(
                        write_bronze_manifest(
                            cfg.bronze_dir,
                            "jquants",
                            slug,
                            {
                                "ok": False,
                                "date": obs_date.isoformat(),
                                "endpoint": endpoint,
                                "message": _redact_error(exc),
                            },
                        )
                    )
        return dataset, manifests

    try:
        token = jquants_id_token(client)
    except Exception as exc:
        manifests.append(
            write_bronze_manifest(
                cfg.bronze_dir,
                "jquants",
                "auth",
                {"ok": False, "message": _redact_error(exc)},
            )
        )
        return dataset, manifests

    for obs_date in _bounded_business_dates(start, end, cfg.max_jp_option_dates):
        for endpoint in ("/derivatives/options", "/option/index_option"):
            slug = endpoint.strip("/").replace("/", "_")
            try:
                payload = jquants_get_json(
                    client,
                    endpoint,
                    token,
                    {"date": obs_date.strftime("%Y%m%d")},
                )
                manifests.append(
                    write_bronze_payload(
                        cfg.bronze_dir, "jquants", slug, obs_date.isoformat(), payload
                    )
                )
                rows = _first_list(payload)
                quotes = jquants_rows_to_option_quotes(rows, endpoint)
                if quotes:
                    dataset[f"JP_{slug}_{obs_date.isoformat()}"] = quotes
            except Exception as exc:
                manifests.append(
                    write_bronze_manifest(
                        cfg.bronze_dir,
                        "jquants",
                        slug,
                        {
                            "ok": False,
                            "date": obs_date.isoformat(),
                            "endpoint": endpoint,
                            "message": _redact_error(exc),
                        },
                    )
                )

    return dataset, manifests


def load_jp_option_dataset_from_bronze(
    config: DataFetchConfig | None = None,
    *,
    start: date | None = None,
    end: date | None = None,
) -> tuple[dict[str, list[OptionQuote]], list[Path]]:
    """Load the latest per-date J-Quants option payloads already present in bronze."""

    cfg = config or DataFetchConfig.from_env()
    start = start or cfg.start
    end = end or cfg.end
    slug_to_endpoint = {
        "derivatives_bars_daily_options_225": "/derivatives/bars/daily/options/225",
        "derivatives_bars_daily_options": "/derivatives/bars/daily/options",
    }
    latest: dict[tuple[str, date], Path] = {}
    for slug in slug_to_endpoint:
        for path in (cfg.bronze_dir / "jquants" / slug).glob("20*.json"):
            obs_date = _date_from_bronze_payload_name(path)
            if obs_date is None or obs_date < start or obs_date > end:
                continue
            key = (slug, obs_date)
            if key not in latest or path.name > latest[key].name:
                latest[key] = path

    dataset: dict[str, list[OptionQuote]] = {}
    paths: list[Path] = []
    for (slug, obs_date), path in sorted(latest.items(), key=lambda item: (item[0][1], item[0][0])):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        rows = _first_list(payload)
        quotes = jquants_rows_to_option_quotes(rows, slug_to_endpoint[slug])
        if quotes:
            dataset[f"JP_BRONZE_{slug}_{obs_date.isoformat()}"] = quotes
            paths.append(path)
    return dataset, paths


def fetch_or_generate_dataset(
    config: DataFetchConfig | None = None,
    n_surfaces: int = 80,
    n_underlyings: int = 5,
) -> dict[str, list[OptionQuote]]:
    """Fetch real samples, optionally falling back to synthetic data."""

    cfg = config or DataFetchConfig.from_env()
    us_data, _ = fetch_us_option_dataset(cfg)
    jp_data, _ = fetch_jp_option_dataset(cfg)
    dataset = {**us_data, **jp_data}
    if len(dataset) >= n_surfaces or not cfg.use_synthetic_fallback:
        return dataset

    from log_iv.synthetic import (
        SyntheticSurfaceConfig,
        generate_synthetic_surface_dataset,
        synthetic_quotes_to_option_quotes,
    )

    synthetic = generate_synthetic_surface_dataset(
        SyntheticSurfaceConfig(random_seed=42),
        n_surfaces=n_surfaces,
        n_underlyings=n_underlyings,
    )
    for key, syn_quotes in synthetic.items():
        dataset.setdefault(key, synthetic_quotes_to_option_quotes(syn_quotes))
    return dataset


def massive_fetch_option_chain(
    client: httpx.Client,
    ticker: str,
    api_key: str,
    *,
    limit: int = 250,
) -> dict[str, Any]:
    """Fetch the current Massive option-chain snapshot for one underlying."""

    return _massive_get_json(
        client,
        f"/v3/snapshot/options/{ticker.upper()}",
        {
            "expiration_date.gte": (date.today() + timedelta(days=7)).isoformat(),
            "limit": str(limit),
            "sort": "expiration_date",
            "order": "asc",
        },
        api_key,
    )


def massive_fetch_equity_bars(
    client: httpx.Client,
    ticker: str,
    start: date,
    end: date,
    api_key: str,
) -> dict[str, Any]:
    """Fetch Massive daily stock aggregates for the requested window."""

    path = f"/v2/aggs/ticker/{ticker.upper()}/range/1/day/{start.isoformat()}/{end.isoformat()}"
    return _massive_get_json(
        client,
        path,
        {"adjusted": "true", "sort": "asc", "limit": "50000"},
        api_key,
    )


def massive_snapshot_to_option_quotes(
    rows: list[dict[str, Any]],
    ticker: str,
) -> list[OptionQuote]:
    """Normalize Massive option-chain snapshot rows to canonical quotes."""

    quotes: list[OptionQuote] = []
    for row in rows:
        details = _dict(row.get("details"))
        last_quote = _dict(row.get("last_quote"))
        last_trade = _dict(row.get("last_trade"))
        day = _dict(row.get("day"))
        underlying_asset = _dict(row.get("underlying_asset"))

        expiry = _parse_date(
            _pick(details, "expiration_date", "expiry", "expiration", "ExpirationDate")
        )
        strike = _float(_pick(details, "strike_price", "strike", "StrikePrice"))
        option_type = _option_type(_pick(details, "contract_type", "option_type", "type"))
        observation_date = _date_from_vendor_timestamp(
            _pick(
                last_quote,
                "sip_timestamp",
                "participant_timestamp",
                "last_updated",
            )
            or _pick(last_trade, "sip_timestamp", "participant_timestamp", "last_updated")
            or _pick(day, "last_updated"),
        )

        bid = _float(_pick(last_quote, "bid", "bid_price", "bp"))
        ask = _float(_pick(last_quote, "ask", "ask_price", "ap"))
        if bid is None and ask is None:
            price_proxy = _float(_pick(last_trade, "price")) or _float(_pick(day, "close"))
            bid = price_proxy
            ask = price_proxy
        if expiry is None or strike is None or option_type is None or observation_date is None:
            continue
        if bid is None or ask is None or ask < bid:
            continue

        iv_raw = _float(_pick(row, "implied_volatility", "iv"))
        iv = iv_raw / 100.0 if iv_raw is not None and iv_raw > 5.0 else iv_raw
        volume = _int(_pick(day, "volume", "v"))
        open_interest = _int(_pick(row, "open_interest", "openInterest"))
        spot = _float(_pick(underlying_asset, "price", "last_quote", "last_trade", "value"))
        contract_symbol = _pick(details, "ticker", "contract_symbol", "symbol")

        try:
            quotes.append(
                OptionQuote(
                    market="US",
                    underlying=ticker,
                    observation_date=observation_date,
                    expiry=expiry,
                    strike=strike,
                    option_type=option_type,
                    bid=bid,
                    ask=ask,
                    implied_vol=iv if iv is not None and iv > 0 else None,
                    volume=volume,
                    open_interest=open_interest,
                    forward=spot if spot is not None and spot > 0 else None,
                    underlying_price=spot if spot is not None and spot > 0 else None,
                    vendor_symbol=str(contract_symbol) if contract_symbol else None,
                )
            )
        except ValueError:
            continue
    return quotes


def jquants_id_token(client: httpx.Client) -> str:
    """Acquire a J-Quants ID token via refresh token or mail/password."""

    refresh_token = read_secret_file("JQUANTS_REFRESH_TOKEN_FILE")
    if refresh_token is None:
        mail = read_secret_file("JQUANTS_MAILADDRESS_FILE")
        password = read_secret_file("JQUANTS_PASSWORD_FILE")
        if mail is None or password is None:
            msg = (
                "missing JQUANTS_REFRESH_TOKEN_FILE or "
                "JQUANTS_MAILADDRESS_FILE/JQUANTS_PASSWORD_FILE"
            )
            raise RuntimeError(msg)
        response = _request_json(
            client,
            "POST",
            "/token/auth_user",
            json_body={"mailaddress": mail, "password": password},
        )
        refresh_token = str(response.get("refreshToken", ""))
        if not refresh_token:
            msg = "J-Quants auth_user response did not include refreshToken"
            raise RuntimeError(msg)

    response = _request_json(
        client,
        "POST",
        "/token/auth_refresh",
        params={"refreshtoken": refresh_token},
    )
    token = str(response.get("idToken", ""))
    if not token:
        msg = "J-Quants auth_refresh response did not include idToken"
        raise RuntimeError(msg)
    return token


def jquants_get_json(
    client: httpx.Client,
    endpoint: str,
    id_token: str,
    params: dict[str, str],
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {id_token}"}
    return _request_json(client, "GET", endpoint, params=params, headers=headers)


def jquants_v2_get_json(
    client: httpx.Client,
    endpoint: str,
    api_key: str,
    params: dict[str, str],
) -> dict[str, Any]:
    headers = {"x-api-key": api_key, "User-Agent": "log-iv-jquants-v2/0.1"}
    return _request_json(client, "GET", endpoint, params=params, headers=headers)


def jquants_rows_to_option_quotes(rows: list[dict[str, Any]], endpoint: str) -> list[OptionQuote]:
    """Normalize J-Quants option rows when enough fields are present."""

    quotes: list[OptionQuote] = []
    underlying = (
        "N225"
        if endpoint in {"/option/index_option", "/derivatives/bars/daily/options/225"}
        else "JP_DERIV"
    )
    for row in rows:
        obs = _parse_date(_pick(row, "Date", "date", "TradingDate"))
        expiry = _parse_date(
            _pick(
                row,
                "LastTradingDay",
                "LTD",
                "ExerciseDate",
                "ContractMonth",
                "CM",
                "contract_month",
            )
        )
        strike = _float(_pick(row, "StrikePrice", "Strike", "strike", "ExercisePrice"))
        option_type = _option_type(_pick(row, "PutCall", "PutCallDivision", "PCDiv", "OptionType"))
        if obs is None or expiry is None or strike is None or option_type is None:
            continue

        bid = _float(_pick(row, "Bid", "bid"))
        ask = _float(_pick(row, "Ask", "ask"))
        close = _float(
            _pick(row, "WholeDayClose", "Close", "C", "SettlementPrice", "Settle", "Theo")
        )
        if bid is None and ask is None and close is not None:
            bid = close
            ask = close
        if bid is not None and ask is not None and ask < bid:
            continue

        iv_raw = _float(_pick(row, "ImpliedVolatility", "IV", "iv"))
        iv = iv_raw / 100.0 if iv_raw is not None and iv_raw > 5.0 else iv_raw
        volume = _int(_pick(row, "Volume", "TradingVolume", "WholeDayVolume", "Vo"))
        oi = _int(_pick(row, "OpenInterest", "OI"))
        symbol = _pick(row, "Code", "ContractCode", "LocalCode")
        row_underlying = _pick(row, "UndSSO")
        underlying_value = str(row_underlying) if row_underlying else underlying
        underlying_price = _float(_pick(row, "UnderPx"))

        try:
            quotes.append(
                OptionQuote(
                    market="JP",
                    underlying=underlying_value,
                    observation_date=obs,
                    expiry=expiry,
                    strike=strike,
                    option_type=option_type,
                    bid=bid,
                    ask=ask,
                    implied_vol=iv if iv is not None and iv > 0 else None,
                    volume=volume,
                    open_interest=oi,
                    underlying_price=underlying_price if underlying_price else None,
                    vendor_symbol=str(symbol) if symbol else None,
                )
            )
        except ValueError:
            continue
    return quotes


def write_silver_option_quotes(
    quotes: list[OptionQuote],
    stem: str,
    silver_dir: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write canonical option quotes to a silver Parquet table."""

    table_dir = silver_dir / "option_quotes"
    table_dir.mkdir(parents=True, exist_ok=True)
    path = table_dir / f"{stem}.parquet"
    rows = [quote.model_dump(mode="json") for quote in quotes]
    pd.DataFrame(rows).to_parquet(path, index=False)
    manifest = {
        "table": str(path),
        "row_count": len(rows),
        "schema_fingerprint": table_schema_fingerprint(path),
        "canonical_schema": list(OptionQuote.model_fields),
    }
    if metadata:
        manifest.update(metadata)
    _write_json(path.with_suffix(".manifest.json"), manifest)
    return path


def write_bronze_payload(
    bronze_dir: Path,
    source: str,
    kind: str,
    slug: str,
    payload: dict[str, Any],
) -> Path:
    """Write one raw vendor payload and manifest under bronze."""

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    base = bronze_dir / source / kind
    base.mkdir(parents=True, exist_ok=True)
    payload_path = base / f"{slug}_{stamp}.json"
    _write_json(payload_path, payload)
    rows = _first_list(payload)
    return write_bronze_manifest(
        bronze_dir,
        source,
        kind,
        {
            "ok": True,
            "payload": str(payload_path),
            "row_count": len(rows),
            "schema_fingerprint": schema_fingerprint(rows),
            "written_at": stamp,
        },
    )


def write_bronze_manifest(
    bronze_dir: Path,
    source: str,
    kind: str,
    payload: dict[str, Any],
) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    manifest_dir = bronze_dir / source / kind
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / f"manifest_{stamp}.json"
    index = 1
    while path.exists():
        path = manifest_dir / f"manifest_{stamp}_{index}.json"
        index += 1
    _write_json(path, payload)
    return path


def schema_fingerprint(rows: list[dict[str, Any]]) -> str:
    """Return a deterministic fingerprint of row keys and primitive value types."""

    schema: dict[str, str] = {}
    for row in rows:
        for key, value in row.items():
            schema.setdefault(str(key), type(value).__name__)
    payload = json.dumps(sorted(schema.items()), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def dedupe_option_quotes(quotes: list[OptionQuote]) -> list[OptionQuote]:
    """Deduplicate canonical option rows by token identity, keeping the richest row."""

    selected: dict[tuple[Any, ...], OptionQuote] = {}
    for quote in quotes:
        key = _option_quote_key(quote)
        current = selected.get(key)
        if current is None or _quote_quality_score(quote) > _quote_quality_score(current):
            selected[key] = quote
    return sorted(selected.values(), key=_option_quote_key)


def option_surface_gate_summary(
    quotes: list[OptionQuote],
    *,
    min_nodes_per_surface: int = DEFAULT_MIN_NODES_PER_SURFACE,
    min_surfaces: int = 0,
    min_observation_dates: int = 0,
) -> dict[str, Any]:
    """Summarize whether canonical rows are usable for the IV benchmark gate."""

    surface_sizes: dict[tuple[str, date], int] = {}
    usable_sizes: dict[tuple[str, date], int] = {}
    dates = sorted({quote.observation_date for quote in quotes})
    underlyings = sorted({quote.underlying for quote in quotes})
    for quote in quotes:
        surface_key = (quote.underlying, quote.observation_date)
        surface_sizes[surface_key] = surface_sizes.get(surface_key, 0) + 1
        if _is_iv_usable_quote(quote):
            usable_sizes[surface_key] = usable_sizes.get(surface_key, 0) + 1

    usable_surface_sizes = [size for size in usable_sizes.values() if size >= min_nodes_per_surface]
    usable_dates = sorted(
        {
            obs_date
            for (_underlying, obs_date), size in usable_sizes.items()
            if size >= min_nodes_per_surface
        }
    )
    return {
        "ok": len(usable_surface_sizes) >= min_surfaces
        and len(usable_dates) >= min_observation_dates,
        "row_count": len(quotes),
        "iv_usable_row_count": sum(1 for quote in quotes if _is_iv_usable_quote(quote)),
        "underlying_count": len(underlyings),
        "underlyings": underlyings,
        "surface_count": len(surface_sizes),
        "usable_surface_count": len(usable_surface_sizes),
        "distinct_observation_date_count": len(dates),
        "distinct_usable_observation_date_count": len(usable_dates),
        "observation_dates": [value.isoformat() for value in dates],
        "usable_observation_dates": [value.isoformat() for value in usable_dates],
        "min_nodes_per_surface": min_nodes_per_surface,
        "min_required_surfaces": min_surfaces,
        "min_required_observation_dates": min_observation_dates,
        "surface_size_min": min(surface_sizes.values()) if surface_sizes else 0,
        "surface_size_p50": (
            float(np.median(list(surface_sizes.values()))) if surface_sizes else 0.0
        ),
        "surface_size_max": max(surface_sizes.values()) if surface_sizes else 0,
        "usable_surface_size_min": min(usable_surface_sizes) if usable_surface_sizes else 0,
        "usable_surface_size_p50": float(np.median(usable_surface_sizes))
        if usable_surface_sizes
        else 0.0,
        "usable_surface_size_max": max(usable_surface_sizes) if usable_surface_sizes else 0,
    }


def data_stage_from_gate(gate: dict[str, Any]) -> str:
    """Classify an expanded silver panel against the data-first benchmark ladder."""

    usable_surfaces = int(gate.get("usable_surface_count", 0))
    usable_dates = int(gate.get("distinct_usable_observation_date_count", 0))
    stage = "under_gate"
    for name, target in DATA_VERSION_TARGETS.items():
        if usable_surfaces >= target["usable_surfaces"] and usable_dates >= target["usable_dates"]:
            stage = name
    return stage


def table_schema_fingerprint(path: Path) -> str:
    df = pd.read_parquet(path)
    return _frame_schema_fingerprint(df)


def _frame_schema_fingerprint(df: pd.DataFrame) -> str:
    payload = json.dumps([(str(col), str(dtype)) for col, dtype in df.dtypes.items()])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _vendor_config(prefix: str) -> dict[str, Any]:
    base_key = f"{prefix}_API_BASE_URL" if prefix == "JQUANTS" else f"{prefix}_BASE_URL"
    return {
        "base_url": os.environ.get(base_key, _default_base_url(prefix)).rstrip("/"),
        "timeout": _env_float(f"{prefix}_REQUEST_TIMEOUT_SECONDS", 30.0),
        "max_retries": _env_int(f"{prefix}_MAX_RETRIES", 2),
        "backoff": _env_float(f"{prefix}_RETRY_BACKOFF_SECONDS", 1.0),
    }


def _is_jquants_v2(config: dict[str, Any]) -> bool:
    return "/v2" in str(config["base_url"]).rstrip("/")


def _default_base_url(prefix: str) -> str:
    if prefix == "MASSIVE":
        return "https://api.massive.com"
    if prefix == "JQUANTS":
        return "https://api.jquants.com/v1"
    return ""


def _client(config: dict[str, Any]) -> httpx.Client:
    return httpx.Client(base_url=str(config["base_url"]), timeout=float(config["timeout"]))


def _request_json(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    json_body: dict[str, str] | None = None,
) -> dict[str, Any]:
    response = client.request(method, path, params=params, headers=headers, json=json_body)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        msg = "expected JSON object response"
        raise RuntimeError(msg)
    return payload


def _massive_get_json(
    client: httpx.Client,
    path: str,
    params: dict[str, str],
    api_key: str,
) -> dict[str, Any]:
    request_params = {**params, "apiKey": api_key}
    return _request_json(client, "GET", path, params=request_params)


def _jquants_v1_endpoint_probe(client: httpx.Client, token: str, endpoint: str) -> ProbeResult:
    try:
        params = (
            {"date": "20260430"}
            if "option" in endpoint or "derivatives" in endpoint
            else {"from": "20260429", "to": "20260430"}
        )
        payload = jquants_get_json(
            client,
            endpoint,
            token,
            params,
        )
        rows = _first_list(payload)
        return ProbeResult(
            "jquants",
            "options" if "option" in endpoint else "equities",
            True,
            f"{endpoint} returned {len(rows)} rows",
            endpoint=endpoint,
            row_count=len(rows),
            schema_fingerprint=schema_fingerprint(rows),
        )
    except Exception as exc:
        return ProbeResult("jquants", "options", False, _redact_error(exc), endpoint=endpoint)


def _jquants_v2_endpoint_probe(client: httpx.Client, api_key: str, endpoint: str) -> ProbeResult:
    try:
        if endpoint == "/equities/master":
            params = {"date": "2026-04-30"}
            mode = "auth"
        elif endpoint == "/equities/bars/daily":
            params = {"code": "72030", "date": "2026-04-30"}
            mode = "equities"
        else:
            params = {"date": "2026-04-30"}
            mode = "options"
        payload = jquants_v2_get_json(client, endpoint, api_key, params)
        rows = _first_list(payload)
        return ProbeResult(
            "jquants",
            mode,
            True,
            f"{endpoint} returned {len(rows)} rows",
            endpoint=endpoint,
            row_count=len(rows),
            schema_fingerprint=schema_fingerprint(rows),
        )
    except Exception as exc:
        return ProbeResult("jquants", "options", False, _redact_error(exc), endpoint=endpoint)


def _secret_from_optional_file(file_env_var: str, explicit_file: str | None) -> str | None:
    if explicit_file:
        path = Path(os.path.expanduser(os.path.expandvars(explicit_file)))
        if path.is_file():
            return path.read_text().strip() or None
    return read_secret_file(file_env_var)


def _resolve_massive_options_flat_file_template(explicit_template: str | None = None) -> str | None:
    template = (
        explicit_template
        or os.environ.get("MASSIVE_OPTIONS_FLAT_FILE_TEMPLATE")
        or os.environ.get("MASSIVE_OPTION_FLAT_FILE_KEY_TEMPLATE")
    )
    if not template:
        return None
    if explicit_template:
        return template
    parsed = urlparse(template)
    if parsed.scheme or Path(os.path.expanduser(os.path.expandvars(template))).is_absolute():
        return template
    if _looks_like_flatfile_key(template):
        bucket = os.environ.get("MASSIVE_FLAT_FILE_BUCKET", "flatfiles").strip("/") or "flatfiles"
        return f"s3://{bucket}/{template.lstrip('/')}"
    return template


def _resolve_massive_underlying_flat_file_template(
    explicit_template: str | None = None,
) -> str | None:
    template = (
        explicit_template
        or os.environ.get("MASSIVE_UNDERLYING_FLAT_FILE_TEMPLATE")
        or os.environ.get("MASSIVE_UNDERLYING_FLAT_FILE_KEY_TEMPLATE")
    )
    if not template:
        return None
    if explicit_template:
        return template
    parsed = urlparse(template)
    if parsed.scheme or Path(os.path.expanduser(os.path.expandvars(template))).is_absolute():
        return template
    if _looks_like_flatfile_key(template):
        bucket = os.environ.get("MASSIVE_FLAT_FILE_BUCKET", "flatfiles").strip("/") or "flatfiles"
        return f"s3://{bucket}/{template.lstrip('/')}"
    return template


def _looks_like_flatfile_key(template: str) -> bool:
    return template.startswith(
        (
            "us_options_opra/",
            "us_stocks_sip/",
            "global_crypto/",
            "global_forex/",
            "us_indices/",
        )
    )


def _format_flat_file_template(template: str, obs_date: date) -> str:
    dataset = (
        os.environ.get("MASSIVE_OPTIONS_FLAT_FILE_DATASET")
        or os.environ.get("MASSIVE_OPTION_FLAT_FILE_DATASET")
        or DEFAULT_OPTION_FLAT_FILE_DATASET
    )
    return template.format(
        date=obs_date.isoformat(),
        yyyymmdd=obs_date.strftime("%Y%m%d"),
        year=obs_date.strftime("%Y"),
        month=obs_date.strftime("%m"),
        day=obs_date.strftime("%d"),
        dataset=dataset,
    )


def _read_flat_file_rows(path_or_url: str) -> list[dict[str, Any]]:
    parsed = urlparse(path_or_url)
    if parsed.scheme == "s3":
        frame = _read_s3_flat_file_frame(path_or_url)
        return [_dict(row) for row in frame.to_dict(orient="records")]
    path_text = parsed.path if parsed.scheme == "file" else path_or_url
    if parsed.scheme in {"", "file"} and not Path(path_text).is_file():
        msg = f"flat file not found: {path_or_url}"
        raise FileNotFoundError(msg)
    if path_or_url.endswith(".parquet"):
        frame = pd.read_parquet(path_or_url)
    elif path_or_url.endswith((".jsonl", ".ndjson")):
        frame = pd.read_json(path_or_url, lines=True)
    else:
        frame = pd.read_csv(path_or_url)
    return [_dict(row) for row in frame.to_dict(orient="records")]


def _read_s3_flat_file_frame(uri: str) -> pd.DataFrame:
    endpoint = os.environ.get("MASSIVE_FLAT_FILE_ENDPOINT_URL", "https://files.polygon.io")
    env = os.environ.copy()
    credentials = _read_massive_flat_file_credentials()
    if credentials is not None:
        env["AWS_ACCESS_KEY_ID"] = credentials[0]
        env["AWS_SECRET_ACCESS_KEY"] = credentials[1]
    command = ["aws", "s3", "cp", uri, "-", "--endpoint-url", endpoint]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=env,
            timeout=max(_env_float("MASSIVE_REQUEST_TIMEOUT_SECONDS", 30.0), 30.0) * 4,
        )
    except FileNotFoundError as exc:
        msg = "aws CLI is required for s3:// Massive flat-file templates"
        raise RuntimeError(msg) from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip()
        msg = f"aws s3 cp failed for {uri}: {detail}"
        raise RuntimeError(msg)
    payload = gzip.decompress(completed.stdout) if uri.endswith(".gz") else completed.stdout
    return pd.read_csv(io.BytesIO(payload))


def _read_massive_flat_file_credentials() -> tuple[str, str] | None:
    path_value = os.environ.get("MASSIVE_FLAT_FILE_KEY_FILE")
    if not path_value:
        return None
    path = Path(os.path.expanduser(os.path.expandvars(path_value)))
    if not path.is_file():
        return None
    text = path.read_text().strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        access = payload.get("access_key_id") or payload.get("access_key") or payload.get("key")
        secret = (
            payload.get("secret_access_key") or payload.get("secret_key") or payload.get("secret")
        )
        if access and secret:
            return str(access).strip(), str(secret).strip()
    pairs: dict[str, str] = {}
    values: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            pairs[key.strip().lower()] = value.strip().strip('"').strip("'")
        else:
            values.append(line)
    access = pairs.get("access_key_id") or pairs.get("aws_access_key_id") or pairs.get("access_key")
    secret = (
        pairs.get("secret_access_key")
        or pairs.get("aws_secret_access_key")
        or pairs.get("secret_key")
    )
    if access and secret:
        return access, secret
    if len(values) >= 2:
        return values[0], values[1]
    return None


def _quotes_by_underlying(quotes: list[OptionQuote]) -> dict[str, list[OptionQuote]]:
    grouped: dict[str, list[OptionQuote]] = {}
    for quote in quotes:
        grouped.setdefault(quote.underlying, []).append(quote)
    return grouped


def _underlying_from_option_symbol(symbol: str) -> str | None:
    parsed = _parse_option_symbol(symbol)
    return str(parsed["underlying"]) if parsed else None


def _parse_option_symbol(symbol: str) -> dict[str, Any] | None:
    text = symbol.upper().removeprefix("O:")
    match = re.match(r"([A-Z.]+)(\d{2})(\d{2})(\d{2})([CP])(\d{8})$", text)
    if not match:
        return None
    year = 2000 + int(match.group(2))
    expiry = date(year, int(match.group(3)), int(match.group(4)))
    return {
        "underlying": match.group(1),
        "expiry": expiry,
        "option_type": "C" if match.group(5) == "C" else "P",
        "strike": int(match.group(6)) / 1000.0,
    }


def _option_type_from_symbol(symbol: Any) -> OptionType | None:
    if symbol in (None, ""):
        return None
    text = str(symbol).upper()
    match = re.search(r"\d{6}([CP])\d+", text)
    if not match:
        return None
    return "C" if match.group(1) == "C" else "P"


def _extract_list(payload: dict[str, Any], keys: list[str]) -> list[dict[str, Any]]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [_dict(row) for row in value if isinstance(row, dict)]
    return []


def _first_list(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for value in payload.values():
        if isinstance(value, list):
            return [_dict(row) for row in value if isinstance(row, dict)]
    return []


def _dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _pick(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _parse_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    text = str(value)
    if len(text) == 6 and text.isdigit():
        return date(int(text[:4]), int(text[4:6]), 1)
    if len(text) == 8 and text.isdigit():
        return date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _date_from_vendor_timestamp(value: Any) -> date | None:
    if value in (None, ""):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return _parse_date(value)
    if number > 10**17:
        seconds = number / 1_000_000_000
    elif number > 10**14:
        seconds = number / 1_000_000
    elif number > 10**11:
        seconds = number / 1_000
    else:
        seconds = float(number)
    return datetime.fromtimestamp(seconds, tz=UTC).date()


def _date_from_bronze_payload_name(path: Path) -> date | None:
    return _parse_date(path.name.split("_", 1)[0])


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int:
    if value in (None, ""):
        return 0
    try:
        return max(int(float(value)), 0)
    except (TypeError, ValueError):
        return 0


def _option_type(value: Any) -> OptionType | None:
    if value in (None, ""):
        return None
    text = str(value).strip().upper()
    if text in {"C", "CALL", "2"} or "CALL" in text:
        return "C"
    if text in {"P", "PUT", "1"} or "PUT" in text:
        return "P"
    return None


def _has_valid_bid_ask(quote: OptionQuote) -> bool:
    return quote.bid is not None and quote.ask is not None and quote.ask >= quote.bid


def _is_iv_usable_quote(quote: OptionQuote) -> bool:
    return _has_valid_bid_ask(quote) and quote.implied_vol is not None and quote.implied_vol > 0


def _option_quote_key(quote: OptionQuote) -> tuple[Any, ...]:
    return (
        quote.market,
        quote.underlying,
        quote.observation_date,
        quote.expiry,
        float(quote.strike),
        quote.option_type,
    )


def _quote_quality_score(quote: OptionQuote) -> tuple[int, int, int, int, int, str]:
    return (
        int(_is_iv_usable_quote(quote)),
        int(_has_valid_bid_ask(quote)),
        int((quote.forward or quote.underlying_price or 0) > 0),
        int(quote.volume),
        int(quote.open_interest),
        quote.vendor_symbol or "",
    )


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _redact_error(exc: Exception) -> str:
    text = str(exc)
    text = re.sub(r"refreshtoken=[^&\s']+", "refreshtoken=<redacted>", text)
    text = re.sub(r"apiKey=[^&\s']+", "apiKey=<redacted>", text)
    parsed = urlparse(text)
    if parsed.query:
        text = text.replace(parsed.query, "<redacted-query>")
    return f"{type(exc).__name__}: {text}"


def _date_range(start: date, end: date) -> list[date]:
    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def _bounded_business_dates(start: date, end: date, max_dates: int) -> list[date]:
    dates = [value for value in _date_range(start, end) if value.weekday() < 5]
    if max_dates <= 0 or len(dates) <= max_dates:
        return dates
    return dates[-max_dates:]
