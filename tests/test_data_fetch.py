from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from log_iv import data_fetch
from log_iv.data_fetch import (
    DataFetchConfig,
    black_forward_price,
    fetch_us_flat_file_option_dataset,
    implied_vol_from_option_price,
    jquants_rows_to_option_quotes,
    massive_flat_rows_to_option_quotes,
    massive_snapshot_to_option_quotes,
    probe_imports,
    run_source_probe,
    schema_fingerprint,
    write_data_expansion_report,
)
from log_iv.schema import OptionQuote


def test_massive_snapshot_row_to_canonical_option_quote() -> None:
    timestamp_ns = int(datetime(2026, 4, 30, tzinfo=UTC).timestamp() * 1_000_000_000)
    rows = [
        {
            "details": {
                "expiration_date": "2026-06-19",
                "strike_price": 500.0,
                "contract_type": "call",
                "ticker": "O:SPY260619C00500000",
            },
            "last_quote": {"bid": 3.1, "ask": 3.4, "sip_timestamp": timestamp_ns},
            "day": {"volume": 10},
            "open_interest": 250,
            "implied_volatility": 0.22,
            "underlying_asset": {"price": 505.0},
        }
    ]

    quotes = massive_snapshot_to_option_quotes(rows, "SPY")

    assert len(quotes) == 1
    quote = quotes[0]
    assert quote.market == "US"
    assert quote.underlying == "SPY"
    assert quote.observation_date.isoformat() == "2026-04-30"
    assert quote.expiry.isoformat() == "2026-06-19"
    assert quote.option_type == "C"
    assert quote.bid == 3.1
    assert quote.ask == 3.4
    assert quote.vendor_symbol == "O:SPY260619C00500000"


def test_massive_flat_rows_to_canonical_option_quote() -> None:
    rows = [
        {
            "contract_symbol": "O:SPY260619C00500000",
            "expiration_date": "2026-06-19",
            "strike_price": 500.0,
            "contract_type": "call",
            "bid": 3.1,
            "ask": 3.4,
            "volume": 10,
            "open_interest": 250,
            "implied_volatility": 0.22,
            "underlying_price": 505.0,
        }
    ]

    quotes = massive_flat_rows_to_option_quotes(rows, date(2026, 4, 30), {"SPY"})

    assert len(quotes) == 1
    assert quotes[0].underlying == "SPY"
    assert quotes[0].observation_date == date(2026, 4, 30)


def test_massive_day_agg_symbol_supplies_contract_geometry() -> None:
    rows = [
        {
            "ticker": "O:SPY260619C00500000",
            "close": 3.25,
            "volume": 10,
            "transactions": 7,
        }
    ]

    quotes = massive_flat_rows_to_option_quotes(rows, date(2026, 4, 30), {"SPY"})

    assert len(quotes) == 1
    quote = quotes[0]
    assert quote.underlying == "SPY"
    assert quote.expiry == date(2026, 6, 19)
    assert quote.strike == 500.0
    assert quote.option_type == "C"
    assert quote.bid == 3.25
    assert quote.ask == 3.25


def test_us_flat_file_ingestion_writes_per_date_manifest(tmp_path: Path) -> None:
    flat_path = tmp_path / "options_20260430.csv"
    flat_path.write_text(
        "\n".join(
            [
                "contract_symbol,expiration_date,strike_price,contract_type,bid,ask,implied_volatility",
                "O:SPY260619C00500000,2026-06-19,500,call,3.1,3.4,0.22",
            ]
        )
    )
    cfg = DataFetchConfig(
        bronze_dir=tmp_path / "bronze",
        silver_dir=tmp_path / "silver",
        gold_dir=tmp_path / "gold",
        reports_dir=tmp_path / "reports",
        us_tickers=["SPY"],
    )

    dataset, manifests = fetch_us_flat_file_option_dataset(
        cfg,
        start=date(2026, 4, 30),
        end=date(2026, 4, 30),
        path_template=str(flat_path),
    )

    assert len(dataset) == 1
    assert manifests


def test_us_flat_file_ingestion_infers_iv_from_underlying_close(tmp_path: Path) -> None:
    option_price = black_forward_price(
        option_type="C",
        forward=100.0,
        strike=100.0,
        tenor_years=(date(2026, 6, 19) - date(2026, 4, 30)).days / 365.25,
        sigma=0.2,
    )
    option_path = tmp_path / "options_20260430.csv"
    option_path.write_text(
        "\n".join(
            [
                "ticker,close,volume",
                f"O:SPY260619C00100000,{option_price},10",
            ]
        )
    )
    stock_path = tmp_path / "stocks_20260430.csv"
    stock_path.write_text("ticker,close\nSPY,100.0\n")
    cfg = DataFetchConfig(
        bronze_dir=tmp_path / "bronze",
        silver_dir=tmp_path / "silver",
        gold_dir=tmp_path / "gold",
        reports_dir=tmp_path / "reports",
        us_tickers=["SPY"],
    )

    dataset, _manifests = fetch_us_flat_file_option_dataset(
        cfg,
        start=date(2026, 4, 30),
        end=date(2026, 4, 30),
        path_template=str(option_path),
        underlying_path_template=str(stock_path),
    )

    quote = dataset["US_FLAT_SPY_2026-04-30"][0]
    assert quote.underlying_price == 100.0
    assert quote.forward == 100.0
    assert quote.implied_vol == pytest.approx(0.2, abs=1e-4)


def test_black_forward_implied_vol_inversion_round_trips() -> None:
    price = black_forward_price(
        option_type="P",
        forward=95.0,
        strike=100.0,
        tenor_years=45 / 365.25,
        sigma=0.35,
    )

    implied_vol = implied_vol_from_option_price(
        option_type="P",
        forward=95.0,
        strike=100.0,
        tenor_years=45 / 365.25,
        price=price,
    )

    assert implied_vol == pytest.approx(0.35, abs=1e-5)


def test_data_expansion_writes_deduped_expanded_silver_and_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def quote(market: str, underlying: str, obs: date, strike: float) -> OptionQuote:
        return OptionQuote(
            market=market,
            underlying=underlying,
            observation_date=obs,
            expiry=date(2026, 6, 19),
            strike=strike,
            option_type="C",
            bid=1.0,
            ask=1.1,
            implied_vol=0.2,
            volume=10,
            open_interest=100,
            underlying_price=100.0,
        )

    us_quotes = [quote("US", "SPY", date(2026, 4, 30), 80.0 + i) for i in range(20)]
    us_quotes.append(us_quotes[0])
    jp_quotes = [
        quote("JP", "N225", date(2026, 4, 1 + day), 30000.0 + i)
        for day in range(20)
        for i in range(20)
    ]

    monkeypatch.setattr(
        data_fetch,
        "fetch_us_flat_file_option_dataset",
        lambda *_args, **_kwargs: ({"us": us_quotes}, [tmp_path / "us_manifest.json"]),
    )
    monkeypatch.setattr(
        data_fetch,
        "fetch_jp_option_dataset",
        lambda *_args, **_kwargs: ({"jp": jp_quotes}, [tmp_path / "jp_manifest.json"]),
    )
    cfg = DataFetchConfig(
        bronze_dir=tmp_path / "bronze",
        silver_dir=tmp_path / "silver",
        gold_dir=tmp_path / "gold",
        reports_dir=tmp_path / "reports",
    )

    report_path = write_data_expansion_report(
        start=date(2026, 4, 1), end=date(2026, 4, 30), config=cfg
    )

    report = json.loads(report_path.read_text())
    us_table = Path(report["expanded_silver_tables"]["us"])
    jp_table = Path(report["expanded_silver_tables"]["jp"])
    assert us_table.is_file()
    assert jp_table.is_file()
    assert len(pd.read_parquet(us_table)) == 20
    assert len(pd.read_parquet(jp_table)) == 400
    us_manifest = json.loads(us_table.with_suffix(".manifest.json").read_text())
    assert us_manifest["iv_source"] == "option_mid_price_with_underlying_daily_close"
    assert report["us_flat"]["iv_method"] == "black_forward_bisection_zero_rate_zero_dividend"
    assert report["us_flat"]["surface_gate"]["usable_surface_count"] == 1
    assert report["jp_date_loop"]["surface_gate"]["distinct_usable_observation_date_count"] == 20


def test_jquants_row_to_canonical_option_quote() -> None:
    rows = [
        {
            "Date": "20260430",
            "LTD": "20260612",
            "Strike": "39000",
            "PCDiv": "1",
            "C": "215.0",
            "Vo": "42",
            "OI": "1000",
            "UnderPx": "38000",
            "Code": "N225_TEST",
        }
    ]

    quotes = jquants_rows_to_option_quotes(rows, "/derivatives/bars/daily/options/225")

    assert len(quotes) == 1
    quote = quotes[0]
    assert quote.market == "JP"
    assert quote.underlying == "N225"
    assert quote.option_type == "P"
    assert quote.bid == 215.0
    assert quote.ask == 215.0
    assert quote.underlying_price == 38000
    assert quote.open_interest == 1000


def test_probe_imports_reports_ml_import_failure() -> None:
    real_import = __import__("importlib").import_module

    def fake_import(name: str, package: str | None = None) -> Any:
        if name == "torch_geometric":
            raise ModuleNotFoundError("No module named torch_geometric")
        return real_import(name, package)

    with patch("importlib.import_module", side_effect=fake_import):
        results = probe_imports()

    failures = [result for result in results if not result.ok]
    assert any("torch_geometric" in result.message for result in failures)


def test_unknown_source_probe_is_explicit_failure() -> None:
    results = run_source_probe("unknown", "auto")

    assert len(results) == 1
    assert not results[0].ok


def test_schema_fingerprint_is_stable_for_key_order() -> None:
    left = schema_fingerprint([{"a": 1, "b": "x"}])
    right = schema_fingerprint([{"b": "y", "a": 2}])

    assert left == right
