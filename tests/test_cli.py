from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from log_iv.cli import _load_option_surface_graphs, _read_baseline_rows, build_parser, main
from log_iv.schema import OptionQuote


def test_status_command_reports_scaffold_state(monkeypatch: Any) -> None:
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.delenv("MASSIVE_API_KEY_FILE", raising=False)

    with patch.object(sys, "argv", ["log-iv", "status"]):
        # Will run without error and print status
        try:
            main(["status"])
        except SystemExit as e:
            assert e.code == 0
        else:
            pass  # no exception is also fine


def test_source_probe_selects_all_sources() -> None:
    parser = build_parser()
    args = parser.parse_args(["source-probe", "massive", "rest"])

    assert args.source == "massive"
    assert args.mode == "rest"


def test_data_expansion_exposes_market_and_date_loop_controls() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "data-expansion",
            "--market",
            "jp",
            "--max-jp-option-dates",
            "35",
            "--tickers",
            "SPY,QQQ",
            "--jp-tickers",
            "7203,8306",
            "--max-workers",
            "4",
            "--no-bronze-cache",
            "--refresh-failed",
            "--refresh-all",
        ]
    )

    assert args.market == "jp"
    assert args.max_jp_option_dates == 35
    assert args.tickers == "SPY,QQQ"
    assert args.jp_tickers == "7203,8306"
    assert args.max_workers == 4
    assert args.no_bronze_cache
    assert args.refresh_failed
    assert args.refresh_all


def test_benchmark_protocol_exposes_postprocess_controls() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark-protocol",
            "--us-data",
            "us.parquet",
            "--baseline-preset",
            "full",
            "--baseline-eval-splits",
            "val,test",
            "--svi-timeout-seconds",
            "0.2",
            "--svi-maxiter",
            "50",
            "--no-arb-diagnostics-mode",
            "sampled_surface",
            "--no-arb-eval-splits",
            "val,test",
            "--no-arb-max-surfaces-per-split",
            "25",
            "--quiet-postprocess",
        ]
    )

    assert args.baseline_preset == "full"
    assert args.baseline_eval_splits == "val,test"
    assert args.svi_timeout_seconds == pytest.approx(0.2)
    assert args.svi_maxiter == 50
    assert args.no_arb_diagnostics_mode == "sampled_surface"
    assert args.no_arb_eval_splits == "val,test"
    assert args.no_arb_max_surfaces_per_split == 25
    assert args.quiet_postprocess


def test_read_baseline_rows_skips_scope_labels(tmp_path: Path) -> None:
    path = tmp_path / "baselines_summary.csv"
    path.write_text(
        "\n".join(
            [
                "baseline,fit_scope,eval_scope,eval_rows,iv_mae,model_delta_mae",
                "train_knn_moneyness_tenor,train_only,masked_nodes,10,0.12,-0.02",
            ]
        )
    )

    rows = _read_baseline_rows(path)

    assert rows["train_knn_moneyness_tenor"]["iv_mae"] == pytest.approx(0.12)
    assert rows["train_knn_moneyness_tenor"]["model_delta_mae"] == pytest.approx(-0.02)
    assert "fit_scope" not in rows["train_knn_moneyness_tenor"]


def test_option_surface_loader_reuses_gold_graph_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    quotes = [
        OptionQuote(
            market="US",
            underlying="SPY",
            observation_date=date(2026, 4, 30),
            expiry=date(2026, 6, 19),
            strike=100.0 + index,
            option_type="C",
            bid=1.0,
            ask=1.1,
            implied_vol=0.2,
        )
        for index in range(3)
    ]
    path = tmp_path / "quotes.parquet"
    pd.DataFrame([quote.model_dump(mode="json") for quote in quotes]).to_parquet(path, index=False)
    monkeypatch.setenv("GOLD_DATA_DIR", str(tmp_path / "gold"))

    graphs, surface_ids, stats = _load_option_surface_graphs(
        path,
        min_nodes_per_surface=2,
        max_nodes_per_surface=2,
        max_surfaces=None,
    )

    assert len(graphs) == 1
    assert len(graphs[0]) == 2
    assert surface_ids == ["SPY_2026-04-30"]
    assert stats["graph_cache_hit"] is False
    assert Path(str(stats["graph_cache_path"])).is_file()

    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("cache miss")),
    )

    cached_graphs, cached_ids, cached_stats = _load_option_surface_graphs(
        path,
        min_nodes_per_surface=2,
        max_nodes_per_surface=2,
        max_surfaces=None,
    )

    assert cached_ids == surface_ids
    assert len(cached_graphs[0]) == 2
    assert cached_stats["graph_cache_hit"] is True


def test_source_probe_exits_nonzero_on_failure() -> None:
    with patch("log_iv.data_fetch.run_source_probe") as probe:
        from log_iv.data_fetch import ProbeResult

        probe.return_value = [ProbeResult("massive", "rest", False, "boom")]
        with pytest.raises(SystemExit) as exc_info:
            main(["source-probe", "massive", "rest"])

    assert exc_info.value.code == 1


def test_toy_graph_command_reports_edge_types() -> None:
    import io

    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    try:
        main(["toy-graph"])
    except SystemExit:
        pass
    finally:
        sys.stdout = old_stdout

    output = buffer.getvalue()
    payload = json.loads(output)
    assert payload["nodes"] == 3
    assert payload["edges"] == 8
    assert set(payload["edge_types"]) == {
        "liquidity_similarity",
        "maturity_neighbor",
        "strike_neighbor",
    }
