from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from log_iv.cli import _read_baseline_rows, build_parser, main


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
            "--no-bronze-cache",
        ]
    )

    assert args.market == "jp"
    assert args.max_jp_option_dates == 35
    assert args.tickers == "SPY,QQQ"
    assert args.jp_tickers == "7203,8306"
    assert args.no_bronze_cache


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
