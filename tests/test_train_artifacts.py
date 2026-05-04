from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
import pytest
import torch

from log_iv.cli import build_parser
from log_iv.schema import OptionQuote
from log_iv.train import (
    OptionTokenGNN,
    TrainingConfig,
    _baseline_summary,
    _metrics_summary,
    _no_arbitrage_diagnostics,
    _price_diagnostics,
    _train_only_baseline_summary,
    _write_run_artifacts,
    compute_losses,
    prepare_registered_splits,
)


def _quote(
    *,
    strike: float,
    expiry: date,
    option_type: Literal["C", "P"] = "C",
    iv: float = 0.2,
    bid: float = 2.0,
    ask: float = 2.2,
    surface: int = 0,
    obs: date = date(2026, 1, 2),
) -> OptionQuote:
    return OptionQuote(
        market="US",
        underlying=f"T{surface}",
        observation_date=obs,
        expiry=expiry,
        strike=strike,
        option_type=option_type,
        bid=bid,
        ask=ask,
        implied_vol=iv,
        volume=100 + surface,
        open_interest=500 + surface,
        forward=100.0,
        underlying_price=100.0,
    )


def _surface(surface: int) -> list[OptionQuote]:
    return [
        _quote(strike=95.0, expiry=date(2026, 2, 20), surface=surface, iv=0.20),
        _quote(strike=100.0, expiry=date(2026, 2, 20), surface=surface, iv=0.21),
        _quote(strike=105.0, expiry=date(2026, 3, 20), surface=surface, iv=0.22),
    ]


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def encode_graph(
        self, surface: list[OptionQuote], device: torch.device
    ) -> dict[str, torch.Tensor]:
        quotes = getattr(surface, "target_quotes", surface)
        mask = getattr(surface, "target_mask", tuple(False for _ in quotes))
        iv = torch.tensor(
            [[quote.implied_vol or 0.0] for quote in quotes],
            dtype=torch.float32,
            device=device,
        )
        return {
            "iv_pred": iv + 0.01,
            "target_price": iv,
            "target_mask": torch.tensor(mask, dtype=torch.bool, device=device),
        }


def test_write_run_artifacts_exports_all_val_and_test_predictions(tmp_path: Path) -> None:
    surfaces = [_surface(i) for i in range(6)]
    output_dir = tmp_path / "run"

    _write_run_artifacts(
        output_dir=output_dir,
        model=DummyModel(),  # type: ignore[arg-type]
        config=TrainingConfig(n_epochs=1, experiment_name="unit"),
        run_label="engineering_smoke",
        dataset_label="unit",
        surface_ids=[f"s{i}" for i in range(6)],
        all_graphs=surfaces,
        train_data=surfaces[:1],
        val_data=surfaces[1:5],
        test_data=surfaces[5:],
        metrics=[],
        final_val_loss=0.1,
        final_test_loss=0.2,
        source_surface_count=6,
        filtered_surface_count=0,
    )

    predictions = pd.read_parquet(output_dir / "predictions.parquet")
    splits = json.loads((output_dir / "splits.json").read_text())

    assert len(predictions) == 15
    assert set(predictions["surface_id"]) == {"s1", "s2", "s3", "s4", "s5"}
    assert splits["val_rows"] == 12
    assert splits["test_rows"] == 3
    assert (output_dir / "diagnostics_price.json").is_file()
    assert (output_dir / "diagnostics_no_arbitrage.json").is_file()


def test_baseline_summary_contains_first_matrix_baselines() -> None:
    frame = pd.DataFrame(
        {
            "split": ["val", "val", "test", "test"],
            "surface_id": ["a", "a", "b", "b"],
            "underlying": ["SPY", "SPY", "QQQ", "QQQ"],
            "option_type": ["C", "P", "C", "P"],
            "log_moneyness": [-0.02, 0.02, -0.01, 0.01],
            "tenor_days": [30, 30, 60, 60],
            "iv_true": [0.2, 0.22, 0.3, 0.32],
            "iv_pred": [0.21, 0.21, 0.29, 0.31],
        }
    )

    baselines = _baseline_summary(frame)

    assert set(baselines["baseline"]) == {
        "mean_iv_global",
        "mean_iv_by_underlying",
        "mean_iv_by_surface",
        "knn_moneyness_tenor",
    }


def test_temporal_split_is_deterministic_without_date_overlap() -> None:
    surfaces = [
        [_quote(strike=100.0, expiry=date(2026, 6, 20), surface=i % 2, obs=date(2026, 1, 2 + i))]
        for i in range(6)
    ]
    cfg = TrainingConfig(split_mode="temporal", task_mode="observed_reconstruction", seed=7)

    left = prepare_registered_splits(surfaces, cfg, [f"s{i}" for i in range(6)])
    right = prepare_registered_splits(surfaces, cfg, [f"s{i}" for i in range(6)])

    assert left.manifest["split_id"] == right.manifest["split_id"]
    split_dates = {
        split: {
            quote.observation_date
            for surface in getattr(left, split)
            for quote in getattr(surface, "target_quotes", surface)
        }
        for split in ("train", "val", "test")
    }
    assert split_dates["train"].isdisjoint(split_dates["val"])
    assert split_dates["train"].isdisjoint(split_dates["test"])
    assert split_dates["val"].isdisjoint(split_dates["test"])


def test_ticker_holdout_excludes_heldout_from_train() -> None:
    surfaces = [_surface(i) for i in range(4)]
    cfg = TrainingConfig(split_mode="ticker_holdout", heldout_tickers=("T3",))

    split = prepare_registered_splits(surfaces, cfg, [f"s{i}" for i in range(4)])

    train_tickers = {quote.underlying for surface in split.train for quote in surface}
    test_tickers = {quote.underlying for surface in split.test for quote in surface}
    assert "T3" not in train_tickers
    assert test_tickers == {"T3"}


def test_masked_reconstruction_removes_target_values_from_inputs() -> None:
    surfaces = [_surface(i) for i in range(3)]
    cfg = TrainingConfig(task_mode="masked_reconstruction", mask_fraction=0.5, seed=11)

    split = prepare_registered_splits(surfaces, cfg, [f"s{i}" for i in range(3)])
    masked_surface = next(
        surface for surface in split.train + split.val + split.test if any(surface.target_mask)
    )
    masked_index = next(index for index, masked in enumerate(masked_surface.target_mask) if masked)

    assert masked_surface.target_quotes[masked_index].implied_vol is not None
    assert masked_surface.input_quotes[masked_index].implied_vol is None
    assert masked_surface.input_quotes[masked_index].bid is None
    assert masked_surface.input_quotes[masked_index].ask is None
    assert masked_surface.input_quotes[masked_index].volume == 0
    assert masked_surface.input_quotes[masked_index].open_interest == 0


def test_masked_metrics_use_masked_nodes_for_headline() -> None:
    frame = pd.DataFrame(
        {
            "iv_true": [0.2, 0.2],
            "iv_pred": [0.3, 1.2],
            "is_masked_target": [True, False],
            "split": ["test", "test"],
            "option_type": ["C", "C"],
            "underlying": ["SPY", "SPY"],
            "volume": [1, 1],
            "open_interest": [1, 1],
            "log_moneyness": [0.0, 0.0],
            "tenor_days": [30, 30],
        }
    )

    metrics = _metrics_summary(frame, 0.0, 0.0, "unit")

    assert metrics["headline_metric"] == "masked_iv_mae"
    assert metrics["masked_iv_mae"] == pytest.approx(0.1)
    assert metrics["masked_iv_p90_abs_error"] == pytest.approx(0.1)
    assert metrics["iv_mae"] == pytest.approx(0.55)


def test_train_only_baselines_do_not_use_eval_targets() -> None:
    train_surface = [_quote(strike=100.0, expiry=date(2026, 2, 20), iv=0.2)]
    predictions = pd.DataFrame(
        {
            "split": ["test"],
            "surface_id": ["eval"],
            "underlying": ["T0"],
            "option_type": ["C"],
            "log_moneyness": [0.0],
            "tenor_days": [30],
            "iv_true": [9.0],
            "iv_pred": [9.0],
            "is_masked_target": [True],
        }
    )

    baselines = _train_only_baseline_summary([train_surface], predictions)
    global_row = baselines[baselines["baseline"] == "train_mean_iv_global"].iloc[0]

    assert global_row["iv_mae"] == pytest.approx(8.8)
    assert "train_only" in set(baselines["fit_scope"])
    assert "eval_visible_only" in set(baselines["fit_scope"])


def test_within_surface_baselines_use_only_visible_nodes_after_mask() -> None:
    train_surface = [_quote(strike=100.0, expiry=date(2026, 2, 20), iv=0.2)]
    predictions = pd.DataFrame(
        {
            "split": ["test", "test", "test"],
            "surface_id": ["eval", "eval", "eval"],
            "underlying": ["T0", "T0", "T0"],
            "option_type": ["C", "C", "C"],
            "expiry": ["2026-02-20", "2026-02-20", "2026-02-20"],
            "log_moneyness": [-0.2, 0.0, 0.2],
            "tenor_days": [30, 30, 30],
            "tenor_years": [30 / 365.25, 30 / 365.25, 30 / 365.25],
            "iv_true": [0.2, 9.0, 0.3],
            "iv_pred": [0.2, 9.0, 0.3],
            "is_masked_target": [False, True, False],
            "input_target_visible": [True, False, True],
        }
    )

    baselines = _train_only_baseline_summary([train_surface], predictions)
    knn = baselines[baselines["baseline"] == "within_surface_knn_moneyness_tenor"].iloc[0]
    svi = baselines[baselines["baseline"] == "within_surface_svi_raw_per_expiry"].iloc[0]

    assert knn["visible_context_policy"] == "visible_nodes_after_mask"
    assert bool(knn["uses_masked_quote_features"]) is False
    assert knn["iv_mae"] == pytest.approx(8.75)
    assert svi["underidentified_rate"] == pytest.approx(1.0)
    assert svi["predicted_rows"] == 0


def test_fast_baseline_preset_scopes_eval_splits_and_skips_svi() -> None:
    train_surface = [_quote(strike=100.0, expiry=date(2026, 2, 20), iv=0.2)]
    predictions = pd.DataFrame(
        {
            "split": ["val", "val", "ood_jp"],
            "surface_id": ["eval", "eval", "jp_eval"],
            "underlying": ["T0", "T0", "7203"],
            "option_type": ["C", "C", "C"],
            "expiry": ["2026-02-20", "2026-02-20", "2026-02-20"],
            "log_moneyness": [-0.1, 0.1, 0.0],
            "tenor_days": [30, 30, 30],
            "tenor_years": [30 / 365.25, 30 / 365.25, 30 / 365.25],
            "iv_true": [0.2, 0.3, 5.0],
            "iv_pred": [0.2, 0.3, 5.0],
            "is_masked_target": [False, True, False],
            "input_target_visible": [True, False, True],
        }
    )

    baselines = _train_only_baseline_summary(
        [train_surface],
        predictions,
        baseline_preset="fast",
        baseline_eval_splits=("val", "test"),
    )

    assert "within_surface_svi_raw_per_expiry" not in set(baselines["baseline"])
    assert set(baselines["baseline_eval_splits"]) == {"val,test"}
    assert set(baselines["eval_rows"]) == {1}


def test_encoder_mlp_model_kind_skips_gnn() -> None:
    model = OptionTokenGNN(
        TrainingConfig(model_kind="encoder_mlp", n_gnn_layers=0, d_model=16, n_encoder_layers=1)
    )

    out = model.encode_graph(_surface(1), torch.device("cpu"))

    assert model.gnn is None
    assert out["layer_outputs"] == []
    assert out["iv_pred"].shape == (3, 1)


def test_decoded_regularizer_does_not_call_embedding_proxy_paths() -> None:
    model = OptionTokenGNN(
        TrainingConfig(
            d_model=16,
            n_encoder_layers=1,
            n_gnn_layers=0,
            no_arb_weight=1.0,
            calendar_weight=1.0,
            butterfly_weight=1.0,
            convexity_weight=1.0,
            put_call_weight=1.0,
        )
    )

    def fail_forward(*_args: object, **_kwargs: object) -> dict[str, torch.Tensor]:
        raise AssertionError("embedding proxy regularizer should not be called")

    model.regularizer.forward = fail_forward  # type: ignore[method-assign]
    outputs = [model.encode_graph(_surface(1), torch.device("cpu"))]
    losses = compute_losses(model, outputs, model.config)

    assert losses["no_arb"].item() >= 0.0


def test_zero_target_greeks_loss_is_disabled_by_default() -> None:
    model = OptionTokenGNN(
        TrainingConfig(d_model=16, n_encoder_layers=1, n_gnn_layers=0, no_arb_weight=0.0)
    )

    outputs = [model.encode_graph(_surface(1), torch.device("cpu"))]
    losses = compute_losses(model, outputs, model.config)

    assert losses["greeks"].item() == 0.0


def test_diagnostics_detect_price_and_no_arb_violations() -> None:
    predictions = pd.DataFrame(
        {
            "split": ["test"] * 6,
            "surface_id": ["s"] * 6,
            "underlying": ["SPY"] * 6,
            "expiry": [
                "2026-02-20",
                "2026-02-20",
                "2026-02-20",
                "2026-03-20",
                "2026-02-20",
                "2026-02-20",
            ],
            "strike": [95.0, 100.0, 105.0, 100.0, 100.0, 100.0],
            "option_type": ["C", "C", "C", "C", "P", "P"],
            "iv_true": [0.2, 0.6, 0.2, 0.1, 0.2, 0.2],
            "iv_pred": [0.2, 0.7, 0.2, 0.05, 0.2, 0.2],
            "bid": [1.0] * 6,
            "ask": [1.2] * 6,
            "mid_price": [1.1] * 6,
            "forward": [100.0] * 6,
            "underlying_price": [100.0] * 6,
            "tenor_years": [
                30 / 365.25,
                30 / 365.25,
                30 / 365.25,
                60 / 365.25,
                30 / 365.25,
                30 / 365.25,
            ],
        }
    )

    price = _price_diagnostics(predictions)
    no_arb = _no_arbitrage_diagnostics(predictions)

    assert price["count"] == 6
    assert no_arb["pred_iv"]["calendar"]["violations"] >= 1
    assert no_arb["pred_iv"]["butterfly_convexity"]["violations"] >= 1
    assert no_arb["pred_iv"]["put_call_parity"]["pairs"] >= 1


def test_no_arb_diagnostics_sample_complete_surfaces() -> None:
    rows = []
    for surface_id in ["s0", "s1", "s2"]:
        for strike in [95.0, 100.0, 105.0]:
            rows.append(
                {
                    "split": "test",
                    "surface_id": surface_id,
                    "underlying": "SPY",
                    "expiry": "2026-02-20",
                    "strike": strike,
                    "option_type": "C",
                    "iv_true": 0.2,
                    "iv_pred": 0.2,
                    "forward": 100.0,
                    "underlying_price": 100.0,
                    "tenor_years": 30 / 365.25,
                }
            )
    predictions = pd.DataFrame(rows)

    diagnostics = _no_arbitrage_diagnostics(
        predictions,
        mode="sampled_surface",
        eval_splits=("test",),
        max_surfaces_per_split=1,
        sample_seed=123,
    )

    assert diagnostics["metadata"]["sampling_unit"] == "surface_id"
    assert diagnostics["metadata"]["surface_count"] == 1
    assert diagnostics["metadata"]["row_count"] == 3
    assert diagnostics["pred_iv"]["butterfly_convexity"]["triples"] == 1


def test_cli_exposes_matrix_and_requires_ood_data() -> None:
    parser = build_parser()

    matrix_args = parser.parse_args(
        [
            "experiment-matrix",
            "--us-data",
            "us.parquet",
            "--jp-data",
            "jp.parquet",
            "--model-kind",
            "gnn",
            "--device",
            "cpu",
            "--log-every",
            "1",
            "--no-arb-weight",
            "0.25",
        ]
    )

    assert matrix_args.command == "experiment-matrix"
    assert matrix_args.device == "cpu"
    assert matrix_args.log_every == 1
    assert matrix_args.no_arb_weight == 0.25

    benchmark_args = parser.parse_args(
        [
            "benchmark-protocol",
            "--us-data",
            "us.parquet",
            "--seeds",
            "1,2,3",
            "--variants",
            "gnn_liq,gnn_no_liq",
            "--torch-threads",
            "4",
            "--mask-regime",
            "liquidity_correlated",
            "--min-us-dates",
            "60",
            "--decoded-regularizer-max-terms",
            "32",
            "--refresh-graph-cache",
        ]
    )
    assert benchmark_args.task == "masked_reconstruction"
    assert benchmark_args.split_mode == "temporal"
    assert benchmark_args.epochs == 100
    assert benchmark_args.variants == "gnn_liq,gnn_no_liq"
    assert benchmark_args.torch_threads == 4
    assert benchmark_args.mask_regime == "liquidity_correlated"
    assert benchmark_args.min_us_dates == 60
    assert benchmark_args.decoded_regularizer_max_terms == 32
    assert benchmark_args.use_graph_cache is True
    assert benchmark_args.refresh_graph_cache is True
    with pytest.raises(SystemExit):
        parser.parse_args(["ood-transfer"])
