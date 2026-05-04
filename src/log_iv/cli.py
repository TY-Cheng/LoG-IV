"""CLI for LoG-IV research project.

Provides subcommands:
- status: Print project metadata.
- source-probe: Verify source tree and imports.
- toy-graph: Build a minimal 3-node graph.
- synth: Generate synthetic datasets.
- train: Run training pipeline (synthetic or real data).
- ood-transfer: Run OOD transfer experiment.
- fetch: Fetch real data from APIs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from dataclasses import replace
from datetime import date, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

try:
    __version__ = importlib_metadata.version("log-iv")
except importlib_metadata.PackageNotFoundError:
    __version__ = "0.1.0"

GRAPH_CACHE_VERSION = 1


def _cmd_status(args: argparse.Namespace) -> None:
    """Print project metadata and config overview."""
    from log_iv.config import ProjectSettings, env_key_status, load_research_config

    cfg = ProjectSettings.from_env()
    research = load_research_config()
    key_status = env_key_status(
        [
            "MASSIVE_API_KEY_FILE",
            "MASSIVE_FLAT_FILE_KEY_FILE",
            "JQUANTS_API_KEY_FILE",
            "FRED_API_KEY_FILE",
        ]
    )

    print("LoG-IV status")
    print(f"  version    : {__version__}")
    print(f"  time       : {datetime.now().isoformat(sep=' ', timespec='seconds')}")
    print(f"  python     : {sys.version.split()[0]}")
    print(f"  project    : {cfg.project_name}")
    print(f"  data_dir   : {cfg.data_dir}  exists={cfg.data_dir.exists()}")
    for d in ("bronze", "silver", "gold"):
        p = cfg.data_dir / d
        print(f"  data/{d:9s} : {p}  exists={p.exists()}")
    print(f"  reports    : {cfg.reports_dir}  exists={cfg.reports_dir.exists()}")
    print(f"  log_level  : {cfg.log_level}")
    print(f"  research   : {research.get('project', {}).get('name', 'N/A')}")
    for k, v in key_status.items():
        print(f"  env        : {k}={'set' if v else 'missing'}")
    print("status ok")


def _cmd_source_probe(args: argparse.Namespace) -> None:
    """Verify imports and/or vendor source readiness."""
    from log_iv.data_fetch import run_source_probe

    source = args.source_option or args.source or "all"
    mode = args.mode_option or args.mode or "auto"
    results = run_source_probe(source=source, mode=mode)
    failures = [result for result in results if not result.ok]
    for result in results:
        status = "OK" if result.ok else "FAIL"
        suffix = f" rows={result.row_count}" if result.row_count is not None else ""
        endpoint = f" endpoint={result.endpoint}" if result.endpoint else ""
        print(f"  {status:4s} {result.source}:{result.mode}{endpoint}{suffix} - {result.message}")
    print(f"\n{len(results) - len(failures)} ok, {len(failures)} fail")
    if failures:
        raise SystemExit(1)


def _cmd_toy_graph(args: argparse.Namespace) -> None:
    """Build a minimal 3-node graph and print its statistics."""
    from log_iv.graph import build_option_surface_graph, summarize_graph
    from log_iv.schema import OptionQuote

    # Minimal quotes — use the actual pydantic model field names
    today = date.today()
    quotes = [
        OptionQuote(
            market="US",
            underlying="TEST",
            observation_date=today,
            expiry=date(2026, 6, 15),
            strike=95.0,
            option_type="C",
            bid=7.0,
            ask=7.2,
            implied_vol=0.25,
            volume=100,
            open_interest=500,
            forward=100.0,
            underlying_price=100.0,
        ),
        OptionQuote(
            market="US",
            underlying="TEST",
            observation_date=today,
            expiry=date(2026, 6, 15),
            strike=100.0,
            option_type="C",
            bid=3.0,
            ask=3.2,
            implied_vol=0.22,
            volume=200,
            open_interest=800,
            forward=100.0,
            underlying_price=100.0,
        ),
        OptionQuote(
            market="US",
            underlying="TEST",
            observation_date=today,
            expiry=date(2026, 9, 15),
            strike=100.0,
            option_type="C",
            bid=5.0,
            ask=5.3,
            implied_vol=0.23,
            volume=150,
            open_interest=600,
            forward=100.0,
            underlying_price=100.0,
        ),
    ]

    g = build_option_surface_graph(quotes, similarity_edges_per_node=1)
    summary = summarize_graph(g, quotes)
    print(json.dumps(summary, indent=2, default=str))


def _cmd_synth(args: argparse.Namespace) -> None:
    """Generate synthetic option surface datasets."""
    from log_iv.synthetic import (
        SyntheticSurfaceConfig,
        generate_synthetic_surface_dataset,
        synthetic_quotes_to_option_quotes,
    )

    config = SyntheticSurfaceConfig(
        n_maturities=args.maturities,
        n_strikes=args.strikes,
        min_tenor_days=args.min_tenor,
        max_tenor_days=args.max_tenor,
        moneyness_range=args.moneyness_range,
        missing_wing_prob=args.missing_prob,
        stale_quote_prob=args.stale_prob,
        random_seed=args.seed,
    )

    print(f"Generating {args.surfaces} surfaces with {args.underlyings} underlyings...")
    dataset = generate_synthetic_surface_dataset(
        config,
        n_surfaces=args.surfaces,
        n_underlyings=args.underlyings,
    )

    # Count stats
    total_quotes = sum(len(synthetic_quotes_to_option_quotes(v)) for v in dataset.values())

    print(f"Generated {len(dataset)} surfaces, {total_quotes} total quotes")
    print(f"Keys: {list(dataset.keys())[:5]}...")

    # Save if output specified
    if args.output:
        import pickle
        from pathlib import Path

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(dataset, f)
        print(f"Saved to {out_path}")


def _cmd_train(args: argparse.Namespace) -> None:
    """Run the training pipeline."""
    from log_iv.train import (
        run_option_quote_dataset_experiment,
        run_synthetic_experiment,
    )

    config = _training_config_from_args(args)

    print(f"Starting training: experiment={args.experiment_name}, epochs={args.epochs}")
    if args.data:
        all_graphs, surface_ids, stats = _load_option_surface_graphs(
            args.data,
            min_nodes_per_surface=args.min_nodes_per_surface,
            max_nodes_per_surface=args.max_nodes_per_surface,
            max_surfaces=args.max_surfaces,
            use_graph_cache=args.use_graph_cache,
            refresh_graph_cache=args.refresh_graph_cache,
        )
        print(json.dumps(stats, indent=2, sort_keys=True))
        run_label = args.run_label
        if run_label is None:
            run_label = "real_jp_mvp" if "jp" in str(args.data).lower() else "real_us_mvp"
        _model, metrics = run_option_quote_dataset_experiment(
            all_graphs,
            surface_ids,
            config,
            run_label=run_label,
            dataset_label=str(args.data),
            source_surface_count=int(stats["source_surface_count"]),
            filtered_surface_count=int(stats["filtered_surface_count"]),
            claim_label=args.claim_label,
        )
    else:
        _model, metrics = run_synthetic_experiment(config)

    final_loss = metrics[-1].val_loss if metrics else float("nan")
    print(f"\nTraining complete. Final val loss: {final_loss:.4f}")


def _training_config_from_args(
    args: argparse.Namespace,
    *,
    experiment_name: str | None = None,
    model_kind: str | None = None,
    similarity_edges_per_node: int | None = None,
    no_arb_weight: float | None = None,
    calendar_weight: float | None = None,
    butterfly_weight: float | None = None,
    put_call_weight: float | None = None,
    convexity_weight: float | None = None,
    smoothness_weight: float | None = None,
    contrastive_weight: float | None = None,
    heteroscedastic_weight: float | None = None,
    reliability_gate_weight: float | None = None,
    use_liquidity_features: bool | None = None,
    use_liquidity_gate: bool | None = None,
) -> Any:
    from log_iv.train import TrainingConfig

    return TrainingConfig(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        d_model=args.d_model,
        n_encoder_layers=args.encoder_layers,
        n_gnn_layers=args.gnn_layers,
        model_kind=model_kind or args.model_kind,
        task_mode=args.task,
        split_mode=args.split_mode,
        heldout_tickers=tuple(
            item.strip().upper() for item in args.heldout_tickers.split(",") if item.strip()
        ),
        mask_fraction=args.mask_fraction,
        mask_regime=getattr(args, "mask_regime", "stratified"),
        output_dir=args.output_dir,
        experiment_name=experiment_name or args.experiment_name,
        seed=args.seed,
        log_every=args.log_every,
        similarity_edges_per_node=(
            args.similarity_edges_per_node
            if similarity_edges_per_node is None
            else similarity_edges_per_node
        ),
        no_arb_weight=args.no_arb_weight if no_arb_weight is None else no_arb_weight,
        smoothness_weight=args.smoothness_weight
        if smoothness_weight is None
        else smoothness_weight,
        contrastive_weight=args.contrastive_weight
        if contrastive_weight is None
        else contrastive_weight,
        calendar_weight=args.calendar_weight if calendar_weight is None else calendar_weight,
        butterfly_weight=args.butterfly_weight if butterfly_weight is None else butterfly_weight,
        put_call_weight=args.put_call_weight if put_call_weight is None else put_call_weight,
        convexity_weight=args.convexity_weight if convexity_weight is None else convexity_weight,
        heteroscedastic_weight=args.heteroscedastic_weight
        if heteroscedastic_weight is None
        else heteroscedastic_weight,
        reliability_gate_weight=args.reliability_gate_weight
        if reliability_gate_weight is None
        else reliability_gate_weight,
        use_liquidity_features=args.use_liquidity_features
        if use_liquidity_features is None
        else use_liquidity_features,
        use_liquidity_gate=args.use_liquidity_gate
        if use_liquidity_gate is None
        else use_liquidity_gate,
        greeks_weight=args.greeks_weight,
        device=args.device,
        torch_num_threads=args.torch_threads,
        decoded_regularizer_max_terms=args.decoded_regularizer_max_terms,
        baseline_preset=args.baseline_preset,
        baseline_eval_splits=_parse_split_scope(args.baseline_eval_splits),
        svi_timeout_seconds=args.svi_timeout_seconds,
        svi_maxiter=args.svi_maxiter,
        postprocess_verbose=not args.quiet_postprocess,
        no_arb_diagnostics_mode=args.no_arb_diagnostics_mode,
        no_arb_eval_splits=_parse_split_scope(args.no_arb_eval_splits),
        no_arb_max_surfaces_per_split=args.no_arb_max_surfaces_per_split,
        no_arb_sample_seed=args.no_arb_sample_seed,
        synthetic_surfaces=getattr(args, "synthetic_surfaces", 8),
        synthetic_underlyings=getattr(args, "synthetic_underlyings", 2),
        synthetic_maturities=getattr(args, "synthetic_maturities", 5),
        synthetic_strikes=getattr(args, "synthetic_strikes", 9),
    )


def _load_option_surface_graphs(
    path: str | Path,
    *,
    min_nodes_per_surface: int,
    max_nodes_per_surface: int | None,
    max_surfaces: int | None,
    use_graph_cache: bool = True,
    refresh_graph_cache: bool = False,
) -> tuple[list[list[Any]], list[str], dict[str, int | float | str]]:
    import pandas as pd

    from log_iv.schema import OptionQuote

    cache_payload, cache_path = _graph_cache_payload_and_path(
        path,
        min_nodes_per_surface=min_nodes_per_surface,
        max_nodes_per_surface=max_nodes_per_surface,
        max_surfaces=max_surfaces,
    )
    if use_graph_cache and not refresh_graph_cache:
        cached = _read_graph_cache(cache_path, cache_payload)
        if cached is not None:
            return cached

    df = pd.read_parquet(path)
    input_rows = int(len(df))
    if "implied_vol" in df.columns:
        missing_iv_row_count = int(df["implied_vol"].isna().sum())
        df = df[df["implied_vol"].notna()].copy()
    else:
        missing_iv_row_count = input_rows
        df = df.iloc[0:0].copy()
    canonical_fields = list(OptionQuote.model_fields)
    all_graphs: list[list[Any]] = []
    surface_ids: list[str] = []
    source_surface_count = 0
    below_min_count = 0
    capped_count = 0
    max_surface_excluded_count = 0
    kept_rows = 0
    kept_sizes: list[int] = []
    kept_dates: set[str] = set()

    for key, group in df.groupby(["underlying", "observation_date"], sort=True):
        source_surface_count += 1
        if len(group) < min_nodes_per_surface:
            below_min_count += 1
            continue
        if max_surfaces and len(all_graphs) >= max_surfaces:
            max_surface_excluded_count += 1
            continue
        if max_nodes_per_surface and len(group) > max_nodes_per_surface:
            group = _evenly_sample_group(group, max_nodes_per_surface)
            capped_count += 1
        raw_rows = group[canonical_fields].to_dict(orient="records")
        rows = [
            {field: (None if pd.isna(value) else value) for field, value in row.items()}
            for row in raw_rows
        ]
        quotes = [OptionQuote(**row) for row in rows]  # type: ignore[arg-type]
        if len(quotes) < min_nodes_per_surface:
            below_min_count += 1
            continue
        all_graphs.append(quotes)
        surface_ids.append("_".join(str(part) for part in key))
        kept_rows += len(quotes)
        kept_sizes.append(len(quotes))
        kept_dates.add(str(key[1]))

    if kept_sizes:
        sizes = pd.Series(kept_sizes)
        min_size = int(sizes.min())
        median_size = float(sizes.median())
        mean_size = float(sizes.mean())
        max_size = int(sizes.max())
    else:
        min_size = 0
        median_size = 0.0
        mean_size = 0.0
        max_size = 0
    stats: dict[str, int | float | str] = {
        "path": str(path),
        "input_rows": input_rows,
        "iv_usable_input_rows": int(len(df)),
        "missing_iv_row_count": missing_iv_row_count,
        "source_surface_count": source_surface_count,
        "kept_surface_count": len(all_graphs),
        "filtered_surface_count": source_surface_count - len(all_graphs),
        "below_min_surface_count": below_min_count,
        "max_surface_excluded_count": max_surface_excluded_count,
        "capped_surface_count": capped_count,
        "kept_rows": kept_rows,
        "distinct_observation_date_count": len(kept_dates),
        "surface_size_min": min_size,
        "surface_size_median": median_size,
        "surface_size_mean": mean_size,
        "surface_size_max": max_size,
        "graph_cache_hit": False,
        "graph_cache_path": str(cache_path),
    }
    if use_graph_cache:
        _write_graph_cache(cache_path, cache_payload, all_graphs, surface_ids, stats)
    return all_graphs, surface_ids, stats


def _graph_cache_payload_and_path(
    path: str | Path,
    *,
    min_nodes_per_surface: int,
    max_nodes_per_surface: int | None,
    max_surfaces: int | None,
) -> tuple[dict[str, Any], Path]:
    source_path = Path(path)
    try:
        stat = source_path.stat()
        resolved_path = str(source_path.resolve())
        source_size = int(stat.st_size)
        source_mtime_ns = int(stat.st_mtime_ns)
    except OSError:
        resolved_path = str(source_path)
        source_size = -1
        source_mtime_ns = -1
    payload = {
        "cache_version": GRAPH_CACHE_VERSION,
        "source_path": resolved_path,
        "source_size": source_size,
        "source_mtime_ns": source_mtime_ns,
        "min_nodes_per_surface": min_nodes_per_surface,
        "max_nodes_per_surface": max_nodes_per_surface,
        "max_surfaces": max_surfaces,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    cache_dir = Path(os.environ.get("GOLD_DATA_DIR", "data/gold")) / "graph_cache"
    return payload, cache_dir / f"{source_path.stem}_{digest}.pkl"


def _read_graph_cache(
    cache_path: Path,
    expected_payload: dict[str, Any],
) -> tuple[list[list[Any]], list[str], dict[str, int | float | str]] | None:
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError):
        return None
    if not isinstance(payload, dict) or payload.get("cache_key") != expected_payload:
        return None
    graphs = payload.get("graphs")
    surface_ids = payload.get("surface_ids")
    stats = payload.get("stats")
    if (
        not isinstance(graphs, list)
        or not isinstance(surface_ids, list)
        or not isinstance(stats, dict)
    ):
        return None
    stats = dict(stats)
    stats["graph_cache_hit"] = True
    stats["graph_cache_path"] = str(cache_path)
    return graphs, surface_ids, stats


def _write_graph_cache(
    cache_path: Path,
    cache_payload: dict[str, Any],
    graphs: list[list[Any]],
    surface_ids: list[str],
    stats: dict[str, int | float | str],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp{cache_path.suffix}")
    payload = {
        "cache_key": cache_payload,
        "graphs": graphs,
        "surface_ids": surface_ids,
        "stats": stats,
    }
    with tmp_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(cache_path)


def _evenly_sample_group(group: Any, max_rows: int) -> Any:
    import numpy as np

    sort_cols = [
        col
        for col in ["expiry", "option_type", "strike", "implied_vol", "volume", "open_interest"]
        if col in group.columns
    ]
    ordered = group.sort_values(sort_cols) if sort_cols else group
    positions = np.linspace(0, len(ordered) - 1, max_rows).round().astype(int)
    return ordered.iloc[positions].drop_duplicates()


def _cmd_ood_transfer(args: argparse.Namespace) -> None:
    """Run OOD transfer experiment (US → Japan)."""
    from log_iv.train import run_ood_transfer_experiment

    config = _training_config_from_args(args, model_kind=args.model_kind)
    us_graphs, us_ids, us_stats = _load_option_surface_graphs(
        args.us_data,
        min_nodes_per_surface=args.min_nodes_per_surface,
        max_nodes_per_surface=args.max_nodes_per_surface,
        max_surfaces=args.max_us_surfaces,
        use_graph_cache=args.use_graph_cache,
        refresh_graph_cache=args.refresh_graph_cache,
    )
    jp_graphs, jp_ids, jp_stats = _load_option_surface_graphs(
        args.jp_data,
        min_nodes_per_surface=args.min_nodes_per_surface,
        max_nodes_per_surface=args.max_nodes_per_surface,
        max_surfaces=args.max_jp_surfaces,
        use_graph_cache=args.use_graph_cache,
        refresh_graph_cache=args.refresh_graph_cache,
    )

    print("U.S. OOD source stats:")
    print(json.dumps(us_stats, indent=2, sort_keys=True))
    print("Japan OOD source stats:")
    print(json.dumps(jp_stats, indent=2, sort_keys=True))
    _model, metrics = run_ood_transfer_experiment(
        us_graphs,
        us_ids,
        jp_graphs,
        jp_ids,
        config,
        dataset_label=f"{args.us_data} -> {args.jp_data}",
        source_surface_count=int(us_stats["source_surface_count"])
        + int(jp_stats["source_surface_count"]),
        filtered_surface_count=int(us_stats["filtered_surface_count"])
        + int(jp_stats["filtered_surface_count"]),
    )
    final_loss = metrics[-1].val_loss if metrics else float("nan")
    print(f"\nOOD training complete. Final U.S. val loss: {final_loss:.4f}")


def _cmd_experiment_matrix(args: argparse.Namespace) -> None:
    """Run the first real-data baseline/model matrix."""
    import pandas as pd

    from log_iv.train import run_option_quote_dataset_experiment

    us_graphs, us_ids, us_stats = _load_option_surface_graphs(
        args.us_data,
        min_nodes_per_surface=args.min_nodes_per_surface,
        max_nodes_per_surface=args.max_nodes_per_surface,
        max_surfaces=args.max_us_surfaces,
        use_graph_cache=args.use_graph_cache,
        refresh_graph_cache=args.refresh_graph_cache,
    )
    jp_graphs, jp_ids, jp_stats = _load_option_surface_graphs(
        args.jp_data,
        min_nodes_per_surface=args.min_nodes_per_surface,
        max_nodes_per_surface=args.max_nodes_per_surface,
        max_surfaces=args.max_jp_surfaces,
        use_graph_cache=args.use_graph_cache,
        refresh_graph_cache=args.refresh_graph_cache,
    )
    print("U.S. matrix source stats:")
    print(json.dumps(us_stats, indent=2, sort_keys=True))
    print("Japan matrix source stats:")
    print(json.dumps(jp_stats, indent=2, sort_keys=True))

    variants = [
        {
            "name": "encoder_mlp",
            "model_kind": "encoder_mlp",
            "similarity_edges_per_node": 0,
            "no_arb_weight": 0.0,
            "calendar_weight": 0.0,
            "butterfly_weight": 0.0,
            "put_call_weight": 0.0,
            "convexity_weight": 0.0,
            "smoothness_weight": 0.0,
            "contrastive_weight": 0.0,
            "claim_label": None,
        },
        {
            "name": "gnn_no_liq",
            "model_kind": "gnn",
            "similarity_edges_per_node": 0,
            "no_arb_weight": 0.0,
            "calendar_weight": 0.0,
            "butterfly_weight": 0.0,
            "put_call_weight": 0.0,
            "convexity_weight": 0.0,
            "smoothness_weight": args.smoothness_weight,
            "contrastive_weight": args.contrastive_weight,
            "claim_label": None,
        },
        {
            "name": "gnn_liq",
            "model_kind": "gnn",
            "similarity_edges_per_node": args.similarity_edges_per_node,
            "no_arb_weight": 0.0,
            "calendar_weight": 0.0,
            "butterfly_weight": 0.0,
            "put_call_weight": 0.0,
            "convexity_weight": 0.0,
            "smoothness_weight": args.smoothness_weight,
            "contrastive_weight": args.contrastive_weight,
            "claim_label": None,
        },
        {
            "name": "gnn_calendar_reg",
            "model_kind": "gnn",
            "similarity_edges_per_node": args.similarity_edges_per_node,
            "no_arb_weight": args.no_arb_weight,
            "calendar_weight": 1.0,
            "butterfly_weight": 0.0,
            "put_call_weight": 0.0,
            "convexity_weight": 0.0,
            "smoothness_weight": args.smoothness_weight,
            "contrastive_weight": args.contrastive_weight,
            "claim_label": None,
        },
        {
            "name": "gnn_all_reg_diag",
            "model_kind": "gnn",
            "similarity_edges_per_node": args.similarity_edges_per_node,
            "no_arb_weight": args.no_arb_weight,
            "calendar_weight": args.calendar_weight,
            "butterfly_weight": args.butterfly_weight,
            "put_call_weight": args.put_call_weight,
            "convexity_weight": args.convexity_weight,
            "smoothness_weight": args.smoothness_weight,
            "contrastive_weight": args.contrastive_weight,
            "claim_label": "diagnostic_only",
        },
    ]

    matrix_rows: list[dict[str, Any]] = []
    for market_name, graphs, surface_ids, stats, run_label in [
        ("us", us_graphs, us_ids, us_stats, "real_us_mvp"),
        ("jp", jp_graphs, jp_ids, jp_stats, "japan_ood_probe"),
    ]:
        for variant in variants:
            experiment_name = f"real_{market_name}_{variant['name']}_{args.epochs}e_seed{args.seed}"
            print(f"\n=== Running {experiment_name} ===")
            config = _training_config_from_args(
                args,
                experiment_name=experiment_name,
                model_kind=str(variant["model_kind"]),
                similarity_edges_per_node=int(variant["similarity_edges_per_node"]),
                no_arb_weight=float(variant["no_arb_weight"]),
                calendar_weight=float(variant["calendar_weight"]),
                butterfly_weight=float(variant["butterfly_weight"]),
                put_call_weight=float(variant["put_call_weight"]),
                convexity_weight=float(variant["convexity_weight"]),
                smoothness_weight=float(variant["smoothness_weight"]),
                contrastive_weight=float(variant["contrastive_weight"]),
            )
            run_option_quote_dataset_experiment(
                graphs,
                surface_ids,
                config,
                run_label=run_label,
                dataset_label=str(args.us_data if market_name == "us" else args.jp_data),
                source_surface_count=int(stats["source_surface_count"]),
                filtered_surface_count=int(stats["filtered_surface_count"]),
                claim_label=str(variant["claim_label"] or run_label),
            )
            matrix_rows.append(
                _matrix_row(
                    Path(args.output_dir) / experiment_name,
                    market=market_name,
                    variant=str(variant["name"]),
                    claim_label=str(variant["claim_label"] or run_label),
                )
            )

    matrix = pd.DataFrame(matrix_rows)
    output_path = Path(args.output_dir) / "matrix_summary.csv"
    matrix.to_csv(output_path, index=False)
    print(f"\nWrote {output_path}")


def _cmd_benchmark_protocol(args: argparse.Namespace) -> None:
    """Run the credible fixed-split masked reconstruction benchmark protocol."""
    import pandas as pd

    from log_iv.train import run_option_quote_dataset_experiment

    seeds = _parse_int_list(args.seeds)
    us_graphs, us_ids, us_stats = _load_option_surface_graphs(
        args.us_data,
        min_nodes_per_surface=args.min_nodes_per_surface,
        max_nodes_per_surface=args.max_nodes_per_surface,
        max_surfaces=args.max_us_surfaces,
        use_graph_cache=args.use_graph_cache,
        refresh_graph_cache=args.refresh_graph_cache,
    )
    jp_stats: dict[str, int | float | str] | None = None
    jp_graphs: list[list[Any]] = []
    jp_ids: list[str] = []
    if args.jp_data:
        jp_graphs, jp_ids, jp_stats = _load_option_surface_graphs(
            args.jp_data,
            min_nodes_per_surface=args.min_nodes_per_surface,
            max_nodes_per_surface=args.max_nodes_per_surface,
            max_surfaces=args.max_jp_surfaces,
            use_graph_cache=args.use_graph_cache,
            refresh_graph_cache=args.refresh_graph_cache,
        )

    acceptance = _benchmark_data_acceptance(args, us_stats, jp_stats)
    print("Benchmark data acceptance:")
    print(json.dumps(acceptance, indent=2, sort_keys=True))
    print(
        "Benchmark postprocess: "
        f"baseline_preset={args.baseline_preset}, "
        f"baseline_eval_splits={args.baseline_eval_splits}, "
        f"svi_timeout_seconds={args.svi_timeout_seconds}, "
        f"svi_maxiter={args.svi_maxiter}, "
        f"variant_suite={args.variant_suite}, "
        f"heteroscedastic_weight={args.heteroscedastic_weight}, "
        f"reliability_gate_weight={args.reliability_gate_weight}, "
        f"use_liquidity_features={args.use_liquidity_features}, "
        f"use_liquidity_gate={args.use_liquidity_gate}, "
        f"no_arb_mode={args.no_arb_diagnostics_mode}, "
        f"no_arb_splits={args.no_arb_eval_splits}, "
        f"no_arb_max_surfaces_per_split={args.no_arb_max_surfaces_per_split}",
        flush=True,
    )
    report_path = Path(args.output_dir) / "data_expansion_report.json"
    if not bool(acceptance["ok"]) and not args.allow_diagnostic_under_threshold:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n")
        raise SystemExit(1)

    variants = [
        {
            "name": "encoder_mlp",
            "model_kind": "encoder_mlp",
            "similarity_edges_per_node": 0,
            "no_arb_weight": 0.0,
            "calendar_weight": 0.0,
            "butterfly_weight": 0.0,
            "put_call_weight": 0.0,
            "convexity_weight": 0.0,
            "smoothness_weight": 0.0,
            "contrastive_weight": 0.0,
            "heteroscedastic_weight": 0.0,
            "reliability_gate_weight": 0.0,
            "use_liquidity_features": True,
            "use_liquidity_gate": False,
        },
        {
            "name": "gnn_no_liq",
            "model_kind": "gnn",
            "similarity_edges_per_node": 0,
            "no_arb_weight": 0.0,
            "calendar_weight": 0.0,
            "butterfly_weight": 0.0,
            "put_call_weight": 0.0,
            "convexity_weight": 0.0,
            "smoothness_weight": args.smoothness_weight,
            "contrastive_weight": args.contrastive_weight,
            "heteroscedastic_weight": 0.0,
            "reliability_gate_weight": 0.0,
            "use_liquidity_features": False,
            "use_liquidity_gate": False,
        },
        {
            "name": "gnn_liq",
            "model_kind": "gnn",
            "similarity_edges_per_node": args.similarity_edges_per_node,
            "no_arb_weight": 0.0,
            "calendar_weight": 0.0,
            "butterfly_weight": 0.0,
            "put_call_weight": 0.0,
            "convexity_weight": 0.0,
            "smoothness_weight": args.smoothness_weight,
            "contrastive_weight": args.contrastive_weight,
            "heteroscedastic_weight": 0.0,
            "reliability_gate_weight": 0.0,
            "use_liquidity_features": True,
            "use_liquidity_gate": True,
        },
        {
            "name": "gnn_decoded_calendar_convexity",
            "model_kind": "gnn",
            "similarity_edges_per_node": args.similarity_edges_per_node,
            "no_arb_weight": args.no_arb_weight,
            "calendar_weight": args.calendar_weight,
            "butterfly_weight": args.butterfly_weight,
            "put_call_weight": 0.0,
            "convexity_weight": args.convexity_weight,
            "smoothness_weight": args.smoothness_weight,
            "contrastive_weight": args.contrastive_weight,
            "heteroscedastic_weight": 0.0,
            "reliability_gate_weight": 0.0,
            "use_liquidity_features": True,
            "use_liquidity_gate": True,
        },
    ]
    if args.variant_suite == "lagos_v2":
        variants = [
            *variants,
            {
                "name": "lagos_no_liquidity",
                "model_kind": "gnn",
                "similarity_edges_per_node": 0,
                "no_arb_weight": 0.0,
                "calendar_weight": 0.0,
                "butterfly_weight": 0.0,
                "put_call_weight": 0.0,
                "convexity_weight": 0.0,
                "smoothness_weight": args.smoothness_weight,
                "contrastive_weight": args.contrastive_weight,
                "heteroscedastic_weight": 0.0,
                "reliability_gate_weight": 0.0,
                "use_liquidity_features": False,
                "use_liquidity_gate": False,
            },
            {
                "name": "lagos_liq_feature_only",
                "model_kind": "gnn",
                "similarity_edges_per_node": 0,
                "no_arb_weight": 0.0,
                "calendar_weight": 0.0,
                "butterfly_weight": 0.0,
                "put_call_weight": 0.0,
                "convexity_weight": 0.0,
                "smoothness_weight": args.smoothness_weight,
                "contrastive_weight": args.contrastive_weight,
                "heteroscedastic_weight": 0.0,
                "reliability_gate_weight": 0.0,
                "use_liquidity_features": True,
                "use_liquidity_gate": False,
            },
            {
                "name": "lagos_scalar_gate",
                "model_kind": "gnn",
                "similarity_edges_per_node": args.similarity_edges_per_node,
                "no_arb_weight": 0.0,
                "calendar_weight": 0.0,
                "butterfly_weight": 0.0,
                "put_call_weight": 0.0,
                "convexity_weight": 0.0,
                "smoothness_weight": args.smoothness_weight,
                "contrastive_weight": args.contrastive_weight,
                "heteroscedastic_weight": 0.0,
                "reliability_gate_weight": 0.0,
                "use_liquidity_features": True,
                "use_liquidity_gate": True,
            },
            {
                "name": "lagos_loss_only",
                "model_kind": "gnn",
                "similarity_edges_per_node": args.similarity_edges_per_node,
                "no_arb_weight": 0.0,
                "calendar_weight": 0.0,
                "butterfly_weight": 0.0,
                "put_call_weight": 0.0,
                "convexity_weight": 0.0,
                "smoothness_weight": args.smoothness_weight,
                "contrastive_weight": args.contrastive_weight,
                "heteroscedastic_weight": args.heteroscedastic_weight or 1.0,
                "reliability_gate_weight": 0.0,
                "use_liquidity_features": True,
                "use_liquidity_gate": False,
            },
            {
                "name": "lagos_attn_only",
                "model_kind": "gnn",
                "similarity_edges_per_node": args.similarity_edges_per_node,
                "no_arb_weight": 0.0,
                "calendar_weight": 0.0,
                "butterfly_weight": 0.0,
                "put_call_weight": 0.0,
                "convexity_weight": 0.0,
                "smoothness_weight": args.smoothness_weight,
                "contrastive_weight": args.contrastive_weight,
                "heteroscedastic_weight": 0.0,
                "reliability_gate_weight": args.reliability_gate_weight or 1.0,
                "use_liquidity_features": True,
                "use_liquidity_gate": False,
            },
            {
                "name": "lagos_hetero_full",
                "model_kind": "gnn",
                "similarity_edges_per_node": args.similarity_edges_per_node,
                "no_arb_weight": 0.0,
                "calendar_weight": 0.0,
                "butterfly_weight": 0.0,
                "put_call_weight": 0.0,
                "convexity_weight": 0.0,
                "smoothness_weight": args.smoothness_weight,
                "contrastive_weight": args.contrastive_weight,
                "heteroscedastic_weight": args.heteroscedastic_weight or 1.0,
                "reliability_gate_weight": args.reliability_gate_weight or 1.0,
                "use_liquidity_features": True,
                "use_liquidity_gate": True,
            },
        ]
    if args.variants:
        selected = {item.strip() for item in args.variants.split(",") if item.strip()}
        known = {str(variant["name"]) for variant in variants}
        unknown = selected - known
        if unknown:
            msg = f"unknown variants: {sorted(unknown)}; expected subset of {sorted(known)}"
            raise ValueError(msg)
        variants = [variant for variant in variants if str(variant["name"]) in selected]

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        args.seed = seed
        for variant in variants:
            experiment_name = (
                f"benchmark_us_{variant['name']}_{args.task}_{args.split_mode}_"
                f"{args.epochs}e_seed{seed}"
            )
            print(f"\n=== Running {experiment_name} ===")
            config = _training_config_from_args(
                args,
                experiment_name=experiment_name,
                model_kind=str(variant["model_kind"]),
                similarity_edges_per_node=int(variant["similarity_edges_per_node"]),
                no_arb_weight=float(variant["no_arb_weight"]),
                calendar_weight=float(variant["calendar_weight"]),
                butterfly_weight=float(variant["butterfly_weight"]),
                put_call_weight=float(variant["put_call_weight"]),
                convexity_weight=float(variant["convexity_weight"]),
                smoothness_weight=float(variant["smoothness_weight"]),
                contrastive_weight=float(variant["contrastive_weight"]),
                heteroscedastic_weight=float(variant["heteroscedastic_weight"]),
                reliability_gate_weight=float(variant["reliability_gate_weight"]),
                use_liquidity_features=bool(variant["use_liquidity_features"]),
                use_liquidity_gate=bool(variant["use_liquidity_gate"]),
            )
            run_option_quote_dataset_experiment(
                us_graphs,
                us_ids,
                config,
                run_label="real_us_mvp",
                dataset_label=str(args.us_data),
                source_surface_count=int(us_stats["source_surface_count"]),
                filtered_surface_count=int(us_stats["filtered_surface_count"]),
                claim_label="real_us_mvp" if bool(acceptance["ok"]) else "diagnostic_only",
                extra_prediction_splits={"ood_jp": jp_graphs} if jp_graphs else None,
                extra_prediction_surface_ids={"ood_jp": jp_ids} if jp_ids else None,
            )
            row = _matrix_row(
                Path(args.output_dir) / experiment_name,
                market="us",
                variant=str(variant["name"]),
                claim_label="real_us_mvp" if bool(acceptance["ok"]) else "diagnostic_only",
            )
            row["seed"] = seed
            rows.append(row)

    summary = _aggregate_benchmark_rows(pd.DataFrame(rows))
    output_path = Path(args.output_dir) / "benchmark_summary.csv"
    summary.to_csv(output_path, index=False)
    print(f"\nWrote {output_path}")


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_split_scope(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items or any(item.lower() == "all" for item in items):
        return ()
    return items


def _benchmark_data_acceptance(
    args: argparse.Namespace,
    us_stats: dict[str, int | float | str],
    jp_stats: dict[str, int | float | str] | None,
) -> dict[str, Any]:
    us_surface_ok = int(us_stats["kept_surface_count"]) >= args.min_us_surfaces
    us_date_ok = int(us_stats["distinct_observation_date_count"]) >= args.min_us_dates
    us_ok = us_surface_ok and us_date_ok
    jp_ok = (
        True
        if jp_stats is None
        else int(jp_stats["distinct_observation_date_count"]) >= args.min_jp_dates
    )
    return {
        "ok": us_ok and jp_ok,
        "us": {
            "ok": us_ok,
            "kept_surface_count": int(us_stats["kept_surface_count"]),
            "min_required_surfaces": args.min_us_surfaces,
            "distinct_observation_date_count": int(us_stats["distinct_observation_date_count"]),
            "min_required_observation_dates": args.min_us_dates,
            "min_nodes_per_surface": args.min_nodes_per_surface,
        },
        "jp": None
        if jp_stats is None
        else {
            "ok": jp_ok,
            "distinct_observation_date_count": int(jp_stats["distinct_observation_date_count"]),
            "min_required_observation_dates": args.min_jp_dates,
        },
        "action": "run_benchmark" if us_ok and jp_ok else "expand_data_first",
    }


def _aggregate_benchmark_rows(rows: Any) -> Any:
    import pandas as pd

    if rows.empty:
        return rows
    numeric_cols = [
        col
        for col in rows.columns
        if col not in {"run", "market", "variant", "claim_label"}
        and rows[col].dtype.kind in {"f", "i"}
    ]
    grouped = rows.groupby(["market", "variant", "claim_label"], dropna=False)
    output_rows: list[dict[str, Any]] = []
    for key, group in grouped:
        market, variant, claim_label = key
        row: dict[str, Any] = {
            "market": market,
            "variant": variant,
            "claim_label": claim_label,
            "n_seeds": int(group["seed"].nunique()) if "seed" in group else len(group),
        }
        for col in numeric_cols:
            if col == "seed":
                continue
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=0))
        output_rows.append(row)
    return pd.DataFrame(output_rows)


def _matrix_row(run_dir: Path, *, market: str, variant: str, claim_label: str) -> dict[str, Any]:
    metrics = json.loads((run_dir / "metrics_summary.json").read_text())
    baselines = _read_baseline_rows(run_dir / "baselines_summary.csv")
    price = json.loads((run_dir / "diagnostics_price.json").read_text())
    no_arb = json.loads((run_dir / "diagnostics_no_arbitrage.json").read_text())
    no_arb_metadata = no_arb.get("metadata", {})
    row: dict[str, Any] = {
        "run": run_dir.name,
        "market": market,
        "variant": variant,
        "claim_label": claim_label,
        "iv_mae": metrics.get("iv_mae"),
        "iv_rmse": metrics.get("iv_rmse"),
        "masked_iv_mae": metrics.get("masked_iv_mae"),
        "masked_iv_rmse": metrics.get("masked_iv_rmse"),
        "masked_iv_p90_abs_error": metrics.get("masked_iv_p90_abs_error"),
        "masked_count": metrics.get("masked_count"),
        "headline_metric": metrics.get("headline_metric", "iv_mae"),
        "final_val_loss": metrics.get("final_val_loss"),
        "final_test_loss": metrics.get("final_test_loss"),
        "price_pred_mid_norm_mae": price.get("pred_mid_norm_mae"),
        "price_true_mid_norm_mae": price.get("true_mid_norm_mae"),
        "no_arb_diagnostics_mode": no_arb_metadata.get("mode"),
        "no_arb_diagnostics_rows": no_arb_metadata.get("row_count"),
        "no_arb_diagnostics_surfaces": no_arb_metadata.get("surface_count"),
        "no_arb_diagnostics_sampling_unit": no_arb_metadata.get("sampling_unit"),
        "pred_calendar_violations": no_arb.get("pred_iv", {}).get("calendar", {}).get("violations"),
        "pred_convexity_violations": no_arb.get("pred_iv", {})
        .get("butterfly_convexity", {})
        .get("violations"),
        "pred_vertical_spread_violations": no_arb.get("pred_iv", {})
        .get("vertical_spread", {})
        .get("violations"),
        "pred_put_call_pairs": no_arb.get("pred_iv", {}).get("put_call_parity", {}).get("pairs"),
    }
    for baseline, values in baselines.items():
        for metric_name in (
            "iv_mae",
            "model_delta_mae",
            "fit_success_rate",
            "failure_rate",
            "underidentified_rate",
            "fit_failed_rate",
            "timeout_rate",
            "no_visible_context_rate",
        ):
            row[f"baseline_{baseline}_{metric_name}"] = values.get(metric_name)
    return row


def _read_baseline_rows(path: Path) -> dict[str, dict[str, float]]:
    import pandas as pd

    frame = pd.read_csv(path)
    rows: dict[str, dict[str, float]] = {}
    for record in frame.to_dict(orient="records"):
        metrics: dict[str, float] = {}
        for key, value in record.items():
            if key == "baseline" or value != value:
                continue
            try:
                metrics[key] = float(value)
            except (TypeError, ValueError):
                continue
        rows[str(record["baseline"])] = metrics
    return rows


def _cmd_fetch(args: argparse.Namespace) -> None:
    """Fetch real option data from APIs."""
    _cmd_fetch_sample(args)


def _cmd_fetch_sample(args: argparse.Namespace) -> None:
    """Fetch the first real-data sample into bronze/silver tables."""
    from log_iv.data_fetch import DataFetchConfig, fetch_sample_dataset

    cfg = DataFetchConfig.from_env()
    if getattr(args, "tickers", None):
        cfg = DataFetchConfig(
            bronze_dir=cfg.bronze_dir,
            silver_dir=cfg.silver_dir,
            gold_dir=cfg.gold_dir,
            reports_dir=cfg.reports_dir,
            us_tickers=[item.strip().upper() for item in args.tickers.split(",") if item.strip()],
            jp_tickers=cfg.jp_tickers,
            start=date.fromisoformat(args.start),
            end=date.fromisoformat(args.end),
        )
    if getattr(args, "jp_tickers", None):
        cfg = DataFetchConfig(
            bronze_dir=cfg.bronze_dir,
            silver_dir=cfg.silver_dir,
            gold_dir=cfg.gold_dir,
            reports_dir=cfg.reports_dir,
            us_tickers=cfg.us_tickers,
            jp_tickers=[item.strip() for item in args.jp_tickers.split(",") if item.strip()],
            start=date.fromisoformat(args.start),
            end=date.fromisoformat(args.end),
        )

    summary = fetch_sample_dataset(
        market=args.market,
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        config=cfg,
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    if summary.stopped_reason:
        raise SystemExit(1)


def _cmd_data_expansion(args: argparse.Namespace) -> None:
    """Probe historical expansion paths and write a data expansion report."""
    from log_iv.data_fetch import DataFetchConfig, write_data_expansion_report

    cfg = DataFetchConfig.from_env()
    if args.tickers:
        cfg = replace(
            cfg,
            us_tickers=[item.strip().upper() for item in args.tickers.split(",") if item.strip()],
        )
    if args.jp_tickers:
        cfg = replace(
            cfg,
            jp_tickers=[item.strip() for item in args.jp_tickers.split(",") if item.strip()],
        )
    if args.max_jp_option_dates is not None:
        cfg = replace(cfg, max_jp_option_dates=args.max_jp_option_dates)
    if args.max_workers is not None:
        cfg = replace(cfg, max_workers=args.max_workers)
    path = write_data_expansion_report(
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        config=cfg,
        market=args.market,
        use_bronze_cache=not args.no_bronze_cache,
        max_workers=args.max_workers,
        refresh_failed=args.refresh_failed,
        refresh_all=args.refresh_all,
    )
    print(f"Wrote {path}")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared by multiple subcommands."""
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--gnn-layers", type=int, default=3)
    parser.add_argument("--model-kind", choices=["encoder_mlp", "gnn"], default="gnn")
    parser.add_argument(
        "--task",
        choices=["observed_reconstruction", "masked_reconstruction"],
        default="observed_reconstruction",
    )
    parser.add_argument(
        "--split-mode",
        choices=["random", "temporal", "ticker_holdout", "temporal_ticker_holdout"],
        default="random",
    )
    parser.add_argument("--heldout-tickers", type=str, default="")
    parser.add_argument("--mask-fraction", type=float, default=0.2)
    parser.add_argument(
        "--mask-regime",
        choices=["stratified", "random", "liquidity_correlated", "block_wing"],
        default="stratified",
    )
    parser.add_argument("--output-dir", type=str, default="reports/runs")
    parser.add_argument("--experiment-name", type=str, default="log-iv-v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--similarity-edges-per-node", type=int, default=3)
    parser.add_argument("--no-arb-weight", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-weight", type=float, default=0.2)
    parser.add_argument("--heteroscedastic-weight", type=float, default=0.0)
    parser.add_argument("--reliability-gate-weight", type=float, default=0.0)
    parser.add_argument(
        "--use-liquidity-features", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--use-liquidity-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--calendar-weight", type=float, default=1.0)
    parser.add_argument("--butterfly-weight", type=float, default=1.0)
    parser.add_argument("--put-call-weight", type=float, default=0.5)
    parser.add_argument("--convexity-weight", type=float, default=0.5)
    parser.add_argument("--greeks-weight", type=float, default=0.0)
    parser.add_argument("--decoded-regularizer-max-terms", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument(
        "--baseline-preset",
        choices=["none", "fast", "full"],
        default="fast",
        help=(
            "Post-training baseline scope. fast skips raw SVI; full includes raw SVI; "
            "none writes model artifacts without P0 baselines."
        ),
    )
    parser.add_argument(
        "--baseline-eval-splits",
        type=str,
        default="val,test",
        help=(
            "Comma-separated prediction splits used for baseline rows; "
            "use 'all' to disable filtering."
        ),
    )
    parser.add_argument("--svi-timeout-seconds", type=float, default=1.0)
    parser.add_argument("--svi-maxiter", type=int, default=200)
    parser.add_argument(
        "--no-arb-diagnostics-mode",
        choices=["none", "sampled_surface", "full"],
        default="sampled_surface",
        help=(
            "Post-training no-arbitrage diagnostics scope. sampled_surface keeps complete "
            "surface/date graphs for a deterministic sample."
        ),
    )
    parser.add_argument(
        "--no-arb-eval-splits",
        type=str,
        default="val,test",
        help="Comma-separated prediction splits for no-arb diagnostics; use 'all' for all splits.",
    )
    parser.add_argument("--no-arb-max-surfaces-per-split", type=int, default=100)
    parser.add_argument("--no-arb-sample-seed", type=int, default=0)
    parser.add_argument("--quiet-postprocess", action="store_true")
    parser.add_argument("--min-nodes-per-surface", type=int, default=3)
    parser.add_argument("--max-nodes-per-surface", type=int, default=None)
    parser.add_argument("--no-graph-cache", dest="use_graph_cache", action="store_false")
    parser.add_argument("--refresh-graph-cache", action="store_true")
    parser.set_defaults(use_graph_cache=True)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="log-iv",
        description="LoG-IV: Option Surface Learning Research",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # status
    p_status = sub.add_parser("status", help="Print project status and metadata")
    p_status.set_defaults(func=_cmd_status)

    # source-probe
    p_probe = sub.add_parser("source-probe", help="Verify source tree and imports")
    p_probe.add_argument("source", nargs="?", default=None)
    p_probe.add_argument("mode", nargs="?", default=None)
    p_probe.add_argument("--source", dest="source_option", default=None)
    p_probe.add_argument("--mode", dest="mode_option", default=None)
    p_probe.set_defaults(func=_cmd_source_probe)

    # toy-graph
    p_toy = sub.add_parser("toy-graph", help="Build minimal 3-node graph")
    p_toy.set_defaults(func=_cmd_toy_graph)

    # synth
    p_synth = sub.add_parser("synth", help="Generate synthetic option surface datasets")
    p_synth.add_argument(
        "--surfaces", type=int, default=20, help="Number of surfaces per underlying"
    )
    p_synth.add_argument("--underlyings", type=int, default=3, help="Number of underlyings")
    p_synth.add_argument("--maturities", type=int, default=12)
    p_synth.add_argument("--strikes", type=int, default=21)
    p_synth.add_argument("--min-tenor", type=int, default=7)
    p_synth.add_argument("--max-tenor", type=int, default=730)
    p_synth.add_argument("--moneyness-range", type=float, default=0.4)
    p_synth.add_argument("--missing-prob", type=float, default=0.05)
    p_synth.add_argument("--stale-prob", type=float, default=0.02)
    p_synth.add_argument("--seed", type=int, default=42)
    p_synth.add_argument("--output", type=str, default=None, help="Save dataset to file")
    p_synth.set_defaults(func=_cmd_synth)

    # train
    p_train = sub.add_parser("train", help="Run training pipeline")
    _add_common_args(p_train)
    p_train.add_argument("--data", type=str, default=None, help="Path to pickled dataset")
    p_train.add_argument("--run-label", type=str, default=None)
    p_train.add_argument("--claim-label", type=str, default=None)
    p_train.add_argument("--max-surfaces", type=int, default=None)
    p_train.add_argument("--synthetic-surfaces", type=int, default=8)
    p_train.add_argument("--synthetic-underlyings", type=int, default=2)
    p_train.add_argument("--synthetic-maturities", type=int, default=5)
    p_train.add_argument("--synthetic-strikes", type=int, default=9)
    p_train.set_defaults(func=_cmd_train)

    # ood-transfer
    p_ood = sub.add_parser("ood-transfer", help="Run OOD transfer experiment")
    _add_common_args(p_ood)
    p_ood.set_defaults(experiment_name="log-iv-ood")
    p_ood.add_argument("--us-data", type=str, required=True)
    p_ood.add_argument("--jp-data", type=str, required=True)
    p_ood.add_argument("--max-us-surfaces", type=int, default=None)
    p_ood.add_argument("--max-jp-surfaces", type=int, default=None)
    p_ood.set_defaults(func=_cmd_ood_transfer)

    # experiment-matrix
    p_matrix = sub.add_parser("experiment-matrix", help="Run first real-data experiment matrix")
    _add_common_args(p_matrix)
    p_matrix.add_argument("--us-data", type=str, required=True)
    p_matrix.add_argument("--jp-data", type=str, required=True)
    p_matrix.add_argument("--max-us-surfaces", type=int, default=None)
    p_matrix.add_argument("--max-jp-surfaces", type=int, default=None)
    p_matrix.set_defaults(func=_cmd_experiment_matrix)

    # benchmark-protocol
    p_benchmark = sub.add_parser(
        "benchmark-protocol",
        help="Run fixed-split masked reconstruction benchmark protocol",
    )
    _add_common_args(p_benchmark)
    p_benchmark.add_argument("--us-data", type=str, required=True)
    p_benchmark.add_argument("--jp-data", type=str, default=None)
    p_benchmark.add_argument("--seeds", type=str, default="1,2,3")
    p_benchmark.add_argument("--max-us-surfaces", type=int, default=None)
    p_benchmark.add_argument("--max-jp-surfaces", type=int, default=None)
    p_benchmark.add_argument("--min-us-surfaces", type=int, default=1000)
    p_benchmark.add_argument("--min-us-dates", type=int, default=31)
    p_benchmark.add_argument("--min-jp-dates", type=int, default=20)
    p_benchmark.add_argument(
        "--variants",
        type=str,
        default="",
        help=(
            "Comma-separated subset of the active variant suite. Core variants: "
            "encoder_mlp,gnn_no_liq,gnn_liq,gnn_decoded_calendar_convexity. "
            "lagos_v2 additionally exposes lagos_no_liquidity,lagos_liq_feature_only,"
            "lagos_scalar_gate,lagos_loss_only,lagos_attn_only,lagos_hetero_full."
        ),
    )
    p_benchmark.add_argument(
        "--variant-suite",
        choices=["core", "lagos_v2"],
        default="core",
        help="Variant registry to use. core keeps the local A1 matrix compact.",
    )
    p_benchmark.add_argument("--allow-diagnostic-under-threshold", action="store_true")
    p_benchmark.set_defaults(
        func=_cmd_benchmark_protocol,
        epochs=100,
        task="masked_reconstruction",
        split_mode="temporal",
        min_nodes_per_surface=20,
    )

    # fetch
    p_fetch = sub.add_parser("fetch", help="Fetch real data from APIs")
    p_fetch.add_argument("--market", type=str, default="all", choices=["us", "jp", "all"])
    p_fetch.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers")
    p_fetch.add_argument("--jp-tickers", type=str, default=None, help="Comma-separated JP tickers")
    p_fetch.add_argument("--start", type=str, default="2026-02-02", help="Start date")
    p_fetch.add_argument("--end", type=str, default="2026-04-30", help="End date")
    p_fetch.add_argument("--lookback", type=int, default=90, help=argparse.SUPPRESS)
    p_fetch.add_argument("--no-fallback", action="store_true", help="Disable synthetic fallback")
    p_fetch.add_argument("--synth-surfaces", type=int, default=80)
    p_fetch.add_argument("--output", type=str, default=None)
    p_fetch.set_defaults(func=_cmd_fetch)

    # fetch-sample
    p_fetch_sample = sub.add_parser("fetch-sample", help="Fetch MVP real-data sample")
    p_fetch_sample.add_argument("--market", type=str, default="all", choices=["us", "jp", "all"])
    p_fetch_sample.add_argument("--start", type=str, default="2026-02-02")
    p_fetch_sample.add_argument("--end", type=str, default="2026-04-30")
    p_fetch_sample.add_argument(
        "--tickers",
        type=str,
        default="SPY,QQQ,AAPL,MSFT,NVDA",
        help="Comma-separated U.S. tickers",
    )
    p_fetch_sample.add_argument(
        "--jp-tickers",
        type=str,
        default="7203,8306,6758,9984,9432",
        help="Comma-separated Japan codes",
    )
    p_fetch_sample.set_defaults(func=_cmd_fetch_sample)

    # data-expansion
    p_expansion = sub.add_parser(
        "data-expansion",
        help="Probe historical flat-file and JP date-loop expansion readiness",
    )
    p_expansion.add_argument("--market", choices=["us", "jp", "all"], default="all")
    p_expansion.add_argument("--start", type=str, default="2026-02-02")
    p_expansion.add_argument("--end", type=str, default="2026-04-30")
    p_expansion.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated U.S. tickers for flat-file expansion",
    )
    p_expansion.add_argument(
        "--jp-tickers",
        type=str,
        default=None,
        help="Comma-separated Japan codes for date-loop expansion",
    )
    p_expansion.add_argument("--max-jp-option-dates", type=int, default=None)
    p_expansion.add_argument("--max-workers", type=int, default=None)
    p_expansion.add_argument("--no-bronze-cache", action="store_true")
    p_expansion.add_argument("--refresh-failed", action="store_true")
    p_expansion.add_argument("--refresh-all", action="store_true")
    p_expansion.set_defaults(func=_cmd_data_expansion)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the log-iv CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
