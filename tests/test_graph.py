from datetime import date

import pytest

from log_iv.graph import build_option_surface_graph
from log_iv.schema import EdgeKind, OptionQuote


def _quote(strike: float, expiry: date, volume: int = 100, open_interest: int = 500) -> OptionQuote:
    return OptionQuote(
        market="us",
        underlying="spy",
        observation_date=date(2026, 1, 2),
        expiry=expiry,
        strike=strike,
        option_type="C",
        bid=1.0,
        ask=1.2,
        implied_vol=0.2,
        volume=volume,
        open_interest=open_interest,
        forward=100.0,
    )


def test_quote_validation_rejects_crossed_market() -> None:
    with pytest.raises(ValueError, match="ask"):
        OptionQuote(
            market="US",
            underlying="SPY",
            observation_date=date(2026, 1, 2),
            expiry=date(2026, 2, 20),
            strike=100.0,
            option_type="C",
            bid=2.0,
            ask=1.5,
        )


def test_build_graph_adds_strike_and_maturity_edges() -> None:
    quotes = [
        _quote(95.0, date(2026, 2, 20)),
        _quote(100.0, date(2026, 2, 20)),
        _quote(100.0, date(2026, 3, 20)),
    ]

    graph = build_option_surface_graph(quotes)
    edge_kinds = {edge.kind for edge in graph.edges}

    assert len(graph.nodes) == 3
    assert EdgeKind.STRIKE_NEIGHBOR in edge_kinds
    assert EdgeKind.MATURITY_NEIGHBOR in edge_kinds


def test_graph_node_order_matches_input_order() -> None:
    quotes = [
        _quote(105.0, date(2026, 3, 20)),
        _quote(95.0, date(2026, 2, 20)),
        _quote(100.0, date(2026, 2, 20)),
    ]

    graph = build_option_surface_graph(quotes)

    assert [node.quote for node in graph.nodes] == quotes
    assert [node.node_id for node in graph.nodes] == [0, 1, 2]


def test_strike_and_maturity_edges_are_bidirectional() -> None:
    quotes = [
        _quote(95.0, date(2026, 2, 20)),
        _quote(100.0, date(2026, 2, 20)),
        _quote(100.0, date(2026, 3, 20)),
    ]

    graph = build_option_surface_graph(quotes)
    typed_pairs = {(edge.source, edge.target, edge.kind) for edge in graph.edges}

    assert (0, 1, EdgeKind.STRIKE_NEIGHBOR) in typed_pairs
    assert (1, 0, EdgeKind.STRIKE_NEIGHBOR) in typed_pairs
    assert (1, 2, EdgeKind.MATURITY_NEIGHBOR) in typed_pairs
    assert (2, 1, EdgeKind.MATURITY_NEIGHBOR) in typed_pairs


def test_build_graph_can_add_liquidity_similarity_edges() -> None:
    quotes = [
        _quote(95.0, date(2026, 2, 20), volume=10, open_interest=20),
        _quote(100.0, date(2026, 2, 20), volume=11, open_interest=21),
        _quote(105.0, date(2026, 2, 20), volume=1000, open_interest=2000),
    ]

    graph = build_option_surface_graph(quotes, similarity_edges_per_node=1)

    assert any(edge.kind == EdgeKind.LIQUIDITY_SIMILARITY for edge in graph.edges)


def test_build_graph_handles_empty_input() -> None:
    graph = build_option_surface_graph([])

    assert graph.nodes == ()
    assert graph.edges == ()


def test_liquidity_similarity_handles_missing_spread_and_moneyness() -> None:
    quotes = [
        OptionQuote(
            market="US",
            underlying="SPY",
            observation_date=date(2026, 1, 2),
            expiry=date(2026, 2, 20),
            strike=95.0,
            option_type="C",
            volume=10,
            open_interest=20,
        ),
        OptionQuote(
            market="US",
            underlying="SPY",
            observation_date=date(2026, 1, 2),
            expiry=date(2026, 2, 20),
            strike=100.0,
            option_type="C",
            volume=10,
            open_interest=20,
        ),
    ]

    graph = build_option_surface_graph(quotes, similarity_edges_per_node=1)

    assert len(graph.nodes) == 2
    assert any(edge.kind == EdgeKind.LIQUIDITY_SIMILARITY for edge in graph.edges)
