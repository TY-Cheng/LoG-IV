from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from log_iv.schema import EdgeKind, OptionEdge, OptionNode, OptionQuote, OptionSurfaceGraph


def summarize_graph(
    graph: OptionSurfaceGraph,
    quotes: Iterable[OptionQuote],
) -> dict[str, Any]:
    """Return statistics dict for a built graph."""
    edge_types = sorted(edge.kind.value for edge in graph.edges)
    unique_types = sorted(set(edge.kind.value for edge in graph.edges))
    return {
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "edge_types": unique_types,
        "edge_type_counts": {kind: edge_types.count(kind) for kind in unique_types},
        "n_quotes": len(list(quotes)),
    }


def build_option_surface_graph(
    quotes: Iterable[OptionQuote],
    *,
    similarity_edges_per_node: int = 0,
    liquidity_epsilon: float = 1e-8,
) -> OptionSurfaceGraph:
    """Build a small option-surface graph from canonical quote observations."""

    ordered_quotes = tuple(quotes)
    nodes = tuple(
        OptionNode(node_id=index, quote=quote) for index, quote in enumerate(ordered_quotes)
    )
    edges: list[OptionEdge] = []
    edges.extend(_strike_neighbor_edges(nodes))
    edges.extend(_maturity_neighbor_edges(nodes))
    if similarity_edges_per_node > 0:
        edges.extend(
            _liquidity_similarity_edges(
                nodes,
                edges_per_node=similarity_edges_per_node,
                epsilon=liquidity_epsilon,
            )
        )
    return OptionSurfaceGraph(nodes=nodes, edges=tuple(edges))


def _strike_neighbor_edges(nodes: tuple[OptionNode, ...]) -> list[OptionEdge]:
    groups: dict[tuple[object, ...], list[OptionNode]] = defaultdict(list)
    for node in nodes:
        quote = node.quote
        key = (
            quote.market,
            quote.underlying,
            quote.observation_date,
            quote.expiry,
            quote.option_type,
        )
        groups[key].append(node)

    edges: list[OptionEdge] = []
    for group in groups.values():
        ordered = sorted(group, key=lambda node: node.quote.strike)
        for left, right in zip(ordered, ordered[1:], strict=False):
            edges.extend(
                _bidirectional_edges(left.node_id, right.node_id, EdgeKind.STRIKE_NEIGHBOR, 1.0)
            )
    return edges


def _maturity_neighbor_edges(nodes: tuple[OptionNode, ...]) -> list[OptionEdge]:
    groups: dict[tuple[object, ...], list[OptionNode]] = defaultdict(list)
    for node in nodes:
        quote = node.quote
        key = (
            quote.market,
            quote.underlying,
            quote.observation_date,
            quote.strike,
            quote.option_type,
        )
        groups[key].append(node)

    edges: list[OptionEdge] = []
    for group in groups.values():
        ordered = sorted(group, key=lambda node: node.quote.expiry)
        for near, far in zip(ordered, ordered[1:], strict=False):
            tenor_gap = far.quote.tenor_days - near.quote.tenor_days
            weight = 1.0 / max(tenor_gap, 1)
            edges.extend(
                _bidirectional_edges(near.node_id, far.node_id, EdgeKind.MATURITY_NEIGHBOR, weight)
            )
    return edges


def _liquidity_similarity_edges(
    nodes: tuple[OptionNode, ...],
    *,
    edges_per_node: int,
    epsilon: float,
) -> list[OptionEdge]:
    candidates_by_scope: dict[tuple[object, ...], list[OptionNode]] = defaultdict(list)
    for node in nodes:
        quote = node.quote
        key = (quote.market, quote.underlying, quote.observation_date, quote.option_type)
        candidates_by_scope[key].append(node)

    edges: list[OptionEdge] = []
    seen: set[tuple[int, int, EdgeKind]] = set()
    for group in candidates_by_scope.values():
        for node in group:
            ranked = sorted(
                (
                    (_geometry_liquidity_distance(node.quote, other.quote, epsilon), other)
                    for other in group
                    if other.node_id != node.node_id
                ),
                key=lambda item: item[0],
            )
            for distance, other in ranked[:edges_per_node]:
                weight = 1.0 / (1.0 + distance)
                for edge in _bidirectional_edges(
                    node.node_id, other.node_id, EdgeKind.LIQUIDITY_SIMILARITY, weight
                ):
                    edge_key = (edge.source, edge.target, edge.kind)
                    if edge_key not in seen:
                        seen.add(edge_key)
                        edges.append(edge)
    return edges


def _geometry_liquidity_distance(left: OptionQuote, right: OptionQuote, epsilon: float) -> float:
    left_money = left.log_moneyness if left.log_moneyness is not None else 0.0
    right_money = right.log_moneyness if right.log_moneyness is not None else 0.0
    moneyness_distance = abs(left_money - right_money)
    tenor_distance = abs(left.tenor_years - right.tenor_years)
    liquidity_distance = abs(_liquidity_score(left, epsilon) - _liquidity_score(right, epsilon))
    return moneyness_distance + tenor_distance + liquidity_distance


def _liquidity_score(quote: OptionQuote, epsilon: float) -> float:
    spread = quote.spread if quote.spread is not None else 0.0
    spread_penalty = math.log1p(spread + epsilon)
    support = math.log1p(quote.volume) + math.log1p(quote.open_interest)
    return support - spread_penalty


def _edge(source: int, target: int, kind: EdgeKind, weight: float) -> OptionEdge:
    return OptionEdge(source=source, target=target, kind=kind, weight=weight)


def _bidirectional_edges(
    left: int,
    right: int,
    kind: EdgeKind,
    weight: float,
) -> list[OptionEdge]:
    if left == right:
        return []
    return [
        _edge(left, right, kind, weight),
        _edge(right, left, kind, weight),
    ]
