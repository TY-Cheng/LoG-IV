from __future__ import annotations

import hashlib
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
    graph_style: str = "default",
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
    if graph_style == "hexagon":
        edges.extend(_hexagon_style_edges(nodes, liquidity_epsilon))
    elif graph_style in {"random_edges", "shuffled_edges"}:
        edges = _rewire_edges(nodes, edges, graph_style)
    elif graph_style != "default":
        msg = "graph_style must be one of: default, hexagon, random_edges, shuffled_edges"
        raise ValueError(msg)
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


def _hexagon_style_edges(nodes: tuple[OptionNode, ...], epsilon: float) -> list[OptionEdge]:
    edges: list[OptionEdge] = []
    edges.extend(
        _adjacent_edges_by_scope(nodes, ("expiry", "option_type"), "strike", EdgeKind.SAME_EXPIRY)
    )
    edges.extend(
        _adjacent_edges_by_scope(nodes, ("strike", "option_type"), "tenor", EdgeKind.SAME_STRIKE)
    )
    edges.extend(
        _nearest_coordinate_edges(nodes, "moneyness", EdgeKind.NEAR_MONEYNESS, edges_per_node=2)
    )
    edges.extend(_nearest_coordinate_edges(nodes, "tenor", EdgeKind.NEAR_TENOR, edges_per_node=2))
    edges.extend(_put_call_pair_edges(nodes))
    edges.extend(_same_liquidity_bucket_edges(nodes, epsilon))
    return _dedupe_edges(edges)


def _adjacent_edges_by_scope(
    nodes: tuple[OptionNode, ...],
    scope_fields: tuple[str, ...],
    order_field: str,
    kind: EdgeKind,
) -> list[OptionEdge]:
    groups: dict[tuple[object, ...], list[OptionNode]] = defaultdict(list)
    for node in nodes:
        quote = node.quote
        key_parts: list[object] = [quote.market, quote.underlying, quote.observation_date]
        for field in scope_fields:
            key_parts.append(_quote_group_value(quote, field))
        groups[tuple(key_parts)].append(node)
    edges: list[OptionEdge] = []
    for group in groups.values():
        ordered = sorted(group, key=lambda node: _quote_order_value(node.quote, order_field))
        for left, right in zip(ordered, ordered[1:], strict=False):
            distance = abs(
                _quote_order_value(left.quote, order_field)
                - _quote_order_value(right.quote, order_field)
            )
            weight = 1.0 / (1.0 + max(distance, 0.0))
            edges.extend(_bidirectional_edges(left.node_id, right.node_id, kind, weight))
    return edges


def _nearest_coordinate_edges(
    nodes: tuple[OptionNode, ...],
    coordinate: str,
    kind: EdgeKind,
    *,
    edges_per_node: int,
) -> list[OptionEdge]:
    groups: dict[tuple[object, ...], list[OptionNode]] = defaultdict(list)
    for node in nodes:
        quote = node.quote
        groups[(quote.market, quote.underlying, quote.observation_date, quote.option_type)].append(
            node
        )
    edges: list[OptionEdge] = []
    for group in groups.values():
        for node in group:
            origin = _quote_order_value(node.quote, coordinate)
            ranked = sorted(
                (
                    (abs(origin - _quote_order_value(other.quote, coordinate)), other)
                    for other in group
                    if other.node_id != node.node_id
                ),
                key=lambda item: item[0],
            )
            for distance, other in ranked[:edges_per_node]:
                edges.extend(
                    _bidirectional_edges(
                        node.node_id,
                        other.node_id,
                        kind,
                        1.0 / (1.0 + max(distance, 0.0)),
                    )
                )
    return edges


def _put_call_pair_edges(nodes: tuple[OptionNode, ...]) -> list[OptionEdge]:
    groups: dict[tuple[object, ...], list[OptionNode]] = defaultdict(list)
    for node in nodes:
        quote = node.quote
        key = (
            quote.market,
            quote.underlying,
            quote.observation_date,
            quote.expiry,
            quote.strike,
        )
        groups[key].append(node)
    edges: list[OptionEdge] = []
    for group in groups.values():
        calls = [node for node in group if node.quote.option_type == "C"]
        puts = [node for node in group if node.quote.option_type == "P"]
        for call in calls:
            for put in puts:
                edges.extend(
                    _bidirectional_edges(call.node_id, put.node_id, EdgeKind.PUT_CALL_PAIR, 1.0)
                )
    return edges


def _same_liquidity_bucket_edges(nodes: tuple[OptionNode, ...], epsilon: float) -> list[OptionEdge]:
    groups: dict[tuple[object, ...], list[OptionNode]] = defaultdict(list)
    scores = [_liquidity_score(node.quote, epsilon) for node in nodes]
    median = sorted(scores)[len(scores) // 2] if scores else 0.0
    for node in nodes:
        quote = node.quote
        bucket = "high" if _liquidity_score(quote, epsilon) >= median else "low"
        groups[
            (quote.market, quote.underlying, quote.observation_date, quote.option_type, bucket)
        ].append(node)
    edges: list[OptionEdge] = []
    for group in groups.values():
        ordered = sorted(group, key=lambda node: (node.quote.tenor_days, node.quote.strike))
        for left, right in zip(ordered, ordered[1:], strict=False):
            edges.extend(
                _bidirectional_edges(
                    left.node_id, right.node_id, EdgeKind.SAME_LIQUIDITY_BUCKET, 1.0
                )
            )
    return edges


def _quote_group_value(quote: OptionQuote, field: str) -> object:
    if field == "expiry":
        return quote.expiry
    if field == "strike":
        return round(float(quote.strike), 8)
    if field == "option_type":
        return quote.option_type
    msg = f"unknown quote group field: {field}"
    raise ValueError(msg)


def _quote_order_value(quote: OptionQuote, field: str) -> float:
    if field == "strike":
        return float(quote.strike)
    if field == "tenor":
        return float(quote.tenor_days) / 365.25
    if field == "moneyness":
        return float(quote.log_moneyness or 0.0)
    msg = f"unknown quote order field: {field}"
    raise ValueError(msg)


def _dedupe_edges(edges: list[OptionEdge]) -> list[OptionEdge]:
    result: list[OptionEdge] = []
    seen: set[tuple[int, int, EdgeKind]] = set()
    for edge in edges:
        key = (edge.source, edge.target, edge.kind)
        if key not in seen:
            seen.add(key)
            result.append(edge)
    return result


def _rewire_edges(
    nodes: tuple[OptionNode, ...],
    edges: list[OptionEdge],
    graph_style: str,
) -> list[OptionEdge]:
    """Return a deterministic graph-necessity ablation with geometry broken."""

    if len(nodes) < 2:
        return []
    node_ids = [node.node_id for node in nodes]
    if graph_style == "shuffled_edges":
        return _dedupe_edges([_shuffled_target_edge(edge, node_ids) for edge in edges])
    if graph_style == "random_edges":
        return _dedupe_edges(
            [_random_target_edge(edge, node_ids, position) for position, edge in enumerate(edges)]
        )
    msg = f"unsupported rewiring style: {graph_style}"
    raise ValueError(msg)


def _shuffled_target_edge(edge: OptionEdge, node_ids: list[int]) -> OptionEdge:
    shifted = {
        node_id: node_ids[(index + 1) % len(node_ids)] for index, node_id in enumerate(node_ids)
    }
    target = shifted.get(edge.target, node_ids[0])
    if target == edge.source:
        target = shifted.get(target, node_ids[0])
    return _edge(edge.source, target, edge.kind, edge.weight)


def _random_target_edge(edge: OptionEdge, node_ids: list[int], position: int) -> OptionEdge:
    candidates = [node_id for node_id in node_ids if node_id != edge.source]
    seed = f"{position}:{edge.source}:{edge.target}:{edge.kind.value}:{len(node_ids)}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    target = candidates[int(digest[:12], 16) % len(candidates)]
    return _edge(edge.source, target, edge.kind, edge.weight)


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
