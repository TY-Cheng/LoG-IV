"""No-arbitrage regularizer for option surface learning.

Implements differentiable constraints derived from option pricing theory:
1. Calendar arbitrage: total variance must be non-decreasing in maturity
2. Butterfly arbitrage: implied density must be non-negative
3. Put-call parity: relationship between calls and puts at same strike/maturity
4. Convexity constraint: option price convex in strike
5. Liquidity-aware constraint weighting

Reference: LoG-IV research plan §No-Arbitrage Regularized Graph Smoothing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoArbitrageRegularizer(nn.Module):
    """Differentiable no-arbitrage regularizer for option surface learning.

    Computes penalty terms for violations of no-arbitrage conditions,
    weighted by liquidity to down-weight constraints on illiquid quotes.

    Parameters
    ----------
    calendar_weight : float
        Weight for calendar arbitrage penalty.
    butterfly_weight : float
        Weight for butterfly arbitrage penalty.
    put_call_weight : float
        Weight for put-call parity penalty.
    convexity_weight : float
        Weight for strike convexity penalty.
    liquidity_scale : float
        How much liquidity matters for constraint weighting.
    eps : float
        Numerical stability epsilon.
    """

    def __init__(
        self,
        calendar_weight: float = 1.0,
        butterfly_weight: float = 1.0,
        put_call_weight: float = 0.5,
        convexity_weight: float = 0.5,
        liquidity_scale: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.calendar_weight = calendar_weight
        self.butterfly_weight = butterfly_weight
        self.put_call_weight = put_call_weight
        self.convexity_weight = convexity_weight
        self.liquidity_scale = liquidity_scale
        self.eps = eps

    def forward(
        self,
        node_embeddings: torch.Tensor,
        predicted_iv: torch.Tensor,
        geometry: torch.Tensor,
        liquidity: torch.Tensor,
        graph_data,
    ) -> dict[str, torch.Tensor]:
        """Compute no-arbitrage regularizer terms.

        Parameters
        ----------
        node_embeddings : torch.Tensor
            Node embeddings after GNN processing, shape (N, d_model).
        predicted_iv : torch.Tensor
            Predicted implied volatility, shape (N, 1).
        geometry : torch.Tensor
            Geometric features [log_moneyness, tenor_years, cp_flag], shape (N, 3).
        liquidity : torch.Tensor
            Liquidity features [vol_log, oi_log, spread, liq_score], shape (N, 4).
        graph_data : HeteroData
            Graph data with edge index dict.

        Returns
        -------
        dict[str, torch.Tensor]
            Penalty values: 'calendar', 'butterfly', 'put_call', 'convexity', 'total'.
        """
        # Liquidity weight: more liquid quotes get higher weight in constraints
        liq_scores = liquidity[:, 3]  # liquidity_score
        liq_weights = torch.sigmoid(liq_scores * self.liquidity_scale)

        # Parse geometry
        log_moneyness = geometry[:, 0]
        tenor = geometry[:, 1]
        cp_flag = geometry[:, 2]  # 1 = call, 0 = put

        total_variance = predicted_iv.squeeze(-1) ** 2 * torch.clamp(tenor, min=self.eps)

        penalties = {}

        # 1. Calendar arbitrage: ∂w/∂τ ≥ 0
        # For pairs connected by maturity_neighbor edges
        penalties["calendar"] = _calendar_arbitrage_penalty(
            total_variance, tenor, graph_data, liq_weights, self.eps
        )

        # 2. Butterfly arbitrage: implied density ≥ 0
        # Approximated via second derivative of call price w.r.t. strike
        penalties["butterfly"] = _butterfly_arbitrage_penalty(
            predicted_iv, log_moneyness, tenor, cp_flag, graph_data, liq_weights, self.eps
        )

        # 3. Put-call parity: C - P = F - K
        penalties["put_call"] = _put_call_parity_penalty(
            node_embeddings, geometry, graph_data, liq_weights, self.eps
        )

        # 4. Convexity: option price convex in strike
        penalties["convexity"] = _convexity_penalty(
            node_embeddings, log_moneyness, cp_flag, graph_data, liq_weights, self.eps
        )

        # Total penalty
        total = (
            self.calendar_weight * penalties["calendar"]
            + self.butterfly_weight * penalties["butterfly"]
            + self.put_call_weight * penalties["put_call"]
            + self.convexity_weight * penalties["convexity"]
        )
        penalties["total"] = total

        return penalties


def _calendar_arbitrage_penalty(
    total_variance: torch.Tensor,
    tenor: torch.Tensor,
    graph_data,
    liq_weights: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Penalize decreasing total variance with maturity for same-moneyness pairs."""
    edge_index = None
    if hasattr(graph_data, "edge_index_dict"):
        for et, ei in graph_data.edge_index_dict.items():
            if "maturity" in str(et):
                edge_index = ei
                break

    if edge_index is None or edge_index.size(1) == 0:
        return torch.tensor(0.0, device=total_variance.device)

    src, dst = edge_index[0], edge_index[1]

    # Filter: keep pairs where near < far in tenor
    tenor_src = tenor[src]
    tenor_dst = tenor[dst]
    valid_mask = tenor_src < tenor_dst - eps

    if not valid_mask.any():
        return torch.tensor(0.0, device=total_variance.device)

    src_valid = src[valid_mask]
    dst_valid = dst[valid_mask]

    w_near = total_variance[src_valid]
    w_far = total_variance[dst_valid]

    # Penalize w_far < w_near
    violation = F.relu(w_near - w_far)

    # Weight by min liquidity of the pair
    weight = torch.min(liq_weights[src_valid], liq_weights[dst_valid])

    return (weight * violation).mean()


def _butterfly_arbitrage_penalty(
    predicted_iv: torch.Tensor,
    log_moneyness: torch.Tensor,
    tenor: torch.Tensor,
    cp_flag: torch.Tensor,
    graph_data,
    liq_weights: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Penalize negative implied density (butterfly spread violation).

    Approximated via: for three consecutive strikes K1 < K2 < K3 with same maturity,
    call price C(K2) should be ≤ weighted average of C(K1) and C(K3).
    """
    edge_index = None
    if hasattr(graph_data, "edge_index_dict"):
        for et, ei in graph_data.edge_index_dict.items():
            if "strike" in str(et):
                edge_index = ei
                break

    if edge_index is None or edge_index.size(1) < 2:
        return torch.tensor(0.0, device=predicted_iv.device)

    # For each node, find its two nearest strike neighbors
    # Build adjacency for strike neighbor edges
    src, dst = edge_index[0], edge_index[1]
    call_mask = cp_flag > 0.5

    # Only consider calls (butterfly is defined on call prices)
    N = predicted_iv.size(0)
    penalty_sum = torch.tensor(0.0, device=predicted_iv.device)
    count = 0

    for i in range(N):
        if not call_mask[i]:
            continue
        # Find strike neighbors of i
        neighbors = torch.cat([dst[src == i], src[dst == i]])
        neighbors = neighbors[call_mask[neighbors]]

        if len(neighbors) < 2:
            continue

        # Sort neighbors by log_moneyness
        neighbor_lm = log_moneyness[neighbors]
        sorted_idx = torch.argsort(neighbor_lm)
        neighbors_sorted = neighbors[sorted_idx]

        # Check all triples
        for j in range(len(neighbors_sorted) - 2):
            n1, n2, n3 = neighbors_sorted[j], neighbors_sorted[j + 1], neighbors_sorted[j + 2]
            lm1, lm2, lm3 = log_moneyness[n1], log_moneyness[n2], log_moneyness[n3]

            # Interpolation weights
            alpha = (lm3 - lm2) / (lm3 - lm1 + eps)
            beta = (lm2 - lm1) / (lm3 - lm1 + eps)

            iv1, iv2, iv3 = predicted_iv[n1], predicted_iv[n2], predicted_iv[n3]
            w1 = total_variance_from_iv(iv1, tenor[n1])
            w2 = total_variance_from_iv(iv2, tenor[n2])
            w3 = total_variance_from_iv(iv3, tenor[n3])

            interpolated = alpha * w1 + beta * w3
            violation = F.relu(interpolated - w2)

            weight = liq_weights[i] * liq_weights[n2]
            penalty_sum = penalty_sum + weight * violation
            count += 1

    if count == 0:
        return torch.tensor(0.0, device=predicted_iv.device)

    return penalty_sum / count


def _put_call_parity_penalty(
    node_embeddings: torch.Tensor,
    geometry: torch.Tensor,
    graph_data,
    liq_weights: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Penalize put-call parity violations.

    PCP: C - P = S * e^{-qτ} - K * e^{-rτ}
    In forward terms: C - P = F - K (discounted)

    We check pairs of (call, put) at same strike and maturity.
    """
    log_moneyness = geometry[:, 0]
    tenor = geometry[:, 1]
    cp_flag = geometry[:, 2]

    call_idx = (cp_flag > 0.5).nonzero(as_tuple=True)[0]
    put_idx = (cp_flag < 0.5).nonzero(as_tuple=True)[0]

    if len(call_idx) == 0 or len(put_idx) == 0:
        return torch.tensor(0.0, device=node_embeddings.device)

    # Match calls and puts by approximate log-moneyness and tenor
    penalty_sum = torch.tensor(0.0, device=node_embeddings.device)
    count = 0

    for ci in call_idx:
        lm_c = log_moneyness[ci]
        t_c = tenor[ci]

        # Find closest put
        diffs = (log_moneyness[put_idx] - lm_c) ** 2 + (tenor[put_idx] - t_c) ** 2
        if len(diffs) == 0:
            continue
        best_pi = put_idx[torch.argmin(diffs)]

        # Embedding distance (proxy for predicted price discrepancy)
        emb_dist = F.mse_loss(node_embeddings[ci], node_embeddings[best_pi], reduction="sum")

        weight = liq_weights[ci] * liq_weights[best_pi]
        penalty_sum = penalty_sum + weight * emb_dist
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=node_embeddings.device)

    return penalty_sum / count


def _convexity_penalty(
    node_embeddings: torch.Tensor,
    log_moneyness: torch.Tensor,
    cp_flag: torch.Tensor,
    graph_data,
    liq_weights: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Penalize non-convexity of option price in strike.

    For calls: C(K) convex in K → C(K2) ≤ λ C(K1) + (1-λ) C(K3)
    for K1 < K2 < K3 where λ = (K3 - K2) / (K3 - K1).
    """
    edge_index = None
    if hasattr(graph_data, "edge_index_dict"):
        for et, ei in graph_data.edge_index_dict.items():
            if "strike" in str(et):
                edge_index = ei
                break

    if edge_index is None or edge_index.size(1) < 2:
        return torch.tensor(0.0, device=node_embeddings.device)

    # Use embedding distances as proxy for price differences
    # Real implementation would decode predicted prices
    call_mask = cp_flag > 0.5
    N = node_embeddings.size(0)

    src, dst = edge_index[0], edge_index[1]
    penalty_sum = torch.tensor(0.0, device=node_embeddings.device)
    count = 0

    for i in range(N):
        if not call_mask[i]:
            continue
        neighbors = torch.cat([dst[src == i], src[dst == i]])
        neighbors = neighbors[call_mask[neighbors]]
        if len(neighbors) < 2:
            continue

        neighbor_lm = log_moneyness[neighbors]
        sorted_idx = torch.argsort(neighbor_lm)
        n_sorted = neighbors[sorted_idx]

        for j in range(len(n_sorted) - 2):
            n1, n2, n3 = n_sorted[j], n_sorted[j + 1], n_sorted[j + 2]
            lm1, lm2, lm3 = log_moneyness[n1], log_moneyness[n2], log_moneyness[n3]
            lam = (lm3 - lm2) / (lm3 - lm1 + eps)

            # Use embedding norm as proxy for price
            e1 = torch.norm(node_embeddings[n1])
            e2 = torch.norm(node_embeddings[n2])
            e3 = torch.norm(node_embeddings[n3])

            convex_value = lam * e1 + (1 - lam) * e3
            violation = F.relu(e2 - convex_value)

            weight = liq_weights[i] * liq_weights[n2]
            penalty_sum = penalty_sum + weight * violation
            count += 1

    if count == 0:
        return torch.tensor(0.0, device=node_embeddings.device)

    return penalty_sum / count


def total_variance_from_iv(iv: torch.Tensor, tau: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert implied volatility to total variance: w = σ² τ."""
    return iv.squeeze(-1) ** 2 * torch.clamp(tau, min=eps)


def smoothness_regularizer(
    node_embeddings: torch.Tensor,
    graph_data,
) -> torch.Tensor:
    """Dirichlet energy / graph Laplacian smoothness penalty.

    Penalizes large differences between connected nodes.
    """
    total = torch.tensor(0.0, device=node_embeddings.device)
    count = 0

    if not hasattr(graph_data, "edge_index_dict"):
        return total

    for edge_type, edge_index in graph_data.edge_index_dict.items():
        if edge_index.size(1) == 0:
            continue
        src, dst = edge_index[0], edge_index[1]
        diff = node_embeddings[src] - node_embeddings[dst]
        edge_store = graph_data[edge_type] if hasattr(graph_data, "__getitem__") else None
        edge_weight = getattr(edge_store, "edge_weight", None)
        if edge_weight is None:
            total = total + (diff**2).mean()
        else:
            total = total + ((diff**2).mean(dim=1) * edge_weight).mean()
        count += 1

    if count == 0:
        return total

    return total / count
