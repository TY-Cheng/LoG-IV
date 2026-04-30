"""Liquidity-aware graph neural operator for option surface learning.

Implements the LiquidityGraphOperator: a message-passing GNN that operates
on irregular option surface graphs with:
- Heterogeneous edge types (strike_neighbor, maturity_neighbor, liquidity_similarity)
- Liquidity-weighted message aggregation
- No-arbitrage regularized graph smoothing
- Graph-level readout for surface representation

Reference: LoG-IV research plan §Liquidity-Aware Graph Neural Operators.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv, global_mean_pool


class LiquidityGraphOperator(nn.Module):
    """Liquidity-aware graph neural operator for option surface graphs.

    Operates on heterogeneous graphs with edge types:
    - strike_neighbor: connects options with same maturity, adjacent strikes
    - maturity_neighbor: connects options with same strike, adjacent maturities
    - liquidity_similarity: connects options with similar liquidity profiles

    Parameters
    ----------
    node_dim : int
        Dimension of input node features (from OptionTokenEncoder).
    hidden_dim : int
        Hidden dimension for message passing.
    n_layers : int
        Number of message-passing layers.
    edge_types : list[tuple[str, str, str]]
        Heterogeneous edge type triples.
    dropout : float
        Dropout rate.
    use_attention : bool
        Whether to use GAT-style attention in message passing.
    n_heads : int
        Number of attention heads if use_attention is True.
    """

    def __init__(
        self,
        node_dim: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 3,
        edge_types: list[tuple[str, str, str]] | None = None,
        dropout: float = 0.1,
        use_attention: bool = True,
        n_heads: int = 4,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

        if edge_types is None:
            edge_types = [
                ("option", "strike_neighbor", "option"),
                ("option", "maturity_neighbor", "option"),
                ("option", "liquidity_similarity", "option"),
            ]
        self.edge_types = edge_types

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Message passing layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            conv_dict = {}
            for et in edge_types:
                if use_attention:
                    conv_dict[et] = GATConv(
                        hidden_dim,
                        hidden_dim // n_heads,
                        heads=n_heads,
                        dropout=dropout,
                        add_self_loops=False,
                    )
                else:
                    conv_dict[et] = SAGEConv(
                        hidden_dim,
                        hidden_dim,
                        aggr="mean",
                    )
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            self.norms.append(
                nn.ModuleDict(
                    {
                        et[0]: nn.LayerNorm(hidden_dim)
                        for et in {(et[0], et[2]) for et in edge_types} | {("option", "option")}
                    }
                )
            )

        # Attention-based edge-type fusion
        self.edge_type_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=2, batch_first=True, dropout=dropout
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim),
        )

        # Surface-level readout
        self.readout_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Liquidity gate: adjusts message weight based on node liquidity
        self.liquidity_gate = nn.Sequential(
            nn.Linear(hidden_dim + 4, 1),  # hidden features + [vol, OI, spread, score]
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        data: HeteroData,
        liquidity_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the liquidity-aware graph operator.

        Parameters
        ----------
        x : torch.Tensor
            Node features from encoder, shape (N, node_dim).
        data : HeteroData
            PyG HeteroData with edge_index_dict for each edge type.
        liquidity_features : torch.Tensor, optional
            Liquidity marks, shape (N, 4).

        Returns
        -------
        dict[str, torch.Tensor]
            - 'node_embeddings': updated node embeddings (N, node_dim)
            - 'surface_embedding': graph-level surface representation (hidden_dim,)
            - 'layer_outputs': list of intermediate node features
        """
        h = self.input_proj(x)
        node_type = "option"

        # Build node dict for HeteroConv
        x_dict = {node_type: h}
        layer_outputs = []

        for i, conv in enumerate(self.convs):
            # Apply message passing
            x_dict_new = conv(x_dict, data.edge_index_dict)

            edge_weight_gate = _incoming_edge_weight_gate(
                data,
                num_nodes=x_dict[node_type].size(0),
                device=x_dict[node_type].device,
            )

            # Apply liquidity gating if liquidity features available
            if liquidity_features is not None:
                for nt in x_dict_new:
                    gate_input = torch.cat([x_dict_new[nt], liquidity_features], dim=-1)
                    gate = self.liquidity_gate(gate_input)
                    x_dict_new[nt] = gate * x_dict_new[nt] + (1 - gate) * x_dict[nt]
                    if nt == node_type:
                        x_dict_new[nt] = x_dict_new[nt] * edge_weight_gate

            # Residual connection + layer norm
            for nt in x_dict_new:
                if nt in x_dict:
                    x_dict_new[nt] = x_dict_new[nt] + x_dict[nt]
                if nt in self.norms[i]:
                    x_dict_new[nt] = self.norms[i][nt](x_dict_new[nt])
                x_dict_new[nt] = F.gelu(x_dict_new[nt])

            x_dict = x_dict_new
            layer_outputs.append(x_dict[node_type].clone())

        # Output projection
        h_out = self.output_proj(x_dict[node_type])

        # Surface-level readout (pool over nodes)
        batch = getattr(data, node_type, None)
        if batch is not None and hasattr(batch, "batch"):
            batch_idx = getattr(
                data[node_type],
                "batch",
                torch.zeros(h_out.size(0), dtype=torch.long, device=h_out.device),
            )
        else:
            batch_idx = torch.zeros(h_out.size(0), dtype=torch.long, device=h_out.device)
        surface_embedding = global_mean_pool(h_out, batch_idx).squeeze(0)
        surface_embedding = self.readout_proj(surface_embedding)

        return {
            "node_embeddings": h_out,
            "surface_embedding": surface_embedding,
            "layer_outputs": layer_outputs,
        }


def build_hetero_data_from_graph(
    quotes: list,
    node_embeddings: torch.Tensor,
    graph,
    device: torch.device | None = None,
) -> HeteroData:
    """Build PyG HeteroData from OptionSurfaceGraph and node embeddings.

    Parameters
    ----------
    quotes : list[OptionQuote]
        Option quotes.
    node_embeddings : torch.Tensor
        Encoded node features, shape (N, d_model).
    graph : OptionSurfaceGraph
        Built graph with edges.
    device : torch.device, optional

    Returns
    -------
    HeteroData
        PyG heterogeneous graph data object.
    """
    data = HeteroData()
    device = device or node_embeddings.device

    # Node features
    data["option"].x = node_embeddings.to(device)

    # Liquidity features per node
    from log_iv.encoder import extract_features_from_quotes

    _, _, liq_features = extract_features_from_quotes(quotes, device)
    data["option"].liquidity = liq_features

    # Build edge index dict by edge type
    edge_index_dict: dict[tuple[str, str, str], list[list[int]]] = {}
    edge_weight_dict: dict[tuple[str, str, str], list[float]] = {}

    for edge in graph.edges:
        et = (
            "option",
            edge.kind.value,  # "strike_neighbor", "maturity_neighbor", "liquidity_similarity"
            "option",
        )
        if et not in edge_index_dict:
            edge_index_dict[et] = []
            edge_weight_dict[et] = []

        edge_index_dict[et].append([edge.source, edge.target])
        edge_weight_dict[et].append(edge.weight)

    # Convert to PyG edge stores. Edge weights are also attached and consumed by
    # LiquidityGraphOperator through an incoming-edge gate.
    for et in edge_index_dict:
        data[et].edge_index = (
            torch.tensor(edge_index_dict[et], dtype=torch.long, device=device).t().contiguous()
        )
        data[et].edge_weight = torch.tensor(
            edge_weight_dict[et], dtype=torch.float32, device=device
        )

    return data


def _incoming_edge_weight_gate(
    data: HeteroData,
    *,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Aggregate incoming edge weights into a node-wise multiplicative gate."""

    weight_sum = torch.zeros(num_nodes, 1, device=device)
    degree = torch.zeros(num_nodes, 1, device=device)
    for edge_type in data.edge_types:
        edge_store = data[edge_type]
        edge_index = edge_store.edge_index
        edge_weight = getattr(edge_store, "edge_weight", None)
        if edge_weight is None or edge_index.numel() == 0:
            continue
        dst = edge_index[1]
        weight_sum.index_add_(0, dst, edge_weight.reshape(-1, 1))
        degree.index_add_(0, dst, torch.ones_like(edge_weight).reshape(-1, 1))
    mean_weight = weight_sum / degree.clamp_min(1.0)
    return 0.5 + torch.sigmoid(mean_weight)
