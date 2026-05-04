"""Training pipeline for OptionToken-GNN model.

Implements the full training loop with:
- Graph construction from option quotes
- Option-token encoding
- Liquidity-aware message passing
- Multi-objective loss with no-arbitrage regularization
- Checkpointing, logging, and evaluation

Loss objectives:
1. IV reconstruction (MSE on implied volatility)
2. Geometry reconstruction (for self-supervision)
3. Liquidity reconstruction (auxiliary)
4. Greeks prediction (for no-arbitrage consistency)
5. No-arbitrage regularizer (calendar, butterfly, put-call, convexity)
6. Graph smoothness (Dirichlet energy)
7. Contrastive loss (across surfaces for OOD transfer)
"""

from __future__ import annotations

import hashlib
import json
import math
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from log_iv.encoder import OptionTokenEncoder, extract_features_from_quotes
from log_iv.gnn import LiquidityGraphOperator, build_hetero_data_from_graph
from log_iv.graph import build_option_surface_graph
from log_iv.regularizer import NoArbitrageRegularizer, smoothness_regularizer
from log_iv.schema import OptionQuote


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Model architecture
    d_geom: int = 3
    d_price: int = 4
    d_liquidity: int = 4
    d_model: int = 128
    n_encoder_layers: int = 3
    n_gnn_layers: int = 3
    model_kind: str = "gnn"
    use_fourier: bool = True
    use_attention: bool = True

    # Training
    n_epochs: int = 200
    batch_size: int = 4  # graphs per batch
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    dropout: float = 0.1
    torch_num_threads: int | None = None

    # Loss weights
    iv_recon_weight: float = 1.0
    geom_recon_weight: float = 0.5
    liq_recon_weight: float = 0.3
    heteroscedastic_weight: float = 0.0
    reliability_gate_weight: float = 0.0
    reliability_epsilon: float = 1e-4
    greeks_weight: float = 0.0
    no_arb_weight: float = 1.0
    smoothness_weight: float = 0.1
    contrastive_weight: float = 0.2
    contrastive_temperature: float = 0.1

    # No-arbitrage weights
    calendar_weight: float = 1.0
    butterfly_weight: float = 1.0
    put_call_weight: float = 0.5
    convexity_weight: float = 0.5
    decoded_regularizer_max_terms: int = 256

    # Data
    similarity_edges_per_node: int = 3
    use_liquidity_features: bool = True
    use_liquidity_gate: bool = True
    val_split: float = 0.15
    test_split: float = 0.15
    split_mode: str = "random"
    heldout_tickers: tuple[str, ...] = ()
    task_mode: str = "observed_reconstruction"
    mask_fraction: float = 0.2
    mask_regime: str = "stratified"
    target_space: str = "iv"
    query_node_feature_policy: str = "contract_time_geometry_only"
    visible_context_policy: str = "visible_nodes_after_mask"
    normalizer_scope: str = "train_or_visible_only"
    synthetic_surfaces: int = 8
    synthetic_underlyings: int = 2
    synthetic_maturities: int = 5
    synthetic_strikes: int = 9

    # Output
    output_dir: str = "reports/runs"
    experiment_name: str = "log-iv-v1"
    save_every: int = 20
    log_every: int = 10
    seed: int = 42

    # Hardware
    device: str = "auto"

    # Post-processing / baseline artifacts
    baseline_preset: str = "full"
    baseline_eval_splits: tuple[str, ...] = ("val", "test")
    svi_timeout_seconds: float = 1.0
    svi_maxiter: int = 200
    postprocess_verbose: bool = True
    no_arb_diagnostics_mode: str = "sampled_surface"
    no_arb_eval_splits: tuple[str, ...] = ("val", "test")
    no_arb_max_surfaces_per_split: int = 100
    no_arb_sample_seed: int = 0


@dataclass
class TrainingMetrics:
    """Container for training metrics."""

    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    iv_recon_loss: float = 0.0
    geom_recon_loss: float = 0.0
    liq_recon_loss: float = 0.0
    heteroscedastic_loss: float = 0.0
    greeks_loss: float = 0.0
    no_arb_loss: float = 0.0
    smoothness_loss: float = 0.0
    contrastive_loss: float = 0.0
    grad_norm: float = 0.0
    lr: float = 0.0
    epoch_time: float = 0.0


@dataclass(frozen=True)
class SurfaceData:
    """One train/eval surface with separate model inputs and paper targets."""

    surface_id: str
    target_quotes: list[OptionQuote]
    input_quotes: list[OptionQuote]
    target_mask: tuple[bool, ...]


@dataclass(frozen=True)
class RegisteredSplits:
    """Deterministic benchmark split plus its auditable manifest payload."""

    train: list[Any]
    val: list[Any]
    test: list[Any]
    manifest: dict[str, Any]


def _surface_target_quotes(surface: Any) -> list[OptionQuote]:
    return surface.target_quotes if isinstance(surface, SurfaceData) else surface


def _surface_input_quotes(surface: Any) -> list[OptionQuote]:
    return surface.input_quotes if isinstance(surface, SurfaceData) else surface


def _surface_mask(surface: Any) -> tuple[bool, ...]:
    quotes = _surface_target_quotes(surface)
    if isinstance(surface, SurfaceData):
        return surface.target_mask
    return tuple(False for _ in quotes)


def _surface_observed_flags(surface: Any) -> tuple[bool, ...]:
    return tuple(not value for value in _surface_mask(surface))


def _masked_query_quote(quote: OptionQuote) -> OptionQuote:
    return quote.model_copy(
        update={
            "bid": None,
            "ask": None,
            "implied_vol": None,
            "volume": 0,
            "open_interest": 0,
        }
    )


def _surface_id(surface: Any, fallback: str) -> str:
    return surface.surface_id if isinstance(surface, SurfaceData) else fallback


_GRAPH_CACHE: dict[tuple[int, int, int], Any] = {}


def _cached_option_surface_graph(
    quotes: list[OptionQuote],
    *,
    similarity_edges_per_node: int,
) -> Any:
    key = (id(quotes), len(quotes), similarity_edges_per_node)
    graph = _GRAPH_CACHE.get(key)
    if graph is None:
        graph = build_option_surface_graph(
            quotes,
            similarity_edges_per_node=similarity_edges_per_node,
        )
        _GRAPH_CACHE[key] = graph
    return graph


class OptionTokenGNN(nn.Module):
    """Full OptionToken-GNN model combining encoder, GNN, and decoder.

    Architecture:
    1. OptionTokenEncoder: quote → token embedding
    2. LiquidityGraphOperator: token graph → smoothed embeddings
    3. Decoder heads for IV prediction, geometry reconstruction, etc.
    """

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = cfg = config
        if cfg.model_kind not in {"gnn", "encoder_mlp"}:
            msg = "model_kind must be one of: gnn, encoder_mlp"
            raise ValueError(msg)
        self.use_gnn = cfg.model_kind == "gnn" and cfg.n_gnn_layers > 0

        self.encoder = OptionTokenEncoder(
            d_geom=cfg.d_geom,
            d_price=cfg.d_price,
            d_liquidity=cfg.d_liquidity,
            d_model=cfg.d_model,
            n_layers=cfg.n_encoder_layers,
            dropout=cfg.dropout,
            use_fourier=cfg.use_fourier,
        )

        self.gnn = (
            LiquidityGraphOperator(
                node_dim=cfg.d_model,
                hidden_dim=cfg.d_model * 2,
                n_layers=cfg.n_gnn_layers,
                dropout=cfg.dropout,
                use_attention=cfg.use_attention,
            )
            if self.use_gnn
            else None
        )

        self.regularizer = NoArbitrageRegularizer(
            calendar_weight=cfg.calendar_weight,
            butterfly_weight=cfg.butterfly_weight,
            put_call_weight=cfg.put_call_weight,
            convexity_weight=cfg.convexity_weight,
        )

        # Surface projection head (for contrastive learning)
        surface_dim = cfg.d_model * 2 if self.use_gnn else cfg.d_model
        self.surface_proj = nn.Sequential(
            nn.Linear(surface_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.reliability_head = nn.Linear(cfg.d_model, 1)

    def encode_graph(
        self,
        quotes: Any,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Encode a single option surface graph."""
        input_quotes = _surface_input_quotes(quotes)
        target_quotes = _surface_target_quotes(quotes)
        target_mask_tuple = _surface_mask(quotes)
        input_quotes = [
            _masked_query_quote(quote) if is_masked else quote
            for quote, is_masked in zip(input_quotes, target_mask_tuple, strict=True)
        ]
        observed_flags = _surface_observed_flags(quotes)

        # Extract model inputs and separate target features. Masked reconstruction
        # removes held-out IV/price values from ``input_quotes`` only.
        geom, price, liq = extract_features_from_quotes(input_quotes, device, observed_flags)
        if not self.config.use_liquidity_features:
            liq = torch.zeros_like(liq)
        target_geom, target_price, target_liq = extract_features_from_quotes(
            target_quotes,
            device,
            [True] * len(target_quotes),
        )
        target_mask = torch.tensor(target_mask_tuple, dtype=torch.bool, device=device)

        # Encode
        enc_out = self.encoder(geom, price, liq)
        node_embeddings = enc_out["h"]
        input_log_precision = self.reliability_head(node_embeddings).clamp(-8.0, 8.0)
        message_embeddings = _reliability_gated_embeddings(
            node_embeddings,
            input_log_precision,
            self.config.reliability_gate_weight,
        )

        # Build graph
        graph = _cached_option_surface_graph(
            input_quotes,
            similarity_edges_per_node=self.config.similarity_edges_per_node,
        )
        hetero_data = build_hetero_data_from_graph(input_quotes, message_embeddings, graph, device)

        if self.use_gnn and self.gnn is not None:
            # GNN forward
            gnn_liquidity = liq if self.config.use_liquidity_gate else None
            gnn_out = self.gnn(message_embeddings, hetero_data, gnn_liquidity)

            # Re-decode from GNN outputs
            refined_embeddings = gnn_out["node_embeddings"]
            surface_embedding = gnn_out["surface_embedding"]
            layer_outputs = gnn_out["layer_outputs"]
            log_precision = self.reliability_head(refined_embeddings).clamp(-8.0, 8.0)
            decode_embeddings = _reliability_gated_embeddings(
                refined_embeddings,
                log_precision,
                self.config.reliability_gate_weight,
            )
            iv_pred = self.encoder.iv_head(decode_embeddings)
            geom_recon_g = self.encoder.geom_head(decode_embeddings)
            liq_recon_g = self.encoder.liquidity_head(decode_embeddings)
            greeks_pred_g = self.encoder.greeks_head(decode_embeddings)
        else:
            refined_embeddings = message_embeddings
            surface_embedding = message_embeddings.mean(dim=0)
            layer_outputs = []
            log_precision = input_log_precision
            decode_embeddings = _reliability_gated_embeddings(
                refined_embeddings,
                log_precision,
                self.config.reliability_gate_weight,
            )
            iv_pred = self.encoder.iv_head(decode_embeddings)
            geom_recon_g = self.encoder.geom_head(decode_embeddings)
            liq_recon_g = self.encoder.liquidity_head(decode_embeddings)
            greeks_pred_g = self.encoder.greeks_head(decode_embeddings)

        return {
            "node_embeddings": refined_embeddings,
            "surface_embedding": surface_embedding,
            "iv_pred": iv_pred,
            "geom_recon": geom_recon_g,
            "liq_recon": liq_recon_g,
            "greeks_pred": greeks_pred_g,
            "log_precision": log_precision,
            "geom": geom,
            "price": price,
            "liquidity": liq,
            "target_geom": target_geom,
            "target_price": target_price,
            "target_liquidity": target_liq,
            "target_mask": target_mask,
            "target_quotes": target_quotes,
            "hetero_data": hetero_data,
            "layer_outputs": layer_outputs,
        }

    def forward(
        self,
        batch_quotes: list[Any],
        device: torch.device,
    ) -> list[dict[str, torch.Tensor]]:
        """Forward pass on a batch of graphs."""
        return [self.encode_graph(quotes, device) for quotes in batch_quotes]


def _reliability_gated_embeddings(
    embeddings: torch.Tensor,
    log_precision: torch.Tensor,
    gate_weight: float,
) -> torch.Tensor:
    if gate_weight == 0:
        return embeddings
    weight = float(np.clip(gate_weight, 0.0, 1.0))
    gate = torch.sigmoid(log_precision)
    factor = 1.0 + weight * (gate - 0.5)
    return embeddings * factor


def compute_losses(
    model: OptionTokenGNN,
    batch_outputs: list[dict[str, torch.Tensor]],
    config: TrainingConfig,
) -> dict[str, torch.Tensor]:
    """Compute all loss terms from batch outputs.

    Parameters
    ----------
    model : OptionTokenGNN
    batch_outputs : list[dict]
        List of per-graph outputs from encode_graph.
    config : TrainingConfig

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary of loss terms and total loss.
    """
    device = batch_outputs[0]["node_embeddings"].device
    B = len(batch_outputs)

    total_iv_recon = torch.tensor(0.0, device=device)
    total_geom_recon = torch.tensor(0.0, device=device)
    total_liq_recon = torch.tensor(0.0, device=device)
    total_heteroscedastic = torch.tensor(0.0, device=device)
    total_greeks = torch.tensor(0.0, device=device)
    total_no_arb = torch.tensor(0.0, device=device)
    total_smoothness = torch.tensor(0.0, device=device)

    for out in batch_outputs:
        # IV reconstruction: decode masked/observed targets without leaking masked IV inputs.
        iv_pred = out["iv_pred"]  # (N, 1)
        iv_true = out["target_price"][:, 0:1]  # first price target feature = mid IV
        if config.task_mode == "masked_reconstruction":
            iv_mask = out["target_mask"].unsqueeze(-1) & (iv_true > 0)
        else:
            iv_mask = iv_true > 0
        if bool(iv_mask.any().item()):
            iv_error = iv_pred[iv_mask] - iv_true[iv_mask]
            iv_loss = torch.abs(iv_error).mean()
            if config.heteroscedastic_weight != 0:
                precision = F.softplus(out["log_precision"][iv_mask]) + config.reliability_epsilon
                hetero_loss = 0.5 * (precision * iv_error.square() - torch.log(precision)).mean()
            else:
                hetero_loss = torch.tensor(0.0, device=device)
        else:
            iv_loss = torch.tensor(0.0, device=device)
            hetero_loss = torch.tensor(0.0, device=device)
        total_iv_recon = total_iv_recon + iv_loss
        total_heteroscedastic = total_heteroscedastic + hetero_loss

        # Geometry reconstruction
        geom_pred = out["geom_recon"]
        geom_true = out["target_geom"]
        geom_loss = F.mse_loss(geom_pred, geom_true, reduction="mean")
        total_geom_recon = total_geom_recon + geom_loss

        # Liquidity reconstruction
        liq_pred = out["liq_recon"]
        liq_true = out["target_liquidity"]
        liq_loss = F.mse_loss(liq_pred, liq_true, reduction="mean")
        total_liq_recon = total_liq_recon + liq_loss

        # Disabled by default: zero-target Greeks are diagnostic only until BS Greek targets exist.
        greeks_loss = (
            (out["greeks_pred"] ** 2).mean() * 0.01
            if config.greeks_weight != 0
            else torch.tensor(0.0, device=device)
        )
        total_greeks = total_greeks + greeks_loss

        # Decoded-price no-arbitrage regularizer. Embedding-distance PCP/convexity
        # proxies are intentionally not used for paper-facing benchmark runs.
        if config.no_arb_weight != 0 and any(
            weight != 0
            for weight in (
                config.calendar_weight,
                config.butterfly_weight,
                config.convexity_weight,
            )
        ):
            total_no_arb = total_no_arb + _decoded_no_arb_loss(out, config)

        # Graph smoothness
        sm_loss = (
            smoothness_regularizer(
                out["node_embeddings"],
                out["hetero_data"],
            )
            if config.smoothness_weight != 0
            else torch.tensor(0.0, device=device)
        )
        total_smoothness = total_smoothness + sm_loss

    # Average over batch
    n = max(B, 1)
    losses = {
        "iv_recon": total_iv_recon / n,
        "geom_recon": total_geom_recon / n,
        "liq_recon": total_liq_recon / n,
        "heteroscedastic": total_heteroscedastic / n,
        "greeks": total_greeks / n,
        "no_arb": total_no_arb / n,
        "smoothness": total_smoothness / n,
    }

    # Contrastive loss: across surfaces
    contr_loss = torch.tensor(0.0, device=device)
    if B > 1:
        surface_embs = torch.stack([out["surface_embedding"].squeeze() for out in batch_outputs])
        surface_embs = model.surface_proj(surface_embs)
        surface_embs = F.normalize(surface_embs, dim=-1)

        # SimCLR-style contrastive: all pairs are negatives except same-surface
        sim_matrix = torch.matmul(surface_embs, surface_embs.T) / config.contrastive_temperature
        labels = torch.arange(B, device=device)
        contr_loss = F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)
        contr_loss = contr_loss / 2.0
    losses["contrastive"] = contr_loss

    # Total loss
    total_loss = (
        config.iv_recon_weight * losses["iv_recon"]
        + config.geom_recon_weight * losses["geom_recon"]
        + config.liq_recon_weight * losses["liq_recon"]
        + config.heteroscedastic_weight * losses["heteroscedastic"]
        + config.greeks_weight * losses["greeks"]
        + config.no_arb_weight * losses["no_arb"]
        + config.smoothness_weight * losses["smoothness"]
        + config.contrastive_weight * losses["contrastive"]
    )
    losses["total"] = total_loss

    return losses


def _decoded_no_arb_loss(out: dict[str, Any], config: TrainingConfig) -> torch.Tensor:
    """Decoded Black-forward calendar/convexity penalties from predicted IV."""

    device = out["iv_pred"].device
    quotes: list[OptionQuote] = out["target_quotes"]
    iv_pred = out["iv_pred"].squeeze(-1).clamp_min(1e-6)
    penalties: list[torch.Tensor] = []
    max_terms = config.decoded_regularizer_max_terms

    if config.calendar_weight != 0:
        for indices in _group_quote_indices(quotes, ("option_type", "strike")).values():
            if 0 < max_terms <= len(penalties):
                break
            if len(indices) < 2:
                continue
            ordered = sorted(indices, key=lambda idx: quotes[idx].tenor_years)
            tenors = torch.tensor(
                [max(quotes[idx].tenor_years, 1e-8) for idx in ordered],
                dtype=iv_pred.dtype,
                device=device,
            )
            total_variance = iv_pred[ordered].square() * tenors
            penalties.extend(
                _take_penalty_terms(
                    (
                        torch.relu(total_variance[:-1] - total_variance[1:])
                        * config.calendar_weight
                    ).unbind(),
                    max_terms=max_terms,
                    current_terms=len(penalties),
                )
            )

    convexity_weight = config.convexity_weight + config.butterfly_weight
    if convexity_weight != 0:
        for indices in _group_quote_indices(quotes, ("expiry", "option_type")).values():
            if 0 < max_terms <= len(penalties):
                break
            if len(indices) < 3:
                continue
            ordered = sorted(indices, key=lambda idx: quotes[idx].strike)
            for left, middle, right in zip(ordered[:-2], ordered[1:-1], ordered[2:], strict=False):
                if 0 < max_terms <= len(penalties):
                    break
                q1, q2, q3 = quotes[left], quotes[middle], quotes[right]
                if q3.strike <= q1.strike:
                    continue
                p1 = _torch_black_forward_norm_price(q1, iv_pred[left])
                p2 = _torch_black_forward_norm_price(q2, iv_pred[middle])
                p3 = _torch_black_forward_norm_price(q3, iv_pred[right])
                alpha = (q3.strike - q2.strike) / (q3.strike - q1.strike)
                beta = (q2.strike - q1.strike) / (q3.strike - q1.strike)
                penalties.append(torch.relu(p2 - (alpha * p1 + beta * p3)) * convexity_weight)

    if not penalties:
        return torch.tensor(0.0, device=device)
    return torch.stack([penalty.square() for penalty in penalties]).mean()


def _take_penalty_terms(
    terms: tuple[torch.Tensor, ...],
    *,
    max_terms: int,
    current_terms: int,
) -> tuple[torch.Tensor, ...]:
    if max_terms <= 0:
        return terms
    remaining = max(max_terms - current_terms, 0)
    return terms[:remaining]


def _group_quote_indices(
    quotes: list[OptionQuote],
    fields: tuple[str, ...],
) -> dict[tuple[Any, ...], list[int]]:
    groups: dict[tuple[Any, ...], list[int]] = {}
    for index, quote in enumerate(quotes):
        key = tuple(getattr(quote, field) for field in fields)
        groups.setdefault(key, []).append(index)
    return groups


def _torch_black_forward_norm_price(quote: OptionQuote, sigma: torch.Tensor) -> torch.Tensor:
    reference = quote.forward or quote.underlying_price
    if reference is None or reference <= 0:
        return sigma.new_tensor(0.0)
    forward = sigma.new_tensor(float(reference))
    strike = sigma.new_tensor(float(quote.strike))
    tenor = sigma.new_tensor(max(float(quote.tenor_years), 1e-8))
    vol_sqrt_t = sigma.clamp_min(1e-6) * torch.sqrt(tenor)
    d1 = (torch.log(forward / strike) + 0.5 * sigma.square() * tenor) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    if quote.option_type == "C":
        price = forward * _torch_normal_cdf(d1) - strike * _torch_normal_cdf(d2)
    else:
        price = strike * _torch_normal_cdf(-d2) - forward * _torch_normal_cdf(-d1)
    return price.clamp_min(0.0) / forward


def _torch_normal_cdf(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(value / math.sqrt(2.0)))


def train_epoch(
    model: OptionTokenGNN,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_quotes: list[list[OptionQuote]],
    config: TrainingConfig,
    device: torch.device,
) -> TrainingMetrics:
    """Train for one epoch."""
    model.train()
    metrics = TrainingMetrics()

    n_graphs = len(train_quotes)
    indices = np.random.permutation(n_graphs)
    total_loss_sum = 0.0
    loss_detail_sum = {
        k: 0.0
        for k in [
            "iv_recon",
            "geom_recon",
            "liq_recon",
            "heteroscedastic",
            "greeks",
            "no_arb",
            "smoothness",
            "contrastive",
        ]
    }
    n_batches = 0
    grad_norms = []

    for start in range(0, n_graphs, config.batch_size):
        batch_indices = indices[start : start + config.batch_size]
        batch = [train_quotes[i] for i in batch_indices]

        optimizer.zero_grad()
        batch_outputs = model(batch, device)
        losses = compute_losses(model, batch_outputs, config)

        losses["total"].backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norms.append(grad_norm.item())

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss_sum += losses["total"].item()
        for k in loss_detail_sum:
            loss_detail_sum[k] += losses[k].item()
        n_batches += 1

    n = max(n_batches, 1)
    metrics.train_loss = total_loss_sum / n
    metrics.iv_recon_loss = loss_detail_sum["iv_recon"] / n
    metrics.geom_recon_loss = loss_detail_sum["geom_recon"] / n
    metrics.liq_recon_loss = loss_detail_sum["liq_recon"] / n
    metrics.heteroscedastic_loss = loss_detail_sum["heteroscedastic"] / n
    metrics.greeks_loss = loss_detail_sum["greeks"] / n
    metrics.no_arb_loss = loss_detail_sum["no_arb"] / n
    metrics.smoothness_loss = loss_detail_sum["smoothness"] / n
    metrics.contrastive_loss = loss_detail_sum["contrastive"] / n
    metrics.grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
    metrics.lr = optimizer.param_groups[0]["lr"]

    return metrics


@torch.no_grad()
def validate(
    model: OptionTokenGNN,
    val_quotes: list[list[OptionQuote]],
    config: TrainingConfig,
    device: torch.device,
) -> float:
    """Compute validation loss."""
    model.eval()
    total_loss = 0.0

    for quotes in val_quotes:
        batch_outputs = model([quotes], device)
        losses = compute_losses(model, batch_outputs, config)
        total_loss += losses["total"].item()

    return total_loss / max(len(val_quotes), 1)


def prepare_data_splits(
    all_quotes: list[list[OptionQuote]],
    config: TrainingConfig,
) -> tuple[list[list], list[list], list[list]]:
    """Split data into train/val/test sets.

    Parameters
    ----------
    all_quotes : list[list[OptionQuote]]
        Each element is a list of quotes for one surface/date.
    config : TrainingConfig

    Returns
    -------
    tuple[list, list, list]: train, val, test splits
    """
    registered = prepare_registered_splits(all_quotes, config)
    return registered.train, registered.val, registered.test


def prepare_registered_splits(
    all_quotes: list[list[OptionQuote]],
    config: TrainingConfig,
    surface_ids: list[str] | None = None,
) -> RegisteredSplits:
    """Create deterministic benchmark splits and apply task-specific masking."""

    if len(all_quotes) < 3:
        msg = "need at least three option surfaces for train/val/test split"
        raise ValueError(msg)
    surface_ids = surface_ids or [str(index) for index in range(len(all_quotes))]
    mode = config.split_mode
    if mode == "random":
        split_indices, split_meta = _random_split_indices(all_quotes, config)
    elif mode == "temporal":
        split_indices, split_meta = _temporal_split_indices(all_quotes, config)
    elif mode == "ticker_holdout":
        split_indices, split_meta = _ticker_holdout_split_indices(all_quotes, config)
    elif mode == "temporal_ticker_holdout":
        split_indices, split_meta = _temporal_ticker_holdout_split_indices(all_quotes, config)
    else:
        msg = "split_mode must be one of: random, temporal, ticker_holdout, temporal_ticker_holdout"
        raise ValueError(msg)

    split_payload: dict[str, Any] = {
        "split_id": _split_id(mode, config, surface_ids),
        "split_mode": mode,
        "task_mode": config.task_mode,
        "mask_fraction": config.mask_fraction
        if config.task_mode == "masked_reconstruction"
        else 0.0,
        "seed": config.seed,
        **split_meta,
    }
    split_surfaces: dict[str, list[Any]] = {}
    for split, indices in split_indices.items():
        split_surfaces[split] = [
            _prepare_surface_for_task(all_quotes[index], surface_ids[index], config, split)
            for index in indices
        ]
    split_payload.update(_split_count_payload(split_surfaces))
    split_payload["surface_ids"] = {
        split: [surface_ids[index] for index in indices] for split, indices in split_indices.items()
    }
    return RegisteredSplits(
        train=split_surfaces["train"],
        val=split_surfaces["val"],
        test=split_surfaces["test"],
        manifest=split_payload,
    )


def _random_split_indices(
    all_quotes: list[list[OptionQuote]],
    config: TrainingConfig,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    n = len(all_quotes)
    indices = list(np.random.RandomState(config.seed).permutation(n))
    n_test = max(1, int(n * config.test_split))
    n_val = max(1, int(n * config.val_split))
    n_train = n - n_val - n_test
    if n_train <= 0:
        msg = "split fractions leave no training surfaces"
        raise ValueError(msg)
    return (
        {
            "train": indices[:n_train],
            "val": indices[n_train : n_train + n_val],
            "test": indices[n_train + n_val :],
        },
        {"date_cutoffs": {}, "heldout_tickers": []},
    )


def _temporal_split_indices(
    all_quotes: list[list[OptionQuote]],
    config: TrainingConfig,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    dates = sorted({surface[0].observation_date for surface in all_quotes if surface})
    if len(dates) < 3:
        msg = "temporal split requires at least three observation dates"
        raise ValueError(msg)
    n_test = max(1, math.ceil(len(dates) * config.test_split))
    n_val = max(1, math.ceil(len(dates) * config.val_split))
    if len(dates) - n_test - n_val <= 0:
        msg = "temporal split fractions leave no training dates"
        raise ValueError(msg)
    test_dates = set(dates[-n_test:])
    val_dates = set(dates[-(n_test + n_val) : -n_test])
    train_dates = set(dates[: -(n_test + n_val)])
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for index, surface in enumerate(all_quotes):
        obs = surface[0].observation_date
        if obs in test_dates:
            splits["test"].append(index)
        elif obs in val_dates:
            splits["val"].append(index)
        elif obs in train_dates:
            splits["train"].append(index)
    return (
        splits,
        {
            "date_cutoffs": {
                "train_end": max(train_dates).isoformat(),
                "val_start": min(val_dates).isoformat(),
                "val_end": max(val_dates).isoformat(),
                "test_start": min(test_dates).isoformat(),
            },
            "heldout_tickers": [],
        },
    )


def _ticker_holdout_split_indices(
    all_quotes: list[list[OptionQuote]],
    config: TrainingConfig,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    tickers = sorted({surface[0].underlying for surface in all_quotes if surface})
    if len(tickers) < 3 and not config.heldout_tickers:
        msg = "ticker holdout split requires at least three underlyings or explicit heldouts"
        raise ValueError(msg)
    test_tickers = set(config.heldout_tickers or (tickers[-1],))
    val_tickers = set(tickers[-2:-1]) if not config.heldout_tickers else set()
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for index, surface in enumerate(all_quotes):
        ticker = surface[0].underlying
        if ticker in test_tickers:
            splits["test"].append(index)
        elif ticker in val_tickers:
            splits["val"].append(index)
        else:
            splits["train"].append(index)
    if not splits["val"]:
        train_tail = splits["train"][-max(1, len(splits["train"]) // 6) :]
        splits["val"] = train_tail
        splits["train"] = splits["train"][: -len(train_tail)]
    return (
        splits,
        {"date_cutoffs": {}, "heldout_tickers": sorted(test_tickers)},
    )


def _temporal_ticker_holdout_split_indices(
    all_quotes: list[list[OptionQuote]],
    config: TrainingConfig,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    temporal, meta = _temporal_split_indices(all_quotes, config)
    heldout = set(config.heldout_tickers)
    if not heldout:
        tickers = sorted({surface[0].underlying for surface in all_quotes if surface})
        heldout = {tickers[-1]} if tickers else set()
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for split, indices in temporal.items():
        for index in indices:
            ticker = all_quotes[index][0].underlying
            if ticker in heldout:
                splits["test"].append(index)
            else:
                splits[split].append(index)
    meta["heldout_tickers"] = sorted(heldout)
    return splits, meta


def _split_id(mode: str, config: TrainingConfig, surface_ids: list[str]) -> str:
    payload = json.dumps(
        {
            "mode": mode,
            "seed": config.seed,
            "val_split": config.val_split,
            "test_split": config.test_split,
            "heldout_tickers": config.heldout_tickers,
            "surface_ids": surface_ids,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _split_count_payload(split_surfaces: dict[str, list[Any]]) -> dict[str, Any]:
    return (
        {f"{split}_count": len(surfaces) for split, surfaces in split_surfaces.items()}
        | {
            f"{split}_rows": sum(len(_surface_target_quotes(surface)) for surface in surfaces)
            for split, surfaces in split_surfaces.items()
        }
        | {
            f"{split}_masked_rows": sum(sum(_surface_mask(surface)) for surface in surfaces)
            for split, surfaces in split_surfaces.items()
        }
    )


def _prepare_surface_for_task(
    quotes: list[OptionQuote],
    surface_id: str,
    config: TrainingConfig,
    split: str,
) -> Any:
    if config.task_mode != "masked_reconstruction":
        return quotes
    mask = _deterministic_surface_mask(quotes, config, surface_id, split)
    input_quotes = [
        _masked_query_quote(quote) if is_masked else quote
        for quote, is_masked in zip(quotes, mask, strict=True)
    ]
    return SurfaceData(
        surface_id=surface_id,
        target_quotes=quotes,
        input_quotes=input_quotes,
        target_mask=tuple(mask),
    )


def _deterministic_surface_mask(
    quotes: list[OptionQuote],
    config: TrainingConfig,
    surface_id: str,
    split: str,
) -> list[bool]:
    candidates = [index for index, quote in enumerate(quotes) if quote.implied_vol is not None]
    if not candidates:
        return [False] * len(quotes)
    target_count = max(1, int(round(len(candidates) * config.mask_fraction)))
    regime = config.mask_regime
    if regime == "random":
        selected = sorted(
            candidates,
            key=lambda idx: _stable_score(config.seed, surface_id, split, "random", quotes[idx]),
        )[:target_count]
        mask = [False] * len(quotes)
        for index in selected:
            mask[index] = True
        return mask
    if regime == "liquidity_correlated":
        selected = sorted(
            candidates,
            key=lambda idx: (
                quotes[idx].volume + quotes[idx].open_interest,
                _stable_score(config.seed, surface_id, split, "liquidity_correlated", quotes[idx]),
            ),
        )[:target_count]
        mask = [False] * len(quotes)
        for index in selected:
            mask[index] = True
        return mask
    if regime == "block_wing":
        selected = sorted(
            candidates,
            key=lambda idx: (
                -abs(float(quotes[idx].log_moneyness or 0.0)),
                _stable_score(config.seed, surface_id, split, "block_wing", quotes[idx]),
            ),
        )[:target_count]
        mask = [False] * len(quotes)
        for index in selected:
            mask[index] = True
        return mask
    if regime != "stratified":
        msg = (
            "mask_regime must be one of stratified/random/liquidity_correlated/block_wing; "
            f"got {regime!r}"
        )
        raise ValueError(msg)
    bucketed: dict[str, list[int]] = {}
    for index in candidates:
        bucketed.setdefault(_mask_bucket(quotes[index], quotes), []).append(index)
    stratified_selected: list[int] = []
    for bucket, indices in bucketed.items():
        quota = max(1, int(round(len(indices) * config.mask_fraction)))
        stratified_selected.extend(
            sorted(
                indices,
                key=lambda idx: _stable_score(config.seed, surface_id, split, bucket, quotes[idx]),
            )[:quota]
        )
    selected = sorted(
        set(stratified_selected),
        key=lambda idx: _stable_score(config.seed, surface_id, split, "final", quotes[idx]),
    )[:target_count]
    mask = [False] * len(quotes)
    for index in selected:
        mask[index] = True
    return mask


def _mask_bucket(quote: OptionQuote, surface: list[OptionQuote]) -> str:
    liquidity_values = [q.volume + q.open_interest for q in surface]
    median_liq = float(np.median(liquidity_values)) if liquidity_values else 0.0
    liquidity_bucket = "low_liq" if quote.volume + quote.open_interest <= median_liq else "high_liq"
    lm = quote.log_moneyness or 0.0
    if abs(lm) <= 0.03:
        moneyness_bucket = "atm"
    elif abs(lm) <= 0.1:
        moneyness_bucket = "near_wing"
    else:
        moneyness_bucket = "far_wing"
    tenor = quote.tenor_days
    tenor_bucket = "short" if tenor <= 45 else "medium" if tenor <= 180 else "long"
    return f"{liquidity_bucket}:{moneyness_bucket}:{tenor_bucket}:{quote.option_type}"


def _stable_score(
    seed: int,
    surface_id: str,
    split: str,
    bucket: str,
    quote: OptionQuote,
) -> str:
    token = "|".join(
        [
            str(seed),
            surface_id,
            split,
            bucket,
            quote.market,
            quote.underlying,
            quote.observation_date.isoformat(),
            quote.expiry.isoformat(),
            f"{quote.strike:.8f}",
            quote.option_type,
            quote.vendor_symbol or "",
        ]
    )
    return hashlib.sha256(token.encode()).hexdigest()


def train(
    train_quotes: list[list[OptionQuote]],
    val_quotes: list[list[OptionQuote]],
    config: TrainingConfig | None = None,
) -> tuple[OptionTokenGNN, list[TrainingMetrics]]:
    """Full training loop.

    Parameters
    ----------
    train_quotes : list[list[OptionQuote]]
        Training surfaces.
    val_quotes : list[list[OptionQuote]]
        Validation surfaces.
    config : TrainingConfig, optional

    Returns
    -------
    tuple[OptionTokenGNN, list[TrainingMetrics]]
        Trained model and metrics history.
    """
    cfg = config or TrainingConfig()
    if cfg.torch_num_threads is not None and cfg.torch_num_threads > 0:
        torch.set_num_threads(cfg.torch_num_threads)

    # Device
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if cfg.device == "mps" and not torch.backends.mps.is_available():
            msg = "MPS device requested but torch.backends.mps.is_available() is false"
            raise RuntimeError(msg)
        if cfg.device == "cuda" and not torch.cuda.is_available():
            msg = "CUDA device requested but torch.cuda.is_available() is false"
            raise RuntimeError(msg)
        device = torch.device(cfg.device)

    # Seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Model
    model = OptionTokenGNN(cfg).to(device)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(cfg.n_epochs // 4, 1),
        T_mult=2,
        eta_min=cfg.learning_rate * 0.01,
    )

    # Output directory
    output_dir = Path(cfg.output_dir) / cfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_history: list[TrainingMetrics] = []
    best_val_loss = float("inf")

    print(f"Training on {device}, {len(train_quotes)} train / {len(val_quotes)} val graphs")

    # Warmup forward to initialize LazyLinear modules
    dummy_quotes = train_quotes[0]
    _ = model.encode_graph(dummy_quotes, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, cfg.n_epochs + 1):
        t0 = time.time()

        metrics = train_epoch(model, optimizer, scheduler, train_quotes, cfg, device)
        metrics.epoch = epoch

        val_loss = validate(model, val_quotes, cfg, device)
        metrics.val_loss = val_loss
        metrics.epoch_time = time.time() - t0

        metrics_history.append(metrics)

        # Logging
        if epoch % cfg.log_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{cfg.n_epochs} | "
                f"train: {metrics.train_loss:.4f} | "
                f"val: {val_loss:.4f} | "
                f"iv: {metrics.iv_recon_loss:.4f} | "
                f"arb: {metrics.no_arb_loss:.4f} | "
                f"grad: {metrics.grad_norm:.3f} | "
                f"lr: {metrics.lr:.2e} | "
                f"time: {metrics.epoch_time:.1f}s"
            )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": cfg,
                },
                output_dir / "best_model.pt",
            )

        # Periodic checkpoint
        if epoch % cfg.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics_history": metrics_history,
                    "config": cfg,
                },
                output_dir / f"checkpoint_epoch_{epoch}.pt",
            )

    # Save final model
    torch.save(model.state_dict(), output_dir / "final_model.pt")

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(
            [
                {
                    "epoch": m.epoch,
                    "train_loss": m.train_loss,
                    "val_loss": m.val_loss,
                    "iv_recon_loss": m.iv_recon_loss,
                    "geom_recon_loss": m.geom_recon_loss,
                    "liq_recon_loss": m.liq_recon_loss,
                    "heteroscedastic_loss": m.heteroscedastic_loss,
                    "no_arb_loss": m.no_arb_loss,
                    "smoothness_loss": m.smoothness_loss,
                    "contrastive_loss": m.contrastive_loss,
                    "grad_norm": m.grad_norm,
                    "lr": m.lr,
                }
                for m in metrics_history
            ],
            f,
            indent=2,
        )
    _write_epoch_jsonl(output_dir / "metrics_epoch.jsonl", metrics_history)

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")

    return model, metrics_history


def run_synthetic_experiment(
    config: TrainingConfig | None = None,
) -> tuple[OptionTokenGNN, list[TrainingMetrics]]:
    """Run a full experiment on synthetic data.

    This is the main entry point for the research pipeline.
    """
    from log_iv.synthetic import (
        SyntheticSurfaceConfig,
        generate_synthetic_surface_dataset,
        synthetic_quotes_to_option_quotes,
    )

    cfg = config or TrainingConfig()

    print("=" * 60)
    print("LoG-IV: OptionToken-GNN Synthetic Experiment")
    print("=" * 60)

    # Generate synthetic data
    print("\n[1/4] Generating synthetic option surface dataset...")
    syn_config = SyntheticSurfaceConfig(
        n_maturities=cfg.synthetic_maturities,
        n_strikes=cfg.synthetic_strikes,
        min_tenor_days=7,
        max_tenor_days=730,
        moneyness_range=0.4,
        missing_wing_prob=0.05,
        stale_quote_prob=0.02,
        random_seed=cfg.seed,
    )
    dataset = generate_synthetic_surface_dataset(
        syn_config,
        n_surfaces=cfg.synthetic_surfaces,
        n_underlyings=cfg.synthetic_underlyings,
    )
    print(f"  Generated {len(dataset)} option surfaces")

    # Convert to OptionQuotes and build graphs
    print("\n[2/4] Converting quotes and building graphs...")
    all_graphs: list[list[OptionQuote]] = []
    surface_ids: list[str] = []
    for key, syn_quotes in dataset.items():
        option_quotes = synthetic_quotes_to_option_quotes(syn_quotes)
        all_graphs.append(option_quotes)
        surface_ids.append(key)
    print(f"  Built {len(all_graphs)} option surface graphs")

    # Split data
    print("\n[3/4] Splitting data...")
    registered = prepare_registered_splits(all_graphs, cfg, surface_ids)
    train_data, val_data, test_data = registered.train, registered.val, registered.test
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Train
    print("\n[4/4] Training...")
    model, metrics = train(train_data, val_data, cfg)

    # Final evaluation
    final_val_loss = validate(model, val_data, cfg, device=next(model.parameters()).device)
    final_test_loss = validate(model, test_data, cfg, device=next(model.parameters()).device)
    print(f"\nFinal: Val loss = {final_val_loss:.4f}, Test loss = {final_test_loss:.4f}")

    output_dir = Path(cfg.output_dir) / cfg.experiment_name
    _write_run_artifacts(
        output_dir=output_dir,
        model=model,
        config=cfg,
        run_label="engineering_smoke",
        dataset_label="synthetic",
        surface_ids=surface_ids,
        all_graphs=all_graphs,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        metrics=metrics,
        final_val_loss=final_val_loss,
        final_test_loss=final_test_loss,
        split_manifest=registered.manifest,
    )

    return model, metrics


def run_option_quote_dataset_experiment(
    all_graphs: list[list[OptionQuote]],
    surface_ids: list[str],
    config: TrainingConfig | None = None,
    *,
    run_label: str = "real_us_mvp",
    dataset_label: str = "real_option_quotes",
    source_surface_count: int | None = None,
    filtered_surface_count: int = 0,
    claim_label: str | None = None,
    extra_prediction_splits: dict[str, list[list[OptionQuote]]] | None = None,
    extra_prediction_surface_ids: dict[str, list[str]] | None = None,
) -> tuple[OptionTokenGNN, list[TrainingMetrics]]:
    """Train/evaluate on already normalized canonical option quote surfaces."""

    cfg = config or TrainingConfig()
    if len(all_graphs) < 3:
        msg = "need at least three option surfaces for train/val/test split"
        raise ValueError(msg)
    registered = prepare_registered_splits(all_graphs, cfg, surface_ids)
    train_data, val_data, test_data = registered.train, registered.val, registered.test
    model, metrics = train(train_data, val_data, cfg)
    device = next(model.parameters()).device
    final_val_loss = validate(model, val_data, cfg, device=device)
    final_test_loss = validate(model, test_data, cfg, device=device)
    output_dir = Path(cfg.output_dir) / cfg.experiment_name
    _write_run_artifacts(
        output_dir=output_dir,
        model=model,
        config=cfg,
        run_label=run_label,
        dataset_label=dataset_label,
        surface_ids=surface_ids,
        all_graphs=all_graphs,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        metrics=metrics,
        final_val_loss=final_val_loss,
        final_test_loss=final_test_loss,
        source_surface_count=source_surface_count,
        filtered_surface_count=filtered_surface_count,
        claim_label=claim_label,
        split_manifest=registered.manifest,
        extra_prediction_splits=extra_prediction_splits,
        extra_prediction_surface_ids=extra_prediction_surface_ids,
    )
    return model, metrics


def run_ood_transfer_experiment(
    us_graphs: list[list[OptionQuote]],
    us_surface_ids: list[str],
    jp_graphs: list[list[OptionQuote]],
    jp_surface_ids: list[str],
    config: TrainingConfig | None = None,
    *,
    dataset_label: str = "us_to_jp_ood",
    source_surface_count: int | None = None,
    filtered_surface_count: int = 0,
) -> tuple[OptionTokenGNN, list[TrainingMetrics]]:
    """Train on U.S. surfaces and evaluate Japan as an OOD split."""

    cfg = config or TrainingConfig()
    if len(us_graphs) < 3:
        msg = "need at least three U.S. option surfaces for OOD train/val/test split"
        raise ValueError(msg)
    if not jp_graphs:
        msg = "need at least one Japan option surface for OOD evaluation"
        raise ValueError(msg)

    registered = prepare_registered_splits(us_graphs, cfg, us_surface_ids)
    train_us, val_us, test_us = registered.train, registered.val, registered.test
    model, metrics = train(train_us, val_us, cfg)
    device = next(model.parameters()).device
    final_val_loss = validate(model, val_us, cfg, device=device)
    final_test_loss = validate(model, test_us, cfg, device=device)
    ood_loss = validate(model, jp_graphs, cfg, device=device)
    print(f"OOD Japan loss: {ood_loss:.4f}")

    output_dir = Path(cfg.output_dir) / cfg.experiment_name
    _write_run_artifacts(
        output_dir=output_dir,
        model=model,
        config=cfg,
        run_label="japan_ood_probe",
        dataset_label=dataset_label,
        surface_ids=us_surface_ids,
        all_graphs=us_graphs,
        train_data=train_us,
        val_data=val_us,
        test_data=test_us,
        metrics=metrics,
        final_val_loss=final_val_loss,
        final_test_loss=final_test_loss,
        source_surface_count=source_surface_count,
        filtered_surface_count=filtered_surface_count,
        claim_label="japan_ood_probe",
        extra_prediction_splits={"ood_jp": jp_graphs},
        extra_prediction_surface_ids={"ood_jp": jp_surface_ids},
        split_manifest=registered.manifest,
    )
    return model, metrics


def _metrics_to_dict(metric: TrainingMetrics) -> dict[str, float | int]:
    return {
        "epoch": metric.epoch,
        "train_loss": metric.train_loss,
        "val_loss": metric.val_loss,
        "iv_recon_loss": metric.iv_recon_loss,
        "geom_recon_loss": metric.geom_recon_loss,
        "liq_recon_loss": metric.liq_recon_loss,
        "heteroscedastic_loss": metric.heteroscedastic_loss,
        "greeks_loss": metric.greeks_loss,
        "no_arb_regularizer_loss": metric.no_arb_loss,
        "smoothness_loss": metric.smoothness_loss,
        "contrastive_loss": metric.contrastive_loss,
        "grad_norm": metric.grad_norm,
        "lr": metric.lr,
        "epoch_time": metric.epoch_time,
    }


def _write_epoch_jsonl(path: Path, metrics: list[TrainingMetrics]) -> None:
    with path.open("w") as handle:
        for metric in metrics:
            handle.write(json.dumps(_metrics_to_dict(metric), sort_keys=True) + "\n")


def _format_split_scope(splits: tuple[str, ...] | list[str] | None) -> str:
    if not splits:
        return "all"
    return ",".join(str(split) for split in splits)


@torch.no_grad()
def _prediction_frame(
    model: OptionTokenGNN,
    surfaces: list[Any],
    device: torch.device,
    split: str,
    surface_ids: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    model.eval()
    for surface_index, surface in enumerate(surfaces):
        quotes = _surface_target_quotes(surface)
        target_mask = _surface_mask(surface)
        out = model.encode_graph(surface, device)
        pred = out["iv_pred"].squeeze(-1).detach().cpu().numpy()
        true = out["target_price"][:, 0].detach().cpu().numpy()
        surface_id = (
            surface_ids[surface_index]
            if surface_ids is not None and surface_index < len(surface_ids)
            else _surface_id(surface, str(surface_index))
        )
        for node_index, quote in enumerate(quotes):
            rows.append(
                {
                    "split": split,
                    "surface_index": surface_index,
                    "surface_id": surface_id,
                    "node_index": node_index,
                    "market": quote.market,
                    "underlying": quote.underlying,
                    "observation_date": quote.observation_date.isoformat(),
                    "expiry": quote.expiry.isoformat(),
                    "strike": quote.strike,
                    "option_type": quote.option_type,
                    "iv_true": float(true[node_index]),
                    "iv_pred": float(pred[node_index]),
                    "abs_error": float(abs(pred[node_index] - true[node_index])),
                    "is_masked_target": bool(target_mask[node_index]),
                    "input_target_visible": bool(not target_mask[node_index]),
                    "bid": float(quote.bid) if quote.bid is not None else np.nan,
                    "ask": float(quote.ask) if quote.ask is not None else np.nan,
                    "mid_price": float(quote.mid) if quote.mid is not None else np.nan,
                    "spread": float(quote.spread or 0.0),
                    "volume": quote.volume,
                    "open_interest": quote.open_interest,
                    "forward": float(quote.forward) if quote.forward is not None else np.nan,
                    "underlying_price": (
                        float(quote.underlying_price)
                        if quote.underlying_price is not None
                        else np.nan
                    ),
                    "tenor_days": quote.tenor_days,
                    "tenor_years": quote.tenor_years,
                    "log_moneyness": float(quote.log_moneyness or 0.0),
                }
            )
    return pd.DataFrame(rows)


def _write_run_artifacts(
    *,
    output_dir: Path,
    model: OptionTokenGNN,
    config: TrainingConfig,
    run_label: str,
    dataset_label: str,
    surface_ids: list[str],
    all_graphs: list[list[OptionQuote]],
    train_data: list[Any],
    val_data: list[Any],
    test_data: list[Any],
    metrics: list[TrainingMetrics],
    final_val_loss: float,
    final_test_loss: float,
    source_surface_count: int | None = None,
    filtered_surface_count: int = 0,
    claim_label: str | None = None,
    extra_prediction_splits: dict[str, list[list[OptionQuote]]] | None = None,
    extra_prediction_surface_ids: dict[str, list[str]] | None = None,
    split_manifest: dict[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    postprocess_started = time.monotonic()
    postprocess_timings: list[dict[str, Any]] = []

    def log_postprocess(message: str) -> None:
        if config.postprocess_verbose:
            elapsed = time.monotonic() - postprocess_started
            print(f"[postprocess {run_label} +{elapsed:.1f}s] {message}", flush=True)

    def start_stage(name: str, detail: str) -> float:
        log_postprocess(f"{name}: {detail}")
        return time.monotonic()

    def finish_stage(name: str, started_at: float, detail: str = "done") -> None:
        elapsed = time.monotonic() - started_at
        postprocess_timings.append(
            {"stage": name, "elapsed_seconds": round(elapsed, 3), "detail": detail}
        )
        log_postprocess(f"{name}: {detail} ({elapsed:.1f}s)")

    device = next(model.parameters()).device
    surface_lookup = {id(quotes): surface_ids[i] for i, quotes in enumerate(all_graphs)}
    prediction_splits = []
    if val_data:
        started = start_stage("predict:val", f"{len(val_data)} surfaces")
        prediction_splits.append(
            _prediction_frame(
                model,
                val_data,
                device,
                "val",
                [
                    _surface_id(surface, surface_lookup.get(id(surface), str(i)))
                    for i, surface in enumerate(val_data)
                ],
            )
        )
        finish_stage("predict:val", started, f"{len(prediction_splits[-1])} rows")
    if test_data:
        started = start_stage("predict:test", f"{len(test_data)} surfaces")
        prediction_splits.append(
            _prediction_frame(
                model,
                test_data,
                device,
                "test",
                [
                    _surface_id(surface, surface_lookup.get(id(surface), str(i)))
                    for i, surface in enumerate(test_data)
                ],
            )
        )
        finish_stage("predict:test", started, f"{len(prediction_splits[-1])} rows")
    if extra_prediction_splits:
        for split_name, split_surfaces in extra_prediction_splits.items():
            started = start_stage(f"predict:{split_name}", f"{len(split_surfaces)} surfaces")
            ids = (
                extra_prediction_surface_ids.get(split_name)
                if extra_prediction_surface_ids is not None
                else None
            )
            prediction_splits.append(
                _prediction_frame(model, split_surfaces, device, split_name, ids)
            )
            finish_stage(
                f"predict:{split_name}",
                started,
                f"{len(prediction_splits[-1])} rows",
            )
    predictions = (
        pd.concat(prediction_splits, ignore_index=True) if prediction_splits else pd.DataFrame()
    )
    predictions_path = output_dir / "predictions.parquet"
    started = start_stage("write:predictions", f"{len(predictions)} rows")
    predictions.to_parquet(predictions_path, index=False)
    finish_stage("write:predictions", started, str(predictions_path.name))

    started = start_stage("metrics", "summarizing model predictions")
    metrics_summary = _metrics_summary(predictions, final_val_loss, final_test_loss, run_label)
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(metrics_summary, indent=2, sort_keys=True) + "\n"
    )
    finish_stage("metrics", started, "metrics_summary.json")

    baseline_scope = _format_split_scope(config.baseline_eval_splits)
    started = start_stage(
        "baselines",
        f"preset={config.baseline_preset} eval_splits={baseline_scope}",
    )
    baselines = _train_only_baseline_summary(
        train_data,
        predictions,
        baseline_preset=config.baseline_preset,
        baseline_eval_splits=config.baseline_eval_splits,
        svi_timeout_seconds=config.svi_timeout_seconds,
        svi_maxiter=config.svi_maxiter,
        progress=log_postprocess,
    )
    baselines.to_csv(output_dir / "baselines_summary.csv", index=False)
    finish_stage("baselines", started, f"{len(baselines)} rows")

    started = start_stage("diagnostic-baselines", "leakage-prone reference only")
    _diagnostic_leakage_prone_baseline_summary(predictions).to_csv(
        output_dir / "diagnostic_leakage_prone_baselines.csv",
        index=False,
    )
    finish_stage("diagnostic-baselines", started, "diagnostic_leakage_prone_baselines.csv")

    started = start_stage("diagnostics:price", f"{len(predictions)} rows")
    price_diagnostics = _price_diagnostics(predictions)
    finish_stage("diagnostics:price", started, "diagnostics_price.json")

    no_arb_scope = (
        f"mode={config.no_arb_diagnostics_mode} "
        f"eval_splits={_format_split_scope(config.no_arb_eval_splits)} "
        f"max_surfaces_per_split={config.no_arb_max_surfaces_per_split}"
    )
    started = start_stage("diagnostics:no-arb", no_arb_scope)
    no_arb_diagnostics = _no_arbitrage_diagnostics(
        predictions,
        mode=config.no_arb_diagnostics_mode,
        eval_splits=config.no_arb_eval_splits,
        max_surfaces_per_split=config.no_arb_max_surfaces_per_split,
        sample_seed=config.no_arb_sample_seed or config.seed,
    )
    metadata = no_arb_diagnostics.get("metadata", {})
    finish_stage(
        "diagnostics:no-arb",
        started,
        f"diagnostics_no_arbitrage.json rows={metadata.get('row_count', len(predictions))}",
    )
    (output_dir / "diagnostics_price.json").write_text(
        json.dumps(price_diagnostics, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "diagnostics_no_arbitrage.json").write_text(
        json.dumps(no_arb_diagnostics, indent=2, sort_keys=True) + "\n"
    )

    postprocess_summary = {
        "run_label": run_label,
        "baseline_preset": config.baseline_preset,
        "baseline_eval_splits": list(config.baseline_eval_splits),
        "svi_timeout_seconds": config.svi_timeout_seconds,
        "svi_maxiter": config.svi_maxiter,
        "no_arb_diagnostics": no_arb_diagnostics.get("metadata", {}),
        "prediction_rows": int(len(predictions)),
        "prediction_rows_by_split": (
            predictions.groupby("split").size().astype(int).to_dict()
            if "split" in predictions and not predictions.empty
            else {}
        ),
        "timings": postprocess_timings,
        "elapsed_seconds": round(time.monotonic() - postprocess_started, 3),
    }
    (output_dir / "postprocess_summary.json").write_text(
        json.dumps(postprocess_summary, indent=2, sort_keys=True) + "\n"
    )

    split_payload = dict(split_manifest or {})
    split_payload.update(
        {
            "train_count": len(train_data),
            "val_count": len(val_data),
            "test_count": len(test_data),
            "train_rows": sum(len(_surface_target_quotes(quotes)) for quotes in train_data),
            "val_rows": sum(len(_surface_target_quotes(quotes)) for quotes in val_data),
            "test_rows": sum(len(_surface_target_quotes(quotes)) for quotes in test_data),
            "extra_split_counts": {
                name: len(surfaces) for name, surfaces in (extra_prediction_splits or {}).items()
            },
            "extra_split_rows": {
                name: sum(len(quotes) for quotes in surfaces)
                for name, surfaces in (extra_prediction_splits or {}).items()
            },
            "surface_count": len(all_graphs),
            "source_surface_count": source_surface_count
            if source_surface_count is not None
            else len(all_graphs),
            "filtered_surface_count": filtered_surface_count,
            "surface_ids_head": surface_ids[:10],
            "seed": config.seed,
        }
    )
    (output_dir / "splits.json").write_text(json.dumps(split_payload, indent=2) + "\n")

    manifest = {
        "run_label": run_label,
        "claim_label": claim_label or run_label,
        "dataset": dataset_label,
        "created_at_unix": int(time.time()),
        "config": asdict(config),
        "epochs_recorded": len(metrics),
        "edge_weights": {
            "constructed": True,
            "consumed": True,
            "mechanism": "incoming-edge-weight gate plus weighted smoothness regularizer",
        },
        "no_arbitrage": {
            "training_regularizer": "decoded_calendar_and_convexity_only",
            "paper_facing_diagnostics": True,
            "diagnostic_mode": config.no_arb_diagnostics_mode,
            "diagnostic_eval_splits": list(config.no_arb_eval_splits),
            "diagnostic_max_surfaces_per_split": config.no_arb_max_surfaces_per_split,
            "diagnostic_sampling_unit": "surface_id",
            "embedding_proxy_regularizers": "disabled_for_benchmark_protocol",
            "put_call_parity": (
                "diagnostic_only_until_rate_forward_dividend_assumptions_are_explicit"
            ),
            "note": "diagnostics_no_arbitrage.json is computed after training on decoded prices",
        },
        "protocol": {
            "task_mode": config.task_mode,
            "target_space": config.target_space,
            "mask_regime": config.mask_regime,
            "query_node_feature_policy": config.query_node_feature_policy,
            "visible_context_policy": config.visible_context_policy,
            "normalizer_scope": config.normalizer_scope,
            "uses_masked_quote_features": False,
        },
        "artifacts": [
            "manifest.json",
            "splits.json",
            "metrics_epoch.jsonl",
            "metrics_summary.json",
            "predictions.parquet",
            "baselines_summary.csv",
            "diagnostic_leakage_prone_baselines.csv",
            "diagnostics_price.json",
            "diagnostics_no_arbitrage.json",
            "postprocess_summary.json",
            "README.md",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _write_run_readme(output_dir, metrics_summary, run_label)


def _metrics_summary(
    predictions: pd.DataFrame,
    final_val_loss: float,
    final_test_loss: float,
    run_label: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "label": run_label,
        "final_val_loss": final_val_loss,
        "final_test_loss": final_test_loss,
    }
    if predictions.empty:
        return summary
    error = predictions["iv_pred"] - predictions["iv_true"]
    summary["iv_mae"] = float(error.abs().mean())
    summary["iv_rmse"] = float(np.sqrt((error**2).mean()))
    if "is_masked_target" in predictions:
        masked = predictions[predictions["is_masked_target"].astype(bool)]
        if not masked.empty:
            masked_error = masked["iv_pred"] - masked["iv_true"]
            masked_abs_error = masked_error.abs()
            summary["masked_iv_mae"] = float(masked_abs_error.mean())
            summary["masked_iv_rmse"] = float(np.sqrt((masked_error**2).mean()))
            summary["masked_iv_p90_abs_error"] = float(masked_abs_error.quantile(0.90))
            summary["masked_count"] = int(len(masked))
            summary["headline_metric"] = "masked_iv_mae"
        else:
            summary["headline_metric"] = "iv_mae"
    summary["by_split"] = (
        predictions.assign(abs_error=error.abs())
        .groupby("split")["abs_error"]
        .mean()
        .round(8)
        .to_dict()
    )
    for bucket_name, series in {
        "option_type": predictions["option_type"],
        "underlying": predictions["underlying"],
    }.items():
        summary[f"by_{bucket_name}"] = (
            predictions.assign(abs_error=error.abs())
            .groupby(series)["abs_error"]
            .mean()
            .round(8)
            .to_dict()
        )
    bucket_specs = {
        "liquidity_bucket": pd.qcut(
            (predictions["volume"].fillna(0) + predictions["open_interest"].fillna(0)).rank(
                method="first"
            ),
            q=min(4, max(1, len(predictions))),
            duplicates="drop",
        ).astype(str),
        "moneyness_bucket": pd.cut(
            predictions["log_moneyness"],
            bins=[-float("inf"), -0.1, -0.03, 0.03, 0.1, float("inf")],
            labels=["deep_itm_put_otm_call", "near_wing_low", "atm", "near_wing_high", "far_wing"],
        ).astype(str),
        "tenor_bucket": pd.cut(
            predictions["tenor_days"],
            bins=[0, 14, 45, 90, 180, 365, float("inf")],
            labels=["0_14d", "15_45d", "46_90d", "91_180d", "181_365d", "365d_plus"],
        ).astype(str),
    }
    for bucket_name, series in bucket_specs.items():
        summary[f"by_{bucket_name}"] = (
            predictions.assign(abs_error=error.abs())
            .groupby(series, observed=True)["abs_error"]
            .mean()
            .round(8)
            .to_dict()
        )
        if "is_masked_target" in predictions:
            masked = predictions[predictions["is_masked_target"].astype(bool)]
            if not masked.empty:
                masked_series = series.loc[masked.index]
                summary[f"masked_by_{bucket_name}"] = (
                    masked.assign(abs_error=(masked["iv_pred"] - masked["iv_true"]).abs())
                    .groupby(masked_series, observed=True)["abs_error"]
                    .mean()
                    .round(8)
                    .to_dict()
                )
    return summary


def _train_only_baseline_summary(
    train_surfaces: list[Any],
    predictions: pd.DataFrame,
    *,
    baseline_preset: str = "full",
    baseline_eval_splits: tuple[str, ...] | list[str] | None = ("val", "test"),
    svi_timeout_seconds: float = 1.0,
    svi_maxiter: int = 200,
    progress: Any | None = None,
) -> pd.DataFrame:
    """Train-fitted baselines for fixed split / masked reconstruction evaluation."""

    def log(message: str) -> None:
        if progress is not None:
            progress(f"baselines: {message}")

    baseline_preset = str(baseline_preset).lower()
    if baseline_preset not in {"none", "fast", "full"}:
        msg = "baseline_preset must be one of: none, fast, full"
        raise ValueError(msg)
    if baseline_preset == "none":
        log("skipped because preset=none")
        return _empty_baseline_frame()
    if predictions.empty:
        return _empty_baseline_frame()
    train_frame = _surface_quote_frame(train_surfaces)
    if train_frame.empty:
        return _empty_baseline_frame()

    all_eval_frame = predictions.reset_index(drop=True).copy()
    if baseline_eval_splits and "split" in all_eval_frame:
        split_set = {str(split) for split in baseline_eval_splits}
        all_eval_frame = all_eval_frame[all_eval_frame["split"].astype(str).isin(split_set)].copy()
    eval_frame = all_eval_frame
    if (
        "is_masked_target" in all_eval_frame
        and all_eval_frame["is_masked_target"].astype(bool).any()
    ):
        eval_frame = all_eval_frame[all_eval_frame["is_masked_target"].astype(bool)].copy()
        eval_scope = "masked_nodes"
    else:
        eval_scope = "all_nodes"
    if eval_frame.empty:
        return _empty_baseline_frame()

    global_mean = float(train_frame["iv_true"].mean())
    train_min = float(train_frame["iv_true"].min())
    train_max = float(train_frame["iv_true"].max())
    by_underlying = train_frame.groupby("underlying", dropna=False)["iv_true"].mean().to_dict()
    by_bucket = train_frame.groupby("bucket", dropna=False)["iv_true"].mean().to_dict()
    log(
        "eval_rows="
        f"{len(eval_frame)} train_rows={len(train_frame)} "
        f"preset={baseline_preset} eval_splits={_format_split_scope(baseline_eval_splits)}"
    )
    baseline_preds = {
        "train_mean_iv_global": pd.Series(global_mean, index=eval_frame.index),
        "train_mean_iv_by_underlying": eval_frame["underlying"]
        .map(by_underlying)
        .fillna(global_mean),
        "train_mean_iv_by_moneyness_tenor_bucket": pd.Series(
            [
                by_bucket.get(_bucket_key_from_row(row), global_mean)
                for _, row in eval_frame.iterrows()
            ],
            index=eval_frame.index,
        ),
        "random_uniform_iv": pd.Series(
            np.random.default_rng(0).uniform(train_min, train_max, size=len(eval_frame)),
            index=eval_frame.index,
        ),
    }
    log("train_knn_moneyness_tenor start")
    started = time.monotonic()
    baseline_preds["train_knn_moneyness_tenor"] = _train_knn_iv_baseline(train_frame, eval_frame)
    log(f"train_knn_moneyness_tenor done ({time.monotonic() - started:.1f}s)")
    rows = []
    for name, pred in baseline_preds.items():
        rows.append(
            _baseline_metric_row(
                name,
                pred,
                eval_frame,
                fit_scope="train_only",
                eval_scope=eval_scope,
                uses_eval_visible_nodes=False,
            )
        )
    within_preds = _within_surface_baseline_predictions(
        all_eval_frame,
        eval_frame,
        include_svi=baseline_preset == "full",
        svi_timeout_seconds=svi_timeout_seconds,
        svi_maxiter=svi_maxiter,
        progress=progress,
    )
    for name, payload in within_preds.items():
        rows.append(
            _baseline_metric_row(
                name,
                payload["pred"],
                eval_frame,
                fit_scope="eval_visible_only",
                eval_scope=eval_scope,
                uses_eval_visible_nodes=True,
                status=payload.get("status"),
            )
            | payload.get("metadata", {})
        )
    result = pd.DataFrame(rows)
    result["baseline_preset"] = baseline_preset
    result["baseline_eval_splits"] = _format_split_scope(baseline_eval_splits)
    return result


def _baseline_metric_row(
    baseline: str,
    pred: pd.Series,
    eval_frame: pd.DataFrame,
    *,
    fit_scope: str,
    eval_scope: str,
    uses_eval_visible_nodes: bool,
    status: pd.Series | None = None,
) -> dict[str, Any]:
    pred = pd.Series(pred, index=eval_frame.index).astype(float)
    valid = pred.notna() & np.isfinite(pred) & eval_frame["iv_true"].notna()
    err = eval_frame.loc[valid, "iv_true"].astype(float) - pred.loc[valid]
    model_err = eval_frame.loc[valid, "iv_true"].astype(float) - eval_frame.loc[
        valid, "iv_pred"
    ].astype(float)
    status = (
        pd.Series("ok", index=eval_frame.index)
        if status is None
        else status.reindex(eval_frame.index).fillna("ok")
    )
    status_counts = status.value_counts(dropna=False).to_dict()
    predicted_rows = int(valid.sum())
    eval_rows = int(len(eval_frame))
    row: dict[str, Any] = {
        "baseline": baseline,
        "fit_scope": fit_scope,
        "eval_scope": eval_scope,
        "eval_rows": eval_rows,
        "predicted_rows": predicted_rows,
        "failed_rows": int(eval_rows - predicted_rows),
        "uses_eval_visible_nodes": bool(uses_eval_visible_nodes),
        "visible_context_policy": "visible_nodes_after_mask"
        if uses_eval_visible_nodes
        else "train_only",
        "uses_masked_quote_features": False,
        "iv_mae": float(err.abs().mean()) if predicted_rows else np.nan,
        "iv_rmse": float(np.sqrt((err**2).mean())) if predicted_rows else np.nan,
        "model_delta_mae": (
            float(model_err.abs().mean() - err.abs().mean()) if predicted_rows else np.nan
        ),
        "fit_success_rate": predicted_rows / eval_rows if eval_rows else np.nan,
        "failure_rate": float(1.0 - predicted_rows / eval_rows) if eval_rows else np.nan,
    }
    for key in ("ok", "underidentified", "fit_failed", "timeout", "no_visible_context"):
        row[f"{key}_rate"] = float(status_counts.get(key, 0) / eval_rows) if eval_rows else np.nan
    return row


def _within_surface_baseline_predictions(
    all_eval_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    *,
    include_svi: bool = True,
    svi_timeout_seconds: float = 1.0,
    svi_maxiter: int = 200,
    progress: Any | None = None,
) -> dict[str, dict[str, Any]]:
    visible = _visible_eval_frame(all_eval_frame)
    if {"split", "surface_id"}.issubset(visible.columns) and {"split", "surface_id"}.issubset(
        eval_frame.columns
    ):
        eval_keys = pd.MultiIndex.from_frame(eval_frame[["split", "surface_id"]]).unique()
        visible_keys = pd.MultiIndex.from_frame(visible[["split", "surface_id"]])
        visible = visible.loc[visible_keys.isin(eval_keys)].copy()

    def log(message: str) -> None:
        if progress is not None:
            progress(f"baselines: {message}")

    result: dict[str, dict[str, Any]] = {}
    for name, fn in [
        ("within_surface_knn_moneyness_tenor", _within_surface_knn_baseline),
        ("within_surface_local_linear", _within_surface_local_linear_baseline),
        ("within_surface_rbf", _within_surface_rbf_baseline),
    ]:
        started = time.monotonic()
        log(f"{name} start")
        result[name] = {"pred": fn(visible, eval_frame)}
        log(f"{name} done ({time.monotonic() - started:.1f}s)")
    if include_svi:
        started = time.monotonic()
        log(
            "within_surface_svi_raw_per_expiry start "
            f"(timeout={svi_timeout_seconds}s maxiter={svi_maxiter})"
        )
        result["within_surface_svi_raw_per_expiry"] = _within_surface_svi_raw_baseline(
            visible,
            eval_frame,
            timeout_seconds=svi_timeout_seconds,
            maxiter=svi_maxiter,
        )
        log(f"within_surface_svi_raw_per_expiry done ({time.monotonic() - started:.1f}s)")
    else:
        log("within_surface_svi_raw_per_expiry skipped by baseline_preset=fast")
    return result


def _visible_eval_frame(all_eval_frame: pd.DataFrame) -> pd.DataFrame:
    frame = all_eval_frame.copy()
    if "input_target_visible" in frame:
        visible = frame["input_target_visible"].astype(bool)
    elif "is_masked_target" in frame:
        visible = ~frame["is_masked_target"].astype(bool)
    else:
        visible = pd.Series(True, index=frame.index)
    return frame[visible & frame["iv_true"].notna()].copy()


def _surface_group_key(row: pd.Series) -> tuple[Any, ...]:
    return (row.get("split"), row.get("surface_id"))


def _baseline_features(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            frame["log_moneyness"].fillna(0.0).to_numpy(dtype=float),
            frame["tenor_days"].fillna(0.0).to_numpy(dtype=float) / 365.25,
            (frame["option_type"].astype(str) == "P").to_numpy(dtype=float) * 0.25,
        ]
    )


def _within_surface_knn_baseline(
    visible_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    k: int = 5,
) -> pd.Series:
    from scipy.spatial import cKDTree

    preds: dict[Any, float] = {}
    grouped = {
        key: group for key, group in visible_frame.groupby(["split", "surface_id"], dropna=False)
    }
    for key, target_group in eval_frame.groupby(["split", "surface_id"], dropna=False):
        candidates = grouped.get(key)
        if candidates is None or candidates.empty:
            for idx in target_group.index:
                preds[idx] = np.nan
            continue
        features = _baseline_features(candidates)
        target = _baseline_features(target_group)
        values = candidates["iv_true"].to_numpy(dtype=float)
        query_k = min(k, len(candidates))
        try:
            _, nearest = cKDTree(features).query(target, k=query_k)
        except ValueError:
            for idx in target_group.index:
                preds[idx] = np.nan
            continue
        nearest = np.asarray(nearest)
        if query_k == 1:
            nearest = nearest.reshape(-1, 1)
        pred_values = np.nanmean(values[nearest], axis=1)
        for idx, value in zip(target_group.index, pred_values, strict=False):
            preds[idx] = float(value)
    return pd.Series(preds, index=eval_frame.index)


def _within_surface_local_linear_baseline(
    visible_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
) -> pd.Series:
    preds: dict[Any, float] = {}
    grouped = {
        key: group for key, group in visible_frame.groupby(["split", "surface_id"], dropna=False)
    }
    for key, target_group in eval_frame.groupby(["split", "surface_id"], dropna=False):
        candidates = grouped.get(key)
        if candidates is None or len(candidates) < 4:
            for idx in target_group.index:
                preds[idx] = np.nan
            continue
        x = np.column_stack([np.ones(len(candidates)), _baseline_features(candidates)])
        y = candidates["iv_true"].to_numpy(dtype=float)
        try:
            beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        except np.linalg.LinAlgError:
            for idx in target_group.index:
                preds[idx] = np.nan
            continue
        target_x = np.column_stack([np.ones(len(target_group)), _baseline_features(target_group)])
        pred_values = np.clip(target_x @ beta, 1e-6, 5.0)
        for idx, value in zip(target_group.index, pred_values, strict=False):
            preds[idx] = float(value)
    return pd.Series(preds, index=eval_frame.index)


def _within_surface_rbf_baseline(
    visible_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
) -> pd.Series:
    from scipy.interpolate import RBFInterpolator

    preds: dict[Any, float] = {}
    grouped = {
        key: group for key, group in visible_frame.groupby(["split", "surface_id"], dropna=False)
    }
    for key, target_group in eval_frame.groupby(["split", "surface_id"], dropna=False):
        candidates = grouped.get(key)
        if candidates is None or len(candidates) < 4:
            for idx in target_group.index:
                preds[idx] = np.nan
            continue
        try:
            model = RBFInterpolator(
                _baseline_features(candidates),
                candidates["iv_true"].to_numpy(dtype=float),
                neighbors=min(32, len(candidates)),
                smoothing=1e-5,
                kernel="thin_plate_spline",
            )
            values = model(_baseline_features(target_group))
        except (ValueError, np.linalg.LinAlgError):
            for idx in target_group.index:
                preds[idx] = np.nan
            continue
        for idx, value in zip(target_group.index, values, strict=False):
            preds[idx] = float(np.clip(value, 1e-6, 5.0))
    return pd.Series(preds, index=eval_frame.index)


def _within_surface_svi_raw_baseline(
    visible_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    *,
    timeout_seconds: float = 1.0,
    maxiter: int = 200,
) -> dict[str, Any]:
    preds = pd.Series(np.nan, index=eval_frame.index, dtype=float)
    status = pd.Series("no_visible_context", index=eval_frame.index, dtype=object)
    required = {"split", "surface_id", "expiry", "iv_true", "log_moneyness", "tenor_years"}
    if not required.issubset(set(visible_frame.columns)) or not required.issubset(
        set(eval_frame.columns)
    ):
        return {
            "pred": preds,
            "status": status,
            "metadata": {
                "fit_count": 0,
                "fit_ok_rate": 0.0,
                "fit_timeout_rate": 0.0,
                "fit_underidentified_rate": 0.0,
                "fit_failed_rate": 0.0,
                "fallback_target": "",
            },
        }
    fit_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    fit_counts: dict[str, int] = {}
    visible_groups = {
        key: group
        for key, group in visible_frame.groupby(["split", "surface_id", "expiry"], dropna=False)
    }
    for key, target_group in eval_frame.groupby(["split", "surface_id", "expiry"], dropna=False):
        fit = fit_cache.get(key)
        if fit is None:
            fit = _fit_svi_slice_raw(
                visible_groups.get(key, pd.DataFrame()),
                timeout_seconds=timeout_seconds,
                maxiter=maxiter,
            )
            fit_cache[key] = fit
            fit_counts[str(fit["status"])] = fit_counts.get(str(fit["status"]), 0) + 1
        if fit["status"] != "ok":
            status.loc[target_group.index] = fit["status"]
            continue
        tau = target_group["tenor_years"].astype(float).clip(lower=1e-8).to_numpy()
        k = target_group["log_moneyness"].fillna(0.0).to_numpy(dtype=float)
        total_variance = _svi_total_variance(k, fit["params"])
        valid = np.isfinite(total_variance) & (total_variance > 0) & np.isfinite(tau)
        values = np.full(len(target_group), np.nan, dtype=float)
        values[valid] = np.sqrt(np.maximum(total_variance[valid] / tau[valid], 1e-12))
        preds.loc[target_group.index] = np.clip(values, 1e-6, 5.0)
        status.loc[target_group.index] = np.where(np.isfinite(values), "ok", "fit_failed")
    total_fits = max(sum(fit_counts.values()), 1)
    return {
        "pred": preds,
        "status": status,
        "metadata": {
            "fit_count": int(sum(fit_counts.values())),
            "fit_ok_rate": float(fit_counts.get("ok", 0) / total_fits),
            "fit_timeout_rate": float(fit_counts.get("timeout", 0) / total_fits),
            "fit_underidentified_rate": float(fit_counts.get("underidentified", 0) / total_fits),
            "fit_failed_rate": float(fit_counts.get("fit_failed", 0) / total_fits),
            "fallback_target": "",
        },
    }


def _fit_svi_slice_raw(
    visible_slice: pd.DataFrame,
    *,
    timeout_seconds: float = 1.0,
    maxiter: int = 200,
) -> dict[str, Any]:
    from scipy.optimize import minimize

    if visible_slice.empty:
        return {"status": "no_visible_context", "params": None}
    frame = visible_slice[
        visible_slice["iv_true"].notna()
        & visible_slice["log_moneyness"].notna()
        & visible_slice["tenor_years"].notna()
    ].copy()
    if len(frame) < 5 or frame["log_moneyness"].nunique() < 5:
        return {"status": "underidentified", "params": None}
    tau = float(np.nanmedian(frame["tenor_years"].astype(float)))
    if not np.isfinite(tau) or tau <= 0:
        return {"status": "underidentified", "params": None}
    k = frame["log_moneyness"].to_numpy(dtype=float)
    y = (np.clip(frame["iv_true"].to_numpy(dtype=float), 1e-8, None) ** 2) * tau
    if not np.all(np.isfinite(k)) or not np.all(np.isfinite(y)):
        return {"status": "fit_failed", "params": None}

    k_min = float(np.min(k))
    k_max = float(np.max(k))
    y_med = float(np.median(y))
    y_max = float(np.max(y))
    bounds = [
        (0.0, max(4.0 * y_max, 0.5)),
        (1e-6, 5.0),
        (-0.999, 0.999),
        (k_min - 1.0, k_max + 1.0),
        (1e-4, 2.0),
    ]
    starts = [
        np.array([max(0.5 * y_med, 1e-5), 0.1, 0.0, float(np.median(k)), 0.1]),
        np.array([max(0.25 * y_med, 1e-5), 0.3, -0.3, k_min, 0.2]),
        np.array([max(0.25 * y_med, 1e-5), 0.3, 0.3, k_max, 0.2]),
    ]

    def objective(params: np.ndarray) -> float:
        w = _svi_total_variance(k, params)
        if not np.all(np.isfinite(w)):
            return 1e12
        penalty = float(np.square(np.minimum(w, 0.0)).sum()) * 1e6
        return float(np.mean((w - y) ** 2) + penalty)

    best_params: np.ndarray | None = None
    best_fun = float("inf")
    try:
        with _wall_time_limit(timeout_seconds):
            for start in starts:
                result = minimize(
                    objective,
                    start,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": maxiter, "ftol": 1e-12},
                )
                if result.success and float(result.fun) < best_fun:
                    best_fun = float(result.fun)
                    best_params = np.asarray(result.x, dtype=float)
    except TimeoutError:
        return {"status": "timeout", "params": None}
    if best_params is None or not np.all(np.isfinite(best_params)):
        return {"status": "fit_failed", "params": None}
    fitted = _svi_total_variance(k, best_params)
    if not np.all(np.isfinite(fitted)) or np.any(fitted <= 0):
        return {"status": "fit_failed", "params": None}
    return {"status": "ok", "params": best_params}


def _svi_total_variance(k: np.ndarray, params: np.ndarray | None) -> np.ndarray:
    if params is None:
        return np.full_like(k, np.nan, dtype=float)
    a, b, rho, m, sigma = [float(value) for value in params]
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


@contextmanager
def _wall_time_limit(seconds: float):
    if (
        seconds <= 0
        or threading.current_thread() is not threading.main_thread()
        or not hasattr(signal, "SIGALRM")
    ):
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def _empty_baseline_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "baseline": "train_mean_iv_global",
                "fit_scope": "train_only",
                "eval_scope": "empty",
                "eval_rows": 0,
                "predicted_rows": 0,
                "failed_rows": 0,
                "uses_eval_visible_nodes": False,
                "visible_context_policy": "train_only",
                "uses_masked_quote_features": False,
                "iv_mae": np.nan,
                "iv_rmse": np.nan,
                "model_delta_mae": np.nan,
                "fit_success_rate": np.nan,
                "failure_rate": np.nan,
            }
        ]
    )


def _surface_quote_frame(surfaces: list[Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for surface in surfaces:
        for quote in _surface_target_quotes(surface):
            if quote.implied_vol is None:
                continue
            rows.append(
                {
                    "underlying": quote.underlying,
                    "option_type": quote.option_type,
                    "log_moneyness": float(quote.log_moneyness or 0.0),
                    "tenor_days": quote.tenor_days,
                    "iv_true": float(quote.implied_vol),
                    "bucket": _bucket_key_from_quote(quote),
                }
            )
    return pd.DataFrame(rows)


def _bucket_key_from_quote(quote: OptionQuote) -> str:
    return _bucket_key(float(quote.log_moneyness or 0.0), int(quote.tenor_days), quote.option_type)


def _bucket_key_from_row(row: pd.Series) -> str:
    return _bucket_key(
        float(row.get("log_moneyness", 0.0)),
        int(row.get("tenor_days", 0)),
        row.get("option_type", "C"),
    )


def _bucket_key(log_moneyness: float, tenor_days: int, option_type: Any) -> str:
    money = (
        "atm"
        if abs(log_moneyness) <= 0.03
        else "near_wing"
        if abs(log_moneyness) <= 0.1
        else "far_wing"
    )
    tenor = "short" if tenor_days <= 45 else "medium" if tenor_days <= 180 else "long"
    return f"{money}:{tenor}:{option_type}"


def _train_knn_iv_baseline(
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    k: int = 5,
) -> pd.Series:
    from scipy.spatial import cKDTree

    global_mean = float(train_frame["iv_true"].mean())
    if train_frame.empty:
        return pd.Series(global_mean, index=eval_frame.index)
    train_features = _baseline_feature_matrix(train_frame)
    eval_features = _baseline_feature_matrix(eval_frame)
    values = train_frame["iv_true"].to_numpy(dtype=float)
    query_k = min(k, len(train_features))
    tree = cKDTree(train_features)
    try:
        _, nearest = tree.query(eval_features, k=query_k, workers=-1)
    except TypeError:
        _, nearest = tree.query(eval_features, k=query_k)
    nearest = np.asarray(nearest)
    if query_k == 1:
        nearest = nearest.reshape(-1, 1)
    preds = np.nanmean(values[nearest], axis=1)
    return pd.Series(np.where(np.isfinite(preds), preds, global_mean), index=eval_frame.index)


def _baseline_feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            frame["log_moneyness"].fillna(0.0).to_numpy(dtype=float),
            frame["tenor_days"].fillna(0.0).to_numpy(dtype=float) / 365.25,
            (frame["option_type"].astype(str) == "P").to_numpy(dtype=float) * 0.25,
        ]
    )


def _baseline_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    return _diagnostic_leakage_prone_baseline_summary(predictions)


def _diagnostic_leakage_prone_baseline_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            [
                {
                    "baseline": "mean_iv_global",
                    "fit_scope": "diagnostic_leakage_prone",
                    "eval_scope": "all_nodes",
                    "eval_rows": 0,
                    "iv_mae": np.nan,
                    "iv_rmse": np.nan,
                    "model_delta_mae": np.nan,
                }
            ]
        )

    frame = predictions.reset_index(drop=True).copy()
    model_mae = float((frame["iv_true"] - frame["iv_pred"]).abs().mean())

    baseline_preds = {
        "mean_iv_global": _leave_one_out_group_mean(frame, []),
        "mean_iv_by_underlying": _leave_one_out_group_mean(frame, ["underlying"]),
        "mean_iv_by_surface": _leave_one_out_group_mean(frame, ["split", "surface_id"]),
        "knn_moneyness_tenor": _knn_iv_baseline(frame),
    }
    rows = []
    for name, pred in baseline_preds.items():
        err = frame["iv_true"] - pred
        baseline_mae = float(err.abs().mean())
        rows.append(
            {
                "baseline": name,
                "fit_scope": "diagnostic_leakage_prone",
                "eval_scope": "all_nodes",
                "eval_rows": int(len(frame)),
                "iv_mae": baseline_mae,
                "iv_rmse": float(np.sqrt((err**2).mean())),
                "model_delta_mae": model_mae - baseline_mae,
            }
        )
    return pd.DataFrame(rows)


def _leave_one_out_group_mean(frame: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    values = frame["iv_true"].astype(float)
    global_mean = float(values.mean())
    if not group_cols:
        if len(values) <= 1:
            return pd.Series([global_mean] * len(frame), index=frame.index)
        total = float(values.sum())
        return (total - values) / (len(values) - 1)

    grouped = frame.groupby(group_cols, dropna=False)["iv_true"]
    group_sum = grouped.transform("sum")
    group_count = grouped.transform("count")
    pred = (group_sum - values) / (group_count - 1).replace(0, np.nan)
    return pred.fillna(global_mean)


def _knn_iv_baseline(frame: pd.DataFrame, k: int = 5) -> pd.Series:
    from scipy.spatial import cKDTree

    if len(frame) <= 1:
        return pd.Series(frame["iv_true"].astype(float).mean(), index=frame.index)
    features = np.column_stack(
        [
            frame["log_moneyness"].fillna(0.0).to_numpy(dtype=float),
            frame["tenor_days"].fillna(0.0).to_numpy(dtype=float) / 365.25,
            (frame["option_type"].astype(str) == "P").to_numpy(dtype=float) * 0.25,
        ]
    )
    values = frame["iv_true"].to_numpy(dtype=float)
    global_mean = float(np.nanmean(values))
    pred_values = np.full(len(frame), global_mean, dtype=float)
    row_positions = {index: position for position, index in enumerate(frame.index)}
    for _, group in frame.groupby("split", dropna=False, sort=False):
        positions = np.asarray([row_positions[index] for index in group.index], dtype=int)
        if len(positions) <= 1:
            continue
        group_features = features[positions]
        query_k = min(k + 1, len(positions))
        try:
            _, nearest_local = cKDTree(group_features).query(group_features, k=query_k, workers=-1)
        except TypeError:
            _, nearest_local = cKDTree(group_features).query(group_features, k=query_k)
        nearest_local = np.asarray(nearest_local)
        if query_k == 1:
            nearest_local = nearest_local.reshape(-1, 1)
        for local_pos, nearest in enumerate(nearest_local):
            neighbor_local = np.asarray(nearest, dtype=int)
            neighbor_local = neighbor_local[neighbor_local != local_pos][:k]
            if len(neighbor_local):
                pred_values[positions[local_pos]] = float(
                    np.nanmean(values[positions[neighbor_local]])
                )
    return pd.Series(pred_values, index=frame.index)


def _price_diagnostics(predictions: pd.DataFrame) -> dict[str, Any]:
    if predictions.empty:
        return {"count": 0}
    rows = []
    for _, row in predictions.iterrows():
        reference = _reference_price(row)
        mid = row.get("mid_price", np.nan)
        tenor = float(row.get("tenor_years", 0.0) or 0.0)
        if not np.isfinite(reference) or reference <= 0 or not np.isfinite(mid):
            continue
        pred_norm = (
            _black_forward_price(
                option_type=str(row["option_type"]),
                forward=reference,
                strike=float(row["strike"]),
                tenor_years=tenor,
                sigma=max(float(row["iv_pred"]), 1e-8),
            )
            / reference
        )
        true_norm = (
            _black_forward_price(
                option_type=str(row["option_type"]),
                forward=reference,
                strike=float(row["strike"]),
                tenor_years=tenor,
                sigma=max(float(row["iv_true"]), 1e-8),
            )
            / reference
        )
        mid_norm = float(mid) / reference
        rows.append((pred_norm, true_norm, mid_norm))
    if not rows:
        return {"count": 0}
    arr = np.asarray(rows, dtype=float)
    pred_error = arr[:, 0] - arr[:, 2]
    true_error = arr[:, 1] - arr[:, 2]
    return {
        "count": int(len(arr)),
        "pred_mid_norm_mae": float(np.abs(pred_error).mean()),
        "pred_mid_norm_rmse": float(np.sqrt((pred_error**2).mean())),
        "true_mid_norm_mae": float(np.abs(true_error).mean()),
        "true_mid_norm_rmse": float(np.sqrt((true_error**2).mean())),
    }


def _no_arbitrage_diagnostics(
    predictions: pd.DataFrame,
    *,
    mode: str = "full",
    eval_splits: tuple[str, ...] | list[str] | None = None,
    max_surfaces_per_split: int | None = None,
    sample_seed: int = 0,
) -> dict[str, Any]:
    mode = str(mode).lower()
    if mode not in {"none", "full", "sampled_surface"}:
        msg = "no-arb diagnostics mode must be one of: none, full, sampled_surface"
        raise ValueError(msg)
    diagnostic_frame, metadata = _select_no_arb_diagnostic_frame(
        predictions,
        mode=mode,
        eval_splits=eval_splits,
        max_surfaces_per_split=max_surfaces_per_split,
        sample_seed=sample_seed,
    )
    if diagnostic_frame.empty:
        return {
            "metadata": metadata,
            "true_iv": _empty_no_arb_payload(),
            "pred_iv": _empty_no_arb_payload(),
        }
    return {
        "metadata": metadata,
        "true_iv": _no_arbitrage_for_iv_column(diagnostic_frame, "iv_true"),
        "pred_iv": _no_arbitrage_for_iv_column(diagnostic_frame, "iv_pred"),
    }


def _select_no_arb_diagnostic_frame(
    predictions: pd.DataFrame,
    *,
    mode: str,
    eval_splits: tuple[str, ...] | list[str] | None,
    max_surfaces_per_split: int | None,
    sample_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = predictions.reset_index(drop=True).copy()
    metadata: dict[str, Any] = {
        "mode": mode,
        "sampling_unit": "surface_id",
        "sample_seed": int(sample_seed),
        "source_row_count": int(len(frame)),
        "source_surface_count": int(frame["surface_id"].nunique()) if "surface_id" in frame else 0,
        "eval_splits": list(eval_splits or ()),
        "max_surfaces_per_split": max_surfaces_per_split,
    }
    if frame.empty or mode == "none":
        metadata.update(
            {
                "row_count": 0,
                "surface_count": 0,
                "surface_share": 0.0,
                "rows_by_split": {},
                "surfaces_by_split": {},
            }
        )
        return frame.iloc[0:0].copy(), metadata
    if eval_splits and "split" in frame:
        split_set = {str(split) for split in eval_splits}
        frame = frame[frame["split"].astype(str).isin(split_set)].copy()
    if mode == "sampled_surface" and max_surfaces_per_split is not None:
        frame = _sample_complete_surfaces(
            frame,
            max_surfaces_per_split=max_surfaces_per_split,
            sample_seed=sample_seed,
        )
    surface_count = int(frame["surface_id"].nunique()) if "surface_id" in frame else 0
    source_surface_count = max(int(metadata["source_surface_count"]), 1)
    metadata.update(
        {
            "row_count": int(len(frame)),
            "surface_count": surface_count,
            "surface_share": float(surface_count / source_surface_count),
            "rows_by_split": (
                frame.groupby("split").size().astype(int).to_dict()
                if "split" in frame and not frame.empty
                else {}
            ),
            "surfaces_by_split": (
                frame.groupby("split")["surface_id"].nunique().astype(int).to_dict()
                if {"split", "surface_id"}.issubset(frame.columns) and not frame.empty
                else {}
            ),
        }
    )
    return frame, metadata


def _sample_complete_surfaces(
    frame: pd.DataFrame,
    *,
    max_surfaces_per_split: int,
    sample_seed: int,
) -> pd.DataFrame:
    if frame.empty or max_surfaces_per_split <= 0:
        return frame.iloc[0:0].copy()
    if not {"split", "surface_id"}.issubset(frame.columns):
        return frame
    selected_keys: set[tuple[str, str]] = set()
    for split_name, split_frame in frame.groupby("split", dropna=False, sort=True):
        surface_ids = sorted(str(value) for value in split_frame["surface_id"].dropna().unique())
        if len(surface_ids) > max_surfaces_per_split:
            surface_ids = sorted(
                surface_ids,
                key=lambda surface_id: _stable_sample_key(str(split_name), surface_id, sample_seed),
            )[:max_surfaces_per_split]
        selected_keys.update((str(split_name), surface_id) for surface_id in surface_ids)
    keys = list(zip(frame["split"].astype(str), frame["surface_id"].astype(str), strict=False))
    keep = pd.Series([key in selected_keys for key in keys], index=frame.index)
    return frame[keep].copy()


def _stable_sample_key(split_name: str, surface_id: str, sample_seed: int) -> str:
    payload = f"{sample_seed}|{split_name}|{surface_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def _no_arbitrage_for_iv_column(predictions: pd.DataFrame, iv_col: str) -> dict[str, Any]:
    frame = predictions.copy()
    frame["reference"] = frame.apply(_reference_price, axis=1)
    valid = frame[(frame["reference"] > 0) & frame[iv_col].notna()].copy()
    if valid.empty:
        return _empty_no_arb_payload()
    valid["total_variance"] = valid[iv_col].clip(lower=1e-8) ** 2 * valid["tenor_years"].clip(
        lower=1e-8
    )
    valid["norm_price"] = valid.apply(
        lambda row: (
            _black_forward_price(
                option_type=str(row["option_type"]),
                forward=float(row["reference"]),
                strike=float(row["strike"]),
                tenor_years=float(row["tenor_years"]),
                sigma=max(float(row[iv_col]), 1e-8),
            )
            / float(row["reference"])
        ),
        axis=1,
    )
    return {
        "calendar": _calendar_diagnostics(valid),
        "butterfly_convexity": _convexity_diagnostics(valid),
        "vertical_spread": _vertical_spread_diagnostics(valid),
        "put_call_parity": _put_call_diagnostics(valid),
    }


def _empty_no_arb_payload() -> dict[str, Any]:
    empty = {"pairs": 0, "violations": 0, "mean_violation": 0.0, "max_violation": 0.0}
    return {
        "calendar": empty,
        "butterfly_convexity": {"triples": 0, "violations": 0, "mean_violation": 0.0},
        "vertical_spread": empty,
        "put_call_parity": {"pairs": 0, "mean_abs_residual": 0.0, "p95_abs_residual": 0.0},
    }


def _calendar_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    magnitudes: list[float] = []
    pairs = 0
    group_cols = ["split", "surface_id", "option_type", "strike"]
    for _, group in frame.groupby(group_cols, dropna=False):
        ordered = group.sort_values("tenor_years")
        if len(ordered) < 2:
            continue
        values = ordered["total_variance"].to_numpy(dtype=float)
        for near, far in zip(values[:-1], values[1:], strict=False):
            pairs += 1
            magnitudes.append(max(float(near - far), 0.0))
    positives = [value for value in magnitudes if value > 0]
    return {
        "pairs": pairs,
        "violations": len(positives),
        "mean_violation": float(np.mean(positives)) if positives else 0.0,
        "max_violation": float(np.max(positives)) if positives else 0.0,
    }


def _convexity_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    magnitudes: list[float] = []
    triples = 0
    group_cols = ["split", "surface_id", "expiry", "option_type"]
    for _, group in frame.groupby(group_cols, dropna=False):
        ordered = group.sort_values("strike")
        if len(ordered) < 3:
            continue
        strikes = ordered["strike"].to_numpy(dtype=float)
        prices = ordered["norm_price"].to_numpy(dtype=float)
        for i in range(1, len(ordered) - 1):
            k1, k2, k3 = strikes[i - 1], strikes[i], strikes[i + 1]
            if k3 <= k1:
                continue
            p1, p2, p3 = prices[i - 1], prices[i], prices[i + 1]
            alpha = (k3 - k2) / (k3 - k1)
            beta = (k2 - k1) / (k3 - k1)
            interpolated = alpha * p1 + beta * p3
            triples += 1
            magnitudes.append(max(float(p2 - interpolated), 0.0))
    positives = [value for value in magnitudes if value > 0]
    return {
        "triples": triples,
        "violations": len(positives),
        "mean_violation": float(np.mean(positives)) if positives else 0.0,
    }


def _vertical_spread_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    """Europeanized call vertical-spread monotonicity diagnostic.

    Uses normalized Black-forward call prices C/F and normalized strikes K/F, so
    the no-arbitrage slope interval is [-1, 0] on European call surfaces.
    """

    magnitudes: list[float] = []
    pairs = 0
    group_cols = ["split", "surface_id", "expiry"]
    calls = frame[frame["option_type"] == "C"].copy()
    if calls.empty:
        return {"pairs": 0, "violations": 0, "mean_violation": 0.0, "max_violation": 0.0}
    calls["norm_strike"] = calls["strike"] / calls["reference"].replace(0, np.nan)
    calls = calls[calls["norm_strike"].notna() & np.isfinite(calls["norm_strike"])]
    for _, group in calls.groupby(group_cols, dropna=False):
        ordered = group.sort_values("norm_strike")
        if len(ordered) < 2:
            continue
        strikes = ordered["norm_strike"].to_numpy(dtype=float)
        prices = ordered["norm_price"].to_numpy(dtype=float)
        for left_k, right_k, left_price, right_price in zip(
            strikes[:-1],
            strikes[1:],
            prices[:-1],
            prices[1:],
            strict=False,
        ):
            if right_k <= left_k:
                continue
            slope = (right_price - left_price) / (right_k - left_k)
            pairs += 1
            magnitudes.append(max(float(slope), 0.0) + max(float(-1.0 - slope), 0.0))
    positives = [value for value in magnitudes if value > 0]
    return {
        "pairs": pairs,
        "violations": len(positives),
        "mean_violation": float(np.mean(positives)) if positives else 0.0,
        "max_violation": float(np.max(positives)) if positives else 0.0,
    }


def _put_call_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    residuals: list[float] = []
    group_cols = ["split", "surface_id", "expiry", "strike"]
    for _, group in frame.groupby(group_cols, dropna=False):
        calls = group[group["option_type"] == "C"]
        puts = group[group["option_type"] == "P"]
        if calls.empty or puts.empty:
            continue
        call = calls.iloc[0]
        put = puts.iloc[0]
        reference = float(np.nanmean([call["reference"], put["reference"]]))
        if not np.isfinite(reference) or reference <= 0:
            continue
        strike = float(call["strike"])
        parity_rhs = (reference - strike) / reference
        residuals.append(abs(float(call["norm_price"] - put["norm_price"] - parity_rhs)))
    return {
        "pairs": len(residuals),
        "mean_abs_residual": float(np.mean(residuals)) if residuals else 0.0,
        "p95_abs_residual": float(np.quantile(residuals, 0.95)) if residuals else 0.0,
    }


def _reference_price(row: pd.Series) -> float:
    forward = float(row.get("forward", np.nan))
    if np.isfinite(forward) and forward > 0:
        return forward
    spot = float(row.get("underlying_price", np.nan))
    if np.isfinite(spot) and spot > 0:
        return spot
    return np.nan


def _black_forward_price(
    *,
    option_type: str,
    forward: float,
    strike: float,
    tenor_years: float,
    sigma: float,
) -> float:
    if tenor_years <= 0 or sigma <= 0:
        intrinsic = forward - strike if option_type == "C" else strike - forward
        return max(intrinsic, 0.0)
    vol_sqrt_t = sigma * np.sqrt(tenor_years)
    if vol_sqrt_t <= 0:
        intrinsic = forward - strike if option_type == "C" else strike - forward
        return max(intrinsic, 0.0)
    d1 = (np.log(forward / strike) + 0.5 * sigma * sigma * tenor_years) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    if option_type == "C":
        return float(forward * _normal_cdf(d1) - strike * _normal_cdf(d2))
    return float(strike * _normal_cdf(-d2) - forward * _normal_cdf(-d1))


def _normal_cdf(value: float) -> float:
    return float(0.5 * (1.0 + math.erf(value / math.sqrt(2.0))))


def _write_run_readme(output_dir: Path, metrics_summary: dict[str, Any], run_label: str) -> None:
    text = "\n".join(
        [
            "# LoG-IV Run",
            "",
            f"Label: {run_label}",
            "",
            "This run is not paper evidence unless promoted in docs/results_snapshot.md.",
            "",
            "Key metrics:",
            "",
            f"- final_val_loss: {metrics_summary.get('final_val_loss')}",
            f"- final_test_loss: {metrics_summary.get('final_test_loss')}",
            f"- iv_mae: {metrics_summary.get('iv_mae', 'NA')}",
            f"- iv_rmse: {metrics_summary.get('iv_rmse', 'NA')}",
            "",
        ]
    )
    (output_dir / "README.md").write_text(text)
