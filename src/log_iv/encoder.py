"""Option-token encoder: maps irregular option quotes to latent representations.

This module implements the OptionTokenEncoder that converts each option quote
(node) into a fixed-dimensional embedding, combining:
- Geometric features (log-moneyness, tenor, call/put flag)
- Price/IV features (mid IV, spread, bid/ask, observed-target flag)
- Liquidity marks (volume, OI, spread_pct, liquidity_score)

The encoder is designed to be the first stage of an OptionTokenModel that
can be used for reconstruction, forecasting, and cross-market transfer.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class OptionTokenEncoder(nn.Module):
    """Encodes option quotes into latent token embeddings.

    Each quote produces a d_model-dimensional token that captures its
    geometric, pricing, and liquidity characteristics.

    Architecture: MLP with residual connections and layer norm.

    Parameters
    ----------
    d_geom : int
        Number of geometric input features (log-moneyness, tenor, cp flag).
    d_price : int
        Number of price/IV features (mid IV, spread, etc.).
    d_liquidity : int
        Number of liquidity mark features (volume, OI, spread_pct, etc.).
    d_model : int
        Output embedding dimension.
    n_layers : int
        Number of MLP layers.
    dropout : float
        Dropout rate for training.
    use_fourier : bool
        Whether to use Fourier feature encoding for periodic/time features.
    n_fourier : int
        Number of Fourier basis functions if use_fourier is True.
    """

    def __init__(
        self,
        d_geom: int = 3,
        d_price: int = 4,
        d_liquidity: int = 4,
        d_model: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1,
        use_fourier: bool = True,
        n_fourier: int = 16,
    ):
        super().__init__()
        self.d_geom = d_geom
        self.d_price = d_price
        self.d_liquidity = d_liquidity
        self.d_model = d_model
        self.use_fourier = use_fourier

        # Fourier feature encoding for periodic/time structure
        if use_fourier:
            self.fourier_geom = FourierFeatures(d_geom, n_fourier)
            d_geom_encoded = d_geom + 2 * n_fourier
        else:
            self.fourier_geom = None
            d_geom_encoded = d_geom

        d_input = d_geom_encoded + d_price + d_liquidity

        # MLP with residual blocks
        layers: list[nn.Module] = []
        in_dim = d_input
        for i in range(n_layers):
            out_dim = d_model if i == n_layers - 1 else d_model
            layers.append(
                ResidualMLPBlock(
                    in_features=in_dim,
                    hidden_features=d_model * 2,
                    out_features=out_dim,
                    dropout=dropout,
                )
            )
            in_dim = d_model
        self.mlp = nn.Sequential(*layers)

        # Final layer norm
        self.layer_norm = nn.LayerNorm(d_model)

        # Per-category projection heads (for reconstruction losses)
        self.geom_head = nn.Linear(d_model, d_geom)
        self.iv_head = nn.Linear(d_model, 1)  # predict mid IV
        self.liquidity_head = nn.Linear(d_model, d_liquidity)

        # Delta/vega prediction head (for no-arbitrage checks)
        self.greeks_head = nn.Linear(d_model, 2)  # delta, vega

    def forward(
        self,
        geom: torch.Tensor,
        price_features: torch.Tensor,
        liquidity_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encode option quotes into latent embeddings.

        Parameters
        ----------
        geom : torch.Tensor
            Geometry features, shape (N, d_geom): [log_moneyness, tenor_years, cp_flag]
        price_features : torch.Tensor
            Price features, shape (N, d_price): [mid_iv, spread_pct, bid_ask_ratio,
            observed_target_flag]
        liquidity_features : torch.Tensor
            Liquidity features, shape (N, d_liquidity): [volume, OI, spread_pct, liq_score]

        Returns
        -------
        dict[str, torch.Tensor]
            - 'h': latent embeddings (N, d_model)
            - 'geom_recon': reconstructed geometry (N, d_geom)
            - 'iv_pred': predicted mid IV (N, 1)
            - 'liq_recon': reconstructed liquidity features (N, d_liquidity)
            - 'greeks_pred': predicted delta, vega (N, 2)
        """
        # Apply Fourier encoding to geometric features if enabled
        if self.use_fourier and self.fourier_geom is not None:
            geom_encoded = self.fourier_geom(geom)
        else:
            geom_encoded = geom

        # Concatenate all features
        x = torch.cat([geom_encoded, price_features, liquidity_features], dim=-1)

        # MLP encoding
        h_raw = self.mlp(x)
        h = self.layer_norm(h_raw)

        # Reconstruction heads
        geom_recon = self.geom_head(h)
        iv_pred = self.iv_head(h)
        liq_recon = self.liquidity_head(h)
        greeks_pred = self.greeks_head(h)

        return {
            "h": h,
            "geom_recon": geom_recon,
            "iv_pred": iv_pred,
            "liq_recon": liq_recon,
            "greeks_pred": greeks_pred,
        }


class FourierFeatures(nn.Module):
    """Fourier feature encoding for capturing periodic structure.

    Maps x ∈ R^d to [x, sin(2π·B₁·x), cos(2π·B₁·x), ..., sin(2π·Bₖ·x), cos(2π·Bₖ·x)]
    where B_i are learnable or fixed frequency bases.
    """

    def __init__(self, d_input: int, n_frequencies: int, learnable: bool = True):
        super().__init__()
        self.d_input = d_input
        self.n_frequencies = n_frequencies

        # Initialize frequency basis (B matrix)
        if learnable:
            self.B = nn.Parameter(torch.randn(d_input, n_frequencies) * 0.1)
        else:
            # Fixed log-spaced frequencies
            freqs = torch.logspace(-2, 2, n_frequencies)
            B = torch.zeros(d_input, n_frequencies)
            for i in range(d_input):
                B[i] = freqs * (i + 1)
            self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier encoding.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (N, d_input)

        Returns
        -------
        torch.Tensor
            Encoded features, shape (N, d_input + 2 * n_frequencies)
        """
        proj = 2.0 * math.pi * torch.matmul(x, self.B)  # (N, n_freq)
        sin_features = torch.sin(proj)
        cos_features = torch.cos(proj)
        return torch.cat([x, sin_features, cos_features], dim=-1)


class ResidualMLPBlock(nn.Module):
    """MLP block with residual connection, layer norm, and dropout."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.layer_norm1 = nn.LayerNorm(in_features)
        self.layer_norm2 = nn.LayerNorm(hidden_features)
        self.dropout = nn.Dropout(dropout)

        # Residual projection if dimensions don't match
        self.residual = (
            nn.Linear(in_features, out_features, bias=False)
            if in_features != out_features
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.layer_norm1(x)
        h = F.gelu(self.fc1(x))
        h = self.layer_norm2(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return h + residual


def extract_features_from_quotes(
    quotes: list,
    device: torch.device | None = None,
    target_observed: list[bool] | tuple[bool, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract geometry, price, and liquidity feature tensors from OptionQuotes.

    Parameters
    ----------
    quotes : list[OptionQuote]
        List of OptionQuote objects.
    device : torch.device, optional
        Device to place tensors on.
    target_observed : sequence of bool, optional
        Per-node indicator that the target IV/price is visible to the model.
        Masked reconstruction sets this to ``False`` for held-out tokens.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        (geom, price, liquidity) each of shape (N, d_*)
    """
    geom_list: list[list[float]] = []
    price_list: list[list[float]] = []
    liq_list: list[list[float]] = []

    observed_flags = target_observed or [True] * len(quotes)
    for q, observed in zip(quotes, observed_flags, strict=True):
        # Geometry: log_moneyness, tenor_years, cp_flag (1 for call, 0 for put)
        lm = q.log_moneyness if q.log_moneyness is not None else 0.0
        ty = q.tenor_years
        cp = 1.0 if q.option_type == "C" else 0.0
        geom_list.append([lm, ty, cp])

        # Price: mid IV, spread (normalized), bid-ask ratio, target visibility flag.
        iv = q.implied_vol if q.implied_vol is not None else 0.0
        spread = q.spread if q.spread is not None else 0.0
        mid = q.mid if q.mid is not None else 0.0
        spread_norm = spread / (mid + 1e-6) if mid > 0 else 0.0
        price_list.append([iv, spread_norm, min(spread_norm, 1.0), 1.0 if observed else 0.0])

        # Liquidity: volume, OI, spread_pct, liquidity_score
        vol_log = math.log1p(q.volume)
        oi_log = math.log1p(q.open_interest)
        spread_ref = spread / (mid + 1e-6) if mid > 0 else 0.0
        liq_score = vol_log + oi_log - math.log1p(spread_ref + 1e-6)
        liq_list.append([vol_log, oi_log, spread_ref, liq_score])

    device = device or torch.device("cpu")
    geom_t = torch.tensor(geom_list, dtype=torch.float32, device=device)
    price_t = torch.tensor(price_list, dtype=torch.float32, device=device)
    liq_t = torch.tensor(liq_list, dtype=torch.float32, device=device)

    return geom_t, price_t, liq_t
