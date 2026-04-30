"""Synthetic arbitrage-free option surface generator for LoG-IV benchmark.

Generates SVI/SSVI surfaces with configurable:
- Grid density and moneyness/maturity ranges
- Liquidity-correlated subsampling (spread, volume, OI)
- Heteroscedastic noise injection
- Stale quote simulation
- Missing wing/maturity corruption

Output format matches the research plan specification:
  u_j = (k_j, τ_j, cp_j)       — geometry (log-moneyness, tenor, call/put)
  y_j = mid IV                  — observation
  m_j = (spread, volume, OI)   — liquidity marks
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from log_iv.schema import OptionQuote

# ---------------------------------------------------------------------------
# SVI parameterization (Gatheral 2004)
# ---------------------------------------------------------------------------


@dataclass
class SVISliceParams:
    """Raw SVI parameters for a single maturity slice (total variance w = σ² τ).

    Parameters
    ----------
    a : float
        Overall level of total variance
    b : float
        Angle of left/right asymptotes, controls smile amplitude
    rho : float
        Correlation/skew parameter in (-1, 1)
    m : float
        Horizontal shift of the smile
    sigma : float
        Smoothness of the smile (> 0)
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate raw SVI total variance at log-moneyness k."""
        return self.a + self.b * (
            self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma**2)
        )

    def implied_vol(self, k: NDArray[np.float64], tau: float) -> NDArray[np.float64]:
        """Evaluate implied volatility at log-moneyness k and tenor tau."""
        w = self.total_variance(k)
        w = np.maximum(w, 1e-12)
        return np.sqrt(w / max(tau, 1e-8))

    def is_arbitrage_free(self) -> bool:
        """Check sufficient conditions for no butterfly arbitrage."""
        cond1 = self.b * (1.0 + abs(self.rho)) <= 2.0
        return cond1 and self.sigma > 1e-6


# ---------------------------------------------------------------------------
# SSVI surface parameterization (Gatheral & Jacquier 2014)
# ---------------------------------------------------------------------------


@dataclass
class SSVISurface:
    """Surface-level SSVI parameterization.

    Uses the power-law form φ(τ) = η τ^{-λ} for ATM variance term structure.

    Parameters
    ----------
    eta : float
        η > 0, controls ATM total variance level at τ = 1
    lam : float
        λ in [0, 1], power-law exponent for term structure
    rho_surface : float
        Global correlation parameter in (-1, 1)
    eta_surface : float
        Controls curvature of the smile
    """

    eta: float = 0.04
    lam: float = 0.5
    rho_surface: float = -0.3
    eta_surface: float = 1.0

    def __post_init__(self) -> None:
        if not (-1.0 < self.rho_surface < 1.0):
            raise ValueError("rho_surface must be in (-1, 1)")
        if not (0.0 <= self.lam <= 1.0):
            raise ValueError("lam must be in [0, 1]")
        if self.eta <= 0:
            raise ValueError("eta must be > 0")
        if self.eta_surface <= 0:
            raise ValueError("eta_surface must be > 0")

    def atm_total_variance(self, tau: float | NDArray) -> NDArray:
        """ATM total variance φ(τ) = η τ^{1-λ}."""
        return self.eta * np.power(np.maximum(tau, 1e-8), 1.0 - self.lam)

    def slice_params(self, tau: float) -> SVISliceParams:
        """Build SVI slice params from SSVI surface at tenor tau."""
        theta = self.atm_total_variance(np.array(tau)).item()

        rho = self.rho_surface
        d = np.sqrt(np.maximum(1.0 - rho**2, 0.0))

        b = theta / (self.eta_surface * np.sqrt(np.maximum(tau, 1e-8)) + 1e-8)
        b = np.clip(b, 0.001, 2.0 / (1.0 + abs(rho)))

        sigma = d * np.sqrt(np.maximum(tau, 1e-8))
        a = theta - b * d * sigma

        return SVISliceParams(a=a, b=b, rho=rho, m=0.0, sigma=sigma)

    def implied_vol_surface(self, k_grid: NDArray, tau_grid: NDArray) -> NDArray:
        """Evaluate full IV surface over k and tau grids.

        Returns array of shape (len(tau), len(k)).
        """
        iv_surface = np.zeros((len(tau_grid), len(k_grid)), dtype=np.float64)
        for i, tau in enumerate(tau_grid):
            sl = self.slice_params(tau)
            iv_surface[i] = sl.implied_vol(k_grid, tau)
        return iv_surface

    def is_arbitrage_free(self, tau_grid: NDArray) -> bool:
        """Check no-arbitrage conditions across all maturities."""
        for tau in tau_grid:
            sl = self.slice_params(tau)
            if not sl.is_arbitrage_free():
                return False
        return True


# ---------------------------------------------------------------------------
# Synthetic option quote generation
# ---------------------------------------------------------------------------


class SyntheticQuote(NamedTuple):
    """A single synthetic option quote with all features."""

    market: str
    underlying: str
    observation_date: date
    expiry: date
    strike: float
    option_type: str  # "C" or "P"
    bid: float
    ask: float
    implied_vol: float
    volume: int
    open_interest: int
    forward: float
    underlying_price: float
    log_moneyness: float
    tenor_years: float
    delta: float
    vega: float
    spread_pct: float
    liquidity_score: float


@dataclass
class SyntheticSurfaceConfig:
    """Configuration for synthetic surface generation."""

    # Grid
    n_maturities: int = 12
    n_strikes: int = 21
    min_tenor_days: int = 7
    max_tenor_days: int = 730
    moneyness_range: float = 0.4  # ±40% log-moneyness

    # SSVI parameters
    eta: float = 0.04
    lam: float = 0.5
    rho_surface: float = -0.3
    eta_surface: float = 1.0

    # Underlying
    spot: float = 500.0
    risk_free_rate: float = 0.04
    dividend_yield: float = 0.015

    # Noise
    bid_ask_spread_pct: float = 0.02  # average relative spread
    vol_noise_scale: float = 0.002  # IV noise std (additive)
    vol_noise_liquidity_corr: float = 0.5  # correlation of noise with illiquidity
    volume_base: int = 100
    volume_range: tuple[int, int] = (0, 10000)
    oi_base: int = 500
    oi_range: tuple[int, int] = (0, 50000)

    # Corruption
    missing_wing_prob: float = 0.05  # prob of missing deep OTM quotes
    stale_quote_prob: float = 0.02
    stale_tenor_shift_days: int = 1

    # Seed
    random_seed: int = 42

    # Market tags
    market: str = "US"
    underlying: str = "SPY"


def generate_synthetic_surface(
    config: SyntheticSurfaceConfig | None = None,
    observation_date: date | None = None,
) -> list[SyntheticQuote]:
    """Generate a synthetic option surface with realistic microstructure noise.

    Parameters
    ----------
    config : SyntheticSurfaceConfig, optional
        Configuration for surface generation.
    observation_date : date, optional
        Observation date, defaults to today.

    Returns
    -------
    list[SyntheticQuote]
        List of synthetic option quotes.
    """
    cfg = config or SyntheticSurfaceConfig()
    rng = np.random.default_rng(cfg.random_seed)

    if observation_date is None:
        observation_date = date.today()

    # Build SSVI surface
    surface = SSVISurface(
        eta=cfg.eta,
        lam=cfg.lam,
        rho_surface=cfg.rho_surface,
        eta_surface=cfg.eta_surface,
    )

    # Log-moneyness grid
    k_grid = np.linspace(-cfg.moneyness_range, cfg.moneyness_range, cfg.n_strikes)

    # Tenor grid (in years)
    tenor_days_grid = np.linspace(cfg.min_tenor_days, cfg.max_tenor_days, cfg.n_maturities)
    tenor_years_grid = tenor_days_grid / 365.25

    # Forward price for each maturity: F = S * exp((r - q) * τ)
    forward_grid = cfg.spot * np.exp((cfg.risk_free_rate - cfg.dividend_yield) * tenor_years_grid)

    # Implied volatility surface
    iv_surface = surface.implied_vol_surface(k_grid, tenor_years_grid)

    quotes: list[SyntheticQuote] = []

    for i, tau in enumerate(tenor_years_grid):
        expiry = observation_date + timedelta(days=int(tenor_days_grid[i]))
        F = forward_grid[i]
        # Strike from log-moneyness: K = F * exp(k)
        strikes = F * np.exp(k_grid)

        for j, (k_val, K) in enumerate(zip(k_grid, strikes, strict=False)):
            iv_clean = iv_surface[i, j]

            # Generate liquidity marks
            moneyness_abs = abs(k_val)
            tenor_factor = np.sqrt(max(tau, 0.001))

            # Spread: wider for OTM and short-dated
            spread_pct = cfg.bid_ask_spread_pct * (1.0 + 2.0 * moneyness_abs + 1.0 / tenor_factor)
            spread_pct *= rng.lognormal(0.0, 0.3)

            # Volume: lower for deep OTM, varies with tenor
            volume_mean = cfg.volume_base * np.exp(-3.0 * moneyness_abs) * tenor_factor
            volume = max(0, int(rng.poisson(max(volume_mean, 0.5))))

            # Open interest: similar pattern
            oi_mean = cfg.oi_base * np.exp(-3.0 * moneyness_abs) * tenor_factor
            open_interest = max(0, int(rng.poisson(max(oi_mean, 0.5))))

            # Apply missing-wing corruption
            if moneyness_abs > 0.3 and rng.random() < cfg.missing_wing_prob:
                continue  # skip this quote

            # Apply stale quote corruption
            actual_observation = observation_date
            if rng.random() < cfg.stale_quote_prob:
                actual_observation = observation_date - timedelta(days=cfg.stale_tenor_shift_days)

            # Liquidity-correlated noise on IV
            liquidity_normalized = np.log1p(max(volume, 0)) / np.log1p(cfg.volume_range[1])
            noise_scale = cfg.vol_noise_scale * (
                1.0 + cfg.vol_noise_liquidity_corr * (1.0 - liquidity_normalized)
            )
            iv_noisy = iv_clean + rng.normal(0.0, noise_scale)
            iv_noisy = max(iv_noisy, 1e-6)

            # Black-Scholes price for midpoint
            option_types = ["C", "P"]
            for opt_type in option_types:
                price = _bs_price(F, K, tau, cfg.risk_free_rate, iv_noisy, opt_type)
                half_spread = price * spread_pct / 2.0
                bid = max(price - half_spread, 0.01)
                ask = max(price + half_spread, bid + 0.01)

                # Greeks
                delta = _bs_delta(F, K, tau, cfg.risk_free_rate, iv_noisy, opt_type)
                vega = _bs_vega(F, K, tau, cfg.risk_free_rate, iv_noisy)

                liquidity_score = _compute_liquidity_score(volume, open_interest, spread_pct)

                quote = SyntheticQuote(
                    market=cfg.market,
                    underlying=cfg.underlying,
                    observation_date=actual_observation,
                    expiry=expiry,
                    strike=round(K, 2),
                    option_type=opt_type,
                    bid=round(bid, 4),
                    ask=round(ask, 4),
                    implied_vol=round(iv_noisy, 6),
                    volume=volume,
                    open_interest=open_interest,
                    forward=round(F, 4),
                    underlying_price=cfg.spot,
                    log_moneyness=round(k_val, 6),
                    tenor_years=round(tau, 6),
                    delta=round(delta, 6),
                    vega=round(vega, 6),
                    spread_pct=round(spread_pct, 6),
                    liquidity_score=round(liquidity_score, 6),
                )
                quotes.append(quote)

    return quotes


def generate_synthetic_surface_dataset(
    config: SyntheticSurfaceConfig | None = None,
    n_surfaces: int = 10,
    n_underlyings: int = 5,
    observation_dates: list[date] | None = None,
) -> dict[str, list[SyntheticQuote]]:
    """Generate multiple synthetic surfaces for different underlyings and dates.

    Returns a dict mapping "{underlying}_{date}" to lists of SyntheticQuotes.
    """
    cfg = config or SyntheticSurfaceConfig()
    rng = np.random.default_rng(cfg.random_seed + 1)

    if observation_dates is None:
        base = date.today()
        observation_dates = [base - timedelta(days=7 * i) for i in range(n_surfaces)]

    underlyings = [f"SYNTH_{i}" for i in range(n_underlyings)]

    dataset: dict[str, list[SyntheticQuote]] = {}
    for ul in underlyings:
        ul_cfg = SyntheticSurfaceConfig(
            n_maturities=cfg.n_maturities,
            n_strikes=cfg.n_strikes,
            min_tenor_days=cfg.min_tenor_days,
            max_tenor_days=cfg.max_tenor_days,
            moneyness_range=cfg.moneyness_range,
            eta=cfg.eta + rng.normal(0, 0.005),
            lam=cfg.lam + rng.uniform(-0.1, 0.1),
            rho_surface=cfg.rho_surface + rng.uniform(-0.1, 0.1),
            eta_surface=cfg.eta_surface + rng.uniform(-0.1, 0.1),
            spot=cfg.spot * rng.lognormal(0, 0.1),
            risk_free_rate=cfg.risk_free_rate,
            dividend_yield=cfg.dividend_yield,
            bid_ask_spread_pct=cfg.bid_ask_spread_pct,
            vol_noise_scale=cfg.vol_noise_scale,
            vol_noise_liquidity_corr=cfg.vol_noise_liquidity_corr,
            volume_base=cfg.volume_base,
            volume_range=cfg.volume_range,
            oi_base=cfg.oi_base,
            oi_range=cfg.oi_range,
            missing_wing_prob=cfg.missing_wing_prob,
            stale_quote_prob=cfg.stale_quote_prob,
            market=cfg.market,
            underlying=ul,
            random_seed=cfg.random_seed + _stable_underlying_offset(ul),
        )
        for obs_date in observation_dates:
            key = f"{ul}_{obs_date.isoformat()}"
            dataset[key] = generate_synthetic_surface(ul_cfg, obs_date)

    return dataset


def _stable_underlying_offset(underlying: str) -> int:
    digest = hashlib.sha256(underlying.encode()).hexdigest()
    return int(digest[:8], 16) % 10000


def synthetic_quotes_to_option_quotes(
    syn_quotes: list[SyntheticQuote],
) -> list[OptionQuote]:
    """Convert SyntheticQuotes to OptionQuote schema objects.

    Imported lazily to avoid circular imports — call from CLI or when graph module needed.
    """
    from log_iv.schema import OptionQuote, OptionType

    return [
        OptionQuote(
            market=q.market,
            underlying=q.underlying,
            observation_date=q.observation_date,
            expiry=q.expiry,
            strike=q.strike,
            option_type=cast(OptionType, q.option_type),
            bid=q.bid,
            ask=q.ask,
            implied_vol=q.implied_vol,
            volume=q.volume,
            open_interest=q.open_interest,
            forward=q.forward,
            underlying_price=q.underlying_price,
        )
        for q in syn_quotes
    ]


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------


def _bs_price(F: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    """Black forward price formula."""
    from math import log, sqrt

    if T <= 0 or sigma <= 0:
        return max(0.0, F - K) if option_type == "C" else max(0.0, K - F)

    d1 = (log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    from scipy.stats import norm

    if option_type == "C":
        price = np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        price = np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return float(price)


def _bs_delta(F: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
    """Black forward delta."""
    from math import log, sqrt

    if T <= 0 or sigma <= 0:
        return 1.0 if (option_type == "C" and F > K) else 0.0

    d1 = (log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt(T))
    from scipy.stats import norm

    if option_type == "C":
        return float(norm.cdf(d1))
    else:
        return float(norm.cdf(d1) - 1.0)


def _bs_vega(F: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black forward vega (÷100 for 1% vol change)."""
    from math import log, sqrt

    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = (log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt(T))
    from scipy.stats import norm

    vega = F * norm.pdf(d1) * sqrt(T) * np.exp(-r * T) / 100.0
    return float(vega)


def _compute_liquidity_score(
    volume: int, open_interest: int, spread_pct: float, epsilon: float = 1e-8
) -> float:
    """Compute a liquidity score from volume, OI, and spread."""
    import math

    support = math.log1p(volume) + math.log1p(open_interest)
    spread_penalty = math.log1p(spread_pct + epsilon)
    return support - spread_penalty
