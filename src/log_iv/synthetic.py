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
import json
from dataclasses import asdict, dataclass
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
    oracle_clean_iv: float
    oracle_clean_price: float
    oracle_latent_log_precision: float
    oracle_noise_std: float
    oracle_surface_family: str
    oracle_event_bump_flag: bool


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

    # Temporal / public benchmark reproducibility
    surface_ar1_phi: float = 0.85
    liquidity_ar1_phi: float = 0.90
    surface_innovation_scale: float = 0.002
    liquidity_innovation_scale: float = 0.08
    event_bump_probability: float = 0.0
    event_bump_amplitude: float = 0.0005  # total-variance bump
    event_bump_width_moneyness: float = 0.08
    event_bump_width_log_tenor: float = 0.35
    event_bump_center_moneyness: float | None = None
    event_bump_center_tenor_years: float | None = None
    synthetic_version: str = "synthetic-log-iv-v0"

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
    event_bump_flag = bool(rng.random() < cfg.event_bump_probability)
    bump_center_k = (
        cfg.event_bump_center_moneyness
        if cfg.event_bump_center_moneyness is not None
        else float(rng.uniform(-0.15, 0.15))
    )
    bump_center_tau = (
        cfg.event_bump_center_tenor_years
        if cfg.event_bump_center_tenor_years is not None
        else float(rng.choice(tenor_years_grid))
    )

    quotes: list[SyntheticQuote] = []

    for i, tau in enumerate(tenor_years_grid):
        expiry = observation_date + timedelta(days=int(tenor_days_grid[i]))
        F = forward_grid[i]
        # Strike from log-moneyness: K = F * exp(k)
        strikes = F * np.exp(k_grid)

        for j, (k_val, K) in enumerate(zip(k_grid, strikes, strict=False)):
            iv_clean = iv_surface[i, j]
            if event_bump_flag:
                total_variance = iv_clean * iv_clean * tau
                bump = cfg.event_bump_amplitude * np.exp(
                    -((k_val - bump_center_k) ** 2) / (2.0 * cfg.event_bump_width_moneyness**2)
                    - ((np.log(max(tau, 1e-8)) - np.log(max(bump_center_tau, 1e-8))) ** 2)
                    / (2.0 * cfg.event_bump_width_log_tenor**2)
                )
                iv_clean = float(np.sqrt(max(total_variance + bump, 1e-12) / max(tau, 1e-8)))

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
            latent_log_precision = -2.0 * np.log(max(noise_scale, 1e-12))
            iv_noisy = iv_clean + rng.normal(0.0, noise_scale)
            iv_noisy = max(iv_noisy, 1e-6)

            # Black-Scholes price for midpoint
            option_types = ["C", "P"]
            for opt_type in option_types:
                clean_price = _bs_price(F, K, tau, cfg.risk_free_rate, iv_clean, opt_type)
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
                    oracle_clean_iv=round(iv_clean, 8),
                    oracle_clean_price=round(clean_price, 8),
                    oracle_latent_log_precision=round(float(latent_log_precision), 8),
                    oracle_noise_std=round(float(noise_scale), 8),
                    oracle_surface_family="ssvi_ar1_bump" if event_bump_flag else "ssvi_ar1",
                    oracle_event_bump_flag=event_bump_flag,
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
    chronological_dates = sorted(observation_dates)
    for ul in underlyings:
        eta = float(np.clip(cfg.eta + rng.normal(0, 0.005), 0.005, 0.25))
        lam = float(np.clip(cfg.lam + rng.uniform(-0.1, 0.1), 0.0, 1.0))
        rho_surface = float(np.clip(cfg.rho_surface + rng.uniform(-0.1, 0.1), -0.95, 0.20))
        eta_surface = float(np.clip(cfg.eta_surface + rng.uniform(-0.1, 0.1), 0.2, 5.0))
        spread_pct = float(max(cfg.bid_ask_spread_pct * rng.lognormal(0.0, 0.1), 1e-5))
        volume_base = int(max(1, cfg.volume_base * rng.lognormal(0.0, 0.15)))
        oi_base = int(max(1, cfg.oi_base * rng.lognormal(0.0, 0.15)))
        ul_cfg = SyntheticSurfaceConfig(
            n_maturities=cfg.n_maturities,
            n_strikes=cfg.n_strikes,
            min_tenor_days=cfg.min_tenor_days,
            max_tenor_days=cfg.max_tenor_days,
            moneyness_range=cfg.moneyness_range,
            eta=eta,
            lam=lam,
            rho_surface=rho_surface,
            eta_surface=eta_surface,
            spot=cfg.spot * rng.lognormal(0, 0.1),
            risk_free_rate=cfg.risk_free_rate,
            dividend_yield=cfg.dividend_yield,
            bid_ask_spread_pct=spread_pct,
            vol_noise_scale=cfg.vol_noise_scale,
            vol_noise_liquidity_corr=cfg.vol_noise_liquidity_corr,
            volume_base=volume_base,
            volume_range=cfg.volume_range,
            oi_base=oi_base,
            oi_range=cfg.oi_range,
            missing_wing_prob=cfg.missing_wing_prob,
            stale_quote_prob=cfg.stale_quote_prob,
            stale_tenor_shift_days=cfg.stale_tenor_shift_days,
            surface_ar1_phi=cfg.surface_ar1_phi,
            liquidity_ar1_phi=cfg.liquidity_ar1_phi,
            surface_innovation_scale=cfg.surface_innovation_scale,
            liquidity_innovation_scale=cfg.liquidity_innovation_scale,
            event_bump_probability=cfg.event_bump_probability,
            event_bump_amplitude=cfg.event_bump_amplitude,
            event_bump_width_moneyness=cfg.event_bump_width_moneyness,
            event_bump_width_log_tenor=cfg.event_bump_width_log_tenor,
            synthetic_version=cfg.synthetic_version,
            market=cfg.market,
            underlying=ul,
            random_seed=cfg.random_seed + _stable_underlying_offset(ul),
        )
        for date_index, obs_date in enumerate(chronological_dates):
            if date_index > 0:
                phi = cfg.surface_ar1_phi
                eta = _ar1_update(eta, cfg.eta, phi, cfg.surface_innovation_scale, rng, 0.005, 0.25)
                lam = _ar1_update(lam, cfg.lam, phi, cfg.surface_innovation_scale, rng, 0.0, 1.0)
                rho_surface = _ar1_update(
                    rho_surface,
                    cfg.rho_surface,
                    phi,
                    cfg.surface_innovation_scale * 4.0,
                    rng,
                    -0.95,
                    0.20,
                )
                eta_surface = _ar1_update(
                    eta_surface,
                    cfg.eta_surface,
                    phi,
                    cfg.surface_innovation_scale * 10.0,
                    rng,
                    0.2,
                    5.0,
                )
                liq_phi = cfg.liquidity_ar1_phi
                spread_pct = _ar1_update(
                    spread_pct,
                    cfg.bid_ask_spread_pct,
                    liq_phi,
                    cfg.liquidity_innovation_scale * cfg.bid_ask_spread_pct,
                    rng,
                    1e-5,
                    0.5,
                )
                volume_base = int(
                    _ar1_update(
                        float(volume_base),
                        float(cfg.volume_base),
                        liq_phi,
                        cfg.liquidity_innovation_scale * max(cfg.volume_base, 1),
                        rng,
                        1.0,
                        float(cfg.volume_range[1]),
                    )
                )
                oi_base = int(
                    _ar1_update(
                        float(oi_base),
                        float(cfg.oi_base),
                        liq_phi,
                        cfg.liquidity_innovation_scale * max(cfg.oi_base, 1),
                        rng,
                        1.0,
                        float(cfg.oi_range[1]),
                    )
                )
            ul_cfg = dataclass_replace(
                ul_cfg,
                eta=eta,
                lam=lam,
                rho_surface=rho_surface,
                eta_surface=eta_surface,
                bid_ask_spread_pct=spread_pct,
                volume_base=volume_base,
                oi_base=oi_base,
                random_seed=cfg.random_seed
                + _stable_underlying_offset(ul)
                + date_index * 1_000_003,
            )
            key = f"{ul}_{obs_date.isoformat()}"
            dataset[key] = generate_synthetic_surface(ul_cfg, obs_date)

    return dataset


def dataclass_replace(config: SyntheticSurfaceConfig, **updates: object) -> SyntheticSurfaceConfig:
    payload = asdict(config)
    payload.update(updates)
    return SyntheticSurfaceConfig(**payload)


def _ar1_update(
    current: float,
    target: float,
    phi: float,
    innovation_scale: float,
    rng: np.random.Generator,
    lower: float,
    upper: float,
) -> float:
    value = phi * current + (1.0 - phi) * target + rng.normal(0.0, innovation_scale)
    return float(np.clip(value, lower, upper))


def stable_synthetic_logical_hash(dataset: dict[str, list[SyntheticQuote]]) -> str:
    """Stable hash over sorted logical synthetic rows, not Parquet bytes."""

    rows: list[dict[str, object]] = []
    for surface_id, quotes in dataset.items():
        for quote in quotes:
            row = quote._asdict()
            row["surface_id"] = surface_id
            rows.append(row)
    canonical_columns = [
        "surface_id",
        "market",
        "underlying",
        "observation_date",
        "expiry",
        "strike",
        "option_type",
        "bid",
        "ask",
        "implied_vol",
        "volume",
        "open_interest",
        "forward",
        "underlying_price",
        "log_moneyness",
        "tenor_years",
        "delta",
        "vega",
        "spread_pct",
        "liquidity_score",
        "oracle_clean_iv",
        "oracle_clean_price",
        "oracle_latent_log_precision",
        "oracle_noise_std",
        "oracle_surface_family",
        "oracle_event_bump_flag",
    ]
    rows = sorted(
        rows,
        key=lambda row: (
            str(row["surface_id"]),
            str(row["option_type"]),
            str(row["expiry"]),
            float(cast(float | int | str, row["strike"])),
        ),
    )
    lines = [",".join(canonical_columns)]
    for row in rows:
        values = []
        for column in canonical_columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{round(value, 12):.12g}")
            elif isinstance(value, date):
                values.append(value.isoformat())
            else:
                values.append(str(value))
        lines.append(",".join(values))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def stable_synthetic_config_hash(config: SyntheticSurfaceConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def synthetic_no_arbitrage_diagnostics(quotes: list[SyntheticQuote]) -> dict[str, object]:
    """Dense-grid diagnostics on oracle clean synthetic prices and IVs."""

    return {
        "calendar": _synthetic_calendar_diagnostics(quotes),
        "butterfly_convexity": _synthetic_convexity_diagnostics(quotes),
        "vertical_spread": _synthetic_vertical_diagnostics(quotes),
    }


def _synthetic_calendar_diagnostics(quotes: list[SyntheticQuote]) -> dict[str, float | int]:
    magnitudes: list[float] = []
    pairs = 0
    grouped: dict[tuple[str, str, str, str, float], list[SyntheticQuote]] = {}
    for quote in quotes:
        key = (
            quote.market,
            quote.underlying,
            quote.observation_date.isoformat(),
            quote.option_type,
            round(quote.log_moneyness, 8),
        )
        grouped.setdefault(key, []).append(quote)
    for group in grouped.values():
        ordered = sorted(group, key=lambda quote: quote.tenor_years)
        for near, far in zip(ordered[:-1], ordered[1:], strict=False):
            pairs += 1
            near_tv = near.oracle_clean_iv * near.oracle_clean_iv * near.tenor_years
            far_tv = far.oracle_clean_iv * far.oracle_clean_iv * far.tenor_years
            magnitudes.append(max(float(near_tv - far_tv), 0.0))
    positives = [value for value in magnitudes if value > 0]
    return {
        "pairs": pairs,
        "violations": len(positives),
        "mean_violation": float(np.mean(positives)) if positives else 0.0,
        "max_violation": float(np.max(positives)) if positives else 0.0,
    }


def _synthetic_convexity_diagnostics(quotes: list[SyntheticQuote]) -> dict[str, float | int]:
    magnitudes: list[float] = []
    triples = 0
    grouped: dict[tuple[str, str, str, date], list[SyntheticQuote]] = {}
    for quote in quotes:
        if quote.option_type != "C":
            continue
        key = (quote.market, quote.underlying, quote.observation_date.isoformat(), quote.expiry)
        grouped.setdefault(key, []).append(quote)
    for group in grouped.values():
        ordered = sorted(group, key=lambda quote: quote.strike)
        if len(ordered) < 3:
            continue
        strikes = np.array([quote.strike / quote.forward for quote in ordered], dtype=float)
        prices = np.array(
            [quote.oracle_clean_price / quote.forward for quote in ordered], dtype=float
        )
        for i in range(1, len(ordered) - 1):
            k1, k2, k3 = strikes[i - 1], strikes[i], strikes[i + 1]
            if k3 <= k1:
                continue
            p1, p2, p3 = prices[i - 1], prices[i], prices[i + 1]
            alpha = (k3 - k2) / (k3 - k1)
            beta = (k2 - k1) / (k3 - k1)
            triples += 1
            magnitudes.append(max(float(p2 - (alpha * p1 + beta * p3)), 0.0))
    positives = [value for value in magnitudes if value > 0]
    return {
        "triples": triples,
        "violations": len(positives),
        "mean_violation": float(np.mean(positives)) if positives else 0.0,
        "max_violation": float(np.max(positives)) if positives else 0.0,
    }


def _synthetic_vertical_diagnostics(quotes: list[SyntheticQuote]) -> dict[str, float | int]:
    magnitudes: list[float] = []
    pairs = 0
    grouped: dict[tuple[str, str, str, date], list[SyntheticQuote]] = {}
    for quote in quotes:
        if quote.option_type != "C":
            continue
        key = (quote.market, quote.underlying, quote.observation_date.isoformat(), quote.expiry)
        grouped.setdefault(key, []).append(quote)
    for group in grouped.values():
        ordered = sorted(group, key=lambda quote: quote.strike)
        for left, right in zip(ordered[:-1], ordered[1:], strict=False):
            left_k = left.strike / left.forward
            right_k = right.strike / right.forward
            if right_k <= left_k:
                continue
            left_price = left.oracle_clean_price / left.forward
            right_price = right.oracle_clean_price / right.forward
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
