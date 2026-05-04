from __future__ import annotations

from typing import cast

import numpy as np

from log_iv.synthetic import (
    SSVISurface,
    SVISliceParams,
    SyntheticSurfaceConfig,
    generate_synthetic_surface,
    generate_synthetic_surface_dataset,
    stable_synthetic_config_hash,
    stable_synthetic_logical_hash,
    synthetic_no_arbitrage_diagnostics,
)


def test_svi_total_variance_is_vectorized() -> None:
    params = SVISliceParams(a=0.02, b=0.3, rho=-0.4, m=0.0, sigma=0.2)
    k = np.array([-0.1, 0.0, 0.1], dtype=np.float64)

    total_variance = params.total_variance(k)

    assert total_variance.shape == k.shape
    assert np.all(total_variance > 0)
    assert total_variance[0] > total_variance[2]


def test_svi_arbitrage_free_sufficient_condition() -> None:
    valid = SVISliceParams(a=0.02, b=0.3, rho=0.0, m=0.0, sigma=0.2)
    too_steep = SVISliceParams(a=0.02, b=2.1, rho=0.0, m=0.0, sigma=0.2)
    degenerate = SVISliceParams(a=0.02, b=0.3, rho=0.0, m=0.0, sigma=1e-8)

    assert valid.is_arbitrage_free()
    assert not too_steep.is_arbitrage_free()
    assert not degenerate.is_arbitrage_free()


def test_ssvi_surface_defaults_are_conservative() -> None:
    surface = SSVISurface()

    assert surface.eta == 0.04
    assert surface.lam == 0.5


def test_synthetic_quotes_include_oracle_fields() -> None:
    quotes = generate_synthetic_surface(
        SyntheticSurfaceConfig(n_maturities=2, n_strikes=5, random_seed=7)
    )

    assert quotes
    first = quotes[0]
    assert first.oracle_clean_iv > 0
    assert first.oracle_clean_price >= 0
    assert np.isfinite(first.oracle_latent_log_precision)
    assert first.oracle_noise_std > 0
    assert first.oracle_surface_family.startswith("ssvi")


def test_synthetic_logical_hash_is_reproducible() -> None:
    config = SyntheticSurfaceConfig(n_maturities=2, n_strikes=5, random_seed=11)
    left = generate_synthetic_surface_dataset(config, n_surfaces=2, n_underlyings=2)
    right = generate_synthetic_surface_dataset(config, n_surfaces=2, n_underlyings=2)

    assert stable_synthetic_logical_hash(left) == stable_synthetic_logical_hash(right)
    assert len(stable_synthetic_config_hash(config)) == 64


def test_synthetic_dataset_uses_temporal_ar1_drift() -> None:
    config = SyntheticSurfaceConfig(n_maturities=2, n_strikes=5, random_seed=19)
    dataset = generate_synthetic_surface_dataset(config, n_surfaces=3, n_underlyings=1)
    surfaces = list(dataset.values())

    mean_ivs = [
        float(np.mean([quote.oracle_clean_iv for quote in surface])) for surface in surfaces
    ]

    assert len(set(mean_ivs)) == 3
    assert (
        max(abs(left - right) for left, right in zip(mean_ivs[:-1], mean_ivs[1:], strict=False))
        < 0.5
    )


def test_synthetic_clean_surface_has_vertical_diagnostics() -> None:
    quotes = generate_synthetic_surface(
        SyntheticSurfaceConfig(
            n_maturities=3,
            n_strikes=7,
            random_seed=23,
            vol_noise_scale=0.0,
            event_bump_probability=0.0,
        )
    )
    diagnostics = synthetic_no_arbitrage_diagnostics(quotes)
    vertical = cast(dict[str, int | float], diagnostics["vertical_spread"])

    assert vertical["pairs"] > 0
    assert vertical["violations"] == 0
