from __future__ import annotations

import numpy as np

from log_iv.synthetic import SSVISurface, SVISliceParams


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
