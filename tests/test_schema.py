from __future__ import annotations

import math
from datetime import date

import pytest

from log_iv.schema import EdgeKind, OptionEdge, OptionQuote, OptionSurfaceGraph


def test_quote_properties_normalize_and_derive_values() -> None:
    quote = OptionQuote(
        market="jp",
        underlying="7203",
        observation_date=date(2026, 1, 2),
        expiry=date(2026, 1, 30),
        strike=110.0,
        option_type="P",
        bid=2.0,
        ask=3.0,
        underlying_price=100.0,
    )

    assert quote.market == "JP"
    assert quote.option_type == "P"
    assert quote.mid == 2.5
    assert quote.spread == 1.0
    assert quote.tenor_days == 28
    assert quote.tenor_years == pytest.approx(28 / 365.25)
    assert quote.log_moneyness == pytest.approx(math.log(1.1))


def test_quote_properties_allow_missing_quote_and_reference_fields() -> None:
    quote = OptionQuote(
        market="US",
        underlying="SPY",
        observation_date=date(2026, 1, 2),
        expiry=date(2026, 2, 20),
        strike=100.0,
        option_type="C",
    )

    assert quote.mid is None
    assert quote.spread is None
    assert quote.log_moneyness is None


def test_quote_validation_rejects_expired_contract() -> None:
    with pytest.raises(ValueError, match="expiry"):
        OptionQuote(
            market="US",
            underlying="SPY",
            observation_date=date(2026, 1, 2),
            expiry=date(2026, 1, 2),
            strike=100.0,
            option_type="C",
        )


def test_graph_models_require_positive_edge_weight() -> None:
    quote = OptionQuote(
        market="US",
        underlying="SPY",
        observation_date=date(2026, 1, 2),
        expiry=date(2026, 2, 20),
        strike=100.0,
        option_type="C",
    )

    with pytest.raises(ValueError, match="greater than 0"):
        OptionSurfaceGraph(
            nodes=(),
            edges=(OptionEdge(source=0, target=1, kind=EdgeKind.STRIKE_NEIGHBOR, weight=0),),
        )
    assert quote.vendor_symbol is None
