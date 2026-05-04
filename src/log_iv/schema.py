from __future__ import annotations

import math
from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

OptionType = Literal["C", "P"]


class OptionQuote(BaseModel):
    """Canonical option-token observation used before vendor-specific adapters exist."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    market: str = Field(min_length=2)
    underlying: str = Field(min_length=1)
    observation_date: date
    expiry: date
    strike: float = Field(gt=0)
    option_type: OptionType
    bid: float | None = Field(default=None, ge=0)
    ask: float | None = Field(default=None, ge=0)
    implied_vol: float | None = Field(default=None, gt=0)
    volume: int = Field(default=0, ge=0)
    open_interest: int = Field(default=0, ge=0)
    forward: float | None = Field(default=None, gt=0)
    underlying_price: float | None = Field(default=None, gt=0)
    vendor_symbol: str | None = None

    @field_validator("market", "underlying", "option_type", mode="before")
    @classmethod
    def normalize_upper(cls, value: str) -> str:
        return value.upper()

    @model_validator(mode="after")
    def validate_quote(self) -> OptionQuote:
        if self.expiry <= self.observation_date:
            msg = "expiry must be after observation_date"
            raise ValueError(msg)
        if self.bid is not None and self.ask is not None and self.ask < self.bid:
            msg = "ask must be greater than or equal to bid"
            raise ValueError(msg)
        return self

    @property
    def mid(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float | None:
        if self.bid is None or self.ask is None:
            return None
        return self.ask - self.bid

    @property
    def tenor_days(self) -> int:
        return (self.expiry - self.observation_date).days

    @property
    def tenor_years(self) -> float:
        return self.tenor_days / 365.25

    @property
    def log_moneyness(self) -> float | None:
        reference = self.forward or self.underlying_price
        if reference is None:
            return None
        return math.log(self.strike / reference)


class EdgeKind(StrEnum):
    STRIKE_NEIGHBOR = "strike_neighbor"
    MATURITY_NEIGHBOR = "maturity_neighbor"
    LIQUIDITY_SIMILARITY = "liquidity_similarity"
    SAME_EXPIRY = "same_expiry"
    SAME_STRIKE = "same_strike"
    NEAR_MONEYNESS = "near_moneyness"
    NEAR_TENOR = "near_tenor"
    PUT_CALL_PAIR = "put_call_pair"
    SAME_LIQUIDITY_BUCKET = "same_liquidity_bucket"


class OptionNode(BaseModel):
    model_config = ConfigDict(frozen=True)

    node_id: int
    quote: OptionQuote


class OptionEdge(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: int
    target: int
    kind: EdgeKind
    weight: float = Field(gt=0)


class OptionSurfaceGraph(BaseModel):
    model_config = ConfigDict(frozen=True)

    nodes: tuple[OptionNode, ...]
    edges: tuple[OptionEdge, ...]
