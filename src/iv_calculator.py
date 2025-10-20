from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

OptionType = Literal["call", "put"]

DEFAULT_TICKER_SYMBOL = "AAPL"
DEFAULT_RISK_FREE_RATE = 0.15


@dataclass(frozen=True)
class SolverBounds:
    """Numerical bounds and tolerances for implied volatility root finding."""
    low_vol: float = 1e-4
    high_vol: float = 4.0
    tolerance: float = 1e-6


@dataclass(frozen=True)
class OptionFilterConfig:
    """Filtering thresholds applied before/after implied volatility estimation."""
    min_volume: int = 1
    max_relative_spread: float = 0.50
    min_days_to_expiry: int = 1
    min_implied_vol: float = 0.01
    max_implied_vol: float = 2.00


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
) -> float:
    """Return the Black-Scholes price for a European option."""
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be either 'call' or 'put'.")

    discount_factor = 1 / (1 + risk_free_rate) ** time_to_expiry

    if volatility <= 0 or time_to_expiry <= 0:
        intrinsic = (
            max(0.0, spot - strike * discount_factor)
            if option_type == "call"
            else max(0.0, strike * discount_factor - spot)
        )
        return intrinsic

    try:
        d1 = (
            np.log(spot / strike)
            + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry
        ) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)

        if option_type == "call":
            price = spot * norm.cdf(d1) - strike * discount_factor * norm.cdf(d2)
        else:
            price = strike * discount_factor * norm.cdf(-d2) - spot * norm.cdf(-d1)

    except (FloatingPointError, OverflowError):
        return np.nan

    return max(price, 0.0)


def implied_volatility(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    market_price: float,
    option_type: OptionType,
    bounds: SolverBounds = SolverBounds(),
) -> float:
    """Estimate implied volatility via Brent's method with robust validation."""
    if time_to_expiry <= 0 or market_price <= 0:
        return np.nan

    option_type = option_type.lower()  # type: ignore[assignment]
    if option_type not in {"call", "put"}:
        return np.nan

    discount_factor = 1 / (1 + risk_free_rate) ** time_to_expiry

    intrinsic_value = (
        max(0.0, spot - strike * discount_factor)
        if option_type == "call"
        else max(0.0, strike * discount_factor - spot)
    )

    if market_price < intrinsic_value - bounds.tolerance:
        warnings.warn(
            f"Market price {market_price:.4f} below intrinsic value "
            f"{intrinsic_value:.4f}.",
            RuntimeWarning,
        )
        return np.nan

    if option_type == "call" and market_price > spot + bounds.tolerance:
        warnings.warn(
            f"Call price {market_price:.4f} exceeds spot {spot:.4f}.",
            RuntimeWarning,
        )
        return np.nan

    if option_type == "put" and market_price > strike * discount_factor + bounds.tolerance:
        warnings.warn(
            f"Put price {market_price:.4f} exceeds discounted strike "
            f"{strike * discount_factor:.4f}.",
            RuntimeWarning,
        )
        return np.nan

    def objective(vol: float) -> float:
        if vol <= 0:
            return 1e12
        price = black_scholes_price(
            spot, strike, time_to_expiry, risk_free_rate, vol, option_type  # type: ignore[arg-type]
        )
        if np.isnan(price):
            return 1e12
        return price - market_price

    try:
        lower_val = objective(bounds.low_vol)
        upper_val = objective(bounds.high_vol)

        price_low = black_scholes_price(
            spot, strike, time_to_expiry, risk_free_rate, bounds.low_vol, option_type  # type: ignore[arg-type]
        )
        price_high = black_scholes_price(
            spot, strike, time_to_expiry, risk_free_rate, bounds.high_vol, option_type  # type: ignore[arg-type]
        )

        if any(np.isnan(val) for val in (price_low, price_high)):
            return np.nan

        if market_price < price_low - bounds.tolerance:
            return np.nan

        if market_price > price_high + bounds.tolerance:
            return np.nan

        if np.sign(lower_val) == np.sign(upper_val):
            return np.nan

        implied_vol = brentq(
            objective,
            bounds.low_vol,
            bounds.high_vol,
            xtol=bounds.tolerance,
            rtol=bounds.tolerance,
        )
    except (ValueError, OverflowError):
        return np.nan
    except Exception as exc:  # pragma: no cover - defensive clause
        warnings.warn(f"Unexpected numerical error: {exc}", RuntimeWarning)
        return np.nan

    if not (bounds.low_vol <= implied_vol <= bounds.high_vol):
        return np.nan

    return float(implied_vol)


class OptionChainIVCalculator:
    """Pipeline for cleaning option chain data and estimating implied volatility."""

    REQUIRED_COLUMNS = {
        "bid",
        "ask",
        "Strike",
        "Expiry",
        "SpotPrice",
        "Type",
        "volume",
        "TimeToExpiry",
    }

    def __init__(
        self,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        filter_config: OptionFilterConfig = OptionFilterConfig(),
        solver_bounds: SolverBounds = SolverBounds(),
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.filter_config = filter_config
        self.solver_bounds = solver_bounds

    def calculate(self, option_chain: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the option chain and compute implied volatility row-wise.

        Returns a dataframe containing: TimeToExpiry, Strike, ImpliedVolatility, Type, Forward.
        """
        df = option_chain.copy()

        self._ensure_required_columns(df)
        self._prepare_columns(df)

        df = self._apply_filters(df)
        if df.empty:
            warnings.warn("No options remain after filtering.", RuntimeWarning)
            return self._empty_result()

        df["ImpliedVolatility"] = df.apply(self._calculate_row_iv, axis=1)
        df = df.dropna(subset=["ImpliedVolatility"])

        cfg = self.filter_config
        df = df[
            (df["ImpliedVolatility"] >= cfg.min_implied_vol)
            & (df["ImpliedVolatility"] <= cfg.max_implied_vol)
        ]

        if df.empty:
            warnings.warn("No valid implied volatilities computed.", RuntimeWarning)
            return self._empty_result()

        return df[["TimeToExpiry", "Strike", "ImpliedVolatility", "Type", "Forward"]].reset_index(drop=True)

    def _calculate_row_iv(self, row: pd.Series) -> float:
        return implied_volatility(
            spot=float(row["SpotPrice"]),
            strike=float(row["Strike"]),
            time_to_expiry=float(row["TimeToExpiry"]),
            risk_free_rate=self.risk_free_rate,
            market_price=float(row["MidPrice"]),
            option_type=row["Type"],
            bounds=self.solver_bounds,
        )

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.filter_config

        df = df[(df["bid"] > 0) & (df["ask"] > 0)]
        df = df[df["MidPrice"] > 0]

        df["Spread"] = df["ask"] - df["bid"]
        df["RelativeSpread"] = df["Spread"] / df["MidPrice"]
        df = df[(df["RelativeSpread"] >= 0) & (df["RelativeSpread"] <= cfg.max_relative_spread)]

        df = df[df["volume"] >= cfg.min_volume]

        df = df[df["TimeToExpiry"] > 0]
        df = df[df["TimeToExpiry"] * 252 >= cfg.min_days_to_expiry]

        discount_factor = (1 + self.risk_free_rate) ** df["TimeToExpiry"]
        df["IntrinsicValue"] = np.where(
            df["Type"] == "call",
            np.maximum(0.0, df["SpotPrice"] - df["Strike"] / discount_factor),
            np.maximum(0.0, df["Strike"] * discount_factor - df["SpotPrice"]),
        )

        df = df[df["MidPrice"] >= df["IntrinsicValue"] - 0.005]
        return df

    def _prepare_columns(self, df: pd.DataFrame) -> None:
        df["Type"] = df["Type"].astype(str).str.lower()
        df["MidPrice"] = (df["bid"] + df["ask"]) / 2.0

        if "Forward" not in df.columns:
            df["Forward"] = np.nan

        df.dropna(
            subset=["SpotPrice", "Strike", "TimeToExpiry", "MidPrice", "Type"],
            inplace=True,
        )

    def _ensure_required_columns(self, df: pd.DataFrame) -> None:
        missing = self.REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            raise KeyError(f"Option chain is missing required columns: {sorted(missing)}")

    @staticmethod
    def _empty_result() -> pd.DataFrame:
        return pd.DataFrame(columns=["TimeToExpiry", "Strike", "ImpliedVolatility", "Type", "Forward"])