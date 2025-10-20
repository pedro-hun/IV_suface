from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import pandas as pd


NestedOptionSurface = Dict[str, Dict[float, pd.DataFrame]]


@dataclass(frozen=True)
class ColumnMapping:
    """Declarative mapping of column names used in the source dataframe."""
    strike: str = "strike"
    implied_volatility: str = "ImpliedVolatility"
    time_to_expiry: str = "TimeToExpiry"
    forward: str = "Forward"
    option_type: str = "Type"


class OptionDataFormatter:
    """
    Format an option chain dataframe into a nested dictionary by option type and time to expiry.

    The structure returned is:
        {
            "call": {
                0.25: pd.DataFrame([...]),
                0.50: pd.DataFrame([...]),
                ...
            },
            "put": {
                ...
            }
        }
    """

    MIN_IMPLIED_VOLATILITY = 1e-5

    def __init__(
        self,
        columns: ColumnMapping = ColumnMapping(),
        option_types: Optional[Sequence[str]] = None,
    ) -> None:
        self.columns = columns
        self.option_types = tuple(t.lower() for t in option_types) if option_types else None

    def format(self, data: pd.DataFrame) -> NestedOptionSurface:
        """
        Transform the dataframe into a nested dictionary keyed by option type and time-to-expiry.

        Parameters
        ----------
        data:
            Option chain dataframe containing the configured columns.

        Returns
        -------
        dict
            Nested dictionary mapping option type → time to expiry → formatted dataframe.
        """
        self._validate_columns(data)

        df = data.copy()
        df[self.columns.option_type] = (
            df[self.columns.option_type].astype(str).str.strip().str.lower()
        )

        option_types = self.option_types or tuple(
            sorted(df[self.columns.option_type].dropna().unique())
        )

        formatted: NestedOptionSurface = {}

        for opt_type in option_types:
            type_df = self._filter_by_type(df, opt_type)
            if type_df.empty:
                continue

            type_dict: Dict[float, pd.DataFrame] = {}
            unique_ttes = sorted(type_df[self.columns.time_to_expiry].dropna().unique())

            for tte_value in unique_ttes:
                tte_df = type_df[type_df[self.columns.time_to_expiry] == tte_value]
                surface_slice = self._build_surface_slice(tte_df)
                if surface_slice is None:
                    continue

                try:
                    tte_key = float(tte_value)
                except (TypeError, ValueError):
                    continue

                type_dict[tte_key] = surface_slice

            if type_dict:
                formatted[opt_type] = type_dict

        return formatted

    def _build_surface_slice(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        subset = df[
            [
                self.columns.strike,
                self.columns.implied_volatility,
                self.columns.forward,
                self.columns.time_to_expiry,
            ]
        ].rename(
            columns={
                self.columns.strike: "Strike",
                self.columns.implied_volatility: "ImpliedVolatility",
                self.columns.forward: "Forward",
                self.columns.time_to_expiry: "TimeToExpiry",
            }
        )

        subset = subset.dropna()
        subset = subset[subset["ImpliedVolatility"] > self.MIN_IMPLIED_VOLATILITY]

        if subset.empty:
            return None

        return subset.sort_values("Strike").reset_index(drop=True)

    def _filter_by_type(self, df: pd.DataFrame, option_type: str) -> pd.DataFrame:
        return df[df[self.columns.option_type] == option_type].copy()

    def _validate_columns(self, df: pd.DataFrame) -> None:
        required = {
            self.columns.strike,
            self.columns.implied_volatility,
            self.columns.time_to_expiry,
            self.columns.forward,
            self.columns.option_type,
        }
        missing = required.difference(df.columns)
        if missing:
            raise KeyError(f"Input dataframe is missing required columns: {sorted(missing)}")