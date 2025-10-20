from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal, Mapping, Optional, Union

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, griddata

SurfaceMethod = Literal["linear", "cubic"]
SliceMethod = Literal["linear", "cubic"]

REQUIRED_SURFACE_COLUMNS = {"TimeToExpiry", "Strike", "ImpliedVolatility"}
FormattedSurface = Mapping[str, Mapping[float, pd.DataFrame]]


@dataclass(frozen=True)
class SurfaceGridConfig:
    """Configuration for two-dimensional surface interpolation."""
    n_strikes: int = 100
    n_expiries: int = 100
    method: SurfaceMethod = "cubic"
    min_points: int = 4  # griddata in 2D needs at least four unique points
    strike_padding: float = 1.0
    expiry_padding: float = 0.01


@dataclass(frozen=True)
class SliceGridConfig:
    """Configuration for one-dimensional (single-expiry) interpolation."""
    n_points: int = 200
    method: SliceMethod = "cubic"
    tolerance: float = 1e-4
    min_points: int = 3  # for cubic; linear will require at least two
    use_nearest: bool = True  # fallback to nearest expiry if within tolerance


@dataclass(frozen=True)
class SurfaceInterpolationResult:
    """Container for a full volatility surface interpolation."""
    time_grid: np.ndarray
    strike_grid: np.ndarray
    implied_volatility: np.ndarray

    @classmethod
    def empty(cls) -> SurfaceInterpolationResult:
        empty = np.array([])
        return cls(empty, empty, empty)

    @property
    def is_empty(self) -> bool:
        return self.implied_volatility.size == 0 or np.isnan(self.implied_volatility).all()


@dataclass(frozen=True)
class SliceInterpolationResult:
    """Container for a single-expiry (1D) implied volatility interpolation."""
    strikes: np.ndarray
    implied_volatility: np.ndarray
    time_to_expiry: float

    @classmethod
    def empty(cls, time_to_expiry: float) -> SliceInterpolationResult:
        empty = np.array([])
        return cls(empty, empty, time_to_expiry)

    @property
    def is_empty(self) -> bool:
        return self.implied_volatility.size == 0 or np.isnan(self.implied_volatility).all()


class VolatilityInterpolator:
    """
    Utility class for interpolating implied volatility data.

    Accepts either:
        • A pandas DataFrame containing TimeToExpiry/Strike/ImpliedVolatility columns.
        • A nested dictionary produced by OptionDataFormatter:
              { option_type: { time_to_expiry: DataFrame } }

    Provides two primary workflows:
      * interpolate_surface: 2D interpolation across strike and expiry.
      * interpolate_slice: 1D interpolation across strike for a single expiry.
    """

    def __init__(
        self,
        surface_config: SurfaceGridConfig = SurfaceGridConfig(),
        slice_config: SliceGridConfig = SliceGridConfig(),
        *,
        enforce_positive_iv: bool = True,
    ) -> None:
        self.surface_config = surface_config
        self.slice_config = slice_config
        self.enforce_positive_iv = enforce_positive_iv

    def interpolate_surface(
        self,
        data: Union[pd.DataFrame, FormattedSurface],
        option_type: Optional[str] = None,
    ) -> SurfaceInterpolationResult:
        """
        Interpolate the implied volatility surface onto a regular grid.

        Parameters
        ----------
        data:
            DataFrame or nested dictionary containing implied volatility data.
        option_type:
            When providing a dictionary (or a DataFrame that still has a 'Type' column),
            specify which option type ("call"/"put"/etc.) to interpolate.

        Returns
        -------
        SurfaceInterpolationResult:
            Structured result containing mesh grids for expiry and strike and the interpolated
            implied volatility matrix. Returns an empty result if interpolation cannot be performed.
        """
        df = self._normalise_to_dataframe(data, option_type)

        if df.shape[0] < self.surface_config.min_points:
            warnings.warn(
                "Insufficient unique data points for 2D interpolation.",
                RuntimeWarning,
            )
            return SurfaceInterpolationResult.empty()

        points = df[["TimeToExpiry", "Strike"]].to_numpy()
        values = df["ImpliedVolatility"].to_numpy()

        t_min, t_max = self._expand_range(points[:, 0], self.surface_config.expiry_padding)
        k_min, k_max = self._expand_range(points[:, 1], self.surface_config.strike_padding)

        time_lin = np.linspace(t_min, t_max, self.surface_config.n_expiries)
        strike_lin = np.linspace(k_min, k_max, self.surface_config.n_strikes)
        time_grid, strike_grid = np.meshgrid(time_lin, strike_lin)

        try:
            iv_surface = griddata(
                points,
                values,
                (time_grid, strike_grid),
                method=self.surface_config.method,
                fill_value=np.nan,
            )
        except Exception as exc:
            warnings.warn(f"Interpolation failed: {exc}", RuntimeWarning)
            return SurfaceInterpolationResult.empty()

        if self.enforce_positive_iv and iv_surface.size:
            iv_surface[iv_surface < 0] = np.nan

        nan_ratio = np.isnan(iv_surface).sum() / iv_surface.size if iv_surface.size else 1.0
        if nan_ratio > 0.8:
            warnings.warn(
                "Interpolated surface contains more than 80% NaN values.",
                RuntimeWarning,
            )

        return SurfaceInterpolationResult(time_grid, strike_grid, iv_surface)

    def interpolate_slice(
        self,
        data: Union[pd.DataFrame, FormattedSurface],
        time_to_expiry: float,
        *,
        option_type: Optional[str] = None,
        strike_grid: Optional[np.ndarray] = None,
        method: Optional[SliceMethod] = None,
        tolerance: Optional[float] = None,
    ) -> SliceInterpolationResult:
        """
        Interpolate implied volatility for a single expiry across strike.

        Parameters
        ----------
        data:
            DataFrame or nested dictionary containing implied volatility data.
        time_to_expiry:
            Target expiry (in years) for which to interpolate a slice.
        option_type:
            Needed when supplying a formatted dictionary or a DataFrame with multiple option types.
        strike_grid:
            Optional array of strike values where IV should be evaluated.
        method:
            Interpolation method ('linear' or 'cubic'). Defaults to the configured method.
        tolerance:
            Maximum absolute difference allowed when matching the target expiry.

        Returns
        -------
        SliceInterpolationResult:
            Structured result containing strike locations and interpolated IV values.
            Returns an empty result if the slice cannot be produced.
        """
        config_method = method or self.slice_config.method
        config_tol = tolerance if tolerance is not None else self.slice_config.tolerance

        df = self._normalise_to_dataframe(data, option_type)
        slice_df = self._filter_by_expiry(df, time_to_expiry, config_tol)

        if slice_df.empty and self.slice_config.use_nearest:
            nearest_slice_df, nearest_expiry = self._nearest_expiry(df, time_to_expiry)
            if nearest_slice_df is not None:
                warnings.warn(
                    f"No points found within ±{config_tol} of expiry {time_to_expiry:.6f}. "
                    f"Using nearest expiry slice at {nearest_expiry:.6f}.",
                    RuntimeWarning,
                )
                slice_df = nearest_slice_df
                time_to_expiry = nearest_expiry

        if slice_df.empty:
            warnings.warn(
                f"No data available to interpolate slice at expiry {time_to_expiry:.6f}.",
                RuntimeWarning,
            )
            return SliceInterpolationResult.empty(time_to_expiry)

        strikes = slice_df["Strike"].to_numpy()
        iv_values = slice_df["ImpliedVolatility"].to_numpy()

        if self.enforce_positive_iv:
            mask = iv_values > 0
            strikes = strikes[mask]
            iv_values = iv_values[mask]

        sorted_idx = np.argsort(strikes)
        strikes = strikes[sorted_idx]
        iv_values = iv_values[sorted_idx]

        unique_strikes, unique_indices = np.unique(strikes, return_index=True)
        strikes = unique_strikes
        iv_values = iv_values[unique_indices]

        if strikes.size < 2:
            warnings.warn(
                "At least two data points are required to interpolate a slice.",
                RuntimeWarning,
            )
            return SliceInterpolationResult.empty(time_to_expiry)

        target_strikes = strike_grid
        if target_strikes is None:
            target_strikes = np.linspace(strikes.min(), strikes.max(), self.slice_config.n_points)

        interpolated_iv = self._interpolate_1d(
            strikes,
            iv_values,
            target_strikes,
            method=config_method,
            required_points=self.slice_config.min_points,
        )

        return SliceInterpolationResult(target_strikes, interpolated_iv, time_to_expiry)

    @staticmethod
    def plot_surface(
        result: SurfaceInterpolationResult,
        *,
        title: str = "Implied Volatility Surface",
        cmap: str = "viridis",
        zlabel: str = "Implied Volatility",
        save_path: Optional[str] = None,
    ) -> None:
        """Visualise an interpolated volatility surface using Matplotlib."""
        if result.is_empty:
            warnings.warn("Empty surface result provided to plot_surface.", RuntimeWarning)
            return

        import matplotlib.pyplot as plt  # Local import to keep module lightweight
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (required for 3D plotting)

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(
            result.time_grid,
            result.strike_grid,
            result.implied_volatility * 100,
            cmap=cmap,
            edgecolor="none",
        )
        ax.set_title(title)
        ax.set_xlabel("Time to Expiry (years)")
        ax.set_ylabel("Strike")
        ax.set_zlabel(f"{zlabel} (%)")

        fig.colorbar(surf, shrink=0.6, aspect=12, pad=0.1, label=f"{zlabel} (%)")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close(fig)

    @staticmethod
    def plot_slice(
        result: SliceInterpolationResult,
        *,
        title: Optional[str] = None,
        ylabel: str = "Implied Volatility",
        save_path: Optional[str] = None,
    ) -> None:
        """Visualise a single-expiry implied volatility slice."""
        if result.is_empty:
            warnings.warn("Empty slice result provided to plot_slice.", RuntimeWarning)
            return

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(result.strikes, result.implied_volatility * 100, label="Interpolated IV")
        ax.set_xlabel("Strike")
        ax.set_ylabel(f"{ylabel} (%)")
        ax.set_title(
            title
            or f"Implied Volatility Slice – Time to Expiry {result.time_to_expiry:.4f} years"
        )
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close(fig)

    def _normalise_to_dataframe(
        self,
        data: Union[pd.DataFrame, FormattedSurface],
        option_type: Optional[str],
    ) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            if option_type and "Type" in df.columns:
                df = df[df["Type"].astype(str).str.lower() == option_type.lower()]
            return self._prepare_dataframe(df)

        if not isinstance(data, Mapping):
            raise TypeError(
                "data must be either a pandas DataFrame or a formatted surface dictionary."
            )

        if option_type is None:
            if len(data) == 1:
                option_type = next(iter(data))
            else:
                raise ValueError(
                    "When providing a formatted surface dictionary with multiple option types, "
                    "the option_type argument must be specified."
                )

        if option_type not in data:
            raise KeyError(f"Option type '{option_type}' not found in formatted surface data.")

        frames: list[pd.DataFrame] = []
        for tte, slice_df in data[option_type].items():
            if slice_df is None or slice_df.empty:
                continue

            working = slice_df.copy()
            if "TimeToExpiry" not in working.columns:
                working["TimeToExpiry"] = tte
            elif working["TimeToExpiry"].isna().all():
                working["TimeToExpiry"] = tte

            frames.append(working)

        if not frames:
            return pd.DataFrame(columns=list(REQUIRED_SURFACE_COLUMNS))

        combined = pd.concat(frames, ignore_index=True)
        return self._prepare_dataframe(combined)

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=list(REQUIRED_SURFACE_COLUMNS))

        missing = REQUIRED_SURFACE_COLUMNS.difference(df.columns)
        if missing:
            raise KeyError(f"Input data is missing required columns: {sorted(missing)}")

        prepared = (
            df[list(REQUIRED_SURFACE_COLUMNS)]
            .dropna(subset=list(REQUIRED_SURFACE_COLUMNS))
            .drop_duplicates(subset=["TimeToExpiry", "Strike"])
            .sort_values(["TimeToExpiry", "Strike"])
            .reset_index(drop=True)
        )
        return prepared

    def _filter_by_expiry(
        self,
        df: pd.DataFrame,
        time_to_expiry: float,
        tolerance: float,
    ) -> pd.DataFrame:
        mask = np.isclose(df["TimeToExpiry"], time_to_expiry, atol=tolerance)
        return df.loc[mask].copy()

    def _nearest_expiry(
        self,
        df: pd.DataFrame,
        target_expiry: float,
    ) -> tuple[Optional[pd.DataFrame], float]:
        if df.empty:
            return None, target_expiry

        nearest_idx = (df["TimeToExpiry"] - target_expiry).abs().idxmin()
        nearest_expiry = df.loc[nearest_idx, "TimeToExpiry"]
        nearest_slice = df[df["TimeToExpiry"] == nearest_expiry].copy()

        return nearest_slice, nearest_expiry

    def _interpolate_1d(
        self,
        strikes: np.ndarray,
        values: np.ndarray,
        target_strikes: np.ndarray,
        *,
        method: SliceMethod,
        required_points: int,
    ) -> np.ndarray:
        if method == "cubic":
            if strikes.size < required_points:
                warnings.warn(
                    "Insufficient points for cubic interpolation; falling back to linear.",
                    RuntimeWarning,
                )
                method = "linear"
            else:
                spline = CubicSpline(strikes, values, bc_type="natural")
                return spline(target_strikes)

        # Default to linear interpolation
        return np.interp(target_strikes, strikes, values)

    @staticmethod
    def _expand_range(values: np.ndarray, padding: float) -> tuple[float, float]:
        v_min = float(np.min(values))
        v_max = float(np.max(values))
        if np.isclose(v_min, v_max):
            v_min -= padding
            v_max += padding
        return v_min, v_max