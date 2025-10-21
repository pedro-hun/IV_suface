from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cubic_interpolation import (
    SliceGridConfig,
    SliceInterpolationResult,
    SurfaceGridConfig,
    SurfaceInterpolationResult,
    VolatilityInterpolator,
)
from svi import (
    SVICalibrationSettings,
    SVICalibrator,
    SVISurfaceCalibration,
    SVIParameters,
)

# --------------------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Configuration dataclasses
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnMapping:
    strike: str = "Strike"
    implied_vol: str = "IV"
    expiry: str = "Expiry"
    option_type: str = "Type"
    forward: Optional[str] = "Forward"
    spot: Optional[str] = "SpotPrice"
    bid: Optional[str] = "bid"
    ask: Optional[str] = "ask"
    last_price: Optional[str] = "lastPrice"
    volume: Optional[str] = "volume"
    open_interest: Optional[str] = "openInterest"


@dataclass(frozen=True)
class ExcelLoadConfig:
    path: Path
    sheet_name: str
    header_row: int = 0
    column_mapping: ColumnMapping = ColumnMapping()
    date_format: str = "%d/%m/%Y"


@dataclass(frozen=True)
class TimeToExpiryConfig:
    valuation_date: str
    use_business_days: bool = True
    holidays_file: Optional[Path] = None
    holidays_column: Optional[str] = None
    trading_days_in_year: int = 252
    calendar_days_in_year: int = 365
    allow_negative_dte: bool = False


@dataclass(frozen=True)
class FilterConfig:
    option_types: Optional[Sequence[str]] = None  # e.g. ("call", "put")
    min_volume: int = 0
    min_open_interest: int = 0
    min_iv: float = 0.01
    max_iv: float = 2.0
    min_mid_price: float = 0.0
    min_days_to_expiry: int = 1
    max_days_to_expiry: Optional[int] = None
    max_relative_spread: Optional[float] = 0.50
    drop_na_columns: Sequence[str] = (
        "Strike",
        "ImpliedVolatility",
        "Expiry",
        "Type",
        "TimeToExpiry",
    )


@dataclass(frozen=True)
class ForwardCurveConfig:
    risk_free_rate: float = 0.15
    compounding: str = "simple"  # "simple" or "continuous"


@dataclass(frozen=True)
class VisualizationConfig:
    option_type: str = "call"
    slice_expiry: Optional[float] = None  # in years; nearest available will be used
    show_plots: bool = True
    export_figures: bool = False
    save_directory: Optional[Path] = None

    def figure_path(self, filename: str) -> Optional[Path]:
        if not self.export_figures or self.save_directory is None:
            return None
        return self.save_directory / filename


@dataclass(frozen=True)
class PipelineConfig:
    load: ExcelLoadConfig
    time: TimeToExpiryConfig
    filters: FilterConfig = FilterConfig()
    forward: ForwardCurveConfig = ForwardCurveConfig()
    cubic_surface: SurfaceGridConfig = SurfaceGridConfig()
    cubic_slice: SliceGridConfig = SliceGridConfig()
    svi_settings: SVICalibrationSettings = SVICalibrationSettings()
    visualization: VisualizationConfig = VisualizationConfig()


@dataclass
class PipelineOutputs:
    clean_data: pd.DataFrame
    option_type_surface: Dict[float, pd.DataFrame]
    cubic_surface: SurfaceInterpolationResult
    cubic_slice: SliceInterpolationResult
    svi_calibration: SVISurfaceCalibration
    svi_surface: np.ndarray
    time_grid: np.ndarray
    strike_grid: np.ndarray
    target_expiry: float
    svi_slice_iv: Optional[np.ndarray]


# --------------------------------------------------------------------------------------
# Data ingestion and preparation
# --------------------------------------------------------------------------------------


def load_raw_dataframe(config: ExcelLoadConfig) -> pd.DataFrame:
    logger.info("Loading Excel file: %s (sheet=%s)", config.path, config.sheet_name)
    df = pd.read_excel(
        io=config.path,
        sheet_name=config.sheet_name,
        header=config.header_row,
        engine="openpyxl",
    )

    rename_map: Dict[str, str] = {
        config.column_mapping.strike: "Strike",
        config.column_mapping.implied_vol: "ImpliedVolatility",
        config.column_mapping.expiry: "Expiry",
        config.column_mapping.option_type: "Type",
    }
    optional_fields = {
        config.column_mapping.forward: "Forward",
        config.column_mapping.spot: "SpotPrice",
        config.column_mapping.bid: "Bid",
        config.column_mapping.ask: "Ask",
        config.column_mapping.last_price: "LastPrice",
        config.column_mapping.volume: "Volume",
        config.column_mapping.open_interest: "OpenInterest",
    }
    rename_map.update({src: dst for src, dst in optional_fields.items() if src})

    df = df.rename(columns=rename_map)

    required = {"Strike", "ImpliedVolatility", "Expiry", "Type"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns after renaming: {sorted(missing)}")

    logger.debug("Raw dataframe shape after loading: %s", df.shape)
    return df


def _load_holidays(config: TimeToExpiryConfig) -> Optional[np.ndarray]:
    if config.holidays_file is None:
        return None

    holidays_path = config.holidays_file
    if not holidays_path.exists():
        logger.warning("Holidays file %s not found. Proceeding without holidays.", holidays_path)
        return None

    holiday_df = pd.read_csv(holidays_path)
    column = config.holidays_column or holiday_df.columns[0]
    dates = pd.to_datetime(holiday_df[column], errors="coerce").dropna()
    if dates.empty:
        logger.warning("No valid holiday dates parsed from %s.", holidays_path)
        return None
    return dates.dt.date.astype("datetime64[D]").to_numpy()


def compute_time_to_expiry(df: pd.DataFrame, config: TimeToExpiryConfig) -> pd.DataFrame:
    valuation_date = pd.Timestamp(config.valuation_date).normalize()
    expiry_series = pd.to_datetime(df["Expiry"], format=config.date_format, errors="coerce")

    df = df.assign(Expiry=expiry_series)
    invalid_dates = df["Expiry"].isna().sum()
    if invalid_dates:
        logger.warning("Dropped %d rows with invalid expiry dates.", invalid_dates)
        df = df.dropna(subset=["Expiry"])

    if df.empty:
        return df

    if config.use_business_days:
        holidays = _load_holidays(config)
        start = np.full(df.shape[0], valuation_date.date(), dtype="datetime64[D]")
        end = df["Expiry"].dt.date.astype("datetime64[D]")
        if holidays is not None:
            bdays = np.busday_count(start, end, holidays=holidays)
        else:
            bdays = np.busday_count(start, end)
        df["DaysToExpiry"] = bdays
        df["TimeToExpiry"] = bdays / config.trading_days_in_year
    else:
        delta_days = (df["Expiry"] - valuation_date).dt.days.astype(float)
        df["DaysToExpiry"] = delta_days
        df["TimeToExpiry"] = delta_days / config.calendar_days_in_year

    if not config.allow_negative_dte:
        before_filter = df.shape[0]
        df = df[df["DaysToExpiry"] >= 0]
        logger.debug("Removed %d rows with negative days-to-expiry.", before_filter - df.shape[0])

    return df.reset_index(drop=True)


def ensure_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_mid_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    if {"Bid", "Ask"}.issubset(df.columns):
        df["MidPrice"] = (df["Bid"] + df["Ask"]) / 2.0
        df["Spread"] = df["Ask"] - df["Bid"]
        df["RelativeSpread"] = np.where(
            df["MidPrice"] > 0,
            df["Spread"] / df["MidPrice"],
            np.nan,
        )
    return df


def add_forward_column(df: pd.DataFrame, config: ForwardCurveConfig) -> pd.DataFrame:
    if "Forward" in df.columns and df["Forward"].notna().any():
        return df

    if "SpotPrice" not in df.columns:
        logger.warning("Forward column missing and SpotPrice unavailable. Forward will remain NaN.")
        return df

    rate = config.risk_free_rate
    t = df["TimeToExpiry"].to_numpy(dtype=float)

    if config.compounding.lower() == "continuous":
        forwards = df["SpotPrice"] * np.exp(rate * t)
    else:
        forwards = df["SpotPrice"] * np.power((1.0 + rate), t)

    df["Forward"] = forwards
    return df


def apply_filters(df: pd.DataFrame, config: FilterConfig) -> pd.DataFrame:
    before = df.shape[0]

    if config.option_types is not None:
        valid_types = {opt.lower() for opt in config.option_types}
        df = df[df["Type"].str.lower().isin(valid_types)]

    if "Volume" in df.columns:
        df = df[df["Volume"] >= config.min_volume]

    if "OpenInterest" in df.columns:
        df = df[df["OpenInterest"] >= config.min_open_interest]

    df = df[(df["ImpliedVolatility"] >= config.min_iv) & (df["ImpliedVolatility"] <= config.max_iv)]

    if config.max_relative_spread is not None and "RelativeSpread" in df.columns:
        df = df[df["RelativeSpread"] <= config.max_relative_spread]

    if "MidPrice" in df.columns:
        df = df[df["MidPrice"] >= config.min_mid_price]

    if config.min_days_to_expiry is not None:
        df = df[df["DaysToExpiry"] >= config.min_days_to_expiry]

    if config.max_days_to_expiry is not None:
        df = df[df["DaysToExpiry"] <= config.max_days_to_expiry]

    if config.drop_na_columns:
        df = df.dropna(subset=list(config.drop_na_columns), how="any")

    logger.info("Applied filters: kept %d of %d rows.", df.shape[0], before)
    return df.reset_index(drop=True)


def prepare_dataframe(
    config: PipelineConfig,
) -> pd.DataFrame:
    df = load_raw_dataframe(config.load)
    df["Type"] = df["Type"].astype(str).str.lower().str.strip()

    numeric_columns = [
        "Strike",
        "ImpliedVolatility",
        "Forward",
        "SpotPrice",
        "Bid",
        "Ask",
        "LastPrice",
        "Volume",
        "OpenInterest",
    ]
    df = ensure_numeric(df, numeric_columns)
    df = compute_time_to_expiry(df, config.time)
    df = add_mid_price_columns(df)
    df = add_forward_column(df, config.forward)
    df = apply_filters(df, config.filters)
    df = df.sort_values(["TimeToExpiry", "Type", "Strike"]).reset_index(drop=True)
    logger.info("Prepared dataframe with shape: %s", df.shape)
    return df


# --------------------------------------------------------------------------------------
# Surface preparation utilities
# --------------------------------------------------------------------------------------


def surface_by_option_type(df: pd.DataFrame) -> Dict[str, Dict[float, pd.DataFrame]]:
    result: Dict[str, Dict[float, pd.DataFrame]] = {}
    grouped_types = df.groupby("Type")

    for opt_type, type_frame in grouped_types:
        slices: Dict[float, pd.DataFrame] = {}
        for ttm, slice_df in type_frame.groupby("TimeToExpiry"):
            cleaned = (
                slice_df[["Strike", "ImpliedVolatility", "Forward"]]
                .dropna()
                .drop_duplicates(subset=["Strike"])
                .sort_values("Strike")
                .reset_index(drop=True)
            )
            if cleaned.empty:
                continue
            cleaned["TimeToExpiry"] = float(ttm)
            slices[float(ttm)] = cleaned
        if slices:
            result[opt_type] = slices
    return result


def build_forward_map(option_surface: Mapping[float, pd.DataFrame]) -> Dict[float, float]:
    forward_map: Dict[float, float] = {}
    for ttm, frame in option_surface.items():
        if "Forward" in frame.columns and frame["Forward"].notna().any():
            forward_map[ttm] = float(frame["Forward"].median())
    return forward_map


def forward_curve_from_map(forward_map: Mapping[float, float]) -> Callable[[float], float]:
    if not forward_map:
        raise ValueError("Forward map is empty; cannot create forward curve.")

    ttms = np.array(sorted(forward_map.keys()), dtype=float)
    forwards = np.array([forward_map[t] for t in ttms], dtype=float)

    if ttms.size == 1:
        level = forwards[0]

        def constant_forward(_: float) -> float:
            return level

        return constant_forward

    def interpolated_forward(target_ttm: float) -> float:
        return float(np.interp(target_ttm, ttms, forwards, left=forwards[0], right=forwards[-1]))

    return interpolated_forward


def build_market_point_matrix(option_surface: Mapping[float, pd.DataFrame]) -> Optional[np.ndarray]:
    points: List[np.ndarray] = []
    for ttm, frame in option_surface.items():
        if frame.empty:
            continue
        strikes = frame["Strike"].to_numpy(dtype=float)
        ivs = frame["ImpliedVolatility"].to_numpy(dtype=float)
        valid_mask = np.isfinite(strikes) & np.isfinite(ivs)
        if not np.any(valid_mask):
            continue
        ttm_column = np.full(valid_mask.sum(), ttm, dtype=float)
        points.append(np.column_stack((ttm_column, strikes[valid_mask], ivs[valid_mask])))
    if not points:
        return None
    return np.vstack(points)


def create_grid_from_market(
    option_surface: Mapping[float, pd.DataFrame],
    config: SurfaceGridConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    ttms = np.array(sorted(option_surface.keys()), dtype=float)
    strikes = np.concatenate(
        [frame["Strike"].to_numpy(dtype=float) for frame in option_surface.values() if not frame.empty],
        dtype=float,
    )

    time_axis = np.linspace(ttms.min(), ttms.max(), config.n_expiries)
    strike_axis = np.linspace(np.min(strikes), np.max(strikes), config.n_strikes)
    time_grid, strike_grid = np.meshgrid(time_axis, strike_axis)
    return time_grid, strike_grid


def choose_target_expiry(
    option_surface: Mapping[float, pd.DataFrame],
    requested_expiry: Optional[float],
) -> float:
    available = np.array(sorted(option_surface.keys()), dtype=float)
    if available.size == 0:
        raise ValueError("No expiries available in formatted surface.")

    if requested_expiry is None:
        median_index = available.size // 2
        return float(available[median_index])

    idx = int(np.argmin(np.abs(available - requested_expiry)))
    chosen = float(available[idx])
    if not np.isclose(chosen, requested_expiry):
        logger.info(
            "Requested slice expiry %.4f not found. Using nearest available expiry %.4f.",
            requested_expiry,
            chosen,
        )
    return chosen


# --------------------------------------------------------------------------------------
# SVI evaluation helpers
# --------------------------------------------------------------------------------------


def build_svi_parameter_resolver(
    calibrator: SVICalibrator,
    calibration: SVISurfaceCalibration,
) -> Callable[[float], Optional[SVIParameters]]:
    parameter_map = calibration.parameters_by_ttm()
    if not parameter_map:
        return lambda _: None

    available_ttms = np.array(sorted(parameter_map.keys()), dtype=float)

    try:
        interpolator = calibrator.build_parameter_interpolator(calibration)
    except ValueError:
        interpolator = None

    def resolver(ttm: float) -> Optional[SVIParameters]:
        if interpolator is not None and available_ttms.size >= 2:
            return interpolator.evaluate(ttm)
        nearest_idx = int(np.argmin(np.abs(available_ttms - ttm)))
        return parameter_map.get(float(available_ttms[nearest_idx]))

    return resolver


def evaluate_svi_surface_on_grid(
    calibrator: SVICalibrator,
    resolver: Callable[[float], Optional[SVIParameters]],
    forward_curve: Callable[[float], float],
    time_grid: np.ndarray,
    strike_grid: np.ndarray,
) -> np.ndarray:
    iv_surface = np.full_like(time_grid, np.nan, dtype=float)

    it = np.ndindex(time_grid.shape)
    for idx in it:
        ttm = float(time_grid[idx])
        strike = float(strike_grid[idx])
        if ttm <= 0 or strike <= 0:
            continue

        params = resolver(ttm)
        if params is None:
            continue

        forward = forward_curve(ttm)
        if forward <= 0:
            continue

        log_moneyness = np.log(strike / forward)
        try:
            iv = calibrator.evaluate_implied_volatility(params, np.array([log_moneyness]), ttm)[0]
            if np.isfinite(iv):
                iv_surface[idx] = iv
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("SVI evaluation failed at t=%.4f, K=%.4f: %s", ttm, strike, exc)

    return iv_surface


def evaluate_svi_slice(
    calibrator: SVICalibrator,
    resolver: Callable[[float], Optional[SVIParameters]],
    forward_curve: Callable[[float], float],
    time_to_expiry: float,
    strikes: np.ndarray,
) -> Optional[np.ndarray]:
    params = resolver(time_to_expiry)
    if params is None:
        return None

    forward = forward_curve(time_to_expiry)
    if forward <= 0:
        logger.warning("Forward price non-positive at t=%.4f. Skipping SVI slice.", time_to_expiry)
        return None

    strike_array = np.asarray(strikes, dtype=float)
    valid_mask = strike_array > 0
    if not np.any(valid_mask):
        return None

    log_moneyness = np.log(strike_array[valid_mask] / forward)
    iv_values = calibrator.evaluate_implied_volatility(params, log_moneyness, time_to_expiry)

    slice_iv = np.full_like(strike_array, np.nan, dtype=float)
    slice_iv[valid_mask] = iv_values
    return slice_iv


def build_expiry_details(
    calibration: SVISurfaceCalibration,
    option_surface: Mapping[float, pd.DataFrame],
) -> List[Dict[str, Any]]:
    details: List[Dict[str, Any]] = []
    for ttm, slice_result in calibration.slices.items():
        params = slice_result.parameters
        market = option_surface.get(ttm)
        if params is None or market is None or market.empty:
            continue
        detail = {
            "T": ttm,
            "params": tuple(params.as_array()),
            "market_data": market.copy(),
            "forward": float(market["Forward"].median()) if "Forward" in market.columns else np.nan,
            "success": bool(slice_result.optimisation.success) if slice_result.optimisation else False,
        }
        details.append(detail)
    return details


# --------------------------------------------------------------------------------------
# Plotting utilities
# --------------------------------------------------------------------------------------


def plot_3d_surface(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    title: str,
    *,
    xlabel: str = "Time to Expiry",
    ylabel: str = "Strike",
    zlabel: str = "Implied Volatility",
    market_points: Optional[np.ndarray] = None,
    show: bool,
    save_path: Optional[Path] = None,
) -> None:
    if not (isinstance(X, np.ndarray) and isinstance(Y, np.ndarray) and isinstance(Z, np.ndarray)):
        logger.warning("Skipping 3D plot '%s': X, Y, Z must all be numpy arrays.", title)
        return
    if not (X.shape == Y.shape == Z.shape):
        logger.warning("Skipping 3D plot '%s': X, Y, Z must align in shape.", title)
        return
    if X.size == 0 or Y.size == 0 or Z.size == 0:
        logger.warning("Skipping 3D plot '%s': arrays cannot be empty.", title)
        return
    if np.all(np.isnan(Z)):
        logger.warning("Skipping 3D plot '%s': Z contains only NaN values.", title)
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    Z_masked = np.ma.masked_invalid(Z)
    surf = ax.plot_surface(
        X,
        Y,
        Z_masked,
        cmap="viridis",
        edgecolor="none",
        rstride=1,
        cstride=1,
        alpha=0.9,
        linewidth=0,
        antialiased=True,
    )

    if market_points is not None:
        valid = market_points[np.all(np.isfinite(market_points), axis=1)]
        if valid.size:
            ax.scatter(
                valid[:, 0],
                valid[:, 1],
                valid[:, 2],
                color="black",
                s=20,
                label="Market",
                depthshade=False,
            )
            ax.legend(loc="upper right")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title, fontsize=14)

    z_finite = Z[np.isfinite(Z)]
    if z_finite.size:
        z_min, z_max = z_finite.min(), z_finite.max()
        ax.set_zlim(max(0.0, z_min * 0.9), z_max * 1.1)

    ax.view_init(elev=25.0, azim=-130.0)
    cbar = fig.colorbar(surf, shrink=0.6, aspect=10, pad=0.1)
    cbar.set_label(zlabel)

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        logger.info("Saved plot '%s' to %s", title, save_path)

    if show:
        plt.show()

    plt.close(fig)


@dataclass(frozen=True)
class SliceCurve:
    label: str
    strikes: np.ndarray
    implied_vols: np.ndarray
    style: str = "-"
    color: Optional[str] = None


def plot_slice_comparison(
    market_data: pd.DataFrame,
    curves: Sequence[SliceCurve],
    *,
    time_to_expiry: float,
    title: Optional[str] = None,
    xlabel: str = "Strike",
    ylabel: str = "Implied Volatility",
    show: bool,
    save_path: Optional[Path] = None,
) -> None:
    if market_data.empty and not curves:
        logger.warning("Skipping slice plot: no market data or model curves provided.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    if not market_data.empty:
        ax.scatter(
            market_data["Strike"],
            market_data["ImpliedVolatility"],
            label="Market",
            color="black",
            marker="o",
            alpha=0.75,
        )

    for curve in curves:
        valid_mask = np.isfinite(curve.strikes) & np.isfinite(curve.implied_vols)
        if not np.any(valid_mask):
            continue
        ax.plot(
            curve.strikes[valid_mask],
            curve.implied_vols[valid_mask],
            curve.style,
            color=curve.color,
            label=curve.label,
            linewidth=2.0,
        )

    default_title = f"IV Slice at T = {time_to_expiry:.4f} years"
    ax.set_title(title or default_title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        logger.info("Saved slice comparison plot to %s", save_path)

    if show:
        plt.show()

    plt.close(fig)


def plot_svi_parameter_evolution(
    expiry_details: Sequence[Mapping[str, Any]],
    *,
    show: bool,
    save_path: Optional[Path] = None,
) -> None:
    valid = [
        detail for detail in expiry_details
        if isinstance(detail.get("T"), (float, int))
        and isinstance(detail.get("params"), (tuple, list, np.ndarray))
        and len(detail["params"]) == 5
    ]

    if len(valid) < 2:
        logger.info("Insufficient SVI slices for parameter evolution plot (need >=2, found %d).", len(valid))
        return

    valid = sorted(valid, key=lambda d: d["T"])
    ttms = np.array([float(d["T"]) for d in valid], dtype=float)
    params = np.array([np.asarray(d["params"], dtype=float) for d in valid])

    m_values = params[:, 3]
    sigma_values = params[:, 4]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color_m = "tab:red"
    ax1.set_xlabel("Time to Expiry (Years)")
    ax1.set_ylabel("SVI parameter m", color=color_m)
    ax1.plot(ttms, m_values, color=color_m, marker="o", linestyle="-", label="m")
    ax1.tick_params(axis="y", labelcolor=color_m)
    ax1.grid(True, linestyle=":", alpha=0.65)

    ax2 = ax1.twinx()
    color_sigma = "tab:blue"
    ax2.set_ylabel("SVI parameter sigma", color=color_sigma)
    ax2.plot(ttms, sigma_values, color=color_sigma, marker="x", linestyle="--", label="sigma")
    ax2.tick_params(axis="y", labelcolor=color_sigma)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")

    fig.suptitle("Evolution of SVI Parameters (m & sigma)")
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        logger.info("Saved SVI parameter evolution plot to %s", save_path)

    if show:
        plt.show()

    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main orchestration
# --------------------------------------------------------------------------------------


def run_pipeline(config: PipelineConfig) -> PipelineOutputs:
    clean_df = prepare_dataframe(config)

    surfaces = surface_by_option_type(clean_df)
    option_type = config.visualization.option_type.lower()
    if option_type not in surfaces:
        raise ValueError(f"Option type '{option_type}' not available. Found: {sorted(surfaces.keys())}")

    option_surface = surfaces[option_type]
    target_expiry = choose_target_expiry(option_surface, config.visualization.slice_expiry)
    forward_map = build_forward_map(option_surface)
    forward_curve = forward_curve_from_map(forward_map)

    vol_interpolator = VolatilityInterpolator(
        surface_config=config.cubic_surface,
        slice_config=config.cubic_slice,
    )
    cubic_surface = vol_interpolator.interpolate_surface(surfaces, option_type=option_type)
    cubic_slice = vol_interpolator.interpolate_slice(
        surfaces,
        time_to_expiry=target_expiry,
        option_type=option_type,
    )

    if cubic_surface.is_empty:
        logger.warning("Cubic interpolation surface returned empty. Building grid from market data.")
        time_grid, strike_grid = create_grid_from_market(option_surface, config.cubic_surface)
        cubic_iv_surface = np.full_like(time_grid, np.nan, dtype=float)
    else:
        time_grid = cubic_surface.time_grid
        strike_grid = cubic_surface.strike_grid
        cubic_iv_surface = cubic_surface.implied_volatility

    calibrator = SVICalibrator(settings=config.svi_settings)
    svi_calibration = calibrator.calibrate_surface(surfaces, option_type=option_type)
    parameter_resolver = build_svi_parameter_resolver(calibrator, svi_calibration)
    svi_iv_surface = evaluate_svi_surface_on_grid(
        calibrator,
        parameter_resolver,
        forward_curve,
        time_grid,
        strike_grid,
    )

    if cubic_slice.is_empty:
        market_slice = option_surface[target_expiry]
        strike_axis = market_slice["Strike"].to_numpy(dtype=float)
        cubic_slice_result = SliceInterpolationResult.empty(target_expiry)
        logger.warning("Cubic slice interpolation failed; using market strikes only.")
    else:
        strike_axis = cubic_slice.strikes
        cubic_slice_result = cubic_slice

    svi_slice_iv = evaluate_svi_slice(
        calibrator,
        parameter_resolver,
        forward_curve,
        target_expiry,
        strike_axis,
    )

    market_points = build_market_point_matrix(option_surface)

    if config.visualization.show_plots or config.visualization.export_figures:
        plot_3d_surface(
            time_grid,
            strike_grid,
            cubic_iv_surface,
            title="Cubic Spline Implied Volatility Surface",
            xlabel="Time to Expiry (Years)",
            ylabel="Strike",
            zlabel="Implied Volatility",
            market_points=market_points,
            show=config.visualization.show_plots,
            save_path=config.visualization.figure_path("cubic_surface.png"),
        )

        plot_3d_surface(
            time_grid,
            strike_grid,
            svi_iv_surface,
            title="SVI Calibrated Implied Volatility Surface",
            xlabel="Time to Expiry (Years)",
            ylabel="Strike",
            zlabel="Implied Volatility",
            market_points=market_points,
            show=config.visualization.show_plots,
            save_path=config.visualization.figure_path("svi_surface.png"),
        )

        market_slice_df = option_surface[target_expiry].copy()
        curves: List[SliceCurve] = []

        if not cubic_slice_result.is_empty:
            curves.append(
                SliceCurve(
                    label="Cubic spline",
                    strikes=cubic_slice_result.strikes,
                    implied_vols=cubic_slice_result.implied_volatility,
                    style="-",
                    color="tab:blue",
                )
            )

        if svi_slice_iv is not None:
            curves.append(
                SliceCurve(
                    label="SVI",
                    strikes=strike_axis,
                    implied_vols=svi_slice_iv,
                    style="--",
                    color="tab:red",
                )
            )

        plot_slice_comparison(
            market_slice_df,
            curves,
            time_to_expiry=target_expiry,
            show=config.visualization.show_plots,
            save_path=config.visualization.figure_path("slice_comparison.png"),
        )

        expiry_details = build_expiry_details(svi_calibration, option_surface)
        plot_svi_parameter_evolution(
            expiry_details,
            show=config.visualization.show_plots,
            save_path=config.visualization.figure_path("svi_parameter_evolution.png"),
        )

    return PipelineOutputs(
        clean_data=clean_df,
        option_type_surface=option_surface,
        cubic_surface=cubic_surface,
        cubic_slice=cubic_slice_result,
        svi_calibration=svi_calibration,
        svi_surface=svi_iv_surface,
        time_grid=time_grid,
        strike_grid=strike_grid,
        target_expiry=target_expiry,
        svi_slice_iv=svi_slice_iv,
    )


# --------------------------------------------------------------------------------------
# Example configuration
# --------------------------------------------------------------------------------------


def build_default_config() -> PipelineConfig:
    return PipelineConfig(
        load=ExcelLoadConfig(
            path=Path("data/options_snapshot.xlsx"),  # TODO: update to your Excel path
            sheet_name="Sheet1",
            header_row=0,
            column_mapping=ColumnMapping(
                strike="Strike",
                implied_vol="IV",
                expiry="Expiry",
                option_type="Type",
                forward="Forward",
                spot="SpotPrice",
                bid="Bid",
                ask="Ask",
                last_price="LastPrice",
                volume="Volume",
                open_interest="OpenInterest",
            ),
            date_format="%d/%m/%Y",
        ),
        time=TimeToExpiryConfig(
            valuation_date="2025-10-17",
            use_business_days=True,
            holidays_file=Path("data/holidays.csv"),  # optional
            holidays_column="Date",
            trading_days_in_year=252,
        ),
        filters=FilterConfig(
            option_types=("call", "put"),
            min_volume=1,
            min_open_interest=0,
            min_iv=0.01,
            max_iv=2.0,
            min_mid_price=0.05,
            min_days_to_expiry=5,
            max_days_to_expiry=None,
            max_relative_spread=0.50,
        ),
        forward=ForwardCurveConfig(
            risk_free_rate=0.15,
            compounding="simple",
        ),
        cubic_surface=SurfaceGridConfig(
            n_strikes=120,
            n_expiries=80,
            method="cubic",
            min_points=4,
            strike_padding=1.5,
            expiry_padding=0.02,
        ),
        cubic_slice=SliceGridConfig(
            n_points=200,
            method="cubic",
            tolerance=1e-4,
            min_points=3,
            use_nearest=True,
        ),
        svi_settings=SVICalibrationSettings(
            min_points_per_slice=5,
            initial_sigma_floor=0.05,
            initial_b_floor=1e-3,
            sigma_bound_floor=1e-4,
            variance_floor=1e-6,
            optimisation_method="L-BFGS-B",
            max_iterations=1_000,
            ftol=1e-8,
            gtol=1e-7,
        ),
        visualization=VisualizationConfig(
            option_type="call",
            slice_expiry=None,
            show_plots=True,
            export_figures=False,
            save_directory=Path("outputs/figures"),
        ),
    )


# --------------------------------------------------------------------------------------
# Script entry point
# --------------------------------------------------------------------------------------


def main() -> None:
    config = build_default_config()

    if not config.load.path.exists():
        logger.error(
            "Excel file not found at %s. Please update the path in build_default_config().",
            config.load.path,
        )
        return

    if config.visualization.save_directory and not config.visualization.save_directory.exists():
        config.visualization.save_directory.mkdir(parents=True, exist_ok=True)

    outputs = run_pipeline(config)
    logger.info(
        "Pipeline complete. Processed %d rows; generated %d expiries for option type '%s'.",
        outputs.clean_data.shape[0],
        len(outputs.option_type_surface),
        config.visualization.option_type.lower(),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    main()