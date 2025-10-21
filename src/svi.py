from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import OptimizeResult, minimize

OptionTypeSurface = Mapping[float, pd.DataFrame]
FormattedSurface = Mapping[str, OptionTypeSurface]
SurfaceInput = Union[pd.DataFrame, OptionTypeSurface, FormattedSurface]

REQUIRED_SLICE_COLUMNS = {"Strike", "ImpliedVolatility", "Forward"}


@dataclass(frozen=True)
class SVIParameters:
    """Container for a calibrated SVI parameter set (a, b, rho, m, sigma)."""
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma], dtype=float)


@dataclass(frozen=True)
class SVICalibrationSettings:
    """
    Numerical settings and thresholds for SVI calibration.
    """
    min_points_per_slice: int = 5
    initial_sigma_floor: float = 0.05
    initial_b_floor: float = 1e-3
    sigma_bound_floor: float = 1e-4
    variance_floor: float = 1e-6
    optimisation_method: str = "L-BFGS-B"
    max_iterations: int = 1_000
    ftol: float = 1e-8
    gtol: float = 1e-7


@dataclass
class SVISliceCalibration:
    """
    Result of calibrating a single expiry slice.
    """
    ttm: float
    parameters: Optional[SVIParameters]
    optimisation: Optional[OptimizeResult]
    message: str = ""
    log_moneyness: Optional[np.ndarray] = None
    total_variance: Optional[np.ndarray] = None


@dataclass
class SVISurfaceCalibration:
    """
    Aggregated calibration outcome across all expiries.
    """
    slices: Dict[float, SVISliceCalibration]

    def parameters_by_ttm(self) -> Dict[float, SVIParameters]:
        return {
            ttm: result.parameters  # type: ignore[misc]
            for ttm, result in self.slices.items()
            if result.parameters is not None
        }

    def successful_ttm(self) -> np.ndarray:
        return np.array(
            [
                ttm
                for ttm, result in self.slices.items()
                if result.parameters is not None and result.optimisation and result.optimisation.success
            ]
        )


@dataclass
class SVIParameterInterpolator:
    """
    Provides smooth parameter curves across maturities using interpolation.
    """
    parameter_functions: Dict[str, interp1d]
    ttms: np.ndarray

    def evaluate(self, ttm: float) -> SVIParameters:
        clipped_ttm = np.clip(ttm, self.ttms.min(), self.ttms.max())
        return SVIParameters(
            a=float(self.parameter_functions["a"](clipped_ttm)),
            b=float(self.parameter_functions["b"](clipped_ttm)),
            rho=float(self.parameter_functions["rho"](clipped_ttm)),
            m=float(self.parameter_functions["m"](clipped_ttm)),
            sigma=float(self.parameter_functions["sigma"](clipped_ttm)),
        )


def svi_total_variance(log_moneyness: np.ndarray, params: SVIParameters, *, variance_floor: float) -> np.ndarray:
    """
    Raw SVI parameterisation for total variance: w(k) = a + b * ( rho*(k-m) + sqrt((k-m)^2 + sigma^2) ).
    Enforces basic stability constraints.
    """
    k = np.asarray(log_moneyness, dtype=float)
    a, b, rho, m, sigma = params.as_array()

    b = max(b, 0.0)
    sigma = max(sigma, variance_floor)
    rho = float(np.clip(rho, -1.0, 1.0))

    root = np.sqrt(np.square(k - m) + sigma**2)
    total_variance = a + b * (rho * (k - m) + root)
    return np.maximum(total_variance, variance_floor)


class SVICalibrator:
    """
    End-to-end SVI calibration pipeline for option volatility smiles/surfaces.

    Accepts:
      • pandas.DataFrame with columns ['TimeToExpiry', 'Strike', 'ImpliedVolatility', 'Forward'].
      • dict[float, DataFrame] produced by OptionDataFormatter.
      • dict[str, dict[float, DataFrame]] when multiple option types are present.

    The calibrator handles per-expiry parameter estimation, interpolation across maturities,
    and provides helpers to evaluate fitted curves.
    """

    def __init__(self, settings: SVICalibrationSettings = SVICalibrationSettings()) -> None:
        self.settings = settings

    def calibrate_surface(
        self,
        data: SurfaceInput,
        *,
        option_type: Optional[str] = None,
        weights: Optional[Mapping[float, np.ndarray]] = None,
        initial_guesses: Optional[Mapping[float, np.ndarray]] = None,
        bounds: Optional[Mapping[float, Tuple[Tuple[float, float], ...]]] = None,
    ) -> SVISurfaceCalibration:
        """
        Calibrate SVI parameters for every expiry available in the input data.
        """
        surface = self._normalise_to_surface_dict(data, option_type)
        calibration_results: Dict[float, SVISliceCalibration] = {}

        for ttm, slice_df in sorted(surface.items()):
            prepared = self._prepare_slice_dataframe(slice_df)
            if prepared.empty or prepared.shape[0] < self.settings.min_points_per_slice:
                warnings.warn(
                    f"Skipping TTM={ttm:.6f}: insufficient valid observations ({prepared.shape[0]}).",
                    RuntimeWarning,
                )
                calibration_results[ttm] = SVISliceCalibration(
                    ttm=ttm,
                    parameters=None,
                    optimisation=None,
                    message="Insufficient data.",
                )
                continue

            forward_price = float(prepared["Forward"].iloc[0])
            strikes = prepared["Strike"].to_numpy(dtype=float)
            implied_vol = prepared["ImpliedVolatility"].to_numpy(dtype=float)

            valid_mask = (strikes > 0.0) & np.isfinite(implied_vol)
            if not np.any(valid_mask):
                warnings.warn(
                    f"Skipping TTM={ttm:.6f}: no valid strike/volatility pairs after filtering.",
                    RuntimeWarning,
                )
                calibration_results[ttm] = SVISliceCalibration(
                    ttm=ttm,
                    parameters=None,
                    optimisation=None,
                    message="No valid data after filtering.",
                )
                continue

            strikes = strikes[valid_mask]
            implied_vol = implied_vol[valid_mask]

            if strikes.size < self.settings.min_points_per_slice:
                warnings.warn(
                    f"Skipping TTM={ttm:.6f}: insufficient data after filtering ({strikes.size}).",
                    RuntimeWarning,
                )
                calibration_results[ttm] = SVISliceCalibration(
                    ttm=ttm,
                    parameters=None,
                    optimisation=None,
                    message="Insufficient data post-filter.",
                )
                continue

            log_moneyness = np.log(strikes / forward_price)
            total_variance = np.square(implied_vol) * ttm

            slice_weights = (
                weights[ttm]
                if weights and ttm in weights
                else np.ones_like(total_variance, dtype=float)
            )
            slice_weights = np.asarray(slice_weights, dtype=float)
            slice_weights = slice_weights / np.sum(slice_weights)

            initial_guess = (
                initial_guesses[ttm]
                if initial_guesses and ttm in initial_guesses
                else self._default_initial_guess(log_moneyness, total_variance)
            )
            slice_bounds = (
                bounds[ttm]
                if bounds and ttm in bounds
                else self._default_bounds(log_moneyness, total_variance)
            )

            optimisation = self._optimise_slice(
                log_moneyness,
                total_variance,
                initial_guess=initial_guess,
                bounds=slice_bounds,
                weights=slice_weights,
            )

            if optimisation.success:
                params = SVIParameters(*optimisation.x)
                message = "Calibration successful."
            else:
                params = SVIParameters(*optimisation.x)
                message = f"Calibration reported failure: {optimisation.message}"
                warnings.warn(
                    f"SVI optimisation at TTM={ttm:.6f} did not converge: {optimisation.message}",
                    RuntimeWarning,
                )

            calibration_results[ttm] = SVISliceCalibration(
                ttm=ttm,
                parameters=params,
                optimisation=optimisation,
                message=message,
                log_moneyness=log_moneyness,
                total_variance=total_variance,
            )

        return SVISurfaceCalibration(calibration_results)

    def build_parameter_interpolator(
        self,
        calibration: SVISurfaceCalibration,
        *,
        kind: str = "linear",
        fill_value: str = "extrapolate",
    ) -> SVIParameterInterpolator:
        """
        Create interpolation functions for each SVI parameter across maturities.
        """
        parameter_map = calibration.parameters_by_ttm()
        if len(parameter_map) < 2:
            raise ValueError("At least two calibrated expiries are required to interpolate parameters.")

        ttms = np.array(sorted(parameter_map.keys()), dtype=float)
        matrix = np.vstack([parameter_map[ttm].as_array() for ttm in ttms])

        columns = ["a", "b", "rho", "m", "sigma"]
        functions = {
            name: interp1d(ttms, matrix[:, idx], kind=kind, fill_value=fill_value)
            for idx, name in enumerate(columns)
        }
        return SVIParameterInterpolator(functions, ttms)

    def evaluate_variance(
        self,
        params: SVIParameters,
        log_moneyness: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate total variance for given SVI parameters and log-moneyness values.
        """
        return svi_total_variance(
            log_moneyness,
            params,
            variance_floor=self.settings.variance_floor,
        )

    def evaluate_implied_volatility(
        self,
        params: SVIParameters,
        log_moneyness: np.ndarray,
        ttm: float,
    ) -> np.ndarray:
        """
        Convert SVI total variance into implied volatility for a fixed maturity.
        """
        ttm = float(ttm)
        if ttm <= 0:
            raise ValueError("Time to maturity must be positive.")
        total_var = self.evaluate_variance(params, log_moneyness)
        return np.sqrt(total_var / ttm)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _optimise_slice(
        self,
        log_moneyness: np.ndarray,
        total_variance: np.ndarray,
        *,
        initial_guess: np.ndarray,
        bounds: Tuple[Tuple[float, float], ...],
        weights: np.ndarray,
    ) -> OptimizeResult:
        def objective(theta: np.ndarray) -> float:
            params = SVIParameters(*theta)
            model_var = self.evaluate_variance(params, log_moneyness)
            residuals = model_var - total_variance
            return float(np.sum(weights * residuals**2))

        return minimize(
            objective,
            x0=initial_guess,
            bounds=bounds,
            method=self.settings.optimisation_method,
            options={
                "maxiter": self.settings.max_iterations,
                "ftol": self.settings.ftol,
                "gtol": self.settings.gtol,
            },
        )

    def _default_initial_guess(
        self,
        log_moneyness: np.ndarray,
        total_variance: np.ndarray,
    ) -> np.ndarray:
        min_var = float(np.min(total_variance))
        max_var = float(np.max(total_variance))
        idx_min = int(np.argmin(np.abs(log_moneyness)))
        atm_var = float(total_variance[idx_min])

        k_spread = float(np.max(log_moneyness) - np.min(log_moneyness))
        sigma_guess = max(self.settings.initial_sigma_floor, k_spread / 4.0)
        b_guess = max((max_var - min_var) / (sigma_guess * 2.0), self.settings.initial_b_floor)

        rho_guess = np.clip(np.sign(total_variance[-1] - total_variance[0]) * 0.5, -0.95, 0.95)
        m_guess = float(log_moneyness[np.argmin(total_variance)])

        return np.array(
            [min_var, b_guess, rho_guess, m_guess, sigma_guess],
            dtype=float,
        )

    def _default_bounds(
        self,
        log_moneyness: np.ndarray,
        total_variance: np.ndarray,
    ) -> Tuple[Tuple[float, float], Tuple[float, None], Tuple[float, float], Tuple[float, float], Tuple[float, None]]:
        max_var = float(np.max(total_variance))
        k_min = float(np.min(log_moneyness))
        k_max = float(np.max(log_moneyness))

        return (
            (self.settings.variance_floor, max_var * 1.5),  # a
            (1e-6, None),                                   # b
            (-0.999, 0.999),                                # rho
            (k_min - abs(k_min), k_max + abs(k_max)),       # m
            (self.settings.sigma_bound_floor, None),        # sigma
        )

    def _prepare_slice_dataframe(self, slice_df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            "Implied Volatility": "ImpliedVolatility",
            "ForwardPrice": "Forward",
            "TimeToExpiry": "TimeToExpiry",
        }
        df = slice_df.rename(columns=rename_map).copy()

        missing = REQUIRED_SLICE_COLUMNS.difference(df.columns)
        if missing:
            raise KeyError(f"Slice is missing required columns: {sorted(missing)}")

        df = df[list(REQUIRED_SLICE_COLUMNS)]
        df = df.dropna(subset=list(REQUIRED_SLICE_COLUMNS))
        df = df[df["ImpliedVolatility"] > 0.0]
        df = df.drop_duplicates(subset=["Strike"])
        df.sort_values("Strike", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _normalise_to_surface_dict(
        self,
        data: SurfaceInput,
        option_type: Optional[str],
    ) -> Dict[float, pd.DataFrame]:
        if isinstance(data, pd.DataFrame):
            return self._surface_from_dataframe(data, option_type)

        if isinstance(data, Mapping):
            return self._surface_from_mapping(data, option_type)

        raise TypeError("Unsupported data type for SVI calibration input.")

    def _surface_from_dataframe(
        self,
        df: pd.DataFrame,
        option_type: Optional[str],
    ) -> Dict[float, pd.DataFrame]:
        rename_map = {"Implied Volatility": "ImpliedVolatility"}
        working = df.rename(columns=rename_map).copy()

        if option_type and "Type" in working.columns:
            working = working[working["Type"].astype(str).str.lower() == option_type.lower()]

        required = REQUIRED_SLICE_COLUMNS.union({"TimeToExpiry"})
        missing = required.difference(working.columns)
        if missing:
            raise KeyError(f"DataFrame must contain: {sorted(required)}. Missing: {sorted(missing)}")

        grouped: Dict[float, pd.DataFrame] = {}
        for ttm, group in working.groupby("TimeToExpiry"):
            grouped[float(ttm)] = group[list(REQUIRED_SLICE_COLUMNS)].copy()

        if not grouped:
            raise ValueError("No data remaining after grouping by TimeToExpiry.")
        return grouped

    def _surface_from_mapping(
        self,
        mapping: Mapping[Any, Any],
        option_type: Optional[str],
    ) -> Dict[float, pd.DataFrame]:
        # Handle the case where mapping is already Dict[float, DataFrame]
        if all(isinstance(key, (float, int)) for key in mapping.keys()):
            return {float(ttm): df.copy() for ttm, df in mapping.items()}

        # Otherwise expect the OptionDataFormatter nested dictionary
        if option_type is None:
            if len(mapping) != 1:
                raise ValueError(
                    "Multiple option types detected; please specify option_type to select one."
                )
            option_type = next(iter(mapping.keys()))

        option_key = option_type.lower()
        for key in mapping.keys():
            if isinstance(key, str) and key.lower() == option_key:
                surface = mapping[key]
                return {float(ttm): df.copy() for ttm, df in surface.items()}

        raise KeyError(f"Option type '{option_type}' not found in supplied mapping.")