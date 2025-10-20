from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import xlwings as xw


@dataclass(frozen=True)
class ExcelColumnConfig:
    """Typed container for all Excel column references used by ExcelReader."""
    implied_volatility: str
    bid: str
    ask: str
    last_price: str
    strike: str
    open_interest: str
    volume: str
    ticker: str
    maturity: str
    spot_price: str


class ExcelReader:
    """Utility class for reading and enriching option data from Excel workbooks."""

    CALL_TICKER_CODES = set("ABCDEFGHIJKL")
    PUT_TICKER_CODES = set("MNOPQRSTUVWX")

    def __init__(
        self,
        excel_file: str,
        sheet_name: str,
        row_start: int,
        row_end: int,
        column_config: ExcelColumnConfig,
    ) -> None:
        if row_start > row_end:
            raise ValueError("row_start must be less than or equal to row_end.")

        self.excel_file = excel_file
        self.sheet_name = sheet_name
        self.row_start = row_start
        self.row_end = row_end
        self.columns = column_config

        self._workbook: Optional[xw.Book] = None
        self._sheet: Optional[xw.Sheet] = None
        self._data: Optional[pd.DataFrame] = None

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying Excel workbook if it is open."""
        if self._workbook is not None:
            try:
                self._workbook.close()
            finally:
                self._workbook = None
                self._sheet = None

    def load_data(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Read the configured columns from Excel and return a cleaned dataframe.

        Parameters
        ----------
        force_reload:
            When True, ignore any cached dataframe and re-query Excel.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the raw option data.
        """
        if not force_reload and self._data is not None:
            return self._data.copy()

        sheet = self._ensure_sheet()

        raw_columns = {
            "bid": self._read_column(sheet, self.columns.bid),
            "ask": self._read_column(sheet, self.columns.ask),
            "lastPrice": self._read_column(sheet, self.columns.last_price),
            "Strike": self._read_column(sheet, self.columns.strike),
            "openInterest": self._read_column(sheet, self.columns.open_interest),
            "volume": self._read_column(sheet, self.columns.volume),
            "ticker": self._read_column(sheet, self.columns.ticker),
            "Expiry": self._read_column(sheet, self.columns.maturity),
            "IV": self._read_column(sheet, self.columns.implied_volatility),
            "SpotPrice": self._read_column(sheet, self.columns.spot_price),
        }

        data = pd.DataFrame(raw_columns).dropna(how="all")
        self._data = data
        return data.copy()

    def attach_option_type(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Add an 'option_type' column (call/put) inferred from the ticker symbol.

        Parameters
        ----------
        force_reload:
            When True, reload the base data from Excel.

        Returns
        -------
        pd.DataFrame
            Dataframe with the new 'option_type' column.
        """
        data = self.load_data(force_reload=force_reload)

        def infer_type(ticker: Optional[str]) -> Optional[str]:
            if not isinstance(ticker, str) or len(ticker) < 5:
                return None
            code = ticker[4].upper()
            if code in self.CALL_TICKER_CODES:
                return "call"
            if code in self.PUT_TICKER_CODES:
                return "put"
            return None

        data["option_type"] = data["ticker"].map(infer_type)
        self._data = data
        return data.copy()

    def calculate_time_to_expiry(
        self,
        holidays_csv: Optional[str] = "feriados_nacionais.csv",
        *,
        today: Optional[date] = None,
        maturity_format: str = "%d/%m/%Y",
        holiday_column: str = "Data",
        holiday_format: str = "%m/%d/%Y",
        trading_days_per_year: int = 252,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """
        Compute business-day and year-fraction time to expiry.

        Parameters
        ----------
        holidays_csv:
            Path to a CSV file containing holiday dates. If None, holidays are ignored.
        today:
            Reference date for the calculation. Defaults to the current date.
        maturity_format:
            strftime/strptime format for the expiry column.
        holiday_column:
            Column name within the holiday CSV containing the dates.
        holiday_format:
            strftime/strptime format for the holiday dates.
        trading_days_per_year:
            Annualisation factor for business days.
        force_reload:
            When True, reload the base data from Excel.

        Returns
        -------
        pd.DataFrame
            Dataframe with 'days_to_expiry' and 'time_to_expiry' columns.
        """
        data = self.attach_option_type(force_reload=force_reload)

        parsed_expiry = pd.to_datetime(
            data["expiry"],
            format=maturity_format,
            errors="coerce",
        )
        if parsed_expiry.isna().any():
            raise ValueError("Unable to parse one or more expiry dates with the provided format.")

        holidays_array: Optional[np.ndarray] = None
        if holidays_csv is not None:
            try:
                holidays_df = pd.read_csv(holidays_csv)
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"Holidays file not found: {holidays_csv}") from exc
            if holiday_column not in holidays_df:
                raise KeyError(
                    f"Holidays file must contain a '{holiday_column}' column."
                )
            holidays_array = (
                pd.to_datetime(
                    holidays_df[holiday_column],
                    format=holiday_format,
                    errors="coerce",
                )
                .dropna()
                .values.astype("datetime64[D]")
            )

        reference_date = today or date.today()
        business_days = []
        annualised = []

        for maturity in parsed_expiry.dt.date:
            maturity_np = np.datetime64(maturity)
            ref_np = np.datetime64(reference_date)
            delta_days = np.busday_count(ref_np, maturity_np, holidays=holidays_array)
            # Ensure non-negative values when expiry is in the past.
            delta_days = max(delta_days, 0)
            business_days.append(int(delta_days))
            annualised.append(delta_days / trading_days_per_year)

        data["days_to_expiry"] = business_days
        data["time_to_expiry"] = annualised
        self._data = data
        return data.copy()

    def add_forward_price(
        self,
        risk_free_rate: float = 0.15,
        *,
        holidays_csv: Optional[str] = "feriados_nacionais.csv",
        today: Optional[date] = None,
        maturity_format: str = "%d/%m/%Y",
        holiday_column: str = "Data",
        holiday_format: str = "%m/%d/%Y",
        trading_days_per_year: int = 252,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """
        Add a 'forward_price' column based on the cost-of-carry model.

        Parameters
        ----------
        risk_free_rate:
            Annualised risk-free rate expressed as a decimal (e.g., 0.05 for 5%).
        holidays_csv, today, maturity_format, holiday_column, holiday_format,
        trading_days_per_year, force_reload:
            Passed through to ``calculate_time_to_expiry``.

        Returns
        -------
        pd.DataFrame
            Dataframe augmented with the 'forward_price' column.
        """
        if risk_free_rate <= -1.0:
            raise ValueError("risk_free_rate must be greater than -100%.")

        data = self.calculate_time_to_expiry(
            holidays_csv=holidays_csv,
            today=today,
            maturity_format=maturity_format,
            holiday_column=holiday_column,
            holiday_format=holiday_format,
            trading_days_per_year=trading_days_per_year,
            force_reload=force_reload,
        )

        if "time_to_expiry" not in data:
            raise RuntimeError("Time-to-expiry data is required before computing forward prices.")

        data["forward_price"] = data["spot_price"] * np.power(
            1 + risk_free_rate,
            data["time_to_expiry"],
        )
        self._data = data
        return data.copy()

    def _ensure_sheet(self) -> xw.Sheet:
        """Open the workbook/sheet if necessary and return the sheet object."""
        if self._sheet is not None:
            return self._sheet

        try:
            self._workbook = xw.Book(self.excel_file)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Excel file not found: {self.excel_file}") from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to open workbook '{self.excel_file}'.") from exc

        try:
            self._sheet = self._workbook.sheets[self.sheet_name]
        except Exception as exc:
            self.close()
            raise KeyError(
                f"Sheet '{self.sheet_name}' not found in '{self.excel_file}'."
            ) from exc

        return self._sheet

    def _read_column(self, sheet: xw.Sheet, column_letter: str) -> list:
        """Read a single column range from Excel as a flat list."""
        address = f"{column_letter}{self.row_start}:{column_letter}{self.row_end}"
        try:
            values = sheet.range(address).options(
                ndim=1,
                numbers=float,
                empty=np.nan,
            ).value
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read range '{address}' from sheet '{self.sheet_name}'."
            ) from exc

        # xlwings returns scalars for single-cell ranges; normalise to list
        if values is None:
            return []
        if not isinstance(values, list):
            return [values]
        return values