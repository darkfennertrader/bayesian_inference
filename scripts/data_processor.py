import os
import json
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from fracdiff import fdiff


class DataProcessor:
    def __init__(self, input_csv_path: str, drop_na: bool = True, output_dir: str = "./output"):
        self.input_csv_path: str = input_csv_path
        self.drop_na: bool = drop_na
        self.output_dir: str = output_dir
        self.data: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None
        self.cumulative_returns: Optional[pd.DataFrame] = None
        self.stationary_returns: Optional[pd.DataFrame] = None
        self.optimal_d: dict = {}

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self) -> None:
        self.data = pd.read_csv(self.input_csv_path, parse_dates=["Datetime"], index_col="Datetime")
        print("Data loaded successfully.")

    def align_data(self) -> None:
        if self.data is None:
            raise ValueError("Data not loaded. Run load_data() first.")

        if self.drop_na:
            self.data.dropna(how="any", inplace=True)
            print("NaN values dropped from original dataset.")
        else:
            self.data.ffill(inplace=True)
            print("NaN values forward-filled in original dataset.")

        if self.data.empty:
            raise ValueError(
                "No data left after alignment. Check your dataset and drop_na setting."
            )
        else:
            print(f"Data aligned successfully. Starting from {self.data.index[0]}.")

    def calculate_returns(self) -> None:
        if self.data is None:
            raise ValueError("Data not loaded or aligned. Run load_data() and align_data() first.")

        self.returns = self.data.pct_change().dropna(how="any")
        print("Returns calculated successfully.")

        # Save returns to static file
        returns_output_path = os.path.join(self.output_dir, "processed_returns.csv")
        self.returns.to_csv(returns_output_path)
        print(f"Processed returns saved successfully to {returns_output_path}.")

    def shift_returns(self) -> None:
        if self.stationary_returns is None:
            raise ValueError("Stationary returns not calculated. Run make_stationary() first.")
        self.stationary_returns = self.stationary_returns.shift(1).dropna()
        print("Stationary returns shifted successfully to avoid look-ahead bias.")

    def adf_test(self, series, signif=0.05):
        result = adfuller(series, autolag="AIC")
        p_value = result[1]
        return p_value < signif

    def find_min_frac_diff(self, series, d_values=np.linspace(0, 1, 21)):
        for d in d_values:
            diff_series = fdiff(series.values, n=d)
            diff_series = diff_series[~np.isnan(diff_series)]
            if len(diff_series) < 10:
                continue
            if self.adf_test(diff_series):
                return d
        return 1.0  # default to 1 if no smaller d found

    def make_stationary(self) -> None:
        if self.returns is None:
            raise ValueError("Returns not calculated. Run calculate_returns() first.")

        stationary_df = pd.DataFrame(index=self.returns.index)
        optimal_d = {}

        for col in self.returns.columns:
            print(f"Processing asset: {col}")
            series = self.returns[col].dropna()
            d = self.find_min_frac_diff(series)
            optimal_d[col] = d
            print(f"Optimal fractional differentiation order for {col}: {d:.2f}")
            diff_series = fdiff(series.values, n=d)
            diff_series = pd.Series(diff_series, index=series.index).dropna()
            stationary_df[col] = diff_series

        self.stationary_returns = stationary_df.dropna(how="any")
        self.optimal_d = optimal_d

        # Save optimal d values to JSON file
        optimal_d_path = os.path.join(self.output_dir, "optimal_d.json")
        with open(optimal_d_path, "w", encoding="UTF-8") as f:
            json.dump(self.optimal_d, f, indent=4)
        print(f"Optimal fractional differentiation orders saved successfully to {optimal_d_path}.")

        print("Fractional differentiation applied successfully.")

    def plot_stationary_returns(self, asset: str) -> None:
        if self.stationary_returns is None:
            raise ValueError("Stationary returns not calculated. Run make_stationary() first.")
        if asset not in self.stationary_returns.columns:
            raise ValueError(f"Asset {asset} not found in stationary returns.")

        plt.figure(figsize=(12, 6))
        plt.plot(
            self.stationary_returns.index,
            self.stationary_returns[asset],
            label=f"{asset} Stationary Returns",
        )
        plt.title(f"Stationary Returns for {asset}")
        plt.xlabel("Datetime")
        plt.ylabel("Differentiated Returns")
        plt.grid(True)
        plt.legend()

        # Save plot as image file
        plot_path = os.path.join(self.output_dir, f"{asset}_stationary_returns.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved successfully to {plot_path}.")

    def save_processed_data(self, output_csv_path: str) -> None:
        if self.stationary_returns is None:
            raise ValueError("Stationary returns not calculated. Run make_stationary() first.")
        self.stationary_returns.to_csv(output_csv_path)
        print(f"Processed stationary data saved successfully to {output_csv_path}.")

    def process_all(self, output_csv_path: str, plot_asset: Optional[str] = None) -> None:
        self.load_data()
        self.align_data()
        self.calculate_returns()
        self.make_stationary()
        self.shift_returns()
        self.save_processed_data(output_csv_path)
        if plot_asset:
            self.plot_stationary_returns(plot_asset)
