import os
import json
from glob import glob
from typing import Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt


class DataAggregator:
    def __init__(self, asset_dirs_dict, output_dir):
        """
        asset_dirs_dict: dict, keys are asset names, values are lists of directories containing csv files
        output_dir: str, directory to save aggregated files
        """
        self.asset_dirs_dict = asset_dirs_dict
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def aggregate_asset_data(self, asset, dirs):
        """
        Aggregates minute-level data to hourly-level for a single asset.
        """
        all_dfs = []
        for directory in dirs:
            csv_files = glob(os.path.join(directory, "*.csv"))
            for file in csv_files:
                df = pd.read_csv(file, parse_dates=["Datetime"])
                df.set_index("Datetime", inplace=True)
                all_dfs.append(df)

        if not all_dfs:
            print(f"No data found for asset {asset}.")
            return None

        # Concatenate all dataframes and sort by datetime
        asset_df = pd.concat(all_dfs).sort_index()

        # Aggregate to hourly data
        hourly_df = (
            asset_df.resample("h")
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
            .dropna(subset=["Close"])
        )  # Drop hours without any close price

        return hourly_df

    def aggregate_all_assets(self):
        """
        Aggregates data for all assets and saves individual asset files and overall file.
        """
        asset_hourly_data = {}

        # Step 1-3: Aggregate and save individual asset files
        for asset, dirs in self.asset_dirs_dict.items():
            print(f"Aggregating data for asset: {asset}")
            hourly_df = self.aggregate_asset_data(asset, dirs)
            if hourly_df is not None:
                asset_hourly_data[asset] = hourly_df
                asset_file_path = os.path.join(self.output_dir, f"{asset}_hourly.csv")
                hourly_df.to_csv(asset_file_path)
                print(f"Saved hourly data for {asset} to {asset_file_path}")

        # Step 4: Aggregate all assets into one dataframe based on datetime index
        print("Aggregating all assets into overall file...")
        overall_df = pd.DataFrame()

        for asset, df in asset_hourly_data.items():
            overall_df[asset] = df["Close"]

        # Sort by datetime index
        overall_df.sort_index(inplace=True)

        # Step 5: Save overall file
        overall_file_path = os.path.join(self.output_dir, "overall_hourly_close.csv")
        overall_df.to_csv(overall_file_path)
        print(f"Saved overall hourly close data to {overall_file_path}")


class DataProcessor:
    def __init__(self, input_csv_path: str, drop_na: bool = True):
        self.input_csv_path: str = input_csv_path
        self.drop_na: bool = drop_na
        self.data: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None
        self.cumulative_returns: Optional[pd.DataFrame] = None

    def load_data(self) -> None:
        self.data = pd.read_csv(self.input_csv_path, parse_dates=["Datetime"], index_col="Datetime")
        print("Data loaded successfully.")

    def align_data(self) -> None:
        if self.data is None:
            raise ValueError("Data not loaded. Run load_data() first.")

        if self.drop_na:
            # Drop rows with any NaN values before any computation
            self.data.dropna(how="any", inplace=True)
            print("NaN values dropped from original dataset.")
        else:
            # Forward-fill missing values to handle NaNs
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

        self.returns = self.data.pct_change()

        if self.drop_na:
            # Drop NaNs resulting from pct_change
            self.returns.dropna(how="any", inplace=True)
            print("NaN values dropped after calculating returns.")
        else:
            # Replace NaNs with zeros
            self.returns.fillna(0, inplace=True)
            print("NaN values replaced with zeros after calculating returns.")

        # Set first returns to 0 to start cumulative returns at 1
        if not self.returns.empty:
            self.returns.iloc[0] = 0
            self.cumulative_returns = (1 + self.returns).cumprod()
            print("Returns calculated successfully.")
        else:
            raise ValueError("No returns data available after calculation.")

    def shift_returns(self) -> None:
        if self.returns is None:
            raise ValueError("Returns not calculated. Run calculate_returns() first.")
        self.returns = self.returns.shift(1).dropna()
        print("Returns shifted successfully to avoid look-ahead bias.")

    def save_processed_data(self, output_csv_path: str) -> None:
        if self.returns is None:
            raise ValueError("Returns not calculated. Run calculate_returns() first.")
        self.returns.to_csv(output_csv_path)
        print(f"Processed data saved successfully to {output_csv_path}.")

    def plot_cumulative_returns(self) -> None:
        if self.cumulative_returns is not None:
            self.cumulative_returns.plot(figsize=(10, 6), title="Cumulative Returns")
            plt.xlabel("Date")
            plt.ylabel("Cumulative Return")
            plt.grid(True)
            plt.show()
        else:
            print("Cumulative returns not calculated yet. Run calculate_returns() first.")

    def process_all(self, output_csv_path: str, plot: bool = False) -> None:
        self.load_data()
        self.align_data()
        self.calculate_returns()
        self.shift_returns()
        self.save_processed_data(output_csv_path)
        if plot:
            self.plot_cumulative_returns()

    #########################################################################


def build_common_index(df, output_file="data/common_index.csv", thresh=0.05):

    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
    df.set_index("Datetime", inplace=True)

    first_day = df.index.date.min()
    last_day = df.index.date.max()

    all_business_days = pd.date_range(start=first_day, end=last_day, freq="B", tz=df.index.tzinfo)

    min_hour = df.index.hour.min()
    max_hour = df.index.hour.max()

    full_datetime_index = [
        pd.Timestamp(day.year, day.month, day.day, hour, tz=df.index.tzinfo)
        for day in all_business_days
        for hour in range(min_hour, max_hour + 1)
    ]

    common_df = pd.DataFrame(index=full_datetime_index)

    asset_columns = df.columns
    full_df = common_df.join(df[asset_columns], how="left")

    # First-day explicit NaN handling
    first_day_mask = full_df.index.date == first_day  # type: ignore
    first_day_df = full_df.loc[first_day_mask].copy()

    for asset in asset_columns:
        asset_series = first_day_df[asset]
        zero_returns_index = asset_series[asset_series == 0.0].index
        if not zero_returns_index.empty:
            first_zero_idx = zero_returns_index[0]
            pos_first_zero = asset_series.index.get_loc(first_zero_idx)
            indices_before_first_zero = asset_series.index[:pos_first_zero]
            first_day_df.loc[indices_before_first_zero, asset] = asset_series.loc[
                indices_before_first_zero
            ].fillna(0.0)

    full_df.loc[first_day_mask, asset_columns] = first_day_df[asset_columns]

    # Hourly sparsity check with percentage printout
    full_df["hour"] = full_df.index.hour  # type: ignore

    assets_count = len(asset_columns)
    hourly_groups = full_df.groupby("hour")[asset_columns]

    non_nan_counts_per_hour = hourly_groups.apply(lambda x: x.notna().sum().sum())
    total_points_per_hour = hourly_groups.size() * assets_count
    fraction_not_nan_per_hour = non_nan_counts_per_hour / total_points_per_hour

    # Create DataFrame for clear reporting
    hourly_summary_df = pd.DataFrame(
        {
            "Total Points": total_points_per_hour,
            "Non-NaN Counts": non_nan_counts_per_hour,
            "Fraction Non-NaN": fraction_not_nan_per_hour,
            "Percentage Non-NaN (%)": fraction_not_nan_per_hour * 100,
        }
    ).round(2)

    print("\nHourly summary of data availability:")
    print(hourly_summary_df)

    hours_to_remove = hourly_summary_df[hourly_summary_df["Fraction Non-NaN"] <= thresh].index
    if len(hours_to_remove) > 0:
        print(
            f"\nRemoving hour(s) with <= {thresh*100:.2f}% data availability:",
            ", ".join(str(h) for h in hours_to_remove),
        )
    else:
        print(f"\nNo hours removed (all have > {thresh*100:.2f}% data availability).")

    full_df = full_df[~full_df["hour"].isin(hours_to_remove)]
    full_df.drop(columns=["hour"], inplace=True)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    full_df.to_csv(output_file, index_label="Datetime")

    return full_df


def prepare_metadata_exact(full_df, output_metadata_file="data/metadata.json"):
    # Make sure datetime is the index and sorted correctly
    if not isinstance(full_df.index, pd.DatetimeIndex):
        full_df["Datetime"] = pd.to_datetime(full_df["Datetime"], utc=True)
        full_df.set_index("Datetime", inplace=True)

    full_df.sort_index(inplace=True)

    # Get basic metadata
    years = sorted(full_df.index.year.unique())
    num_years = len(years)

    # Find all unique business days in entire dataset
    all_business_days = sorted(full_df.index.normalize().unique())
    trading_days_per_year = len(all_business_days) // num_years

    # Find unique trading hours per day
    trading_hours = sorted(full_df.index.hour.unique())
    trading_hours_per_day = len(trading_hours)

    # Estimate trading weeks per year (assuming 5 business days week)
    trading_days_per_week = 5
    trading_weeks_total = len(all_business_days) // trading_days_per_week

    # Compute weeks, dropping incomplete week at end (if any)
    total_days_in_data = trading_weeks_total * trading_days_per_week
    all_business_days_consistent = all_business_days[:total_days_in_data]

    # Prepare metadata dictionary
    metadata = {
        "number_of_years": num_years,
        "trading_weeks_per_year": trading_weeks_total // num_years,
        "trading_days_per_year": trading_days_per_year,
        "trading_hours_per_day": trading_hours_per_day,
        "assets": {},
    }

    # Reshape prep for each asset individually
    for asset in full_df.columns:

        asset_df = full_df[[asset]].copy()

        # Prepare Daily returns tensor with NaNs -> torch.nan
        daily_returns_df = (
            asset_df.resample("B").sum(min_count=1).reindex(all_business_days_consistent)
        )
        daily_returns_array = daily_returns_df[asset].values.reshape(
            trading_weeks_total, trading_days_per_week
        )
        daily_returns_tensor = torch.tensor(daily_returns_array, dtype=torch.float32)
        daily_returns_tensor[torch.isnan(daily_returns_tensor)] = torch.nan

        # Prepare Hourly returns tensor with NaNs -> torch.nan
        hourly_returns_df = asset_df.copy()

        expected_idx = pd.MultiIndex.from_product(
            [all_business_days_consistent, trading_hours], names=["date", "hour"]
        )

        hourly_returns_df = hourly_returns_df.copy()
        hourly_returns_df["date"] = hourly_returns_df.index.normalize()
        hourly_returns_df["hour"] = hourly_returns_df.index.hour
        hourly_returns_df.set_index(["date", "hour"], inplace=True)

        hourly_returns_df = hourly_returns_df.reindex(expected_idx)
        hourly_returns_array = hourly_returns_df[asset].values.reshape(
            trading_weeks_total, trading_days_per_week, trading_hours_per_day
        )
        hourly_returns_tensor = torch.tensor(hourly_returns_array, dtype=torch.float32)
        hourly_returns_tensor[torch.isnan(hourly_returns_tensor)] = torch.nan

        # Store tensors as lists in metadata json
        metadata["assets"][asset] = {
            "daily_return": daily_returns_tensor.tolist(),
            "hourly_return": hourly_returns_tensor.tolist(),
        }

    # Ensure metadata directory before saving JSON
    os.makedirs(os.path.dirname(output_metadata_file), exist_ok=True)
    with open(output_metadata_file, "w", encoding="UTF-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata successfully saved to {output_metadata_file}.")

    # Return tensors directly
    tensor_metadata = {
        "number_of_years": num_years,
        "trading_weeks_per_year": trading_weeks_total // num_years,
        "trading_days_per_year": trading_days_per_year,
        "trading_hours_per_day": trading_hours_per_day,
        "assets": {},
    }

    for asset in metadata["assets"]:
        tensor_metadata["assets"][asset] = {
            "daily_return": torch.tensor(metadata["assets"][asset]["daily_return"]),
            "hourly_return": torch.tensor(metadata["assets"][asset]["hourly_return"]),
        }

    return tensor_metadata


def prepare_metadata_exact2(full_df, output_metadata_file="data/metadata.json"):
    # Make sure datetime is the index and sorted correctly
    if not isinstance(full_df.index, pd.DatetimeIndex):
        full_df["Datetime"] = pd.to_datetime(full_df["Datetime"], utc=True)
        full_df.set_index("Datetime", inplace=True)

    full_df.sort_index(inplace=True)

    # Basic metadata extraction
    years = sorted(full_df.index.year.unique())
    num_years = len(years)

    # All unique business days
    all_business_days = sorted(full_df.index.normalize().unique())
    trading_days_per_year = len(all_business_days) // num_years

    # Unique trading hours
    trading_hours = sorted(full_df.index.hour.unique())
    trading_hours_per_day = len(trading_hours)

    trading_days_per_week = 5
    trading_weeks_total = len(all_business_days) // trading_days_per_week

    total_days_in_data = trading_weeks_total * trading_days_per_week
    all_business_days_consistent = all_business_days[:total_days_in_data]

    # Prepare metadata dictionary
    metadata = {
        "number_of_years": num_years,
        "trading_weeks_per_year": trading_weeks_total // num_years,
        "trading_days_per_year": trading_days_per_year,
        "trading_hours_per_day": trading_hours_per_day,
        "assets": {},
    }

    tensor_metadata = {
        "number_of_years": num_years,
        "trading_weeks_per_year": trading_weeks_total // num_years,
        "trading_days_per_year": trading_days_per_year,
        "trading_hours_per_day": trading_hours_per_day,
        "assets": {},
    }

    for asset in full_df.columns:
        asset_df = full_df[[asset]].copy()

        # Compute daily returns by compounding hourly returns, not summing  <<<< MODIFIED PART
        compounded_daily_returns = []
        hourly_returns_df = asset_df.copy()
        hourly_returns_df["date"] = hourly_returns_df.index.normalize()
        hourly_returns_df["hour"] = hourly_returns_df.index.hour

        for day in all_business_days_consistent:
            day_hourly_returns = hourly_returns_df[hourly_returns_df["date"] == day][asset].dropna()
            if len(day_hourly_returns) > 0:
                compounded_return = np.nanprod(1 + day_hourly_returns.values) - 1
            else:
                compounded_return = np.nan
            compounded_daily_returns.append(compounded_return)

        daily_returns_array = np.array(compounded_daily_returns).reshape(
            trading_weeks_total, trading_days_per_week
        )
        daily_returns_tensor = torch.tensor(daily_returns_array, dtype=torch.float32)
        daily_returns_tensor[torch.isnan(daily_returns_tensor)] = torch.nan

        # Hourly returns tensor preparation (unchanged logic, minor adjustment)
        expected_idx = pd.MultiIndex.from_product(
            [all_business_days_consistent, trading_hours], names=["date", "hour"]
        )
        hourly_returns_df_full = hourly_returns_df.set_index(["date", "hour"]).reindex(expected_idx)
        hourly_returns_array = hourly_returns_df_full[asset].values.reshape(
            trading_weeks_total, trading_days_per_week, trading_hours_per_day
        )
        hourly_returns_tensor = torch.tensor(hourly_returns_array, dtype=torch.float32)
        hourly_returns_tensor[torch.isnan(hourly_returns_tensor)] = torch.nan

        # Save only in returned dataframe, not in JSON
        tensor_metadata["assets"][asset] = {
            "daily_return": daily_returns_tensor,
            "hourly_return": hourly_returns_tensor,
        }

        # Save only basic asset metadata (no data tensors) in JSON
        metadata["assets"][asset] = {
            "asset_name": asset,
            "currency": "USD" if "usd" in asset else "other",  # you may customize this part
        }

    # Ensure metadata directory before saving JSON
    os.makedirs(os.path.dirname(output_metadata_file), exist_ok=True)
    with open(output_metadata_file, "w", encoding="UTF-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata successfully saved to {output_metadata_file}.")

    return tensor_metadata


# set higher precision for tensor printing
torch.set_printoptions(precision=8, sci_mode=False, linewidth=200)


def load_and_prepare_returns(output_csv, output_path):
    df = pd.read_csv(output_csv)
    print(df.head())
    full_df = build_common_index(df)
    data = prepare_metadata_exact2(full_df)
    print(data["assets"]["eur-usd"]["hourly_return"].shape)
    print(data["assets"]["eur-usd"]["daily_return"].shape)

    asset_name = "eur-usd"
    hourly_returns = data["assets"][asset_name]["hourly_return"]
    # First week, second day (week_idx=0, day_idx=1)
    print("Hourly returns for first week's first day:")
    print(hourly_returns[0, 0])
    print()
    print("Hourly returns for first week's second day:")
    print(hourly_returns[0, 1])
    torch.save(data, "data/assets.pt")


if __name__ == "__main__":

    # output_directory = "data"
    # asset_dirs = {
    #     "eur-usd": [
    #         "/home/ray/projects/data_sources/ibkr_complete/EUR-USD/2017",
    #         "/home/ray/projects/data_sources/ibkr_complete/EUR-USD/2018",
    #         # "/home/ray/projects/data_sources/ibkr_complete/EUR-USD/2019",
    #         # "/home/ray/projects/data_sources/ibkr_complete/EUR-USD/2020",
    #         # "/home/ray/projects/data_sources/ibkr_complete/EUR-USD/2021",
    #     ],
    #     "ibus500": [
    #         "/home/ray/projects/data_sources/ibkr_complete/IBUS500/2017",
    #         "/home/ray/projects/data_sources/ibkr_complete/IBUS500/2018",
    #     ],
    # }

    # aggregator = DataAggregator(asset_dirs, output_directory)
    # aggregator.aggregate_all_assets()

    ####################################################################

    # input_csv = "data/overall_hourly_close.csv"
    output_csv = "data/processed_returns.csv"

    # processor = DataProcessor(input_csv, drop_na=True)
    # df = pd.read_csv(input_csv)
    # print(df.head(), "\n")
    # processor.process_all(output_csv, plot=True)

    ####################################################################

    output_path = "data/adjusted_returns.csv"

    load_and_prepare_returns(output_csv, output_path)
