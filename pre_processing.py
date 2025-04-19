# pylint: disable=broad-exception-caught

import os
import numpy as np
import pandas as pd
import torch


def merge_close_only_csvs(base_dir, merge_dict, output_dir):
    """
    For each key in merge_dict, aggregate all CSVs in its subdirs under base_dir,
    keeping only ['Datetime', 'Close'], print duplicate datetimes,
    and output unique sorted CSV for each key to output_dir/{key}.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    for key, subdir_list in merge_dict.items():
        all_dfs = []
        print(f"Processing group '{key}'...")

        for subdir in subdir_list:
            data_path = os.path.join(base_dir, key, subdir)
            if not os.path.isdir(data_path):
                print(f"Warning: {data_path} does not exist")
                continue

            for file in os.listdir(data_path):
                if file.endswith(".csv"):
                    file_path = os.path.join(data_path, file)
                    try:
                        df = pd.read_csv(
                            file_path, usecols=["Datetime", "Close"], parse_dates=["Datetime"]
                        )
                        all_dfs.append(df)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

        if not all_dfs:
            print(f"No CSVs found for group '{key}'. Skipping.")
            continue

        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Sort by Datetime
        combined_df = combined_df.sort_values("Datetime")

        # Check and print duplicates
        duplicate_mask = combined_df.duplicated("Datetime", keep=False)
        if duplicate_mask.any():
            print(f"Duplicate datetimes in '{key}':")
            print(combined_df.loc[duplicate_mask, "Datetime"])

        # Drop duplicates
        combined_df = combined_df.drop_duplicates(["Datetime"])

        # Save with only the two columns
        out_path = os.path.join(output_dir, f"{key}.csv")
        combined_df[["Datetime", "Close"]].to_csv(out_path, index=False)
        print(f"Saved merged file: {out_path}")

    print("All groups processed.")


def prepare_daily_returns_true_close(data_path, output_csv_path=None, device=torch.device("cpu")):

    fnames = [f for f in os.listdir(data_path) if f.endswith(".csv")]
    asset_names = [f.replace(".csv", "") for f in fnames]

    dfs = {}
    for fname, asset in zip(fnames, asset_names):
        df = pd.read_csv(os.path.join(data_path, fname), parse_dates=["Datetime"])
        df = df.set_index("Datetime")
        # Get only the last observation of each day, with the correct timestamp
        df["DateOnly"] = df.index.to_series().dt.date
        df_last = df.groupby("DateOnly").tail(1)
        df_daily = df_last[["Close"]]
        dfs[asset] = df_daily

    # Union of all true close timestamps
    all_close_datetimes = sorted(set().union(*(df.index for df in dfs.values())))
    all_close_datetimes = pd.DatetimeIndex(all_close_datetimes)

    # Align and calculate returns
    all_returns = []
    for asset in asset_names:
        df = dfs[asset].reindex(all_close_datetimes)
        close = df["Close"].values
        ret = np.full_like(close, np.nan, dtype="float32")
        # no explicit ret[0] = 0; leave as NaN
        # ret[0] = 0  # First as zero
        for i in range(1, len(close)):
            if not np.isnan(close[i]) and not np.isnan(close[i - 1]):
                ret[i] = (close[i] / close[i - 1]) - 1
            else:
                ret[i] = np.nan
        all_returns.append(ret)

    returns_arr = np.stack(all_returns, axis=0)  # [assets, times]
    mask_valid_row = ~np.all(np.isnan(returns_arr), axis=0)
    filtered_returns_arr = returns_arr[:, mask_valid_row]
    filtered_dates = all_close_datetimes[mask_valid_row].date

    returns_tensor = torch.from_numpy(filtered_returns_arr).to(device)
    n_assets, max_T = returns_tensor.shape
    lengths_tensor = torch.full((n_assets,), max_T, dtype=torch.long, device=device)

    if output_csv_path is not None:
        out_df = pd.DataFrame(
            np.transpose(filtered_returns_arr), index=filtered_dates, columns=asset_names
        )
        out_df.index.name = "Date"
        out_df.to_csv(output_csv_path, float_format="%.8f", na_rep="NaN")

    return asset_names, returns_tensor, lengths_tensor, filtered_dates


if __name__ == "__main__":

    # # 1) Create e file for each asset
    # base_dir = "/home/ray/projects/data_sources/ibkr_complete"
    # merge_dict = {
    #     "EUR-USD": ["2020", "2021"],
    #     "XAG-USD": ["2020", "2021"],
    #     # add more as needed
    # }
    # output_dir = "assets/"

    # merge_close_only_csvs(base_dir, merge_dict, output_dir)
    ############################################################

    # 2) Combine the different assets from the output dir
    _, _, _, filtered_dates = prepare_daily_returns_true_close(
        data_path="assets/", output_csv_path="output/universe.csv"
    )

    print(filtered_dates)
