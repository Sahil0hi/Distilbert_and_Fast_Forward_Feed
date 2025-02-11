import pandas as pd
import numpy as np
import torch
import os

def get_prepared_data(data_path="data"):
    """
    Load and clean the IMDB dataset from CSV(s) in data_path, then return
    PyTorch tensors (features, target) for modeling.
    """
    # 1. Read all CSV files in 'data/' and merge on 'Series_Title'
    df = get_raw_data(data_path)

    # 2. Clean & parse 'Runtime' from 'XYZ min' to integer
    df["Runtime"] = df["Runtime"].apply(parse_runtime)

    # 3. Clean & parse 'Gross'
    df["Gross"] = df["Gross"].apply(parse_gross)

    # 4. Drop rows that still have NaN in critical columns
    df.dropna(subset=["Gross", "Runtime"], inplace=True)

    # 5. Clean 'Released_Year'
    df["Released_Year"] = df["Released_Year"].apply(safe_int)

    # 6. Extract just the first 'Genre' (e.g. 'Action, Crime' → 'Action').
    df["Genre"] = df["Genre"].apply(lambda x: x.split(",")[0] if isinstance(x, str) else x)

    # 7. Convert categorical columns to one-hot vectors
    df = pd.get_dummies(df, columns=["Genre", "Certificate"], dummy_na=True)

    # 8. Drop columns we don't need
    drop_cols = ["Poster_Link", "Series_Title", "Overview", "Director",
                 "Star1", "Star2", "Star3", "Star4"]
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # 9. Convert *all* remaining columns to numeric (coercing bad values to NaN)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Optionally, you can drop rows that still contain NaNs after coercion.
    # If you'd rather fill them, do df.fillna(...) instead.
    df.dropna(inplace=True)

    # 10. Separate features and target
    target = df["Gross"].values
    features = df.drop(columns=["Gross"]).values

    # 11. Convert to torch tensors
    features = torch.tensor(features, dtype=torch.float32)
    target = torch.tensor(target.reshape(-1, 1), dtype=torch.float32)

    return features, target


def get_raw_data(data_path="data"):
    """
    Reads every CSV in 'data_path' and merges them on 'Series_Title'.
    Returns a single DataFrame of all merged data.
    """
    files = os.listdir(data_path)
    data = pd.DataFrame()

    for file in files:
        if file.endswith(".csv"):
            temp = pd.read_csv(os.path.join(data_path, file), encoding="utf-8",
                               na_values=["NA", "", " "])
            # ensure 'Series_Title' is present
            if "Series_Title" not in temp.columns:
                continue
            # drop empty 'Series_Title' rows
            temp.dropna(subset=["Series_Title"], inplace=True)

            if data.empty:
                data = temp
            else:
                # merge on 'Series_Title' with an outer join
                data = pd.merge(data, temp, on="Series_Title", how="outer")

    return data


def parse_runtime(x):
    """
    Convert a runtime string like '142 min' → integer 142.
    Returns np.nan if it cannot parse.
    """
    if not isinstance(x, str):
        return np.nan
    x = x.strip().lower()
    if x.endswith("min"):
        x = x.replace("min", "").strip()
        try:
            return int(x)
        except ValueError:
            return np.nan
    return np.nan


def parse_gross(x):
    """
    Convert a gross string like '4,360,000' → 4360000 (float).
    If the row has '#####', return NaN.
    """
    if not isinstance(x, str):
        return np.nan
    x = x.strip()
    if "####" in x:
        return np.nan
    x = x.replace(",", "")
    try:
        return float(x)
    except ValueError:
        return np.nan


def safe_int(x):
    """
    Convert a string (or numeric) to int, else return NaN.
    """
    try:
        return int(x)
    except:
        return np.nan


def get_all_titles(data_path="data"):
    df = get_raw_data(data_path)
    return df["Series_Title"]
