import pandas as pd
import numpy as np
import torch
import os
from transformers import DistilBertTokenizer

# We need the DistilBertTokenizerFast to tokenize text
from transformers import DistilBertTokenizerFast

def get_prepared_data(data_path="data"):
    """
    Load and clean the IMDB dataset from CSV(s) in data_path, then return
    a dictionary of {'text_input': {...}, 'tabular_input': ...} plus a target tensor.
    This allows us to feed DistilBERT + numeric features into MyModel.
    """
    # --- 1. Read all CSV files in 'data/' and merge on 'Series_Title' ---
    df = get_raw_data(data_path)

    # Store Series_Title for tokenization before cleaning
    titles = df['Series_Title'].tolist()

    # We'll assume 'Overview' is our text column for BERT
    if "Overview" not in df.columns:
        # If your dataset truly has no 'Overview', remove the BERT usage from model.py
        df["Overview"] = ""
    df["Overview"] = df["Overview"].fillna("")

    # --- 2. Clean & parse 'Runtime' from 'XYZ min' to integer ---
    df["Runtime"] = df["Runtime"].apply(parse_runtime)

    # --- 3. Clean & parse 'Gross' ---
    df["Gross"] = df["Gross"].apply(parse_gross)

    # Drop rows missing essential columns
    df.dropna(subset=["Gross", "Runtime"], inplace=True)

    # --- 4. Clean 'Released_Year' ---
    df["Released_Year"] = df["Released_Year"].apply(safe_int)

    # --- 5. Extract just the first 'Genre' (e.g. 'Action, Crime' → 'Action') ---
    # 6. Extract just the first 'Genre'
    df["Genre"] = df["Genre"].apply(lambda x: x.split(",")[0] if isinstance(x, str) else x)

    # --- 6. One-hot encode 'Genre' & 'Certificate' ---
    df = pd.get_dummies(df, columns=["Genre", "Certificate"], dummy_na=True)

    # --- 7. We'll keep 'Overview' for text,
    #         but drop other columns we don't need. ---
    drop_cols = ["Poster_Link", "Series_Title", "Director",
    # 8. Tokenize the text data
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    text_encodings = tokenizer(titles, 
                             truncation=True, 
                             padding=True,
                             return_tensors='pt')

    # 9. Drop columns we don't need
    drop_cols = ["Poster_Link", "Series_Title", "Overview", "Director",
                 "Star1", "Star2", "Star3", "Star4"]
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # --- 8. Convert all remaining columns (except 'Overview') to numeric ---
    # Separate out the Overviews
    overviews = df["Overview"].values
    df.drop(columns=["Overview"], inplace=True)

    # Convert to numeric, dropping rows with NaNs
    df = df.apply(pd.to_numeric, errors="coerce")
    # 10. Convert remaining columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Make sure to drop NaN values
    df.dropna(inplace=True)

    # --- 9. Separate target = 'Gross' from the numeric features ---
    target = df["Gross"].values
    df.drop(columns=["Gross"], inplace=True)

    # Now 'df' has your tabular numeric features
    tabular_data = df.values  # shape [N, num_features]

    # --- 10. Tokenize Overviews with DistilBERT tokenizer ---
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    encodings = tokenizer(
        list(overviews),
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"  # return PyTorch tensors
    )

    # encodings["input_ids"]  -> shape [N, seq_len]
    # encodings["attention_mask"] -> shape [N, seq_len]
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    # --- 11. Build final feature dict & convert to Torch tensors ---
    tabular_input = torch.tensor(tabular_data, dtype=torch.float32)
    features = {
        "text_input": {
            "input_ids": input_ids,            # shape [N, seq_len]
            "attention_mask": attention_mask   # shape [N, seq_len]
        },
        "tabular_input": tabular_input         # shape [N, num_features]
    }

    # Convert target
    target = torch.tensor(target.reshape(-1, 1), dtype=torch.float32)
    # Prepare tabular features (all numeric columns except target)
    numeric_features = df.select_dtypes(include=['float64', 'int64']).drop(columns=['Gross'])
    tabular_tensor = torch.tensor(numeric_features.values, dtype=torch.float32)

    # Normalize the gross values (target) using log transformation
    # This is common for monetary values which can have large ranges
    target = torch.log(torch.tensor(df['Gross'].values, dtype=torch.float32))

    # Normalize tabular features using standard scaling
    tabular_mean = tabular_tensor.mean(dim=0)
    tabular_std = tabular_tensor.std(dim=0)
    tabular_tensor = (tabular_tensor - tabular_mean) / (tabular_std + 1e-7)

    # Structure the features
    features = {
        "text_input": {
            "input_ids": text_encodings['input_ids'],
            "attention_mask": text_encodings['attention_mask']
        },
        "tabular_input": tabular_tensor
    }

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
            if "Series_Title" not in temp.columns:
                continue
            temp.dropna(subset=["Series_Title"], inplace=True)

            if data.empty:
                data = temp
            else:
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
