import os
import pandas as pd
from transformers import DistilBertTokenizer
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load tokenizer globally
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Scalers (we will save them later)
scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

def get_raw_data(path="data"):
    files = os.listdir(path)
    data = pd.DataFrame()

    for file in files:
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(path, file))
            df = df.dropna()

            if data.empty:
                data = df
            else:
                data = data.merge(df, on="Series_Title")
    return data

def get_prepared_data(data_path="data"):
    data = get_raw_data(data_path)

    # --- Clean text ---
    overviews = data["Overview"].fillna("").tolist()
    text_tokens = tokenizer(
        overviews,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # --- Tabular preprocessing ---
    data["Genre"] = data["Genre"].apply(lambda x: x.split(",")[0])
    data["Gross"] = data["Gross"].apply(lambda x: int(x.replace(",", "")) if isinstance(x, str) else x)
    data["Runtime"] = data["Runtime"].str.replace("min", "").str.strip()
    data["Runtime"] = pd.to_numeric(data["Runtime"], errors='coerce')
    data["Released_Year"] = pd.to_numeric(data["Released_Year"], errors='coerce')
    data["Meta_score"] = pd.to_numeric(data["Meta_score"], errors='coerce')

    # Drop unused columns
    data = data.drop(columns=["Poster_Link", "Series_Title", "Overview", "Director", "Star1", "Star2", "Star3", "Star4"])

    # One-hot encode categoricals
    data = pd.get_dummies(data)

    # Features and target
    features = data.drop(columns=["Gross"])
    target = data["Gross"].values.reshape(-1, 1)

    # Scale features
    features_scaled = scaler.fit_transform(features)

    # Scale target (Gross) — optional: log transform (removed for scaler simplicity)
    target_scaled = target_scaler.fit_transform(target)

    # Convert to tensors
    tabular_features = torch.tensor(features_scaled, dtype=torch.float32)
    target_tensor = torch.tensor(target_scaled, dtype=torch.float32)

    inputs = {
        "text_input": text_tokens,
        "tabular_input": tabular_features
    }

    return inputs, target_tensor
