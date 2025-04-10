import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import numpy as np
import joblib

from prep_data import get_prepared_data, scaler, target_scaler
from model import create_model

def prepare_dataloader(features, target, batch_size=16):
    dataset = TensorDataset(
        features["tabular_input"],
        features["text_input"]["input_ids"],
        features["text_input"]["attention_mask"],
        target
    )

    class CollateWrapper:
        def __call__(self, batch):
            tabular_input, input_ids, attention_mask, target = zip(*batch)
            return ({
                "tabular_input": torch.stack(tabular_input),
                "text_input": {
                    "input_ids": torch.stack(input_ids),
                    "attention_mask": torch.stack(attention_mask)
                }
            }, torch.stack(target))

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=CollateWrapper())

def move_batch_to_device(X_batch, device):
    for key in X_batch:
        if isinstance(X_batch[key], dict):
            for subkey in X_batch[key]:
                X_batch[key][subkey] = X_batch[key][subkey].to(device)
        else:
            X_batch[key] = X_batch[key].to(device)
    return X_batch

def train_model(model, optimizer, criterion, train_loader, val_loader, device, epochs=50):
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            X_batch = move_batch_to_device(X_batch, device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        preds, trues = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = move_batch_to_device(X_batch, device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                total_val_loss += criterion(outputs, y_batch).item()

                preds.append(outputs.cpu().numpy())
                trues.append(y_batch.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        
        # Stack predictions and ground truth
        preds_stacked = np.vstack(preds)
        trues_stacked = np.vstack(trues)
        
        # Check for NaN values before inverse transform
        if np.isnan(preds_stacked).any():
            print(f"Warning: NaN values found in predictions before inverse transform")
            # Replace NaNs with zeros for the inverse transform
            preds_stacked = np.nan_to_num(preds_stacked, nan=0.0)
            
        if np.isnan(trues_stacked).any():
            print(f"Warning: NaN values found in ground truth before inverse transform")
            trues_stacked = np.nan_to_num(trues_stacked, nan=0.0)
        
        # Apply inverse transform
        preds = target_scaler.inverse_transform(preds_stacked)
        trues = target_scaler.inverse_transform(trues_stacked)
        
        # Check for NaN values after inverse transform
        if np.isnan(preds).any():
            print(f"Warning: NaN values found after inverse transform in predictions")
            preds = np.nan_to_num(preds, nan=0.0)
            
        if np.isnan(trues).any():
            print(f"Warning: NaN values found after inverse transform in ground truth")
            trues = np.nan_to_num(trues, nan=0.0)
            
        # Compute metrics with clean data
        mse = mean_squared_error(trues, preds)
        rmse = mse ** 0.5
        mae = mean_absolute_error(trues, preds)
        r2 = r2_score(trues, preds)

        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.4f}")

    return model

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    features, target = get_prepared_data()

    X_tab_train, X_tab_val, X_ids_train, X_ids_val, X_mask_train, X_mask_val, y_train, y_val = train_test_split(
        features["tabular_input"],
        features["text_input"]["input_ids"],
        features["text_input"]["attention_mask"],
        target,
        test_size=0.2,
        random_state=42
    )

    train_features = {
        "tabular_input": X_tab_train,
        "text_input": {
            "input_ids": X_ids_train,
            "attention_mask": X_mask_train
        }
    }

    val_features = {
        "tabular_input": X_tab_val,
        "text_input": {
            "input_ids": X_ids_val,
            "attention_mask": X_mask_val
        }
    }

    train_loader = prepare_dataloader(train_features, y_train)
    val_loader = prepare_dataloader(val_features, y_val)

    model, optimizer = create_model(train_features)
    model.to(device)

    criterion = nn.MSELoss()

    trained_model = train_model(model, optimizer, criterion, train_loader, val_loader, device)

    # Save model and scalers
    torch.save(trained_model.state_dict(), "saved_weights/model.pth")
    joblib.dump(scaler, "saved_weights/feature_scaler.pkl")
    joblib.dump(target_scaler, "saved_weights/target_scaler.pkl")

    print("✅ Model and scalers saved!")
