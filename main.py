import torch
import torch.nn as nn
from sklearn.model_selection import KFold
import numpy as np
import os
import joblib
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


from prep_data import get_prepared_data, scaler, target_scaler
from model import create_model, MyModel
from train import prepare_dataloader, move_batch_to_device

def main():
    """
    Evaluate the multimodal model using 5-fold cross-validation.
    This function is designed specifically for the multimodal architecture.
    """
    print("\n======= MULTIMODAL MODEL EVALUATION =======\n")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data in multimodal format
    features, target = get_prepared_data(multimodal_format=True)
    print("Data loaded successfully in multimodal format")
    
    # Create indices for cross-validation
    all_indices = np.arange(target.shape[0])
    np.random.shuffle(all_indices)  # Shuffle indices
    
    # Define the K-fold Cross Validator
    k_folds = 5
    kf = KFold(n_splits=k_folds)
    
    # Metrics storage
    fold_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(all_indices)):
        print(f"\n----- Fold {fold+1}/{k_folds} -----")
        
        # Split data for this fold
        # For tabular data
        X_tab_train = features["tabular_input"][train_idx]
        X_tab_val = features["tabular_input"][test_idx]
        
        # For text data
        X_ids_train = features["text_input"]["input_ids"][train_idx]
        X_ids_val = features["text_input"]["input_ids"][test_idx]
        X_mask_train = features["text_input"]["attention_mask"][train_idx]
        X_mask_val = features["text_input"]["attention_mask"][test_idx]
        
        # For target
        y_train = target[train_idx]
        y_val = target[test_idx]
        
        # Create feature dictionaries
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
        
        # Create data loaders
        print("Creating dataloaders...")
        train_loader = prepare_dataloader(train_features, y_train)
        val_loader = prepare_dataloader(val_features, y_val, batch_size=32)
        
        # Create model
        print("Initializing model...")
        model, optimizer = create_model(features)
        model.to(device)
        
        # Define loss function
        criterion = nn.MSELoss()
        
        # Training
        print("Training model...")
        num_epochs = 20  # Reduced for cross-validation speed
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = move_batch_to_device(X_batch, device)
                y_batch = y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Only print every few epochs
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss/len(train_loader):.6f}")
        
        # Evaluation
        print("Evaluating model...")
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = move_batch_to_device(X_batch, device)
                y_batch = y_batch.to(device)
                
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
                
                # Store predictions and targets
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        # Calculate metrics
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        # Convert back to original scale
        all_preds_orig = target_scaler.inverse_transform(all_preds)
        all_targets_orig = target_scaler.inverse_transform(all_targets)
        
        # Calculate metrics
        mse = mean_squared_error(all_targets_orig, all_preds_orig)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_targets_orig, all_preds_orig)
        r2 = r2_score(all_targets_orig, all_preds_orig)
        accuracy = 1 - val_loss / y_val.var().item()
        
        print(f"Fold {fold+1} Results:")
        print(f"  MSE: {mse:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        
        # Store metrics for this fold
        fold_metrics.append({
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'accuracy': accuracy
        })
    
    # Calculate average metrics across folds
    avg_metrics = {
        'mse': np.mean([m['mse'] for m in fold_metrics]),
        'rmse': np.mean([m['rmse'] for m in fold_metrics]),
        'mae': np.mean([m['mae'] for m in fold_metrics]),
        'r2': np.mean([m['r2'] for m in fold_metrics]),
        'accuracy': np.mean([m['accuracy'] for m in fold_metrics])
    }
    
    print("\n======= FINAL CROSS-VALIDATION RESULTS =======")
    print(f"Average MSE: {avg_metrics['mse']:.2f}")
    print(f"Average RMSE: {avg_metrics['rmse']:.2f}")
    print(f"Average MAE: {avg_metrics['mae']:.2f}")
    print(f"Average R²: {avg_metrics['r2']:.4f}")
    print(f"Average Accuracy: {avg_metrics['accuracy']:.4f}")
    
    return avg_metrics

def train_and_save_best_model():
    """
    Train the model on the full dataset and save it.
    """
    print("\n======= TRAINING FINAL MODEL =======\n")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    features, target = get_prepared_data(multimodal_format=True)
    
    # Create data loader with all data
    data_loader = prepare_dataloader(features, target, batch_size=32)
    
    # Create model
    model, optimizer = create_model(features)
    model.to(device)
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Training
    num_epochs = 50  # More epochs for final model
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in data_loader:
            X_batch = move_batch_to_device(X_batch, device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Only print every 5 epochs
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss/len(data_loader):.6f}")
    
    # Save model and scalers
    os.makedirs("saved_weights", exist_ok=True)
    torch.save(model.state_dict(), "saved_weights/multimodal_model_final.pth")
    joblib.dump(scaler, "saved_weights/feature_scaler.pkl")
    joblib.dump(target_scaler, "saved_weights/target_scaler.pkl")
    
    print("Final model saved as multimodal_model_final.pth")
    
    # Quick evaluation on training data
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_targets = []
        
        for X_batch, y_batch in data_loader:
            X_batch = move_batch_to_device(X_batch, device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
        
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        # Convert back to original scale
        all_preds_orig = target_scaler.inverse_transform(all_preds)
        all_targets_orig = target_scaler.inverse_transform(all_targets)
        
        # Calculate metrics
        mse = mean_squared_error(all_targets_orig, all_preds_orig)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_targets_orig, all_preds_orig)
        
        print(f"Final Model - Training RMSE: {rmse:.2f}, R²: {r2:.4f}")

if __name__ == "__main__":
    print("===== MULTIMODAL MOVIE GROSS PREDICTION =====")
    
    # Load data to show info
    features, target = get_prepared_data(multimodal_format=True)
    
    print("\n--- Data Information ---")
    print(f"Number of samples: {target.shape[0]}")
    print(f"Tabular features shape: {features['tabular_input'].shape}")
    print(f"Text input shape: {features['text_input']['input_ids'].shape}")
    
    print("\n--- Model Architecture ---")
    model, _ = create_model(features)
    print(model)
    
    # User choice
    choice = input("\nChoose an option:\n1. Run cross-validation evaluation\n2. Train and save final model\n3. Do both\nYour choice (1/2/3): ")
    
    if choice == '1':
        evaluate_multimodal_model()
    elif choice == '2':
        train_and_save_best_model()
    else:
        # Default: do both
        evaluate_multimodal_model()
        train_and_save_best_model() 