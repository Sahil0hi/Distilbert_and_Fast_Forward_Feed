import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import joblib
from torch.serialization import add_safe_globals
import os
import numpy as np

from prep_data import get_prepared_data, get_all_titles
from model import create_model, MyModel  # Add MyModel for safe_globals

# Add your model class to safe globals
add_safe_globals([MyModel])

# you can call this function to test a pre-trained model (might be useful while testing)
def test_saved_model(model_path="saved_weights/test_model.pth"):
    # Check if all files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return
        
    # Create directory if it doesn't exist
    os.makedirs("saved_weights", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    features, target = get_prepared_data()
    
    # get list of movie titles
    titles = get_all_titles()
    
    # Create a new model with the right dimensions
    temp_model, _ = create_model(features)
    
    try:
        # Load model weights
        temp_model.load_state_dict(torch.load(model_path, map_location=device))
        model = temp_model
    except:
        # Try another loading method (for backward compatibility)
        try:
            model = torch.load(model_path, map_location=device, weights_only=False)
        except:
            model = torch.load(model_path, map_location=device)
    
    model.to(device)
    model.eval()
    
    # Define loss function
    criterion = torch.nn.MSELoss()
    
    # Try to load scalers if they exist
    try:
        feature_scaler = joblib.load("saved_weights/feature_scaler.pkl")
        target_scaler = joblib.load("saved_weights/target_scaler.pkl")
        use_scalers = True
    except:
        use_scalers = False
        print("Warning: Scalers not found, using raw predictions")

    # Move features to device
    features["tabular_input"] = features["tabular_input"].to(device)
    features["text_input"]["input_ids"] = features["text_input"]["input_ids"].to(device)
    features["text_input"]["attention_mask"] = features["text_input"]["attention_mask"].to(device)
    target = target.to(device)

    # Predict across all data
    with torch.no_grad():
        output = model(features)
        loss = criterion(output, target)
        print(f"\nTest Loss: {loss.item()}")
        # test accuracy
        print(f"Test Accuracy: {1 - loss.item() / target.var()}")

        # Move outputs back to CPU for numpy conversion
        output = output.cpu()
        target = target.cpu()
        
        # Inverse transform if using scalers
        if use_scalers:
            output_np = target_scaler.inverse_transform(output.numpy())
            target_np = target_scaler.inverse_transform(target.numpy())
        else:
            output_np = output.numpy()
            target_np = target.numpy()

        # print first 10 predictions against actual values
        # For <movie title>, model predicted <prediction>, actual <actual>
        print("\nSample Predictions vs Actual:")
        for i in range(10):
            print(f"For {titles[int(i)]}, model predicted {output_np[i][0]:.2f} vs. actual {target_np[i][0]:.2f}")

        # print best prediction and worst prediction
        # For <movie title>, model predicted <prediction>, actual <actual>
        print("\nBest and Worst Predictions:")
        errors = np.abs(output_np - target_np)
        worst = np.argmax(errors)
        best = np.argmin(errors)
        # convert to numpy arrays
        titles = titles.to_numpy()
        
        print(f"Best Prediction: For {titles[best]}, model predicted {output_np[best][0]:.2f} vs. actual {target_np[best][0]:.2f}")
        print(f"Worst Prediction: For {titles[worst]}, model predicted {output_np[worst][0]:.2f} vs. actual {target_np[worst][0]:.2f}")

# same function but using a new model
def test_new_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    features, target = get_prepared_data()
    
    # Import the training functions here to avoid circular imports
    try:
        from train import train_model, prepare_dataloader
    except ImportError:
        print("Error: Couldn't import train_model and prepare_dataloader from train.py")
        print("Make sure train.py has these functions and they're defined correctly")
        return
    
    # Split the data
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

    # Create data loaders
    train_loader = prepare_dataloader(train_features, y_train)
    val_loader = prepare_dataloader(val_features, y_val)
    
    # Create model
    model, optimizer = create_model(features)
    model.to(device)
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Train the model
    print("Training model...")
    model = train_model(model, optimizer, criterion, train_loader, val_loader, device, epochs=20)  # Increased from 5 to 20
    
    # Save the model weights
    os.makedirs("saved_weights", exist_ok=True)
    torch.save(model.state_dict(), "saved_weights/test_model.pth")
    print("Model saved as test_model.pth")
    
    # Save scalers
    try:
        from prep_data import scaler, target_scaler
        import joblib
        
        joblib.dump(scaler, "saved_weights/feature_scaler.pkl")
        joblib.dump(target_scaler, "saved_weights/target_scaler.pkl")
        print("Scalers saved successfully")
    except Exception as e:
        print(f"Warning: Could not save scalers: {e}")
    
    # Test the model
    test_saved_model("saved_weights/test_model.pth")

if __name__ == '__main__':
    test_new_model()