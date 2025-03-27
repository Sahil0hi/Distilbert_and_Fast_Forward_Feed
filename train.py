import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from prep_data import get_prepared_data
from model import create_model

def train_model(model, optimizer, criterion, X_train, y_train, X_val, y_val, training_updates=True):
    """
    Trains the model with DistilBERT + tabular data.
    X_train and X_val are dicts:
      {
         "text_input": {"input_ids": ..., "attention_mask": ...},
         "tabular_input": ...
      }
    y_train and y_val are tensors of shape [N, 1].
    """
    num_epochs = 2000  # typically fewer is enough

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # forward pass on training
        train_output = model(X_train)
        train_loss = criterion(train_output, y_train)

        train_loss.backward()
        # X_train is already a dictionary with the expected structure
        output = model(X_train)
        loss = criterion(output.squeeze(), y_train)  # squeeze output to match target shape
        loss.backward()
        optimizer.step()

        if training_updates and epoch % max(1, (num_epochs // 10)) == 0:
            model.eval()
            with torch.no_grad():
                val_output = model(X_val)
                val_loss = criterion(val_output, y_val)
            print(f"Epoch {epoch} | Training Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")
            output = model(X_val)
            val_loss = criterion(output.squeeze(), y_val)
            print(f"Epoch {epoch} | Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")

    return model


if __name__ == '__main__':
    # 1. Load data from prep_data.py
    features, target = get_prepared_data()
    
    # Debug prints
    print("Features structure:")
    print(f"Type of features: {type(features)}")
    for key in features:
        print(f"\nKey: {key}")
        if isinstance(features[key], dict):
            for subkey in features[key]:
                print(f"  Subkey: {subkey}")
                print(f"  Shape: {features[key][subkey].shape if hasattr(features[key][subkey], 'shape') else 'No shape'}")
        else:
            print(f"Shape: {features[key].shape if hasattr(features[key], 'shape') else 'No shape'}")
    
    print("\nTarget shape:", target.shape if hasattr(target, 'shape') else 'No shape')

    # Convert target to tensor if it isn't already
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target, dtype=torch.float32)

    # Create custom train/val split that preserves dictionary structure
    indices = list(range(len(target)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2)
    
    # Convert indices to torch tensors for indexing
    train_idx = torch.tensor(train_idx)
    val_idx = torch.tensor(val_idx)

    # Split the features dictionary using proper tensor indexing
    X_train = {
        "text_input": {
            "input_ids": features["text_input"]["input_ids"].index_select(0, train_idx),
            "attention_mask": features["text_input"]["attention_mask"].index_select(0, train_idx)
        },
        "tabular_input": features["tabular_input"].index_select(0, train_idx)
    }
    
    X_val = {
        "text_input": {
            "input_ids": features["text_input"]["input_ids"].index_select(0, val_idx),
            "attention_mask": features["text_input"]["attention_mask"].index_select(0, val_idx)
        },
        "tabular_input": features["tabular_input"].index_select(0, val_idx)
    }

    y_train = target[train_idx]
    y_val = target[val_idx]

    # Create model
    model, optimizer = create_model(X_train)

    # 5. Define loss
    criterion = nn.MSELoss()

    # 6. Train
    model = train_model(model, optimizer, criterion, train_features, y_train, val_features, y_val)

    # 7. Evaluate
    model.eval()
    with torch.no_grad():
        val_output = model(val_features)
        val_loss = criterion(val_output, y_val).item()
        val_var = y_val.var().item()
    val_acc = 1 - (val_loss / val_var)

    print(f"Final Validation Loss: {val_loss:.4f}")
    print(f"Final Validation Accuracy: {val_acc:.4f}")
    output = model(X_val)
    loss = criterion(output.squeeze(), y_val)
    print(f"Final Validation Loss: {loss.item()}")
    # validation accuracy
    print(f"Final Validation Accuracy: {1 - loss.item() / y_val.var()}")

    # 8. Save model
    torch.save(model, "saved_weights/model.pth")
    print("Model saved as model.pth")
