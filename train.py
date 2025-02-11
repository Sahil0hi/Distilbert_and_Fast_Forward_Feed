import torch
import torch.nn as nn

# import test_train_split and cross validation from sklearn
from sklearn.model_selection import train_test_split

# import data processing function from prep_data.py
from prep_data import get_prepared_data

# import model from model.py
from model import create_model


# TODO modify this function however you want to train the model
def train_model(model, optimizer, criterion, X_train, y_train, X_val, y_val, training_updates=True):

    num_epochs = 2000 # hint: you shouldn't need anywhere near this many epochs

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # X_train is already a dictionary with the expected structure
        output = model(X_train)
        loss = criterion(output.squeeze(), y_train)  # squeeze output to match target shape
        loss.backward()
        optimizer.step()
        if training_updates and epoch % (num_epochs // 10) == 0: # print training and validation loss 10 times across training
            with torch.no_grad():
                output = model(X_val)
                val_loss = criterion(output.squeeze(), y_val)
                print(f"Epoch {epoch} | Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")

    return model


# example training loop
if __name__ == '__main__':
    # Load data
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

    # Define loss function and optimizer
    criterion = nn.MSELoss()

    # train model
    model = train_model(model, optimizer, criterion, X_train, y_train, X_val, y_val)

    # basic evaluation (more in test.py)
    with torch.no_grad():
        output = model(X_val)
        loss = criterion(output.squeeze(), y_val)
        print(f"Final Validation Loss: {loss.item()}")
        # validation accuracy
        print(f"Final Validation Accuracy: {1 - loss.item() / y_val.var()}")

    # Save model
    torch.save(model, "saved_weights/model.pth")
    print("Model saved as model.pth")
