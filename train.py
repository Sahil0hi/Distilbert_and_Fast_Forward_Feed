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
        optimizer.step()

        if training_updates and epoch % max(1, (num_epochs // 10)) == 0:
            model.eval()
            with torch.no_grad():
                val_output = model(X_val)
                val_loss = criterion(val_output, y_val)
            print(f"Epoch {epoch} | Training Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

    return model


if __name__ == '__main__':
    # 1. Load data from prep_data.py
    features, target = get_prepared_data()

    # features is dict:
    #   {
    #     "text_input": {
    #       "input_ids": shape [N, seq_len],
    #       "attention_mask": shape [N, seq_len]
    #     },
    #     "tabular_input": shape [N, tab_feat_dim]
    #   }
    # target is shape [N, 1]

    # 2. We split each relevant part for train/val
    X_tab = features["tabular_input"]
    X_ids = features["text_input"]["input_ids"]
    X_mask = features["text_input"]["attention_mask"]

    (X_tab_train, X_tab_val,
     X_ids_train, X_ids_val,
     X_mask_train, X_mask_val,
     y_train, y_val) = train_test_split(
         X_tab,
         X_ids,
         X_mask,
         target,
         test_size=0.2,
         random_state=42
    )

    # 3. Rebuild the dictionary for training & validation
    train_features = {
        "text_input": {
            "input_ids": X_ids_train,
            "attention_mask": X_mask_train
        },
        "tabular_input": X_tab_train
    }

    val_features = {
        "text_input": {
            "input_ids": X_ids_val,
            "attention_mask": X_mask_val
        },
        "tabular_input": X_tab_val
    }

    # 4. Create model & optimizer
    model, optimizer = create_model(train_features)

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

    # 8. Save model
    torch.save(model, "saved_weights/model.pth")
    print("Model saved as model.pth")
