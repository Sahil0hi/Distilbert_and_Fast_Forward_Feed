import torch
import torch.nn as nn

# import test_train_split and cross validation from sklearn
from sklearn.model_selection import train_test_split

# import data processing function from prep_data.py
from prep_data import get_prepared_data

# import model from model.py
from model import create_model

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to train the model
def train_model(model, optimizer, criterion, X_train, y_train, X_val, y_val, training_updates=True):

    num_epochs = 2000  # hint: you shouldn’t need this many epochs

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        if training_updates and epoch % (num_epochs // 10) == 0:
            model.eval()
            with torch.no_grad():
                val_output = model(X_val)
                val_loss = criterion(val_output, y_val)
                print(f"Epoch {epoch} | Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")

    return model

# Example training loop
if __name__ == '__main__':
    # Load data
    features, target = get_prepared_data()

    # Create training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2)

    # Convert data to torch tensors and move to device
    X_train = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
    X_val = torch.tensor(X_val.values, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(device)

    # Define model and move to device
    model, optimizer = create_model(X_train)
    model = model.to(device)

    # Define loss function
    criterion = nn.MSELoss()

    # Train model
    model = train_model(model, optimizer, criterion, X_train, y_train, X_val, y_val)

    # Basic evaluation
    model.eval()
    with torch.no_grad():
        output = model(X_val)
        loss = criterion(output, y_val)
        print(f"Final Validation Loss: {loss.item()}")
        print(f"Final Validation Accuracy: {1 - loss.item() / y_val.var()}")

    # Save model
    torch.save(model, "saved_weights/model.pth")
    print("Model saved as model.pth")
