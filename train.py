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
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        if training_updates and epoch % (num_epochs // 10) == 0: # print training and validation loss 10 times across training
            with torch.no_grad():
                output = model(X_val)
                val_loss = criterion(output, y_val)
                print(f"Epoch {epoch} | Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")

    return model


# example training loop
if __name__ == '__main__':
    # Load data
    features, target = get_prepared_data()

    # create training and validation sets
    # use 80% of the data for training and 20% for validation
    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2)

    # Define model (feed-forward, two hidden layers)
    model, optimizer = create_model(X_train)

    # Define loss function and optimizer
    criterion = nn.MSELoss()

    # train model
    model = train_model(model, optimizer, criterion, X_train, y_train, X_val, y_val)

    # basic evaluation (more in test.py)
    with torch.no_grad():
        output = model(X_val)
        loss = criterion(output, y_val)
        print(f"Final Validation Loss: {loss.item()}")
        # validation accuracy
        print(f"Final Validation Accuracy: {1 - loss.item() / y_val.var()}")

    # Save model
    torch.save(model, "saved_weights/model.pth")
    print("Model saved as model.pth")