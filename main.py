# DO NOT CHANGE ANYTHING IN THIS FILE

from model import create_model
from prep_data import get_prepared_data
from train import train_model

from tqdm import tqdm

def final_cross_validation_test():
    print("Running 5-fold cross validation test...")

    from sklearn.model_selection import KFold, cross_val_score
    import torch
    import torch.nn as nn

    # Load data
    features, target = get_prepared_data()

    # shuffle data
    indices = torch.randperm(features.shape[0])
    features = features[indices]
    target = target[indices]

    # Define loss function
    criterion = nn.MSELoss()

    # Define the K-fold Cross Validator
    k_folds = 5
    kf = KFold(n_splits=k_folds)

    # Store the results
    accuracy_results = []
    loss_results = []
    with tqdm(total=k_folds) as pbar:
        for train_index, test_index in kf.split(features):
            # Split data
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = target[train_index], target[test_index]

            # Define model
            model, optimizer = create_model(X_train)

            # Train model
            model = train_model(model, optimizer, criterion, X_train, y_train, X_test, y_test, training_updates=False)

            # Evaluate
            with torch.no_grad():
                output = model(X_test)
                loss = criterion(output, y_test)
                # store loss and accuracy
                accuracy_results.append(1 - loss.item() / y_test.var())
                loss_results.append(loss.item())

            pbar.update(1)

    # Print out average performance
    print(f"Cross Validation Test Accuracy: {float(sum(accuracy_results)) / k_folds}%")
    print(f"Cross Validation Test Loss: {float(sum(loss_results)) / k_folds}")

# DO NOT CHANGE THIS FUNCTION
if __name__ == '__main__':
    # print information about the final data after processing
    features, target = get_prepared_data()
    print(f"Features shape: {features.shape}")

    # print model summary
    model, optimizer = create_model(features)
    print(model)

    # run final cross validation test
    final_cross_validation_test()
