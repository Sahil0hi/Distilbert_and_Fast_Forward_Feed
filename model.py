import torch
import torch.nn as nn
import torch.optim as optim


# Define model (feed-forward, two hidden layers)
# TODO: This is where most of the work will be done. You can change the model architecture,
#       add more layers, change activation functions, etc.
class MyModel(nn.Module):
    def __init__(self, input_dim):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(64, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.output_layer(x)
        return x


def create_model(features):
    model = MyModel(features.shape[1])

    # define optimizer (feel free to change this)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return model, optimizer

if __name__ == '__main__':
    # create sample model with 228 input features
    model, _ = create_model(torch.zeros(1, 228))
    print(model)