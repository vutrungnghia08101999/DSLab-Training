import torch.nn as nn

class MLP(nn.Module):
    """Multi Layer Perceptron model
    """
    def __init__(self, n_features=14231, hidden_size=256, n_classes=20):
        super(MLP, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=hidden_size),
            nn.Sigmoid(),
            nn.Linear(in_features=hidden_size, out_features=n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):  # x.shape: batch_size x 14231
        return self.features(x)  # shape: batch_size x 20
