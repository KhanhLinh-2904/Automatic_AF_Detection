import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, num_classes=5):
        super(NeuralNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# class BinaryMLP(nn.Module):
#     def __init__(self):
#         super(BinaryMLP, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(3, 8),   # 1st hidden layer: 8 neurons
#             nn.ReLU(),
#             nn.Linear(8, 4),   # 2nd hidden layer: 4 neurons
#             nn.ReLU(),
#             nn.Linear(4, 1),
#             nn.Sigmoid()       # binary classification
#         )

#     def forward(self, x):
#         return self.model(x)
