import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)      # 1 input channel, 6 output, 5x5 kernel
        self.pool = nn.AvgPool2d(2, 2)       # 2x2 average pooling
        self.conv2 = nn.Conv2d(6, 16, 5)     # 6 input, 16 output
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 feature maps
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)         # output layer

    def forward(self, x):
        x = F.relu(self.conv1(x))     # (batch, 6, 28, 28)
        x = self.pool(x)              # (batch, 6, 14, 14)
        x = F.relu(self.conv2(x))     # (batch, 16, 10, 10)
        x = self.pool(x)              # (batch, 16, 5, 5)
        x = x.view(-1, 16 * 5 * 5)     # flatten
        x = F.relu(self.fc1(x))       # (batch, 120)
        x = F.relu(self.fc2(x))       # (batch, 84)
        x = self.fc3(x)               # (batch, 10)
        return x
