import torch
from torch import nn
import torch.nn.functional as F

''' 
The architecure for the DQN is simple and as follows.
There are two convolution layers with kernel size 5 and 32 filters. Both are activated used ReLu function.
There are 3 fully connected layers where the first two are activated using ReLU and the last is just the output layer.
'''

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        #convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)

        #fully connected layers
        self.fc1 = nn.Linear(32 * 6 * 7, 42)
        self.fc2 = nn.Linear(42, 42)
        self.out = nn.Linear(42, 7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        #flattens the input tensor to (batch, 32 * 6 * 7) for transition
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)