import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet96x96(nn.Module):
    def __init__(self, num_classes=1):
        super(ConvNet96x96, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # 96x96x32
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )  # 96x96x64
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )  # 96x96x128

        # Pooling layer
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # Reduces each spatial dimension by half

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 24 * 24, 512)  # Corrected size
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # Dropout (to prevent overfitting)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        residual = x
        # Convolutional layers with ReLU and pooling
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
