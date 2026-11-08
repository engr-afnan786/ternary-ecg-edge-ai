"""FP32 ECG baseline. Input is normalized MLII, shape [batch, 1, 180]."""
from torch import nn


class ECGFP32(nn.Sequential):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, 7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(8, 16, 5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.average = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1)
        self.head = nn.Linear(16, 2)
