import torch.nn as nn
import torch.nn.functional as F


class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, num_classes,  dropout_rate=0.05):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(256, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.output = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout3(x)

        x = self.output(x)
        return x
    
