import torch
import torch.nn as nn


class CricketShotClassifier(nn.Module):
    def __init__(self):
        super(CricketShotClassifier,self).__init__()

        self.lstm = nn.LSTM(input_size = 51, hidden_size = 128, num_layers = 2, batch_first = True)

        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2) # Output layer for the two shot types
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        # Forward pass through the LSTM layer
        lstm_out,_ = self.lstm(x)
        lstm_out = lstm_out[:,-1,:]

        x = self.fc1(lstm_out)
        x = self.relu(x)
        x = self.fc2(x)

        x = self.softmax(x)

        return x

