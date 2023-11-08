import torch
import torch.nn as nn
import torch.nn.functional as F

class charLSTM(nn.Module):
    def __init__(self, emb_dim, n_layers: int = 1):
        super(charLSTM, self).__init__()
        self.embedding = nn.Embedding(emb_dim, 50)
        self.lstm = nn.LSTM(input_size=50, hidden_size=200, num_layers=n_layers, batch_first=True)
        self.fc1 = nn.Linear(200, 300)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(300, 386)
        

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x