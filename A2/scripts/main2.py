from read_data import parse_file
from torch.utils.data import Dataset, DataLoader
import torch
import os
from model import Parser
from model_utilsv2 import train_loop


train_data = parse_file(os.getcwd() + "/A2/data/train.txt")
dev_data = parse_file(os.getcwd() + "/A2/data/dev.txt")
test_data = parse_file(os.getcwd() + "/A2/data/test.txt")

class TokenDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
train_dataset = TokenDataset(train_data)
dev_dataset = TokenDataset(dev_data)
test_dataset = TokenDataset(test_data)

batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = Parser(d_emb=300).to("cuda" if torch.cuda.is_available() else "cpu")

train_loop(model, train_data, dev_loader, 0.001)
