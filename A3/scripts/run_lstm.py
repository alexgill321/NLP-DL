import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from preprocess import get_counts
from utils import characterDataset
from lstm import charLSTM
import pickle
import utils as ut
import tqdm
import numpy as np

def train_loop(model, train_loader, dev_loader, loss_weights, epochs=5, lr=0.0001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_weights = loss_weights.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=loss_weights, ignore_index=384)
    torch.manual_seed(42)
    model.to(device)
    print("Using Device: ", device)
    for epoch in range(epochs):
        print("Training Epoch: ", epoch + 1)
        with tqdm.tqdm(train_loader) as pbar:
            for batch in train_loader:
                model.train()
                optimizer.zero_grad()
                x, y = batch
                tochar(x)
                x = torch.tensor(x).to(device)
                y = torch.tensor(y, dtype=torch.long).to(device)
                emb_x = model.embedding(x)
                emb_x = emb_x.clone().detach().requires_grad_(True)
                out = model(emb_x).to(device)
                out = torch.transpose(out, 1, 2)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()
                pbar.set_description("Loss: %f" % (loss.item()))
                pbar.update(1)
        print("Evaluating on Dev Set")
        dev_losses = []
        with tqdm.tqdm(dev_loader) as pbar:
            for batch in dev_loader:
                model.eval()
                x, y = batch
                x = torch.tensor(x, requires_grad=False).to(device)
                y = torch.tensor(y, dtype=torch.long).to(device)
                emb_x = model.embedding(x)
                emb_x = emb_x.clone().detach().requires_grad_(False)
                out = model(emb_x).to(device)
                out = torch.transpose(out, 1, 2)
                loss = nn.CrossEntropyLoss(weight=loss_weights, ignore_index=384)(out, y)
                dev_losses.append(loss.item())
                pbar.set_description("Loss: %f" % (loss.item()))
                pbar.update(1)
        print("Dev Loss: ", np.mean(dev_losses))
        print("Dev Perplexity: ", 2**np.mean(dev_losses))

def tochar(x):
    for i in range(len(x)):
        x_char = []
        for j in range(len(x[i])):
            x_char.append(inv_vocab[x[i][j]])
        print(x_char)

train_dataset = torch.load(os.getcwd() + '/A3/data/preprocessed/train.pt')
dev_dataset = torch.load(os.getcwd() + '/A3/data/preprocessed/dev.pt')

with open(os.getcwd() + '/A3/data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

inv_vocab = {v: k for k, v in vocab.items()}

raw_train_data = ut.convert_files2idx(ut.get_files(os.getcwd() + '/A3/data/train'), vocab)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
dev_loader = DataLoader(dev_dataset, batch_size=128, shuffle=False)

model = charLSTM(emb_dim=386, n_layers=2)

counts = get_counts(raw_train_data, vocab)

sum = 0
for count in counts.values():
    sum += count

loss_weights = []
for i in range(len(vocab)):
    loss_weights.append(1- counts[i]/sum)

train_loop(model, train_loader=train_loader, dev_loader=dev_loader,
            loss_weights=torch.tensor(loss_weights, dtype=torch.float32))