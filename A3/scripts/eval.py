import torch
import os
import pickle
from run_lstm import train_loop
from preprocess import get_counts
import utils as ut
from utils import characterDataset
from lstm import charLSTM
import tqdm
import torch.nn as nn
import utils as ut

def predict(model, data, vocab, iters = 200):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rev_vocab = {v:k for k,v in vocab.items()}
    model.to(device)
    model.eval()
    predictions = []
    for row in data:
        pred = []
        chars = ut.convert_line2idx(row, vocab)
        int_len = len(chars)
        while len(chars) < int_len + iters:
            x = torch.tensor(chars, dtype=torch.long).to(device)
            x = x.reshape(1, -1)
            emb_x = model.embedding(x)
            emb_x = emb_x.clone().detach().requires_grad_(False)
            out = model(emb_x).to(device)
            last = out[:,-1,:]
            next_char = torch.multinomial(torch.softmax(last, dim=1), 1)
            chars.append(next_char.item())
        for c in chars:
            pred.append(rev_vocab[c])
        predictions.append(''.join(pred))
    return predictions

def main():
    train_dataset = torch.load(os.getcwd() + '/A3/data/preprocessed/train.pt')

    test_dataset = torch.load(os.getcwd() + '/A3/data/preprocessed/test.pt')

    with open(os.getcwd() + '/A3/data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    raw_train_data = ut.convert_files2idx(ut.get_files(os.getcwd() + '/A3/data/train'), vocab)

    counts = get_counts(raw_train_data, vocab)

    sum = 0

    for count in counts.values():
        sum += count
    
    loss_weights = []
    for i in range(len(vocab)):
        loss_weights.append(1- counts[i]/sum)
    
    for n_layers in range(2, 3):
        model = charLSTM(emb_dim=386, n_layers=n_layers)

        sum = 0
        for p in model.parameters():
            sum += p.numel()
        print(sum)

        train_loop(model, train_loader=train_loader, dev_loader=test_loader,
                    loss_weights=torch.tensor(loss_weights, dtype=torch.float32), epochs=5, lr=0.0001)
        
        predictions = []
        data = [
            "The little boy was",
            "Once upon a time in",
            "With the target in",
            "Capitals are big cities. For example,",
            "A cheap alternative to"
        ]
        predictions.append(predict(model, data, vocab))
        for pred in predictions:
            print(pred)
        
if __name__ == "__main__":
    main()
