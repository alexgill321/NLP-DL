#%%
import os
import utils as ut
from torch.utils.data import Dataset, DataLoader
import torch
from model import CBOW
import numpy as np
from model_utils import train_loop, evaluate
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, help='Directory where model checkpoints will be saved')
args = parser.parse_args()

torch.manual_seed(69)
#%%
vocab_path = os.getcwd()+"/vocab.txt"
train_path = os.getcwd()+"/data/train"
test_path = os.getcwd()+"/data/dev"

#%%
vocab_dict = ut.get_word2ix(vocab_path)
train_files = ut.get_files(train_path)
test_files = ut.get_files(test_path)

# %%
context_size = 5
train_lines = ut.process_data(train_files, context_size, vocab_dict)
test_lines = ut.process_data(test_files, context_size, vocab_dict)

#%%
train_x, train_y = ut.generate_contexts(train_lines, context_size)
test_x, test_y = ut.generate_contexts(test_lines, context_size)

# %%
class CBOWDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
train_dataset = CBOWDataset(train_x, train_y)
test_dataset = CBOWDataset(test_x, test_y)
# %%
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# %%
model = CBOW(len(vocab_dict), 100).to("cuda" if torch.cuda.is_available() else "cpu")

#%%
save_dir = args.output_dir
# %%
lrs = [0.01, 0.001, .0001]
eval_losses = []
for lr in lrs:
    print(f"Learning rate: {lr}")
    model = CBOW(len(vocab_dict), 100).to("cuda" if torch.cuda.is_available() else "cpu")
    train_loop(model, train_loader, lr = lr)
    
    eval_losses.append(evaluate(model, test_loader))
    torch.save(model.state_dict(), f"{save_dir}/cbow_lr_{lr}.pt")
    
# %%
print(eval_losses)
# %%
print(f"Best learning rate: {lrs[np.argmin(eval_losses)]}")
# %%
