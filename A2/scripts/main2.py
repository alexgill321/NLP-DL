from read_data import parse_file, get_tag_dict
from torch.utils.data import Dataset, DataLoader
from state import generate_from_data_with_dep
import torch
import os
from model import ParserDep
from model_utils import train_loop_dep
import torchtext


train_data = parse_file(os.getcwd() + "/A2/data/train.txt")
dev_data = parse_file(os.getcwd() + "/A2/data/dev.txt")
test_data = parse_file(os.getcwd() + "/A2/data/test.txt")
label_tags = get_tag_dict(os.getcwd() + "/A2/data/tagset.txt")
pos_tags = get_tag_dict(os.getcwd() + "/A2/data/pos_set.txt")


train_w, train_p, train_dep, train_y = generate_from_data_with_dep(train_data, label_tags, pos_tags)

w_emb = torchtext.vocab.GloVe(name='6B', dim=300)

class TokenDataset(Dataset):
    def __init__(self, w, p, dep, y):
        self.x = [w_emb.get_vecs_by_tokens(w, lower_case_backup=True) for w in w]
        self.p = torch.tensor(p)
        self.l = torch.tensor(dep)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.p[idx], self.l[idx], self.y[idx]
    
train_dataset = TokenDataset(train_w, train_p, train_dep, train_y)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

model = ParserDep(d_emb=300).to("cuda" if torch.cuda.is_available() else "cpu")

train_loop_dep(model, train_loader, dev_data, lr=0.001, emb = ('6B', 300), save_dir=os.getcwd() + "/A2/models/dep/")
