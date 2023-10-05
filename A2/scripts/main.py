from read_data import parse_file, get_tag_dict, get_tag_dict_rev
import os
from state import generate_from_data, generate_from_data_raw
from torch.utils.data import Dataset, DataLoader
import torch
from model import Parser
from model_utils import train_loop
import torchtext


train_data = parse_file(os.getcwd() + "/A2/data/train.txt")
dev_data = parse_file(os.getcwd() + "/A2/data/dev.txt")
test_data = parse_file(os.getcwd() + "/A2/data/test.txt")
label_tags = get_tag_dict(os.getcwd() + "/A2/data/tagset.txt")
pos_tags = get_tag_dict(os.getcwd() + "/A2/data/pos_set.txt")
tags_to_labels = get_tag_dict_rev(os.getcwd() + "/A2/data/tagset.txt")

train_w, train_p, train_y = generate_from_data(train_data, label_tags, pos_tags)
dev_raw_sent, dev_raw_a = generate_from_data_raw(dev_data)
dev_w, dev_p, dev_y = generate_from_data(dev_data, label_tags, pos_tags)
test_w, test_p, test_y = generate_from_data(test_data, label_tags, pos_tags)

w_emb = torchtext.vocab.GloVe(name='840B', dim=300)

class DependencyDataset(Dataset):
    def __init__(self, w, p, y):
        self.x = [w_emb.get_vecs_by_tokens(w, lower_case_backup=True) for w in w]
        self.p = torch.tensor(p)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.p[idx], self.y[idx]
    
train_dataset = DependencyDataset(train_w, train_p, train_y)
dev_dataset = DependencyDataset(dev_w, dev_p, dev_y)
test_dataset = DependencyDataset(test_w, test_p, test_y)

batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = Parser(d_emb=300).to("cuda" if torch.cuda.is_available() else "cpu")

train_loop(model, train_loader, dev_loader, dev_data, dev_raw_sent, dev_raw_a, tags_to_labels, 0.001)