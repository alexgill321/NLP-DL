from read_data import parse_file, emb_parse, get_tag_dict, get_tag_dict_rev
import os
from state import generate_from_data
from torch.utils.data import Dataset, DataLoader
import torch
from model import Parser
from model_utils import train_loop
import torchtext
import json


train_data = parse_file(os.getcwd() + "/A2/data/train.txt")
dev_data = parse_file(os.getcwd() + "/A2/data/dev.txt")
test_data = parse_file(os.getcwd() + "/A2/data/test.txt")
label_tags = get_tag_dict(os.getcwd() + "/A2/data/tagset.txt")
pos_tags = get_tag_dict(os.getcwd() + "/A2/data/pos_set.txt")
tags_to_labels = get_tag_dict_rev(os.getcwd() + "/A2/data/tagset.txt")

train_w, train_p, train_y = generate_from_data(train_data, label_tags, pos_tags)

class DependencyDataset(Dataset):
    def __init__(self, w, p, y):
        self.x = [w_emb.get_vecs_by_tokens(w, lower_case_backup=True) for w in w]
        self.p = torch.tensor(p)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.p[idx], self.y[idx]

batch_size = 512

emb_list = [('6B', 50),('6B', 300), ('42B', 300), ('840B', 300)]
mean = [False, True]

res = []
for m in mean:
    for emb in emb_list:
        w_emb = torchtext.vocab.GloVe(name=emb[0], dim=emb[1])
        train_dataset = DependencyDataset(train_w, train_p, train_y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model = Parser(d_emb=emb[1], mean=m).to("cuda" if torch.cuda.is_available() else "cpu")
        for lr in [0.01, 0.001, 0.0001]:
            uas,las = train_loop(model, train_loader, dev_data, lr, emb=emb, save_dir=os.getcwd() + "/A2/models/")
            params = "Model: Mean: " + str(m) + " Embedding: " + str(emb) + " LR: " + str(lr)
            res.append((params, uas, las))

            with open('A2/models/results.json', 'w') as file:
                json.dump(res, file)
