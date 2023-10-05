from tqdm import tqdm
import torch
import numpy as np
from evaluate import compute_metrics
from state import Token, ParseState, is_final_state, shift, left_arc, right_arc, is_legal

def train_loop(model, train_loader, dev_loader, dev_data, dev_raw_w, dev_raw_a, label_tags, lr, c=2):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    max_epochs = 20

    # Fix random seed for reproducibility
    torch.manual_seed(42)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for ep in range(1, max_epochs+1):
        print(f"Epoch {ep}")
        train_loss = []
        train_acc = []
        for w, p, lab in tqdm(train_loader):
            model.train()
            optimizer.zero_grad()
            w = w.to(device)
            p = p.to(device)
            lab = lab.to(device)
            out = model(w, p)

            loss= loss_fn(out, lab)
            acc = torch.sum(torch.argmax(out, dim=1) == lab)/len(lab)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_acc.append(acc.item())

        dev_iter = iter(dev_loader)
        pred_labels = []
        model.eval()
        for tokens, labels in tqdm(dev_data):
            stack = [Token(idx=-i-1, word="[NULL]", pos="NULL") for i in range(c)]
            parse_buffer = tokens.copy()
            ix = len(parse_buffer)
            parse_buffer.extend([Token(idx=ix+i+1, word="[NULL]", pos="NULL") for i in range(c)])
            dependencies = []
            state = ParseState(stack, parse_buffer, dependencies)
            label_list = []
            for _ in range(len(labels)):
                w, p, lab = next(dev_iter)
                w = w.to(device)
                p = p.to(device)
                out = model(w, p)
                legal = False
                while not legal:
                    pred_a = torch.argmax(out).item()
                    l = label_tags[pred_a]
                    legal = is_legal(l, state, c)
                    if not legal:
                        out[0][pred_a] = -1e9
                if l == "SHIFT":
                    shift(state)
                elif l.startswith("REDUCE_L"):
                    left_arc(state, l[9:])
                else:
                    right_arc(state, l[9:])
                label_list.append(l)
            pred_labels.append(label_list)    
            labels = []
        uas, las = compute_metrics(dev_raw_w, dev_raw_a, pred_labels)
        print(f"Average training batch loss: {np.mean(train_loss)}")
        print(f"Average training batch accuracy: {np.mean(train_acc)}")
        print(f"Dev set UAS: {uas}")
        print(f"Dev set LAS: {las}")

def eval(model, test_loader, dev_raw_w, dev_raw_a):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    
    for w, p, lab in tqdm(test_loader):
        model.eval()
        inp = inp.to(device)
        lab = lab.to(device)
        out = model(inp)
        pred_a = torch.argmax(out, dim=1)
        uas, las = compute_metrics(dev_raw_w, dev_raw_a, pred_a)

        
