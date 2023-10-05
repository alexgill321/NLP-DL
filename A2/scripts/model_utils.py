from tqdm import tqdm
import torch
import numpy as np
from evaluate import compute_metrics
from state import Token, ParseState, is_final_state, shift, left_arc, right_arc, is_legal
from read_data import get_tag_dict, get_tag_dict_rev
import torchtext
import os

def train_loop(model, train_loader, dev_data, lr, c=2, emb=('840B', 300), save_dir=None):
    name, d_emb = emb
    w_emb = torchtext.vocab.GloVe(name=name, dim=d_emb)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pos_tags = get_tag_dict(os.getcwd() + "/A2/data/pos_set.txt")
    tags_to_labels = get_tag_dict_rev(os.getcwd() + "/A2/data/tagset.txt")

    max_epochs = 20

    # Fix random seed for reproducibility
    torch.manual_seed(42)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.NLLLoss()

    uas_list = []
    las_list = []
    for ep in range(1, max_epochs+1):
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
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

        model.eval()
        device = torch.device('cpu')
        model.to(device)
        sentences = []
        pred_actions = []
        gold_actions = []
        for tokens, actions in tqdm(dev_data):
            stack = [Token(idx=-i-1, word="[NULL]", pos="NULL") for i in range(c)]
            parse_buffer = tokens.copy()
            ix = len(parse_buffer)
            parse_buffer.extend([Token(idx=ix+i+1, word="[NULL]", pos="NULL") for i in range(c)])
            dependencies = []
            state = ParseState(stack, parse_buffer, dependencies)
            
            sentences.append([token.word for token in tokens])
            gold_actions.append([action for action in actions])
            action_list = []
            while not is_final_state(state, c):
                w_stack = [t.word for t in state.stack[-c:]]
                w_stack.reverse()
                p_stack = [t.pos for t in state.stack[-c:]]
                p_stack.reverse()
                w_buff = [t.word for t in state.parse_buffer[:c]]
                p_buff = [t.pos for t in state.parse_buffer[:c]]

                w = w_stack + w_buff
                p = p_stack + p_buff
                w_list = [w for w in w]
                w = w_emb.get_vecs_by_tokens(w_list, lower_case_backup=True) 
                w = w.view(1, 2*c, d_emb)
                w = w.to(device)
                p = [[pos_tags[pos] for pos in p]]
                p = torch.tensor(p).to(device)
                out = model(w, p)
                pred_index = torch.argmax(out, dim=1)
                pred_a = tags_to_labels[pred_index.item()]

                while not is_legal(pred_a, state, c):
                    out[0][pred_index] = -100
                    pred_index = torch.argmax(out, dim=1)
                    pred_a = tags_to_labels[pred_index.item()]

                if pred_a == "SHIFT":
                    shift(state)
                elif pred_a.startswith("REDUCE_L"):
                    left_arc(state, pred_a[9:])
                else:
                    right_arc(state, pred_a[9:])
                action_list.append(pred_a)
            pred_actions.append(action_list)
        uas, las = compute_metrics(sentences, gold_actions, pred_actions, cwindow=c)
        uas_list.append(uas)
        las_list.append(las)
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(model.state_dict(), save_dir + f"model_mean_{model.mean_embedding}_{name}_{d_emb}_{lr}_{ep}.pt")
        print(f"Average training batch loss: {np.mean(train_loss)}")
        print(f"Average training batch accuracy: {np.mean(train_acc)}")
        print(f"Dev set UAS: {uas}")
        print(f"Dev set LAS: {las}")
    return uas_list, las_list
        
