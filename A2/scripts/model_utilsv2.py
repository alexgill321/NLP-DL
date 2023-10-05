import torch
from state import Token, ParseState, is_legal, shift, left_arc, right_arc, is_final_state
from evaluate import compute_metrics
from tqdm import tqdm
from read_data import get_tag_dict, get_tag_dict_rev
import torchtext
import os

def train_loop(model, train_loader, dev_loader, lr, c=2, d_emb=300):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    label_tags = get_tag_dict(os.getcwd() + "/A2/data/tagset.txt")
    pos_tags = get_tag_dict(os.getcwd() + "/A2/data/pos_set.txt")
    
    max_epochs = 20


    # Fix random seed for reproducibility
    torch.manual_seed(42)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for ep in range(1, max_epochs+1):
        print(f"Epoch {ep}")
        train_loss = []
        for tokens, labels in tqdm(train_loader):
            model = model.train()
            stack = [Token(idx=-i-1, word="[NULL]", pos="NULL") for i in range(c)]
            parse_buffer = tokens.copy()
            ix = len(parse_buffer)
            parse_buffer.extend([Token(idx=ix+i+1, word="[NULL]", pos="NULL") for i in range(c)])
            dependencies = []
            state = ParseState(stack, parse_buffer, dependencies)

            for label in labels:
                optimizer.zero_grad()
                w_stack = [t.word for t in state.stack[-c:]]
                w_stack.reverse()
                p_stack = [t.pos for t in state.stack[-c:]]
                p_stack.reverse()
                w_buff = [t.word for t in state.parse_buffer[:c]]
                p_buff = [t.pos for t in state.parse_buffer[:c]]

                w = w_stack + w_buff
                w_emb = torchtext.vocab.GloVe(name='840B', dim=d_emb)
                w = w_emb.get_vecs_by_tokens(w, lower_case_backup=True)
                w = w.to(device)
                p = p_stack + p_buff
                p = [pos_tags[pos] for pos in p]
                p = torch.tensor(p).to(device)
                label_tag = label_tags[label]
                label_tag = torch.tensor(label_tag).to(device)
                out = model(w, p)

                loss = loss_fn(out, label_tag)
                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())

                assert is_legal(label, state, c)

                if label == "SHIFT":
                    shift(state)
                elif label.startswith("REDUCE_L"):
                    left_arc(state, label)
                elif label.startswith("REDUCE_R"):
                    right_arc(state, label)
            assert is_final_state(state, c)
        print(f"Average Train Loss: {sum(train_loss)/len(train_loss)}")