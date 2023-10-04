from tqdm import tqdm
import torch
import numpy as np

def train_loop(model, train_loader, lr):
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
        for inp, lab in tqdm(train_loader):
            model.train()
            optimizer.zero_grad()
            inp = inp.to(device)  # Converting to floats
            lab = lab.to(device)
            out = model(inp)

            
            loss= loss_fn(out, lab)
            acc = torch.sum(torch.argmax(out, dim=1) == lab)/len(lab)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_acc.append(acc.item())
        
        print(f"Average training batch loss: {np.mean(train_loss)}")
        print(f"Average training batch accuracy: {np.mean(train_acc)}")