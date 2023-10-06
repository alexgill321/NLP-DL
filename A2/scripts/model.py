import torch
import torchtext


class Parser(torch.nn.Module):
    def __init__(self, d_emb=50, h=200, T=76, P=18, d_pos=50, c=2, mean=True):
        super(Parser, self).__init__()

        self.d_emb = d_emb
        self.h = h
        self.T = T
        self.P = P
        self.d_pos = d_pos
        self.c = c
        self.mean_embedding = mean
        self.w_input_dim = self.d_emb if mean else 2*self.c*self.d_emb
        self.p_input_dim = self.d_pos if mean else 2*self.c*self.d_pos

        self.pos_emb = torch.nn.Embedding(self.P, self.d_pos)

        self.h_w = torch.nn.Linear(self.w_input_dim, self.h, bias=False)
        self.h_p = torch.nn.Linear(self.p_input_dim, self.h, bias=False)

        self.b = torch.nn.Parameter(torch.zeros(self.h))

        self.out = torch.nn.Linear(self.h, self.T, bias=False)
        self.b_out = torch.nn.Parameter(torch.zeros(self.T))


    def forward(self, w, p):
        w_emb = torch.mean(w, dim=1) if self.mean_embedding else torch.flatten(w, start_dim=1)

        pos_emb = self.pos_emb(p)
        pos_emb = torch.mean(pos_emb, dim=1) if self.mean_embedding else torch.flatten(pos_emb, start_dim=1)

        h_w = self.h_w(w_emb)
        h_p = self.h_p(pos_emb)

        h = torch.relu(h_w + h_p + self.b)
        output_logits = self.out(h) + self.b_out
        out = torch.log_softmax(output_logits, dim=1)
        
        return out


class ParserDep(torch.nn.Module):
    def __init__(self, d_emb=50, h=200, T=76, P=18, d_pos=50, d_dep=50, c=2, mean=True):
        super(ParserDep, self).__init__()

        self.d_emb = d_emb
        self.h = h
        self.T = T
        self.P = P
        self.d_pos = d_pos
        self.d_dep = d_dep
        self.c = c
        self.mean_embedding = mean
        self.w_input_dim = self.d_emb if mean else 4*self.c*self.d_emb
        self.p_input_dim = self.d_pos if mean else 2*self.c*self.d_pos
        self.l_input_dim = self.d_dep if mean else 2*self.c*self.d_dep

        self.pos_emb = torch.nn.Embedding(self.P, self.d_pos)
        self.dep_emb = torch.nn.Embedding(self.T, self.d_dep)

        self.h_w = torch.nn.Linear(self.w_input_dim, self.h, bias=False)
        self.h_p = torch.nn.Linear(self.p_input_dim, self.h, bias=False)
        self.h_l = torch.nn.Linear(self.l_input_dim, self.h, bias=False)

        self.b = torch.nn.Parameter(torch.zeros(self.h))

        self.out = torch.nn.Linear(self.h, self.T, bias=False)
        self.b_out = torch.nn.Parameter(torch.zeros(self.T))


    def forward(self, w, p, l):
        w_emb = torch.mean(w, dim=1) if self.mean_embedding else torch.flatten(w, start_dim=1)

        pos_emb = self.pos_emb(p)
        pos_emb = torch.mean(pos_emb, dim=1) if self.mean_embedding else torch.flatten(pos_emb, start_dim=1)

        dep_emb = self.dep_emb(l)
        dep_emb = torch.mean(dep_emb, dim=1) if self.mean_embedding else torch.flatten(dep_emb, start_dim=1)

        h_w = self.h_w(w_emb)
        h_p = self.h_p(pos_emb)
        h_l = self.h_l(dep_emb)

        h = torch.relu(h_w + h_p + h_l + self.b)
        output_logits = self.out(h) + self.b_out
        out = torch.log_softmax(output_logits, dim=1)
        
        return out