import torch

class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        
        # Projection layer: maps one-hot encoded vectors into embedding space
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        
        # Classifier layer: maps from the embedding space back to the vocabulary
        self.fc = torch.nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, context_indices):
        # Context_indices shape: (batch_size, 2*context_window)
        
        # Project the context word indices into the embedding space
        embeddings = self.embedding(context_indices)
        
        # Sum the embeddings along dimension 1 to get h0
        h0 = torch.sum(embeddings, dim=1)
        
        # Pass h0 through the classifier layer
        z = self.fc(h0)
        
        return z