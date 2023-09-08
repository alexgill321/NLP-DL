#%%
import torch
from model import CBOW
import utils as ut
import os
import numpy as np


#%%
vocab_path = os.getcwd()+"/../vocab.txt"
vocab_dict = ut.get_word2ix(vocab_path)


model = CBOW(len(vocab_dict), 100)
model.load_state_dict(torch.load(os.getcwd()+"/../models/cbow_lr_0.001.pt"))

# Retrieve embeddings of every vocabulary word
embeddings = model.embedding.weight.data

# %%
vocab_dict["the"]
# %%
index = vocab_dict["the"]
embeddings[index]
print(len(embeddings[index]))
# %%
vocab_size = len(vocab_dict)
one_hot = torch.zeros(vocab_size)
one_hot[index] = 1.0  # Make it one-hot

# Transpose the weight matrix and perform matmul
W = model.embedding.weight.data
word_embedding_from_one_hot = torch.matmul(one_hot, W)
print(word_embedding_from_one_hot)
print(embeddings[index])
# %%
with open(os.getcwd() + '/../embeddings.txt', 'w') as f:
    f.write(f"{len(vocab_dict)} {100}\n")
    for key, value in vocab_dict.items():
        embedding = embeddings[value]
        f.write(f"{key} {' '.join([str(x) for x in embedding.tolist()])}\n")

# %%
def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

#%%
p1 = "cat"
p2 = "tiger"
p3 = "human"
p4 = "plane"
print(f'Cosine similarity {p1}, {p2}: {cosine_similarity(embeddings[vocab_dict[p1]], embeddings[vocab_dict[p2]])}')
print(f'Cosine similarity {p3}, {p4}: {cosine_similarity(embeddings[vocab_dict[p3]], embeddings[vocab_dict[p4]])}')
print(f'Euclidean distance {p1}, {p2}: {euclidean_distance(embeddings[vocab_dict[p1]], embeddings[vocab_dict[p2]])}')
print(f'Euclidean distance {p3}, {p4}: {euclidean_distance(embeddings[vocab_dict[p3]], embeddings[vocab_dict[p4]])}')


# %%
p1 = "my"
p2 = "mine"
p3 = "happy"
p4 = "human"
print(f'Cosine similarity {p1}, {p2}: {cosine_similarity(embeddings[vocab_dict[p1]], embeddings[vocab_dict[p2]])}')
print(f'Cosine similarity {p3}, {p4}: {cosine_similarity(embeddings[vocab_dict[p3]], embeddings[vocab_dict[p4]])}')
print(f'Euclidean distance {p1}, {p2}: {euclidean_distance(embeddings[vocab_dict[p1]], embeddings[vocab_dict[p2]])}')
print(f'Euclidean distance {p3}, {p4}: {euclidean_distance(embeddings[vocab_dict[p3]], embeddings[vocab_dict[p4]])}')
# %%
p1 = "happy"
p2 = "cat"
p3 = "king"
p4 = "princess"
print(f'Cosine similarity {p1}, {p2}: {cosine_similarity(embeddings[vocab_dict[p1]], embeddings[vocab_dict[p2]])}')
print(f'Cosine similarity {p3}, {p4}: {cosine_similarity(embeddings[vocab_dict[p3]], embeddings[vocab_dict[p4]])}')
print(f'Euclidean distance {p1}, {p2}: {euclidean_distance(embeddings[vocab_dict[p1]], embeddings[vocab_dict[p2]])}')
print(f'Euclidean distance {p3}, {p4}: {euclidean_distance(embeddings[vocab_dict[p3]], embeddings[vocab_dict[p4]])}')

# %%
p1 = "ball"
p2 = "racket"
p3 = "good"
p4 = "ugly"
print(f'Cosine similarity {p1}, {p2}: {cosine_similarity(embeddings[vocab_dict[p1]], embeddings[vocab_dict[p2]])}')
print(f'Cosine similarity {p3}, {p4}: {cosine_similarity(embeddings[vocab_dict[p3]], embeddings[vocab_dict[p4]])}')
print(f'Euclidean distance {p1}, {p2}: {euclidean_distance(embeddings[vocab_dict[p1]], embeddings[vocab_dict[p2]])}')
print(f'Euclidean distance {p3}, {p4}: {euclidean_distance(embeddings[vocab_dict[p3]], embeddings[vocab_dict[p4]])}')

# %%
p1 = "cat"
p2 = "racket"
p3 = "good"
p4 = "bad"
print(f'Cosine similarity {p1}, {p2}: {cosine_similarity(embeddings[vocab_dict[p1]], embeddings[vocab_dict[p2]])}')
print(f'Cosine similarity {p3}, {p4}: {cosine_similarity(embeddings[vocab_dict[p3]], embeddings[vocab_dict[p4]])}')
print(f'Euclidean distance {p1}, {p2}: {euclidean_distance(embeddings[vocab_dict[p1]], embeddings[vocab_dict[p2]])}')
print(f'Euclidean distance {p3}, {p4}: {euclidean_distance(embeddings[vocab_dict[p3]], embeddings[vocab_dict[p4]])}')

# %%
def emb_vec_sim(word1, word2, word3, embeddings, vocab_dict):
    emb1 = embeddings[vocab_dict[word1]]
    emb2 = embeddings[vocab_dict[word2]]
    emb3 = embeddings[vocab_dict[word3]]
    emb4 = emb1 - emb2 + emb3
    
    # Find the closest embedding to the result of the above operation
    max_sim = -1.0
    closest_word = None
    
    for word, index in vocab_dict.items():
        emb = embeddings[index]
        
        # Skip the input words
        if word in [word1, word2, word3]:
            continue
        
        sim = cosine_similarity(emb4, emb)
        
        if sim > max_sim:
            max_sim = sim
            closest_word = word
            
    return closest_word, max_sim
#%%
print(f'Result for queen - king + man: {emb_vec_sim("queen", "king", "man", embeddings, vocab_dict)}')
print(f'Result for queen - king + prince: {emb_vec_sim("queen", "king", "prince", embeddings, vocab_dict)}')
print(f'Result for man - king + queen: {emb_vec_sim("man", "king", "queen", embeddings, vocab_dict)}')
print(f'Result for man - woman + princess: {emb_vec_sim("man", "woman", "princess", embeddings, vocab_dict)}')
print(f'Result for princess - prince + man: {emb_vec_sim("princess", "prince", "man", embeddings, vocab_dict)}')

# %% Word Similarity
print(f'Similarity of cat and dog: {cosine_similarity(embeddings[vocab_dict["cat"]], embeddings[vocab_dict["dog"]])}')
print(f'Similarity of cat and for: {cosine_similarity(embeddings[vocab_dict["cat"]], embeddings[vocab_dict["for"]])}')

print(f'Similarity of spoon and bowl: {cosine_similarity(embeddings[vocab_dict["spoon"]], embeddings[vocab_dict["knife"]])}')
print(f'Similarity of spoon and human: {cosine_similarity(embeddings[vocab_dict["spoon"]], embeddings[vocab_dict["human"]])}')

print(f'Similarity of happy and sad: {cosine_similarity(embeddings[vocab_dict["happy"]], embeddings[vocab_dict["sad"]])}')
print(f'Similarity of happy and in: {cosine_similarity(embeddings[vocab_dict["happy"]], embeddings[vocab_dict["in"]])}')
# %% Word Analogy
print(f'Result for cat - dog + man: {emb_vec_sim("cat", "dog", "man", embeddings, vocab_dict)}')
print(f'Result for school - study + work: {emb_vec_sim("school", "study", "work", embeddings, vocab_dict)}')
print(f'Result for night - dark + light: {emb_vec_sim("night", "dark", "light", embeddings, vocab_dict)}')

# %%
