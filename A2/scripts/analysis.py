import json
from state import Token
import ast
import torch
import os
import numpy as np
from model import Parser
from model_utils import test

with open(os.getcwd() + '/A2/models/results.json', 'r') as file:
    res = json.load(file)

best_model = None
best_LAS = 0
for r in res:
    if max(r[2]) > best_LAS:
        best_model = r
        best_LAS = max(r[2])

def get_params(input_str):
    parts = input_str.split(" ")
    
    # Find the index of the keys and extract the subsequent values
    mean_val = parts[parts.index("Mean:") + 1] == "True"
    
    # Extract the tuple using ast.literal_eval to convert string tuple to actual tuple
    # Find the first opening parenthesis to extract tuple properly
    embedding_start_idx = input_str.find("(")
    # Find the closing parenthesis corresponding to the opening parenthesis
    embedding_end_idx = input_str.find(")", embedding_start_idx) + 1
    embedding_val_str = input_str[embedding_start_idx:embedding_end_idx]
    embedding_val = ast.literal_eval(embedding_val_str)
    
    # Extract learning rate (LR) value
    # Isolating LR value by slicing the string from "LR: " till the end and then splitting
    lr_val = float(input_str.split("LR: ")[1])
    
    return mean_val, embedding_val, lr_val

concat_6b_300 = []
concat_6b_50 = []
concat_42b_300 = []
concat_840b_300 = []
mean_6b_300 = []
mean_6b_50 = []
mean_42b_300 = []
mean_840b_300 = []
for r in res:
    mean_val, embedding_val, lr_val = get_params(r[0])
    if mean_val and embedding_val[0] == '6B' and embedding_val[1] == 300:
        mean_6b_300.append(r)
    elif mean_val and embedding_val[0] == '6B' and embedding_val[1] == 50:
        mean_6b_50.append(r)
    elif mean_val and embedding_val[0] == '42B' and embedding_val[1] == 300:
        mean_42b_300.append(r)
    elif mean_val and embedding_val[0] == '840B' and embedding_val[1] == 300:
        mean_840b_300.append(r)
    elif not mean_val and embedding_val[0] == '6B' and embedding_val[1] == 300:
        concat_6b_300.append(r)
    elif not mean_val and embedding_val[0] == '6B' and embedding_val[1] == 50:
        concat_6b_50.append(r)
    elif not mean_val and embedding_val[0] == '42B' and embedding_val[1] == 300:
        concat_42b_300.append(r)
    elif not mean_val and embedding_val[0] == '840B' and embedding_val[1] == 300:
        concat_840b_300.append(r)

best_model = None
best_LAS = 0
for r in concat_6b_300:
    if max(r[2]) > best_LAS:
        best_model = r
        best_LAS = max(r[2])
print(f"Concat 6B 300 Best Score LAS: {best_LAS}, UAS: {best_model[1][np.argmax(best_model[2])]}")

best_model = None
best_LAS = 0
for r in concat_6b_50:
    if max(r[2]) > best_LAS:
        best_model = r
        best_LAS = max(r[2])
print(f"Concat 6B 50 Best Score LAS: {best_LAS}, UAS: {best_model[1][np.argmax(best_model[2])]}")

best_model = None
best_LAS = 0
for r in concat_42b_300:
    if max(r[2]) > best_LAS:
        best_model = r
        best_LAS = max(r[2])
print(f"Concat 42B 300 Best Score LAS: {best_LAS}, UAS: {best_model[1][np.argmax(best_model[2])]}")

best_model = None
best_LAS = 0
for r in concat_840b_300:
    if max(r[2]) > best_LAS:
        best_model = r
        best_LAS = max(r[2])
print(f"Concat 840B 300 Best Score LAS: {best_LAS}, UAS: {best_model[1][np.argmax(best_model[2])]}")

best_model = None
best_LAS = 0
for r in mean_6b_300:
    if max(r[2]) > best_LAS:
        best_model = r
        best_LAS = max(r[2])
print(f"Mean 6B 300 Best Score LAS: {best_LAS}, UAS: {best_model[1][np.argmax(best_model[2])]}")

best_model = None
best_LAS = 0
for r in mean_6b_50:
    if max(r[2]) > best_LAS:
        best_model = r
        best_LAS = max(r[2])
print(f"Mean 6B 50 Best Score LAS: {best_LAS}, UAS: {best_model[1][np.argmax(best_model[2])]}")

best_model = None
best_LAS = 0
for r in mean_42b_300:
    if max(r[2]) > best_LAS:
        best_model = r
        best_LAS = max(r[2])
print(f"Mean 42B 300 Best Score LAS: {best_LAS}, UAS: {best_model[1][np.argmax(best_model[2])]}")

best_model = None
best_LAS = 0
for r in mean_840b_300:
    if max(r[2]) > best_LAS:
        best_model = r
        best_LAS = max(r[2])
print(f"Mean 840B 300 Best Score LAS: {best_LAS}, UAS: {best_model[1][np.argmax(best_model[2])]}")

word_list = "Mary had a little lamb .".split()
pos_list = "PROPN AUX DET ADJ NOUN PUNCT".split()
tokens = [Token(idx, word, pos) for idx, (word, pos) in enumerate(zip(word_list, pos_list))]
ex1_data = [(tokens)]

word_list = "I ate the fish raw .".split()
pos_list = "PRON VERB DET NOUN ADJ PUNCT".split()
tokens = [Token(idx, word, pos) for idx, (word, pos) in enumerate(zip(word_list, pos_list))]
ex2_data = [(tokens)]

word_list = "With neural networks , I love solving problems .".split()
pos_list = "ADP ADJ NOUN PUNCT PRON VERB VERB NOUN PUNCT".split()
tokens = [Token(idx, word, pos) for idx, (word, pos) in enumerate(zip(word_list, pos_list))]
ex3_data = [(tokens)]

mean_val, embedding_val, lr_val = get_params(best_model[0])
best_epoch = np.argmax(best_model[2])

model = Parser(d_emb=embedding_val[1], mean=mean_val).to("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(os.getcwd() + "/A2/models/model_mean_" + str(mean_val) + "_" + str(embedding_val[0]) +
                                 "_" + str(embedding_val[1]) + "_" + str(lr_val) + "_" + str(best_epoch) + ".pt"))

with open(os.getcwd() + '/A2/data/hidden.txt', 'r', encoding='utf-8') as file:
    data = []
    for line in file.readlines():
        # Split the line into three parts: words, pos, and labels
        words_part, pos_part = line.strip().split(" ||| ")
        
        # Split each part into individual elements
        words = words_part.split()
        pos = pos_part.split()
        
        # Ensure that words and pos have the same length
        if len(words) == len(pos):
            # create tokens
            tokens = [Token(idx, word, pos) for idx, (word, pos) in enumerate(zip(words, pos))]
            data.append((tokens))
        else:
            print("Mismatched lengths in line:", line)
            break

pred_actions, pred_dep = test(model, data, emb=embedding_val)

with open(os.getcwd() + '/A2/data/results.txt', 'w') as file:
    for actions in pred_actions:
        file.write(" ".join(actions) + "\n")

pred_actions, pred_dep_1 = test(model, ex1_data, emb=embedding_val)

pred_actions, pred_dep_2 = test(model, ex2_data, emb=embedding_val)

pred_actions, pred_dep_3 = test(model, ex3_data, emb=embedding_val)

