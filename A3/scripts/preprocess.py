import pickle
import os
import utils as ut
import torch
import numpy as np
import tqdm

from typing import List
from utils import characterDataset

from collections import OrderedDict
    

def preprocess(data: List[List[int]], vocab: dict, k: int = 500):
    """
    Preprocesses the text by replacing the words with their index in the vocab.
    If the word is not in the vocab, it is replaced by the index of the <unk> token.
    """

    out_data = []

    rev_vocab = {v: k for k, v in vocab.items()}
    for row in tqdm.tqdm(data):
        while len(row) % k != 0:
            row.append(vocab['[PAD]'])
        
        inputs = np.array_split(row, len(row)/k)

        rev_inputs = []
        for input in inputs:
            for i in range(len(input)):
                rev_inputs.append(rev_vocab[input[i]])
        

        inputs = [input.tolist() for input in inputs]
        
        row.pop(0)
        row.append(vocab['[PAD]'])

        labels = np.array_split(row, len(row)/k)

        for input, label in zip(inputs, labels):
            out_data.append((input, label))

    return out_data

def get_counts(data: List[List[int]], vocab: dict):
    counts = OrderedDict()
    for i in range(len(vocab)):
        counts[i] = 0
    print("Generating Counts from Data")
    for row in tqdm.tqdm(data):
        for i in range(len(row)):
            counts[row[i]] = counts.get(row[i], 0) + 1
    return counts

def main():
    with open ( os.getcwd() + '/A3/data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    if not os.path.exists(os.getcwd() + '/A3/data/preprocessed'):
        os.mkdir(os.getcwd() + '/A3/data/preprocessed')
    raw_train_data = ut.convert_files2idx(ut.get_files(os.getcwd() + '/A3/data/train'), vocab)
    raw_dev_data = ut.convert_files2idx(ut.get_files(os.getcwd() + '/A3/data/dev'), vocab)
    raw_test_data = ut.convert_files2idx(ut.get_files(os.getcwd() + '/A3/data/test'), vocab)

    print("Processing Training Data")
    train_data = preprocess(raw_train_data, vocab, k=500)
    train_dataset = characterDataset(train_data)
    print("Saving Training Data")
    torch.save(train_dataset, os.getcwd() + '/A3/data/preprocessed/train.pt')

    print("Processing Dev Data")
    dev_data = preprocess(raw_dev_data, vocab, k=500)
    dev_dataset = characterDataset(dev_data)
    print("Saving Dev Data")
    torch.save(dev_dataset, os.getcwd() + '/A3/data/preprocessed/dev.pt')

    print("Processing Test Data")
    test_data = preprocess(raw_test_data, vocab, k=500)
    test_dataset = characterDataset(test_data)
    print("Saving Test Data")
    torch.save(test_dataset, os.getcwd() + '/A3/data/preprocessed/test.pt')

if __name__ == "__main__":
    main()