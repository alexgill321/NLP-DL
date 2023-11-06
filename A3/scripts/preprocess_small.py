from utils import characterDataset
from preprocess import preprocess
import os
import utils as ut
import pickle
import torch

def main():
    with open(os.getcwd() + '/A3/data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    
    if not os.path.exists(os.getcwd() + '/A3/data/preprocessed'):
        os.mkdir(os.getcwd() + '/A3/data/preprocessed')

    raw_train_data = ut.convert_files2idx(ut.get_files(os.getcwd() + '/A3/data/train'), vocab)

    print("Processing Training Data")
    train_data = preprocess(raw_train_data[:1000], vocab, k=500)
    train_dataset = characterDataset(train_data)
    print("Saving Training Data")
    torch.save(train_dataset, os.getcwd() + '/A3/data/preprocessed/train_small.pt')

if __name__ == "__main__":
    main()