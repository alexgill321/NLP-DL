import os
import utils as ut
import pickle
import tqdm
import numpy as np

def n_gram(data, vocab, n=4):
    print("Generating Counts from Training Data")
    n_counts = {}
    n_minus_1_counts = {}
    probs = {}
    for row in tqdm.tqdm(data):
        for i in range(n-1):
            row.insert(0, vocab['[PAD]'])
        for i in range(n-1,len(row)):
            n_counts[tuple(row[i-n+1:i+1])] = n_counts.get(tuple(row[i-n+1:i+1]), 0) + 1
            n_minus_1_counts[tuple(row[i-n+1:i])] = n_minus_1_counts.get(tuple(row[i-n+1:i]), 0) + 1
    print("Count of conditional probabilities from training: ", len(n_counts))
    print("Generating Probabilities")
    for key in tqdm.tqdm(n_minus_1_counts.keys()):
        cond_probs = []
        for v in vocab.values():
            if v == vocab['[PAD]']:
                continue
            seq = list(key)
            seq.append(v)
            cond_probs.append((n_counts.get(tuple(seq), 0) + 1) / (n_minus_1_counts.get(tuple(key), 0) + len(vocab)))
        probs[key] = cond_probs
    print("Count of conditional probabilities from full vocab: ", len(probs))
    return probs

def eval_n_gram(test_data, vocab, probs, n=4):
    print("Evaluating Test Data")

    perplexity = []
    for row in tqdm.tqdm(test_data):
        losses = []
        for i in range(n-1):
            row.insert(0, vocab['[PAD]'])
        for i in range(n-1,len(row)):
            lab = np.zeros(len(vocab)-1)
            index = row[i]
            if index > vocab['[PAD]']:
                lab[index-1] = 1
            elif index == vocab['[PAD]']:
                continue
            else:
                lab[index] = 1

            preds = probs.get(tuple(row[i-n+1:i]), np.ones(len(vocab)-1) * 1/len(vocab))
            preds = np.array(preds)
            preds = preds / np.sum(preds)
            losses.append(-np.sum(lab * np.log(preds)))
        perplexity.append(2**np.mean(losses))
    print("Perplexity: ", np.mean(perplexity))
    return np.mean(perplexity)

def main():
    with open(os.getcwd() + '/A3/data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    raw_train_data = ut.convert_files2idx(ut.get_files(os.getcwd() + '/A3/data/train'), vocab)
    raw_test_data = ut.convert_files2idx(ut.get_files(os.getcwd() + '/A3/data/test'), vocab)

    probs = n_gram(raw_train_data, vocab, n=4)

    perp = eval_n_gram(raw_test_data, vocab, probs, n=4)

if __name__ == "__main__":
    main()

        
    